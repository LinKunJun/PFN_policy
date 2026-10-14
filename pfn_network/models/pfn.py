from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import weakref

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .portable import GCNConv, EdgeAdjacency, normalized_adjacency, PortableEncoderLayer
from .incremental_history import IncrementalHistoryCache
from ._ppo_unique import unique_pairs_packed, unique_states_visited
from ..runtime import resolve_device


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = 4
    width: int = 64
    graph_layers: int = 2
    transformer_layers: int = 2
    heads: int = 4
    context_length: int = 24
    joint_head: bool = True
    history: bool = True
    gcn_backend: str = "auto"
    inference_cache_mb: float = 512.0


@dataclass(frozen=True)
class FactualRecord:
    state: np.ndarray
    global_state: np.ndarray
    allocation: np.ndarray
    exposure: np.ndarray
    reward: float
    next_state: np.ndarray
    next_global: np.ndarray
    done: bool
    episode: int
    time: int

    @classmethod
    def from_step(cls, obs, action, reward, next_obs, done, episode):
        return cls(obs["pfn_state"], obs["global_state"], obs["catalog"][action],
                   obs["catalog_exact_exposure"][action], float(reward),
                   next_obs["pfn_state"], next_obs["global_state"], bool(done),
                   episode, obs["episode_time"])


def _stack_catalog(entries, name, max_n, max_m):
    """Preserve catalog stacking order, shape, dtype and padding semantics."""
    tensors = []
    for entry in entries:
        if name == 'scalars':
            pad = (0, 0, 0, max_m - entry['m'])
        else:
            pad = (0, max_n - entry['n'], 0, max_m - entry['m'])
        tensor = entry[name]
        if any(pad):
            tensor = F.pad(tensor, pad)
        tensors.append(tensor)
    return torch.stack(tensors)


class PFNPolicy(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.gcn_backend not in ('auto', 'dense', 'edge'):
            raise ValueError('gcn_backend must be auto, dense, or edge')
        if not math.isfinite(config.inference_cache_mb) or config.inference_cache_mb <= 0:
            raise ValueError('inference_cache_mb must be positive and finite')
        self._static_cache = OrderedDict()
        self._inference_active = False
        self._inference_bank = None
        self._inference_key_memo = None
        self._incremental_history_cache = None
        self._position_cache = None
        self._causal_cache = None
        self._inference_cache_limit = 512
        self._static_cache_limit = 32
        h = config.width
        self.input = nn.Sequential(nn.Linear(config.feature_dim + 11, h), nn.ReLU(), nn.Linear(h, h))
        self.global_input = nn.Linear(7, h)
        self.gnn = nn.ModuleList([GCNConv(h, h, cached=False, normalize=False, add_self_loops=False) for _ in range(config.graph_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(config.graph_layers)])
        self.record_token = nn.Sequential(
            nn.Linear(4 * h + 4, h), nn.GELU(), nn.Linear(h, h))
        self.query_token = nn.Linear(2 * h + 2, h)
        # Separate construction initializes each layer independently. No dropout:
        # old/new PPO probabilities differ only through parameter updates.
        self.transformer = nn.ModuleList([
            PortableEncoderLayer(h, config.heads, 4 * h, dropout=0.0,
                                       batch_first=True, norm_first=True)
            for _ in range(config.transformer_layers)])
        self.fusion = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.LayerNorm(h))
        self.node_logit = nn.Linear(h, 1)
        self.action_head = nn.Sequential(nn.Linear(5 * h + 3, h), nn.GELU(), nn.Linear(h, 1))
        self.value_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
    @contextmanager
    def _rollout_range(self, name):
        if getattr(self, '_rollout_trace_active', False):
            with torch.profiler.record_function(name):
                yield
        else:
            yield

    @property
    def device(self):
        return next(self.parameters()).device

    def _tensor(self, value, dtype=torch.float32):
        # as_tensor avoids a second host copy; immutable inputs are never mutated.
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._static_cache.clear()
        self._inference_active = False
        self._inference_bank = None
        self._inference_key_memo = None
        self._incremental_history_cache = None
        self._position_cache = None
        self._causal_cache = None
        return result

    @contextmanager
    def cached_inference(self):
        """Cache parameter-independent inference outputs only while weights are frozen."""
        previous_active = self._inference_active
        previous_bank = self._inference_bank
        previous_keys = self._inference_key_memo
        previous_history = self._incremental_history_cache
        self._inference_active = True
        self._inference_bank = None
        self._inference_key_memo = {}
        self._incremental_history_cache = (
            IncrementalHistoryCache(self.config.context_length)
            if self.config.history and self.config.context_length > 0 else None)
        try:
            yield
        finally:
            self._inference_active = previous_active
            self._inference_bank = previous_bank
            self._inference_key_memo = previous_keys
            self._incremental_history_cache = previous_history

    def _normalized_adjacency(self, edges, count):
        return normalized_adjacency(edges, count)

    def _static(self, obs):
        # Catalog references prevent Python object-id reuse while an entry is cached.
        # Only public, parameter-independent tensors are retained between forwards.
        key = (id(obs["catalog"]), obs["edge_index"].tobytes())
        if key in self._static_cache:
            self._static_cache.move_to_end(key)
            return key, self._static_cache[key]
        z = np.asarray(obs["catalog"], dtype=np.float32)
        e = np.asarray(obs["catalog_exact_exposure"], dtype=np.float32)
        s = np.asarray(obs["catalog_majority_exposure"], dtype=np.float32)
        edges = obs["edge_index"]
        edge_fraction = (z[:, edges[0]] * z[:, edges[1]]).sum(1) / max(1, edges.shape[1])
        entry = {"key": key, "reference": obs["catalog"], "n": z.shape[1], "m": z.shape[0],
                 "edges": self._tensor(edges, torch.long),
                 "z": self._tensor(z), "e": self._tensor(e), "s": self._tensor(s),
                 "scalars": self._tensor(np.stack((z.mean(1), e.mean(1), edge_fraction), axis=-1))}
        coo = normalized_adjacency(torch.from_numpy(np.asarray(edges).copy()), entry['n'])
        entry['adj_t'] = EdgeAdjacency(coo.row.to(self.device), coo.col.to(self.device),
                                       coo.value.to(self.device), coo.count)
        if self._dense_graphs(entry['n']):
            adjacency = np.eye(entry['n'], dtype=np.float32)
            adjacency[edges[1], edges[0]] = 1
            inv = adjacency.sum(1).astype(np.float32) ** -0.5
            entry['dense_cpu'] = inv[:, None] * adjacency * inv[None, :]
            entry['dense'] = self._tensor(entry['dense_cpu'])
        self._static_cache[key] = entry
        while len(self._static_cache) > self._static_cache_limit:
            self._static_cache.popitem(last=False)
        return key, entry

    def _dense_graphs(self, max_n):
        return self.config.gcn_backend == 'dense' or (self.config.gcn_backend == 'auto' and max_n <= 128)

    def _encode_unique(self, states, globals_, graph_slots, static):
        """Use one contiguous inference GPU bank; never cache gradient-bearing outputs.

        For variable-node graphs or an insufficient inference budget, directly
        compute the same features without caching. This is a shape/budget
        constraint, not a selectable legacy optimization implementation.
        """
        if not self._inference_active or torch.is_grad_enabled():
            return self._encode_uncached(states, globals_, graph_slots, static)
        packed = self._encode_unique_bank(states, globals_, graph_slots, static)
        return packed if packed is not None else self._encode_uncached(
            states, globals_, graph_slots, static)

    def _bank_content_key(self, state, global_state, static_key):
        """Reuse a content key only for stable, owning, read-only CPU arrays.

        The baseline creates state.tobytes() on every rollout step. In the
        optional sealed-observation scope these arrays cannot change and are
        retained as separate snapshots; identity lookup may safely reuse the
        *same original content key*. Writable or non-owning inputs always use
        the exact baseline content-key path. Weakrefs avoid Python id reuse.
        """
        memo = self._inference_key_memo
        if (memo is None or not isinstance(state, np.ndarray)
                or not isinstance(global_state, np.ndarray)
                or state.flags.writeable or global_state.flags.writeable
                or not state.flags.owndata or not global_state.flags.owndata):
            return (static_key, state.tobytes(), global_state.tobytes())
        identity = (static_key, id(state), id(global_state))
        existing = memo.get(identity)
        if (existing is not None and existing[0]() is state
                and existing[1]() is global_state):
            return existing[2]
        key = (static_key, state.tobytes(), global_state.tobytes())
        memo[identity] = (weakref.ref(state), weakref.ref(global_state), key)
        # Limit Python-side bookkeeping even for extremely long rollouts.
        if len(memo) > max(65536, 4 * self._inference_cache_limit):
            memo.clear()
        return key

    def _encode_unique_bank(self, states, globals_, graph_slots, static):
        """Reuse one contiguous inference-only GPU bank instead of concatenating
        thousands of tiny per-state cache tensors on *every* rollout step.

        Fixed-size graphs are the common case. Irregular shapes use the
        general non-cached encoder instead of a fixed-node bank.
        Model weights must stay fixed throughout cached_inference().
        """
        if not states:
            return None
        nodes = len(states[0])
        if (nodes == 0 or any(len(state) != nodes for state in states)
                or self._inference_cache_limit <= 0):
            return None

        # Stable content identities allow safe reuse while parameters are frozen.
        first = {}
        keys = []
        for index, (state, global_state, slot) in enumerate(zip(states, globals_, graph_slots)):
            key = self._bank_content_key(state, global_state, static[slot]['key'])
            keys.append(key)
            if key not in first:
                first[key] = index
        if len(first) > self._inference_cache_limit:
            return None  # All live keys must fit so an eviction cannot corrupt a gather.

        bank = self._inference_bank
        if bank is None:
            capacity = int(self._inference_cache_limit)
            width = self.config.width
            dtype = next(self.parameters()).dtype
            bank = {
                'nodes_per_state': nodes,
                'capacity': capacity,
                'nodes': torch.empty((capacity, nodes, width),
                                     dtype=dtype, device=self.device),
                'pooled': torch.empty((capacity, width),
                                      dtype=dtype, device=self.device),
                'lru': OrderedDict(),
                'free': list(range(capacity - 1, -1, -1)),
            }
            self._inference_bank = bank
        if bank['nodes_per_state'] != nodes or len(first) > bank['capacity']:
            return None

        lru = bank['lru']
        # Move all live hits to the newest end *before* choosing eviction slots.
        for key in first:
            if key in lru:
                lru.move_to_end(key)
        missing = [key for key in first if key not in lru]
        if missing:
            required = set(first)
            slots = []
            for _ in missing:
                if bank['free']:
                    slot = bank['free'].pop()
                else:
                    old_key, slot = lru.popitem(last=False)
                    if old_key in required:
                        # Defensive recovery if a future change breaks the
                        # capacity invariant; never return corrupted features.
                        self._inference_bank = None
                        return None
                slots.append(slot)
            ids = [first[key] for key in missing]
            new_nodes, new_pooled, _, _ = self._encode_uncached(
                [states[i] for i in ids], [globals_[i] for i in ids],
                [graph_slots[i] for i in ids], static)
            slot_tensor = self._tensor(np.asarray(slots, dtype=np.int64), torch.long)
            bank['nodes'].index_copy_(
                0, slot_tensor, new_nodes.reshape(len(ids), nodes, self.config.width))
            bank['pooled'].index_copy_(0, slot_tensor, new_pooled)
            for key, slot in zip(missing, slots):
                lru[key] = slot

        indices = self._tensor(
            np.fromiter((lru[key] for key in keys), dtype=np.int64, count=len(keys)),
            torch.long)
        flat = bank['nodes'].index_select(0, indices).reshape(-1, self.config.width)
        pooled = bank['pooled'].index_select(0, indices)
        counts = np.full(len(keys), nodes, dtype=np.int64)
        offsets = np.arange(len(keys), dtype=np.int64) * nodes
        return flat, pooled, offsets, counts

    def _encode_uncached(self, states, globals_, graph_slots, static):
        counts_np = np.array([len(state) for state in states], dtype=np.int64)
        offsets_np = np.r_[0, np.cumsum(counts_np)[:-1]]
        # Default graphs have 30 nodes. Small batched GEMMs avoid edge-message
        # tensors and atomic scatter reductions. No dense block diagonal is built.
        if self._dense_graphs(int(counts_np.max())):
            max_n = int(counts_np.max())
            raw = np.zeros((len(states), max_n, states[0].shape[1]), dtype=np.float32)
            for row, state in enumerate(states):
                raw[row, :len(state)] = state
            adjacency = torch.stack([
                static[slot]['dense'] if static[slot]['n'] == max_n
                else F.pad(static[slot]['dense'],
                           (0, max_n-static[slot]['n'], 0, max_n-static[slot]['n']))
                for slot in graph_slots])
            x = self.input(self._tensor(raw))
            for conv, norm in zip(self.gnn, self.norms):
                x = norm(x + F.relu(torch.bmm(adjacency, conv.lin(x)) + conv.bias))
            if np.all(counts_np == max_n):
                pooled = x.mean(1)
                flat = x.reshape(-1, self.config.width)
            else:
                mask = self._tensor(np.arange(max_n)[None] < counts_np[:, None], torch.bool)
                pooled = (x * mask[:, :, None]).sum(1) / self._tensor(counts_np)[:, None]
                indices = np.concatenate([np.arange(size) + row * max_n for row, size in enumerate(counts_np)])
                flat = x.reshape(-1, self.config.width)[self._tensor(indices, torch.long)]
            pooled = pooled + self.global_input(self._tensor(np.stack(globals_)))
            return flat, pooled, offsets_np, counts_np
        offsets = self._tensor(offsets_np, torch.long)
        counts = self._tensor(counts_np, torch.long)
        x = self.input(self._tensor(np.concatenate(states, axis=0)))
        graph_slots = np.asarray(graph_slots)
        rows, columns, weights = [], [], []
        for slot, entry in enumerate(static):
            ids = self._tensor(np.flatnonzero(graph_slots == slot), torch.long)
            row, col, value = entry['adj_t'].coo()
            rows.append((row[None] + offsets[ids, None]).reshape(-1))
            columns.append((col[None] + offsets[ids, None]).reshape(-1))
            weights.append(value[None].expand(ids.numel(), -1).reshape(-1))
        adjacency = EdgeAdjacency(torch.cat(rows), torch.cat(columns), torch.cat(weights), x.shape[0])
        for conv, norm in zip(self.gnn, self.norms):
            x = norm(x + F.relu(conv(x, adjacency)))
        membership = torch.repeat_interleave(torch.arange(len(states), device=self.device), counts,
                                              output_size=x.shape[0])
        pooled = x.new_zeros((len(states), self.config.width)).index_add(0, membership, x)
        pooled = pooled / counts[:, None]
        pooled = pooled + self.global_input(self._tensor(np.stack(globals_)))
        return x, pooled, offsets_np, counts_np

    def _position(self, length, width):
        if self._position_cache is not None and self._position_cache.shape[0] >= length:
            return self._position_cache[:length]
        requested = length
        length = max(length, self.config.context_length + 1)
        pos = torch.arange(length, device=self.device)[:, None]
        freq = torch.exp(torch.arange(0, width, 2, device=self.device) * (-math.log(10000.0) / width))
        encoding = torch.zeros(length, width, device=self.device)
        encoding[:, 0::2] = torch.sin(pos * freq)
        encoding[:, 1::2] = torch.cos(pos * freq[:width // 2])
        self._position_cache = encoding
        return encoding[:requested]

    def _causal(self, length):
        if self._causal_cache is None or self._causal_cache.shape[0] < length:
            capacity = max(length, self.config.context_length + 1)
            self._causal_cache = torch.ones(capacity, capacity, device=self.device, dtype=torch.bool).triu(1)
        return self._causal_cache[:length, :length]

    def _forward_many(self, observations, histories, episodes, history_slots=None):
        batch = len(observations)
        limit = self.config.context_length
        histories = [h[-limit:] if self.config.history and limit > 0 else () for h in histories]
        self._static_cache_limit = max(32, 2 * batch)
        # A 64-task, 384-record rollout needs thousands of states. The old fixed
        # 512-entry LRU thrashed. Bound encoded features by an explicit MiB budget.
        max_n = max(len(obs['pfn_state']) for obs in observations)
        entry_bytes = (max_n + 1) * self.config.width * next(self.parameters()).element_size()
        budget_entries = max(0, int(self.config.inference_cache_mb * 1024**2 / entry_bytes))
        required = max(512, batch * ((limit if self.config.history else 0) + 2))
        self._inference_cache_limit = min(required, budget_entries)
        static, static_ids, slots = [], {}, []
        with self._rollout_range('PFN/CPU_static_catalogs'):
            for obs in observations:
                if not np.any(obs["valid_action_mask"]):
                    raise ValueError("A terminal observation has no action distribution.")
                key, entry = self._static(obs)
                if key not in static_ids:
                    static_ids[key] = len(static)
                    static.append(entry)
                slots.append(static_ids[key])

        # Reuse identical state objects within this forward only. Gradients from
        # all their uses accumulate through the same current-parameter GNN graph.
        states, globals_, graph_slots, state_ids = [], [], [], {}
        # Inference-only packing avoids affecting gradient-enabled forward paths.
        inference_only = not torch.is_grad_enabled()
        def register(state, global_state, slot):
            key = (id(state), id(global_state), slot)
            existing = state_ids.get(key)
            if existing is not None:
                return existing
            index = len(states)
            state_ids[key] = index
            states.append(state)
            globals_.append(global_state)
            graph_slots.append(slot)
            return index

        current, before, after, records, feedback = [], [], [], [], []
        incremental_pack = None
        with self._rollout_range('PFN/CPU_build_histories'):
            if (history_slots is not None and self._incremental_history_cache is not None
                    and not torch.is_grad_enabled()):
                incremental_pack = self._incremental_history_cache.pack(
                    observations, histories, episodes, history_slots, static, slots,
                    max(entry['n'] for entry in static))
            if incremental_pack is not None:
                # All older FactualRecords have been incrementally indexed and
                # packed in bounded per-task rings. Keep the exact original
                # record order, time feedback, and per-forward state dedup.
                states = incremental_pack['states']
                globals_ = incremental_pack['globals']
                graph_slots = incremental_pack['graph_slots']
                current = incremental_pack['current']
                before = incremental_pack['before']
                after = incremental_pack['after']
                feedback = incremental_pack['feedback']
                record_count = len(before)
            else:
                for obs, history, episode, slot in zip(observations, histories, episodes, slots):
                    current.append(register(obs["pfn_state"], obs["global_state"], slot))
                    for record in history:
                        before.append(register(record.state, record.global_state, slot))
                        after.append(register(record.next_state, record.next_global, slot))
                        records.append(record)
                        scalars = [record.reward, record.done, record.time / 20,
                                   (record.episode - episode) / 10]
                        feedback.append(scalars)
                record_count = len(records)
        with self._rollout_range('PFN/GNN_encode_unique'):
            x, pooled, offsets, counts = self._encode_unique(states, globals_, graph_slots, static)
        max_n = max(entry["n"] for entry in static)
        max_m = max(entry["m"] for entry in static)
        # All rollout graphs have the same number of nodes in the common case.
        # The original node_indices() builds an N x max_n int64 tensor via a
        # Python loop, even when every row is simply a consecutive node range.
        # For inference only, reshape the already ordered flat GNN output, then
        # select by *state*; retain the original gather in gradient-enabled code
        # so no changes to backpropagation or its accumulation order are possible.
        # index. This transfers N indices rather than N*max_n indices, without
        # changing any node values or trainable computation.
        uniform_nodes = bool(len(counts) and np.all(counts == max_n))
        if uniform_nodes and not torch.is_grad_enabled():
            grouped_nodes = x.reshape(len(states), max_n, self.config.width)

            def gather_nodes(ids):
                return grouped_nodes.index_select(
                    0, self._tensor(np.asarray(ids, dtype=np.int64), torch.long))
        else:
            sentinel = x.shape[0]
            x = torch.cat((x, x.new_zeros((1, self.config.width))), dim=0)

            def gather_nodes(ids):
                indices = np.full((len(ids), max_n), sentinel, dtype=np.int64)
                for row, index in enumerate(ids):
                    indices[row, :counts[index]] = offsets[index] + np.arange(counts[index])
                return x[self._tensor(indices, torch.long)]

        current_ids = self._tensor(current, torch.long)
        with self._rollout_range('PFN/Gather_current_nodes'):
            current_nodes = gather_nodes(current)
        n = self._tensor(counts[current])
        node_mask = torch.arange(max_n, device=self.device)[None] < n[:, None]
        current_mean = current_nodes.sum(1) / n[:, None]
        times = self._tensor([[obs["episode_time"] / 20, 0.0] for obs in observations])
        query = self.query_token(torch.cat((pooled[current_ids], current_mean, times), dim=-1))
        if record_count:
            if incremental_pack is not None:
                allocation = incremental_pack['allocations']
                exposure = incremental_pack['exposures']
            else:
                allocation = None
                if inference_only and uniform_nodes:
                    # For standard fixed-size graphs, convert whole columns in
                    # NumPy instead of doing two Python slice assignments per row.
                    # Check every length so irregular inputs retain baseline rules.
                    allocations = [record.allocation for record in records]
                    exposures = [record.exposure for record in records]
                    if all(len(a) == max_n and len(e) == max_n
                           for a, e in zip(allocations, exposures)):
                        allocation = np.asarray(allocations, dtype=np.float32)
                        exposure = np.asarray(exposures, dtype=np.float32)
                if allocation is None:
                    allocation = np.zeros((record_count, max_n), dtype=np.float32)
                    exposure = np.zeros_like(allocation)
                    for row, record in enumerate(records):
                        size = len(record.allocation)
                        allocation[row, :size] = record.allocation
                        exposure[row, :size] = record.exposure
            with self._rollout_range('PFN/Gather_history_nodes'):
                past_nodes = gather_nodes(before)
            sizes = self._tensor(counts[before])[:, None]
            treated = (past_nodes * self._tensor(allocation)[:, :, None]).sum(1) / sizes
            exposed = (past_nodes * self._tensor(exposure)[:, :, None]).sum(1) / sizes
            history_tokens = self.record_token(torch.cat((
                pooled[self._tensor(before, torch.long)], treated, exposed,
                pooled[self._tensor(after, torch.long)],
                self._tensor(np.asarray(feedback, dtype=np.float32) if inference_only else feedback)), dim=-1))
        else:
            history_tokens = query.new_zeros((0, self.config.width))

        lengths = (incremental_pack['lengths'] if incremental_pack is not None else
                   np.array([len(history) for history in histories], dtype=np.int64))
        max_length = int(lengths.max()) + 1
        all_tokens = torch.cat((history_tokens, query, query.new_zeros((1, self.config.width))), dim=0)
        token_indices = np.full((batch, max_length), record_count + batch, dtype=np.int64)
        cursor = 0
        for row, length in enumerate(lengths):
            token_indices[row, :length] = np.arange(cursor, cursor + length)
            token_indices[row, length] = record_count + row
            cursor += length
        seq = all_tokens[self._tensor(token_indices, torch.long)]
        seq = seq + self._position(max_length, self.config.width)[None]
        lengths_tensor = self._tensor(lengths, torch.long)
        # Right padding leaves every valid token's original position unchanged.
        padding = torch.arange(max_length, device=self.device)[None] > lengths_tensor[:, None]
        causal = self._causal(max_length)
        with self._rollout_range('PFN/Transformer'):
            for layer in self.transformer:
                seq = layer(seq, src_mask=causal, src_key_padding_mask=padding)
        global_history = seq[torch.arange(batch, device=self.device), lengths_tensor]
        nodes = self.fusion(torch.cat((current_nodes, global_history[:, None].expand(-1, max_n, -1)), dim=-1))
        nodes = nodes * node_mask[:, :, None]
        raw = self.node_logit(nodes).squeeze(-1)

        # Pad each distinct catalog once, then select it for all batch members.
        slot_tensor = self._tensor(slots, torch.long)
        def catalog_tensor(name):
            bank = _stack_catalog(static, name, max_n, max_m)
            return bank[slot_tensor]
        z = catalog_tensor("z")
        available_nodes = node_mask[:, None].to(z.dtype)
        logits = (z * F.logsigmoid(raw[:, None]) +
                  (available_nodes - z) * F.logsigmoid(-raw[:, None])).sum(-1)
        if self.config.joint_head:
            e, s = catalog_tensor("e"), catalog_tensor("s")
            scalars = _stack_catalog(static, 'scalars', max_n, max_m)[slot_tensor]
            normalizer = n[:, None, None]
            joint = torch.cat((torch.bmm(z, nodes) / normalizer,
                               torch.bmm(available_nodes - z, nodes) / normalizer,
                               torch.bmm(e, nodes) / normalizer,
                               torch.bmm(s, nodes) / normalizer,
                               global_history[:, None].expand(-1, max_m, -1), scalars), dim=-1)
            logits = logits + self.action_head(joint).squeeze(-1)
        masks = np.zeros((batch, max_m), dtype=bool)
        for row, obs in enumerate(observations):
            masks[row, :len(obs["valid_action_mask"])] = obs["valid_action_mask"]
        logits = logits.masked_fill(~self._tensor(masks, torch.bool), -1e9)
        values = self.value_head(torch.cat((pooled[current_ids], global_history), dim=-1)).squeeze(-1)
        return logits, values

    def prepare_rollout(self, rollout):
        """Pack each unique factual record once; histories use integer indices."""
        observations = rollout.observations[:len(rollout)]
        use_history = self.config.history and self.config.context_length > 0
        lengths = rollout.history_lengths[:len(rollout)] if use_history else np.zeros(len(rollout), dtype=np.int64)
        length = int(lengths.max())
        history_ids = rollout.history_indices[:len(rollout), :length]
        entries, graph_edges, slots, catalog_ids = [], [], [], {}
        def catalog_slot(obs):
            key, entry = self._static(obs)
            if key not in catalog_ids:
                catalog_ids[key] = len(entries)
                entries.append(entry)
                graph_edges.append(obs['edge_index'])
            return catalog_ids[key]
        for obs in observations:
            slots.append(catalog_slot(obs))
        max_n = max(e['n'] for e in entries)
        max_m = max(e['m'] for e in entries)
        states, globals_, state_slots, state_ids = [], [], [], {}
        def register(state, global_state, slot):
            key = (id(state), id(global_state), slot)
            if key not in state_ids:
                state_ids[key] = len(states)
                states.append(state)
                globals_.append(global_state)
                state_slots.append(slot)
            return state_ids[key]
        current = [register(obs['pfn_state'], obs['global_state'], slot)
                   for obs, slot in zip(observations, slots)]
        records = rollout.records[:rollout.record_count] if length else []
        before, after = np.zeros(len(records), dtype=np.int64), np.zeros(len(records), dtype=np.int64)
        allocation = np.zeros((len(records), max_n), dtype=np.float32)
        exposure = np.zeros_like(allocation)
        feedback = np.zeros((len(records), 4), dtype=np.float32)
        record_episodes = np.zeros(len(records), dtype=np.int64)
        for row, record in enumerate(records):
            slot = catalog_slot(rollout.record_observations[row])
            before[row] = register(record.state, record.global_state, slot)
            after[row] = register(record.next_state, record.next_global, slot)
            n = len(record.allocation)
            allocation[row, :n] = record.allocation
            exposure[row, :n] = record.exposure
            feedback[row, :3] = record.reward, record.done, record.time / 20
            record_episodes[row] = record.episode
        raw = np.zeros((len(states), max_n, states[0].shape[1]), dtype=np.float32)
        for row, state in enumerate(states):
            raw[row, :len(state)] = state
        masks = np.zeros((len(observations), max_m), dtype=bool)
        for row, obs in enumerate(observations):
            masks[row, :len(obs['valid_action_mask'])] = obs['valid_action_mask']
        bank = {name: _stack_catalog(entries, name, max_n, max_m)
                for name in ('z', 'e', 's')}
        bank['scalars'] = _stack_catalog(entries, 'scalars', max_n, max_m)
        dense = None
        if self._dense_graphs(max_n):
            dense = self._tensor(np.stack([
                e['dense_cpu'] if e['n'] == max_n
                else np.pad(e['dense_cpu'], ((0, max_n-e['n']), (0, max_n-e['n'])))
                for e in entries]))
        return dict(raw=self._tensor(raw), globals=self._tensor(np.stack(globals_)),
                    state_slots_cpu=np.asarray(state_slots), counts=self._tensor([len(s) for s in states]),
                    current_cpu=np.asarray(current), before_cpu=before, after_cpu=after,
                    history_ids_cpu=history_ids.copy(), lengths_cpu=lengths.copy(),
                    episodes_cpu=rollout.episodes[:len(rollout)].copy(),
                    allocations=self._tensor(allocation), exposures=self._tensor(exposure),
                    feedback=self._tensor(feedback), record_episodes=record_episodes,
                    lengths=self._tensor(lengths, torch.long),
                    times=self._tensor(np.column_stack((rollout.times[:len(rollout)] / 20,
                                                         np.zeros(len(rollout))))),
                    slots=self._tensor(slots, torch.long), masks=self._tensor(masks, torch.bool),
                    graph_edges_cpu=graph_edges, bank=bank, length=length, dense=dense,
                    position=self._position(length + 1, self.config.width),
                    causal=self._causal(length + 1))

    def prepare_minibatches(self, data, orders, minibatch):
        """Retain small CPU orders only; graph plans enter the device one batch at a time."""
        if minibatch <= 0:
            raise ValueError('minibatch must be positive')
        return [MinibatchEpoch(self, data, order, minibatch) for order in orders]

    def _prepare_minibatch(self, data, ids):
        lengths = data['lengths_cpu'][ids]
        length = int(lengths.max())
        batch = len(ids)
        valid = np.arange(length)[None] < lengths[:, None]
        history = data['history_ids_cpu'][ids, :length]
        # A record token depends on the current episode as well as the record.
        # Deduplication therefore uses both fields, never just the record id.
        pairs, token_inverse = unique_pairs_packed(
            history[valid], data['episodes_cpu'][ids], lengths,
            len(data['before_cpu']))
        records = pairs[:, 0]
        references = np.concatenate((data['current_cpu'][ids], data['before_cpu'][records],
                                     data['after_cpu'][records]))
        states, inverse = unique_states_visited(
            references, len(data['state_slots_cpu']))
        number = len(records)
        indices = np.full((batch, length + 1), number + batch, dtype=np.int64)
        indices[:, :length][valid] = token_inverse
        indices[np.arange(batch), lengths] = number + np.arange(batch)
        slots = data['state_slots_cpu'][states]
        arrays = dict(ids=ids, states=states, current=inverse[:batch], before=inverse[batch:batch+number],
                      after=inverse[batch+number:], records=records, token_indices=indices, state_slots=slots)
        integers = self._tensor(np.concatenate([array.reshape(-1) for array in arrays.values()]), torch.long)
        plan, cursor = {}, 0
        for name, array in arrays.items():
            plan[name] = integers[cursor:cursor + array.size].reshape(array.shape)
            cursor += array.size
        plan.update(episode_delta=self._tensor((data['record_episodes'][records] - pairs[:, 1]) / 10),
                    length=length, has_history=number > 0)
        if data['dense'] is None:
            # Normalization is performed on CPU, avoiding dynamic boolean indexing
            # and its device synchronization while constructing a graph plan.
            max_n = data['raw'].shape[1]
            edges = np.concatenate([data['graph_edges_cpu'][slot] + row * max_n
                                    for row, slot in enumerate(slots)], axis=1)
            adjacency = normalized_adjacency(torch.from_numpy(edges), len(states) * max_n)
            plan['adj_t'] = EdgeAdjacency(adjacency.row.to(self.device), adjacency.col.to(self.device),
                                          adjacency.value.to(self.device), adjacency.count)
        return plan

    def forward_prepared(self, data, plan, return_logits=False):
        """Recompute all GNN/Transformer features under the current parameters."""
        ids, state_ids = plan['ids'], plan['states']
        raw = data['raw'][state_ids]
        count, max_n, _ = raw.shape
        x = self.input(raw)
        if data['dense'] is not None:
            adjacency = data['dense'][plan['state_slots']]
            for conv, norm in zip(self.gnn, self.norms):
                x = norm(x + F.relu(torch.bmm(adjacency, conv.lin(x)) + conv.bias))
        else:
            x = x.reshape(-1, self.config.width)
            for conv, norm in zip(self.gnn, self.norms):
                x = norm(x + F.relu(conv(x, plan['adj_t'])))
            x = x.reshape(count, max_n, self.config.width)
        counts = data['counts'][state_ids]
        node_masks = torch.arange(max_n, device=self.device)[None] < counts[:, None]
        x = x * node_masks[:, :, None]
        means = x.sum(1) / counts[:, None]
        pooled = means + self.global_input(data['globals'][state_ids])
        current = plan['current']
        current_nodes, n = x[current], counts[current]
        query = self.query_token(torch.cat((pooled[current], means[current], data['times'][ids]), dim=-1))
        length, batch = plan['length'], ids.shape[0]
        before, after = plan['before'], plan['after']
        if plan['has_history']:
            past = x[before]
            record_ids = plan['records']
            treated = (past * data['allocations'][record_ids][..., None]).sum(1) / counts[before][..., None]
            exposed = (past * data['exposures'][record_ids][..., None]).sum(1) / counts[before][..., None]
            feedback = data['feedback'][record_ids].clone()
            feedback[:, 3] = plan['episode_delta']
            tokens = self.record_token(torch.cat((pooled[before], treated, exposed,
                                                 pooled[after], feedback), dim=-1))
        else:
            tokens = query.new_zeros((0, self.config.width))
        lengths = data['lengths'][ids]
        positions = torch.arange(length + 1, device=self.device)[None]
        bank = torch.cat((tokens, query, query.new_zeros((1, self.config.width))))
        seq = bank[plan['token_indices']] + data['position'][None, :length + 1]
        for layer in self.transformer:
            seq = layer(seq, src_mask=data['causal'][:length + 1, :length + 1],
                        src_key_padding_mask=positions > lengths[:, None])
        history = seq[torch.arange(batch, device=self.device), lengths]
        nodes = self.fusion(torch.cat((current_nodes, history[:, None].expand(-1, max_n, -1)), dim=-1))
        nodes = nodes * node_masks[current, :, None]
        raw_logits = self.node_logit(nodes).squeeze(-1)
        slots = data['slots'][ids]
        z = data['bank']['z'][slots]
        available = node_masks[current][:, None].to(z.dtype)
        logits = (z * F.logsigmoid(raw_logits[:, None]) +
                  (available - z) * F.logsigmoid(-raw_logits[:, None])).sum(-1)
        if self.config.joint_head:
            e, s = data['bank']['e'][slots], data['bank']['s'][slots]
            normalizer = n[:, None, None]
            joint = torch.cat((torch.bmm(z, nodes) / normalizer,
                               torch.bmm(available - z, nodes) / normalizer,
                               torch.bmm(e, nodes) / normalizer,
                               torch.bmm(s, nodes) / normalizer,
                               history[:, None].expand(-1, z.shape[1], -1),
                               data['bank']['scalars'][slots]), dim=-1)
            logits = logits + self.action_head(joint).squeeze(-1)
        logits = logits.masked_fill(~data['masks'][ids], -1e9)
        values = self.value_head(torch.cat((pooled[current], history), dim=-1)).squeeze(-1)
        if return_logits:
            # DDP must see every parameter even on an empty-history rank. These
            # exact-zero edges expose no observations and do not alter the loss.
            unused = ([self.record_token] if not plan['has_history'] else [])
            if not self.config.joint_head:
                unused.append(self.action_head)
            values = self._ddp_anchor(values, unused)
            return logits, values
        return Categorical(logits=logits, validate_args=False), values

    def _ddp_anchor(self, values, unused):
        for module in unused:
            for parameter in module.parameters():
                values = values + parameter.reshape(-1)[0] * 0.0
        return values

    def forward_batch(self, observations, histories, episodes, history_slots=None):
        logits, values = self._forward_many(
            observations, histories, episodes, history_slots=history_slots)
        return Categorical(logits=logits), values

    def forward(self, obs, history=(), episode=0):
        logits, values = self._forward_many([obs], [history], [episode])
        return Categorical(logits=logits[0]), values[0]

    @torch.no_grad()
    def act_batch(self, observations, histories, episodes, generator=None, history_slots=None):
        with self._rollout_range('PFN/forward_batch'):
            dist, values = self.forward_batch(
                observations, histories, episodes, history_slots=history_slots)
        with self._rollout_range('PFN/CPU_choices'):
            return self._cpu_choices(dist, values, generator)

    def _cpu_choices(self, dist, values, generator=None, deterministic=False):
        if generator is not None and generator.device.type != 'cpu':
            raise ValueError('Policy sampling requires a CPU generator')
        m = dist.probs.shape[-1]
        # One device-to-host boundary. Sampling and log-probability lookup stay
        # on the CPU; no sampled action is uploaded just to download it again.
        with self._rollout_range('PFN/GPU_to_CPU_copy_and_sync'):
            packed = torch.cat((dist.probs, dist.logits, values[:, None]), dim=-1).cpu()
        probabilities, log_probs = packed[:, :m], packed[:, m:2*m]
        actions = log_probs.argmax(-1) if deterministic else torch.multinomial(
            probabilities, 1, generator=generator).squeeze(-1)
        selected = log_probs.gather(1, actions[:, None]).squeeze(-1)
        return torch.stack((actions.to(values.dtype), selected, packed[:, -1]), dim=-1).tolist()

    @torch.no_grad()
    def act(self, obs, history=(), episode=0, deterministic=False, generator=None):
        dist, values = self.forward_batch([obs], [history], [episode])
        action_value, log_prob, state_value = self._cpu_choices(dist, values, generator, deterministic)[0]
        return int(action_value), log_prob, state_value

    def save(self, path, extra=None):
        torch.save({"model_config": asdict(self.config), "model": self.state_dict(),
                    "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        device = resolve_device(device)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        return model, payload["extra"]


class MinibatchEpoch:
    """Reiterable lazy plans; device memory is bounded by the current minibatch."""
    def __init__(self, model, data, order, minibatch):
        self.model, self.data = model, data
        self.order, self.minibatch = np.asarray(order), minibatch

    def __len__(self):
        return (len(self.order) + self.minibatch - 1) // self.minibatch

    def __iter__(self):
        for start in range(0, len(self.order), self.minibatch):
            yield self.model._prepare_minibatch(self.data, self.order[start:start + self.minibatch])
