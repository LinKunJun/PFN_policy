"""Inference-only incremental factual-history assembly for Sequential Rollout.

The model still reevaluates all historical tokens through its Transformer; only
CPU preparation of immutable FactualRecord metadata and state identity is
incremental. Never cache model outputs (which would become stale after PPO).
"""

import numpy as np


class _TaskWindow:
    """Bounded, insertion-order ring holding one task's recent factual records."""

    def __init__(self, limit, node_count, graph_key):
        self.limit = limit
        self.node_count = node_count
        self.graph_key = graph_key
        self.start = 0
        self.length = 0
        self.refs = [None] * limit
        self.before = np.empty(limit, dtype=np.int64)
        self.after = np.empty(limit, dtype=np.int64)
        self.allocation = np.zeros((limit, node_count), dtype=np.float32)
        self.exposure = np.zeros((limit, node_count), dtype=np.float32)
        self.rewards = np.empty(limit, dtype=np.float32)
        self.done = np.empty(limit, dtype=np.float32)
        self.times = np.empty(limit, dtype=np.float32)
        self.episodes = np.empty(limit, dtype=np.int64)

    def _ref(self, i):
        return self.refs[(self.start + i) % self.limit]

    def reset(self):
        self.start = 0
        self.length = 0
        # Drop strong references to outdated record objects.
        self.refs[:] = [None] * self.limit

    def append(self, record, register):
        size = len(record.allocation)
        if size > self.node_count or len(record.exposure) > self.node_count:
            raise ValueError('Record size exceeds task node count')
        if self.length == self.limit:
            slot = self.start
            self.start = (self.start + 1) % self.limit
        else:
            slot = (self.start + self.length) % self.limit
            self.length += 1
        self.refs[slot] = record
        self.before[slot] = register(record.state, record.global_state, self.graph_key)
        self.after[slot] = register(record.next_state, record.next_global, self.graph_key)
        self.allocation[slot, :] = 0
        self.exposure[slot, :] = 0
        self.allocation[slot, :size] = record.allocation
        self.exposure[slot, :size] = record.exposure
        self.rewards[slot] = record.reward
        self.done[slot] = record.done
        self.times[slot] = record.time / 20
        self.episodes[slot] = record.episode

    def sync(self, history, register, stats):
        """On normal collection, at most one new record is processed per call."""
        size = len(history)
        if size == self.length:
            if size == 0:
                return
            if history[0] is self._ref(0) and history[-1] is self._ref(size - 1):
                return
            # With a full window, one newly appended record drops the oldest.
            if (size == self.limit and history[-1] is not self._ref(size - 1)
                    and (size == 1 or
                         (history[0] is self._ref(1) and
                          history[-2] is self._ref(size - 1)))):
                self.append(history[-1], register)
                stats['appended'] += 1
                return
        elif (size == self.length + 1 and self.length < self.limit
              and (self.length == 0 or
                   (history[0] is self._ref(0) and history[-2] is self._ref(self.length - 1)))):
            self.append(history[-1], register)
            stats['appended'] += 1
            return
        # Unexpected history changes (non-monotonic, context replacement, etc.)
        # rebuild exactly from the supplied history rather than using stale data.
        self.reset()
        for record in history:
            self.append(record, register)
        stats['rebuilt'] += 1
        stats['rebuild_records'] += size

    def ordered(self, source):
        if not self.length:
            return source[:0]
        end = self.start + self.length
        if end <= self.limit:
            return source[self.start:end]
        return np.concatenate((source[self.start:], source[:end - self.limit]), axis=0)


class IncrementalHistoryCache:
    """Per-rollout CPU cache for incremental history metadata and state IDs."""

    def __init__(self, limit):
        self.limit = limit
        self.windows = {}
        self.registry = {}
        self.states = []
        self.globals = []
        self.graph_keys = []
        self._bank_plan_context = None
        self._bank_plan_size = 0
        self._bank_plan_keys = np.empty(0, dtype=np.int64)
        self._bank_plan_nodes = np.empty(0, dtype=np.int64)
        self.stats = {'appended': 0, 'rebuilt': 0, 'rebuild_records': 0,
                      'calls': 0, 'record_rows': 0}

    def register(self, state, global_state, graph_key):
        identity = (id(state), id(global_state), graph_key)
        existing = self.registry.get(identity)
        if existing is not None:
            return existing
        value = len(self.states)
        self.registry[identity] = value
        # Own strong references: Python object IDs cannot be recycled while cached.
        self.states.append(state)
        self.globals.append(global_state)
        self.graph_keys.append(graph_key)
        return value

    def bank_plan(self, registry_ids, content_key, content_epoch, snapshots):
        """Resolve only newly registered, collector-owned immutable snapshots.

        Object-ID dedup in pack() and content dedup in the GNN bank are distinct.
        Keep both, including first occurrence order. A content-table prune or
        a different sealing scope invalidates the ID table. Untrusted inputs
        cannot skip the original per-forward array/weakref checks.
        """
        if content_epoch is None or snapshots is None:
            return None
        context = self._bank_plan_context
        if context is None or context[0] is not content_epoch or context[1] is not snapshots:
            self._bank_plan_context = (content_epoch, snapshots)
            self._bank_plan_size = 0
        size = len(self.states)
        if size > len(self._bank_plan_keys):
            capacity = max(size, 64, 2 * len(self._bank_plan_keys))
            keys = np.empty(capacity, dtype=np.int64)
            nodes = np.empty(capacity, dtype=np.int64)
            prefix = self._bank_plan_size
            keys[:prefix] = self._bank_plan_keys[:prefix]
            nodes[:prefix] = self._bank_plan_nodes[:prefix]
            self._bank_plan_keys, self._bank_plan_nodes = keys, nodes
        for i in range(self._bank_plan_size, size):
            state, glob = self.states[i], self.globals[i]
            trusted = snapshots.get((id(state), id(glob)))
            if (trusted is not None and trusted[0] is state and trusted[1] is glob
                    and type(state) is np.ndarray and type(glob) is np.ndarray
                    and state.flags.owndata and glob.flags.owndata
                    and not state.flags.writeable and not glob.flags.writeable):
                self._bank_plan_keys[i] = content_key(state, glob, self.graph_keys[i])
                self._bank_plan_nodes[i] = len(state)
            else:
                self._bank_plan_keys[i] = -1
                self._bank_plan_nodes[i] = -1
        self._bank_plan_size = size
        keys = self._bank_plan_keys[registry_ids]
        nodes = self._bank_plan_nodes[registry_ids]
        if not len(keys) or np.any(keys < 0) or nodes[0] <= 0 or np.any(nodes != nodes[0]):
            return None
        # Vectorized first-occurrence content dedup replaces the Python scan.
        unique, first = np.unique(keys, return_index=True)
        order = np.argsort(first)
        representatives = dict(zip(unique[order].tolist(), first[order].tolist()))
        return {'keys': keys.tolist(), 'first': representatives,
                'nodes': int(nodes[0]), 'epoch': content_epoch}

    def pack(self, observations, histories, episodes, task_ids, static, slots, max_n):
        """Return same order/values as _forward_many history preparation.

        When a caller supplies unrecognizable task IDs or record shapes, the
        caller can safely take its original path. Do not mutate rollout records.
        """
        batch = len(observations)
        if (self.limit <= 0 or len(task_ids) != batch or
                len(set(task_ids)) != batch or len(histories) != batch):
            return None
        # Unsupported record shapes are checked only when a NEW record is
        # appended to its task ring. Rescanning history here would defeat the
        # purpose of this incremental cache.
        current_rows, before_rows, after_rows = [], [], []
        alloc_rows, exposure_rows, feedback_rows = [], [], []
        lengths = []
        all_references = []
        for obs, history, ep, task_id, slot in zip(observations, histories, episodes, task_ids, slots):
            graph_key = static[slot]['key']
            n = len(obs['pfn_state'])
            window = self.windows.get(task_id)
            if window is None or window.graph_key != graph_key or window.node_count != n:
                window = _TaskWindow(self.limit, n, graph_key)
                self.windows[task_id] = window
            history = history[-self.limit:]
            try:
                window.sync(history, self.register, self.stats)
            except (ValueError, TypeError):
                # Unsupported layouts fall back to the exact old inference path
                # for this forward, without leaving a partly built task window.
                self.windows.pop(task_id, None)
                return None
            size = window.length
            lengths.append(size)
            current = self.register(obs['pfn_state'], obs['global_state'], graph_key)
            before = window.ordered(window.before)
            after = window.ordered(window.after)
            current_rows.append(current)
            before_rows.append(before)
            after_rows.append(after)
            # Original state registration order: current, before0, after0, ...
            refs = np.empty(1 + 2 * size, dtype=np.int64)
            refs[0] = current
            refs[1::2] = before
            refs[2::2] = after
            all_references.append(refs)
            if size:
                # Copy per-task windows in C, not one Python iteration per record.
                alloc = np.zeros((size, max_n), dtype=np.float32)
                expo = np.zeros_like(alloc)
                alloc[:, :n] = window.ordered(window.allocation)
                expo[:, :n] = window.ordered(window.exposure)
                alloc_rows.append(alloc)
                exposure_rows.append(expo)
                feed = np.empty((size, 4), dtype=np.float32)
                feed[:, 0] = window.ordered(window.rewards)
                feed[:, 1] = window.ordered(window.done)
                feed[:, 2] = window.ordered(window.times)
                feed[:, 3] = ((window.ordered(window.episodes).astype(np.float64)
                               - float(ep)) / 10).astype(np.float32)
                feedback_rows.append(feed)
        refs = np.concatenate(all_references)
        # Stable first-occurrence dedup; equivalent to Python's insertion-order
        # register(state, global, slot) sequence for each forward.
        _, first, inverse = np.unique(refs, return_index=True, return_inverse=True)
        order = np.argsort(first)
        local_to_stable = refs[first[order]]
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order), dtype=np.int64)
        local = rank[inverse]
        current, before, after = [], [], []
        offset = 0
        for size in lengths:
            sub = local[offset:offset + 1 + 2 * size]
            current.append(int(sub[0]))
            # Keep index slices as int64 arrays through the model gathers.
            # Iterating them into lists boxes every history index in Python,
            # only for the caller to convert those lists back to arrays.
            before.append(sub[1::2])
            after.append(sub[2::2])
            offset += 1 + 2 * size
        static_ids = {entry['key']: row for row, entry in enumerate(static)}
        gids = local_to_stable.tolist()
        self.stats['calls'] += 1
        self.stats['record_rows'] += sum(lengths)
        return {
            'registry_ids': local_to_stable,
            'states': [self.states[x] for x in gids],
            'globals': [self.globals[x] for x in gids],
            'graph_slots': [static_ids[self.graph_keys[x]] for x in gids],
            'current': current,
            'before': np.concatenate(before),
            'after': np.concatenate(after),
            'allocations': (np.concatenate(alloc_rows, axis=0) if alloc_rows else
                            np.zeros((0, max_n), dtype=np.float32)),
            'exposures': (np.concatenate(exposure_rows, axis=0) if exposure_rows else
                          np.zeros((0, max_n), dtype=np.float32)),
            'feedback': (np.concatenate(feedback_rows, axis=0) if feedback_rows else
                         np.zeros((0, 4), dtype=np.float32)),
            'lengths': np.asarray(lengths, dtype=np.int64),
        }
