"""MAB-N exposure catalogs; drop-in replacement for the original dgp.py.

All original public call signatures and dataclass constructor fields are kept.
Only sample_task's private builder receives graph/dynamics context. The public
four-argument make_catalog has no graph input and uses an empty graph/FC design.

This is the previously benchmarked cluster-homogeneous, realizable JOINT
exposure design, not UCB-TSN or EXP3-N-CS. CD phase-local arms are the project's
extension. There is no KL, entropy, DPP, or reward-based candidate objective.
Engineering fast paths preserve candidate order and RNG consumption on the
supported NumPy Generator backends tested by the regression suite.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from math import comb
from numbers import Integral

import numpy as np

from .core import Dynamics, PublicTask, NetworkEnv, exposures, fatigue_penalty


@dataclass(frozen=True)
class TaskPrior:
    n: int = 30
    feature_dim: int = 4
    budget: int = 8
    catalog_size: int = 48
    horizon: int = 6
    graph_family: str = "mixed"
    edge_probability: float = 0.15
    node_noise: float = 0.5
    fatigue_cost: float = 1.0


# Caches contain only immutable combinatorial data, never a graph, task, RNG,
# or reward. Limits also bound memory in persistent multiprocessing workers.
_INT64_MAX = int(np.iinfo(np.int64).max)
_DP_CACHE_MAX_CELLS = 8192
_DECODE_BATCH_ROWS = 4096
_WORK_BYTES = 16 * 1024 * 1024


def _validate_catalog_request(n, budget, size):
    if not all(isinstance(v, Integral) and not isinstance(v, (bool, np.bool_))
               for v in (n, budget, size)):
        raise ValueError('n, budget, size must be integers')
    if n < 1 or budget < 0 or budget > n or size < 1:
        raise ValueError('Require n>=1, 0<=budget<=n, size>=1')
    capacity = 0
    for k in range(int(budget) + 1):
        capacity += comb(int(n), k)
        if capacity >= size:
            return
    raise ValueError(f'Only {capacity} feasible distinct allocations; requested {size}')


def _validate_horizon(horizon):
    if (not isinstance(horizon, Integral) or isinstance(horizon, (bool, np.bool_))
            or horizon < 1):
        raise ValueError('horizon must be a positive integer')


def _validated_adjacency(adjacency, n):
    if adjacency is None:
        return np.zeros((n, n), dtype=np.int8)
    a = np.asarray(adjacency)
    if (a.shape != (n, n) or not np.array_equal(a, a.T)
            or np.any(a.diagonal()) or not np.all((a == 0) | (a == 1))):
        raise ValueError('adjacency must be a simple undirected binary (n,n) graph')
    return np.ascontiguousarray(a, dtype=np.int8)


def _balanced_weights(n, cluster_count):
    small, large = divmod(n, cluster_count)
    return [small + (i < large) for i in range(cluster_count)]


def _phase_indices(weights, periods):
    phases = [[] for _ in range(periods)]
    loads = [0] * periods
    for i in sorted(range(len(weights)), key=lambda j: (-weights[j], j)):
        p = min(range(periods), key=lambda j: (loads[j], len(phases[j]), j))
        phases[p].append(i)
        loads[p] += weights[i]
    return phases


def _build_subset_dp(weights, budget):
    # Python integers avoid count overflow when the feasible space exceeds 2^63.
    length = len(weights)
    dp = [[0] * (budget + 1) for _ in range(length + 1)]
    dp[length] = [1] * (budget + 1)
    for i in range(length - 1, -1, -1):
        weight = weights[i]
        after, now = dp[i + 1], dp[i]
        now[:min(weight, budget + 1)] = after[:min(weight, budget + 1)]
        for b in range(weight, budget + 1):
            now[b] = after[b] + after[b - weight]
    return tuple(tuple(row) for row in dp)


@lru_cache(maxsize=64)
def _cached_subset_dp(weights, budget):
    return _build_subset_dp(weights, budget)


def _subset_dp(weights, budget):
    weights = tuple(int(w) for w in weights)
    budget = int(budget)
    if (len(weights) + 1) * (budget + 1) <= _DP_CACHE_MAX_CELLS:
        return _cached_subset_dp(weights, budget)
    return _build_subset_dp(weights, budget)


@lru_cache(maxsize=128)
def _cluster_count(n, budget, size, periods):
    # Identical search order and capacity decisions to the benchmarked version.
    first_feasible = None
    for c in range(1, n + 1):
        weights = _balanced_weights(n, c)
        capacity = _subset_dp(weights, budget)[0][budget]
        if capacity >= size and first_feasible is None:
            first_feasible = c
        if periods < 2 or size <= periods:
            if capacity >= size:
                return c, False
            continue
        phases = _phase_indices(weights, periods)
        local_capacity = [(_subset_dp([weights[i] for i in phase], budget)[0][budget] - 1)
                          for phase in phases]
        if min(local_capacity) >= 1 and 1 + sum(local_capacity) >= size:
            return c, True
    if first_feasible is None:
        raise AssertionError('Feasible capacity disappeared')
    return first_feasible, False


def _graph_clusters(adjacency, weights, rng):
    """Same seeded graph traversal, without sorting/re-summing every node step.

    rank encodes the original (degree descending, random priority ascending)
    tie break. Adding n per incident edge encodes the primary link-count key.
    Thus argmax exactly replaces the original repeated three-key lexsort.
    """
    a = np.asarray(adjacency)
    n = len(a)
    degree = a.sum(axis=1)
    priority = np.empty(n, dtype=np.intp)
    priority[rng.permutation(n)] = np.arange(n, dtype=np.intp)
    order = np.lexsort((priority, -degree))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n - 1, -1, -1, dtype=np.int64)
    # Contiguous rows replace repeatedly gathered adjacency columns/submatrices.
    updates = np.ascontiguousarray(a.T, dtype=np.int64)
    updates *= n
    unassigned = np.ones(n, dtype=bool)
    groups = []
    for quota in weights:
        score = rank.copy()
        score[~unassigned] = -1
        picked = []
        for step in range(quota):
            vertex = int(score.argmax())
            picked.append(vertex)
            unassigned[vertex] = False
            if step + 1 < quota:
                score += updates[vertex]
                score[~unassigned] = -1
        groups.append(tuple(picked))
    return groups


def _randbelow(rng, upper):
    upper = int(upper)
    if upper <= 0:
        raise ValueError('Empty random integer range')
    if upper <= _INT64_MAX:
        draw = getattr(rng, 'integers', None)
        return int(draw(upper) if draw is not None else rng.randint(upper))
    bits = (upper - 1).bit_length()
    mask = (1 << bits) - 1
    # Integer rejection for unbiased large random numbers (acceptance > 1/2),
    # NOT a retry-until-a-new-candidate catalog construction loop.
    while True:
        sample = int.from_bytes(rng.bytes((bits + 7) // 8), 'little') & mask
        if sample < upper:
            return sample


def _sample_distinct_nonzero_ranks(population, wanted, rng):
    """Floyd sampling: exactly wanted insertions, including near-full spaces."""
    population, wanted = int(population), int(wanted)
    if wanted < 0 or wanted > population:
        raise ValueError('Rank request exceeds capacity')
    selected = set()
    start = population - wanted
    if population <= _INT64_MAX:
        # Vector bounds preserve the scalar Generator.integers sequence.
        upper = np.arange(wanted, dtype=np.int64) + (start + 1)
        draw = getattr(rng, 'integers', None)
        if draw is None:
            draw = rng.randint  # Legacy RandomState/np.random compatibility.
        candidates = draw(upper) if wanted else ()
    else:
        candidates = (_randbelow(rng, j + 1) for j in range(start, population))
    for j, candidate in zip(range(start, population), candidates):
        candidate = int(candidate)
        selected.add(j if candidate in selected else candidate)
    result = [rank + 1 for rank in selected]
    rng.shuffle(result)
    return result


def _unrank_exposure_arm(groups, weights, dp, budget, rank):
    """Scalar reference/general large-integer path; no enumeration of 2**N."""
    if rank < 0 or rank >= dp[0][budget]:
        raise ValueError('Exposure arm rank is out of range')
    mask, left = 0, int(budget)
    for i, (nodes, weight) in enumerate(zip(groups, weights)):
        nskip = dp[i + 1][left]
        if rank >= nskip:
            rank -= nskip
            left -= int(weight)
            for node in nodes:
                mask |= 1 << int(node)
    return mask


def _unrank_switches(weights, dp, budget, ranks):
    """Batch-decode ranks to cluster switches, keeping arbitrary-int support."""
    length = len(weights)
    x = np.zeros((len(ranks), length), dtype=np.int32)
    if dp[0][budget] <= _INT64_MAX:
        r = np.asarray(ranks, dtype=np.int64).copy()
        left = np.full(len(r), budget, dtype=np.intp)
        counts = np.asarray(dp, dtype=np.int64)
        for i, weight in enumerate(weights):
            skip = counts[i + 1, left]
            take = r >= skip
            x[:, i] = take
            np.subtract(r, skip, out=r, where=take)
            np.subtract(left, weight, out=left, where=take)
    else:
        for row, rank in enumerate(ranks):
            left = budget
            for i, weight in enumerate(weights):
                skip = dp[i + 1][left]
                if rank >= skip:
                    rank -= skip
                    left -= weight
                    x[row, i] = 1
    return x


def _switches_to_masks(x, groups, n):
    group_masks = [sum(1 << int(v) for v in nodes) for nodes in groups]
    if n <= 64:
        # Cluster bitsets are disjoint: the exact uint64 sum equals bitwise OR.
        # Explicit casting avoids signed-int/uint64 promotion to float64.
        packed = x.astype(np.uint64) @ np.asarray(group_masks, dtype=np.uint64)
        return packed.tolist()
    # Python int bitsets are exact on >64-node networks; only selected groups
    # are visited, rather than every node of every candidate.
    return [sum(group_masks[int(i)] for i in np.flatnonzero(row)) for row in x]


def _masks_to_binary_catalog(masks, n):
    if n <= 64:
        packed = np.asarray(masks, dtype=np.uint64)
        shifts = np.arange(n, dtype=np.uint64)
        return ((packed[:, None] >> shifts[None, :]) & 1).astype(np.int8)
    nbytes = (n + 7) // 8
    raw = b''.join(int(bits).to_bytes(nbytes, 'little') for bits in masks)
    decoded = np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(-1, nbytes),
                            axis=1, bitorder='little')
    return np.ascontiguousarray(decoded[:, :n], dtype=np.int8)


def _joint_exposure_plan(groups, adjacency):
    """Compile only comparisons needed for nonsingleton exposure clusters."""
    nodes, left, anchor = [], [], []
    for group in groups:
        if len(group) > 1:
            start = len(nodes)
            nodes.extend(group)
            left.extend(range(start + 1, start + len(group)))
            anchor.extend([start] * (len(group) - 1))
    nodes = np.asarray(nodes, dtype=np.intp)
    a = np.asarray(adjacency)
    weights = np.fromiter((len(g) for g in groups), dtype=np.intp, count=len(groups))
    starts = np.r_[0, np.cumsum(weights[:-1])]
    ordered = np.fromiter((v for group in groups for v in group), dtype=np.intp,
                          count=int(weights.sum()))
    if len(nodes):
        # One reduceat replaces a Python loop of repeated np.ix_ and sums.
        contributions = np.ascontiguousarray(np.add.reduceat(
            a[nodes][:, ordered], starts, axis=1, dtype=np.int32).T)
    else:
        contributions = np.empty((len(groups), 0), dtype=np.int32)
    return (nodes, np.asarray(left, dtype=np.intp), np.asarray(anchor, dtype=np.intp),
            a.sum(axis=1)[nodes] // 2, contributions)


def _valid_joint_exposures(mask_list, n, adjacency, groups):
    """General reference-compatible check of BOTH own and neighbor exposure."""
    if not len(mask_list):
        return np.empty(0, dtype=bool)
    nodes, left, anchor, threshold, _ = _joint_exposure_plan(groups, adjacency)
    if not len(nodes):
        return np.ones(len(mask_list), dtype=bool)
    z = _masks_to_binary_catalog(mask_list, n).astype(np.int32)
    direct = z[:, nodes]
    majority = z @ np.asarray(adjacency, dtype=np.int32)[nodes].T > threshold
    return (np.all(direct[:, left] == direct[:, anchor], axis=1)
            & np.all(majority[:, left] == majority[:, anchor], axis=1))


def _feasible_exposure_candidates(groups, weights, n, budget,
                                  adjacency, original_groups, rng, limit, *, plan=None, active=None):
    """Same finite candidate scan as the reference; compressed exposure math.

    Every assignment switches whole original clusters, so direct-treatment
    equality holds by construction. Joint majority equality is checked for ALL
    clusters, including untreated clusters outside the chosen CD phase.
    """
    dp = _subset_dp(weights, budget)
    capacity = dp[0][budget] - 1
    count = min(capacity, limit)
    ranks = (range(1, capacity + 1) if capacity <= limit else
             _sample_distinct_nonzero_ranks(capacity, count, rng))
    if plan is None:
        plan = _joint_exposure_plan(original_groups, adjacency)
    nodes, left, anchor, threshold, all_contributions = plan
    variable = np.empty(0, dtype=np.intp)
    contributions = None
    if len(nodes):
        if active is None:
            index = {tuple(group): i for i, group in enumerate(original_groups)}
            active = [index[tuple(group)] for group in groups]
        contributions = all_contributions[np.asarray(active, dtype=np.intp)]
        # A node cannot have majority exposure when even the entire active
        # region (or the total budget) cannot exceed its majority threshold.
        possible = np.minimum(contributions.sum(axis=0), budget) > threshold
        variable = np.flatnonzero(possible)
        contributions = np.ascontiguousarray(contributions[:, variable])
    # Target ~16 MiB of batch-local arrays; this is not a process memory cap.
    row_bytes = max(1, 20 * len(groups) + 8 * len(nodes))
    batch_rows = max(1, min(_DECODE_BATCH_ROWS, _WORK_BYTES // row_bytes))
    result = []
    for start in range(0, count, batch_rows):
        batch = ranks[start:start + batch_rows]
        x = _unrank_switches(weights, dp, budget, batch)
        if len(variable):
            majority = np.zeros((len(x), len(nodes)), dtype=bool)
            majority[:, variable] = (x @ contributions) > threshold[variable]
            good = np.all(majority[:, left] == majority[:, anchor], axis=1)
            x = x[good]
        # If variable is empty all neighborhood exposure values equal zero.
        if len(x):
            result.extend(_switches_to_masks(x, groups, n))
    return result


def _balanced_quotas(capacities, total):
    """Exact capped round-robin quotas in O(P log(total)), not O(M P)."""
    if total < 0 or total > sum(capacities):
        raise ValueError('Insufficient phase capacity')
    low, high = 0, total
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(c, middle) for c in capacities) <= total:
            low = middle
        else:
            high = middle - 1
    quotas = [min(c, low) for c in capacities]
    remaining = total - sum(quotas)
    for i, c in enumerate(capacities):
        if remaining and quotas[i] < c:
            quotas[i] += 1
            remaining -= 1
    return quotas


def _paper_exposure_catalog(n, budget, size, rng, adjacency,
                            dynamics, horizon):
    """Construct an exact-size catalog from real MAB-N exposure arms.

    Unlike the (direct-treatment only) special case, this enforces both own
    treatment and neighborhood-majority exposure inside each graph cluster.
    These categories constrain the catalog only. The oracle still evaluates
    exact fractional neighbor exposures; no reward-sufficiency claim is made.
    """
    _validate_catalog_request(n, budget, size)
    _validate_horizon(horizon)
    n, budget, size, horizon = map(int, (n, budget, size, horizon))
    dynamics = Dynamics(str(dynamics).upper())
    adjacency = _validated_adjacency(adjacency, n)
    if size == 1:
        return np.zeros((1, n), dtype=np.int8)
    cd = str(dynamics).upper() == 'CD'
    periods = min(3, horizon) if cd and budget else 1
    # This cap is only a computational batching choice, not a reward,
    # diversity, KL, entropy, or surrogate-effect selection criterion.
    scan_limit = max(2048, 32 * size)
    initial_count, _ = _cluster_count(n, budget, size, periods)

    def instantiate(count):
        weights = _balanced_weights(n, count)
        groups = _graph_clusters(adjacency, weights, rng)
        return groups, weights

    if periods >= 2 and size > periods:
        # CD-only extension: generate from three (or two if T=2) disjoint
        # node regions, each of which contains cluster-consistent, realizable
        # exposure arms. No two selected actions across regions intersect.
        for count in range(initial_count, n + 1):
            groups, weights = instantiate(count)
            phases = _phase_indices(weights, periods)
            plan = _joint_exposure_plan(groups, adjacency)
            candidates = [
                _feasible_exposure_candidates(
                    [groups[i] for i in phase], [weights[i] for i in phase],
                    n, budget, adjacency, groups, rng, scan_limit, plan=plan, active=phase)
                for phase in phases
            ]
            if min(map(len, candidates)) == 0 or 1 + sum(map(len, candidates)) < size:
                continue
            # Balanced exposure-arm sampling over CD rotation phases.
            quotas = _balanced_quotas([len(c) for c in candidates], size - 1)
            masks = [0]
            for quota, options in zip(quotas, candidates):
                if quota:
                    ranks = _sample_distinct_nonzero_ranks(len(options), quota, rng)
                    masks.extend(options[rank - 1] for rank in ranks)
            assert len(masks) == size and len(set(masks)) == size
            return _masks_to_binary_catalog(masks, n)

    # CD edge cases with too few nodes/actions for phase-only capacity, as
    # well as FC/NONE: sample real, cluster-wise exposure super arms.
    global_start, _ = _cluster_count(n, budget, size, 1)
    for count in range(global_start, n + 1):
        groups, weights = instantiate(count)
        candidates = _feasible_exposure_candidates(
            groups, weights, n, budget, adjacency, groups, rng, scan_limit)
        if len(candidates) >= size - 1:
            ranks = _sample_distinct_nonzero_ranks(len(candidates), size - 1, rng)
            masks = [0] + [candidates[rank - 1] for rank in ranks]
            assert len(set(masks)) == size
            return _masks_to_binary_catalog(masks, n)
    # At C=N all clusters are singletons. Every original budget-feasible
    # assignment satisfies U_C, and the entire capacity >= size by validation.
    raise AssertionError('Singleton clusters failed to reproduce feasible space')


def make_catalog(n, budget, size, rng):
    """Original four-argument API, now materializing MAB-N exposure arms.

    No graph or dynamics was present in this API. Direct calls therefore use
    an empty graph and a non-CD design. sample_task supplies the real graph
    and FC/CD setting to the private builder below, without changing ANY
    caller signature. The result is a C-contiguous binary int8 (size,n) array,
    with one all-zero first row and no duplicate actions.
    """
    return _paper_exposure_catalog(n, budget, size, rng, None, 'FC', 6)


class EnvModel:
    """Private DGP; supplied only to the simulator and evaluator/oracle planner."""

    def __init__(self, coefficients: np.ndarray, node_noise: float):
        self.__coefficients = coefficients.copy()
        self.node_noise = node_noise

    def mean_outcomes(self, features, assignments, exact):
        x = np.column_stack((np.ones(len(features)), features))
        b, direct, spill, interact, curve = (x @ self.__coefficients.T).T
        b = 0.2 * np.tanh(b)
        direct, spill = 0.5 + np.tanh(direct), 0.5 + np.tanh(spill)
        return (b + direct * assignments + spill * exact
                + np.tanh(interact) * assignments * exact
                + np.tanh(curve) * exact ** 2)

    def prepare(self, task, exact):
        return self.mean_outcomes(task.features, task.catalog, exact).mean(axis=-1)

    def sample_noise(self, rng, n):
        # Independent N(0, sigma^2) node noise averages to sigma/sqrt(N).
        return float(rng.normal(0, self.node_noise / np.sqrt(n)))

    def randomized_experiment(self, task, rounds, seed):
        rng = np.random.default_rng(seed)
        assignments = rng.binomial(1, 0.5, (rounds, task.n)).astype(np.int8)
        exact, majority = exposures(task.adjacency, assignments)
        y = self.mean_outcomes(task.features, assignments, exact)
        y += rng.normal(0, self.node_noise, y.shape)
        # Every row is one executed assignment, with factual node outcomes.
        return assignments, majority, y

    def majority_means(self, task):
        """Evaluation only: exact Bernoulli(1/2) design-standardized means."""
        result = np.full((task.n, 2, 2), np.nan)
        for i, d in enumerate(task.adjacency.sum(1).astype(int)):
            k = np.arange(d + 1)
            prob = np.array([comb(d, int(j)) / 2 ** d for j in k])
            e = k / d if d else np.zeros(1)
            s = k > d // 2
            for a in (0, 1):
                outcomes = self.mean_outcomes(task.features[i:i + 1],
                                               np.full((len(k), 1), a), e[:, None])[:, 0]
                for group in (0, 1):
                    selected = s == group
                    if prob[selected].sum() > 0:
                        result[i, a, group] = np.average(outcomes[selected], weights=prob[selected])
        return result


class ScoreReward:
    """Known predicted objective used exclusively for static PFN pretraining."""
    def __init__(self, prediction):
        self.prediction = prediction

    def prepare(self, task, exact):
        s = exact > 0.5
        return self.prediction[np.arange(task.n), task.catalog, s.astype(int)].mean(1)

    def sample_noise(self, rng, n):
        return 0.0


def _catalog_design_key(dynamics, horizon, budget):
    return min(3, int(horizon)) if dynamics == Dynamics.CD and budget else 1


def _attach_catalog_state(bundle, seed, cache=None):
    # Private state, not new dataclass fields or constructor parameters.
    # Pickle/spawn transports it with the object. At most three catalog variants
    # are cached per task; no cross-task global array cache can leak data.
    bundle._catalog_seed = int(seed)
    bundle._catalog_context = (bundle.public.adjacency, bundle.public.n,
                               bundle.public.budget, bundle.public.m)
    bundle._catalog_cache = {} if cache is None else cache
    key = _catalog_design_key(bundle.public.dynamics, bundle.public.horizon,
                              bundle.public.budget)
    bundle._catalog_cache[key] = bundle.public.catalog
    return bundle


@dataclass
class TaskBundle:
    public: PublicTask
    oracle: EnvModel
    task_id: str

    def env(self, seed, prediction=None, score_objective=False):
        source = ScoreReward(prediction) if score_objective else self.oracle
        return NetworkEnv(self.public, source, seed, prediction)

    def with_setting(self, dynamics, horizon):
        setting = Dynamics(dynamics)
        _validate_horizon(horizon)
        # Legacy/manual TaskBundle construction still preserves its given
        # catalog. Only seed-addressed sample_task bundles rebuild a CD design.
        seed = getattr(self, '_catalog_seed', getattr(self, 'catalog_seed', None))
        catalog = self.public.catalog
        cache = getattr(self, '_catalog_cache', {})
        context = getattr(self, '_catalog_context', None)
        if (context is None or context[0] is not self.public.adjacency
                or context[1:] != (self.public.n, self.public.budget, self.public.m)):
            cache = {}
        if seed is not None:
            current_key = _catalog_design_key(self.public.dynamics,
                                              self.public.horizon, self.public.budget)
            target_key = _catalog_design_key(setting, horizon, self.public.budget)
            if current_key in cache and cache[current_key] is not self.public.catalog:
                # A caller replaced the public catalog: do not reuse stale variants.
                cache = {}
            cache[current_key] = self.public.catalog
            if target_key not in cache:
                cache[target_key] = _paper_exposure_catalog(
                    self.public.n, self.public.budget, self.public.m,
                    np.random.default_rng(seed), self.public.adjacency,
                    setting, horizon)
            catalog = cache[target_key]
        result = replace(self, public=replace(self.public, catalog=catalog,
                         dynamics=setting, horizon=horizon))
        return _attach_catalog_state(result, seed, cache) if seed is not None else result


def task_seed(root_seed: int, split: str, index: int, stream=0):
    split_id = {"train": 11, "validation": 23, "test": 37}[split]
    return int(np.random.SeedSequence([root_seed, split_id, index, stream]).generate_state(1)[0])


def sample_task(prior: TaskPrior, root_seed: int, split: str, index: int,
                dynamics: str = "FC", horizon: int | None = None):
    rng = np.random.default_rng(task_seed(root_seed, split, index))
    n = prior.n
    family = rng.choice(["ER", "SBM"]) if prior.graph_family == "mixed" else prior.graph_family
    if family == "ER":
        probabilities = np.full((n, n), prior.edge_probability)
    elif family == "SBM":
        group = np.arange(n) % 3
        probabilities = np.where(group[:, None] == group[None, :],
                                 min(0.8, 2 * prior.edge_probability), prior.edge_probability / 3)
    else:
        raise ValueError("graph_family must be ER, SBM, or mixed.")
    upper = np.triu(rng.random((n, n)) < probabilities, k=1)
    adjacency = (upper | upper.T).astype(np.int8)
    features = rng.normal(size=(n, prior.feature_dim)).astype(np.float32)
    # Independent generator stream: choosing a different catalog must not
    # change the private SCM coefficients sampled below.
    catalog_seed = task_seed(root_seed, split, index, 101)
    catalog_rng = np.random.default_rng(catalog_seed)
    catalog = _paper_exposure_catalog(
        n, prior.budget, prior.catalog_size, catalog_rng, adjacency, dynamics,
        prior.horizon if horizon is None else horizon)
    public = PublicTask(adjacency, features, catalog, prior.budget,
                        prior.horizon if horizon is None else horizon,
                        Dynamics(dynamics), prior.fatigue_cost)
    coefficients = rng.normal(0, 0.5, (5, prior.feature_dim + 1))
    return _attach_catalog_state(
        TaskBundle(public, EnvModel(coefficients, prior.node_noise),
                   f"{split}:{root_seed}:{index}"), catalog_seed)


def training_task(spec):
    """A seed-addressed job; process scheduling never changes a task's random stream."""
    prior, root, index, objective, rounds, episodes = spec
    dynamics = 'NONE' if objective == 'static' else ('FC' if index % 2 == 0 else 'CD')
    bundle = sample_task(prior, root, 'train', index, dynamics,
                         horizon=1 if objective == 'static' else prior.horizon)
    prediction = None
    if objective == 'static':
        from .score import fit_factual_prediction
        z, s, y = bundle.oracle.randomized_experiment(bundle.public, rounds, task_seed(root, 'train', index, 1))
        prediction = fit_factual_prediction(bundle.public, z, s, y)
    env = bundle.env(task_seed(root, 'train', index, 2), prediction, objective == 'static')
    seeds = [task_seed(root, 'train', index, 10 + ep) for ep in range(episodes)]
    return env, seeds
