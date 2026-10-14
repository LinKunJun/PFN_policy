"""CPU-only PPO deduplication preserving NumPy's sorted unique/inverse semantics.

Only pathological input domains use the mathematically equivalent NumPy path.
Neither function touches torch, RNG, model parameters, or a device.
"""
import numpy as np

_I64_MAX = np.iinfo(np.int64).max


def _baseline_pairs(record_ids, episodes, lengths):
    # The exact original representation and NumPy implementation.
    pairs = np.column_stack((record_ids, np.repeat(episodes, lengths)))
    return np.unique(pairs, axis=0, return_inverse=True)


def unique_pairs_packed(record_ids, episodes, lengths, record_count):
    """Return the same (Nx2 sorted rows, inverse) as np.unique(axis=0).

    For nonnegative record IDs r < record_count and int64 episodes e:
       key = r*span + (e-min_episode), span=max(e)-min(e)+1
    This is injective and strictly order preserving for (r,e) lexicographic
    order, provided record_count*span <= INT64_MAX. Everything else falls back.
    """
    record_ids = np.asarray(record_ids)
    episodes = np.asarray(episodes)
    lengths = np.asarray(lengths)
    if (record_ids.dtype != np.dtype('int64')
            or episodes.dtype != np.dtype('int64')
            or lengths.ndim != 1 or episodes.ndim != 1
            or episodes.size != lengths.size or episodes.size == 0
            or int(record_count) < 0):
        return _baseline_pairs(record_ids, episodes, lengths)
    episode_min = int(episodes.min())
    span = int(episodes.max()) - episode_min + 1
    if (span <= 0 or span > _I64_MAX or
            int(record_count) > _I64_MAX // span):
        return _baseline_pairs(record_ids, episodes, lengths)
    if record_ids.size and (int(record_ids.min()) < 0 or
                            int(record_ids.max()) >= record_count):
        return _baseline_pairs(record_ids, episodes, lengths)
    # The input is already in row-major valid-history order.
    # A broadcasted Nx2 temporary is not needed.
    offsets = np.repeat(episodes - np.int64(episode_min), lengths)
    keys = record_ids * np.int64(span) + offsets
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    pairs = np.empty((unique_keys.size, 2), dtype=np.int64)
    pairs[:, 0] = unique_keys // span
    pairs[:, 1] = unique_keys % span + episode_min
    return pairs, inverse


def unique_states_visited(references, state_count, max_density_ratio=4):
    """Sorted np.unique for bounded int64 state IDs, O(N+K).

    The bitset and dense lookup table are used only if K<=max(4096,4*N).
    A large sparse key space falls back to the original np.unique.
    """
    references = np.asarray(references)
    n = references.size
    k = int(state_count)
    if (references.dtype != np.dtype('int64') or n == 0 or k <= 0
            or k > max(4096, max_density_ratio * n)
            or int(references.min()) < 0 or int(references.max()) >= k):
        return np.unique(references, return_inverse=True)
    used = np.zeros(k, dtype=np.bool_)
    used[references] = True
    states = np.flatnonzero(used).astype(references.dtype, copy=False)
    lookup = np.empty(k, dtype=np.intp)
    lookup[states] = np.arange(states.size, dtype=np.intp)
    return states, lookup[references]
