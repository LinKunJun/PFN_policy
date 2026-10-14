from itertools import combinations

import numpy as np


class CatalogSearch:
    """P1 comparison on the same frozen catalog used by the neural policy.

    A neighbor must itself be a catalog entry. GRASP also starts from sampled
    catalog rows, so disconnected induced neighborhoods remain accessible.
    """
    def __init__(self, catalog, scores, seed=0):
        self.catalog, self.scores = catalog, np.asarray(scores)
        self.rng = np.random.default_rng(seed)
        counts = catalog.sum(1)
        distances = np.abs(catalog[:, None, :] - catalog[None, :, :]).sum(-1)
        self.add = (distances == 1) & (counts[None, :] == counts[:, None] + 1)
        self.local = (distances == 1) | ((distances == 2) & (counts[:, None] == counts[None, :]))
        self.zero = int(np.flatnonzero(counts == 0)[0])

    def _construct(self, rho=0.0, force=False):
        current = self.zero
        while True:
            choices = np.flatnonzero(self.add[current])
            if not len(choices):
                return current
            gains = self.scores[choices] - self.scores[current]
            if not force and gains.max() <= 1e-12:
                return current
            cutoff = gains.max() - rho * np.ptp(gains)
            choices = choices[gains >= cutoff - 1e-12]
            current = int(choices[0] if rho == 0 else self.rng.choice(choices))

    def greedy(self):
        return self._construct()

    def local_swap(self, action):
        while True:
            choices = np.flatnonzero(self.local[action])
            if not len(choices):
                return action
            best = int(choices[np.argmax(self.scores[choices])])
            if self.scores[best] <= self.scores[action] + 1e-12:
                return action
            action = best

    def grasp(self, restarts=16, rho=0.3):
        best = self.local_swap(self.greedy())
        for restart in range(restarts):
            initial = (self._construct(rho, force=True) if restart % 2 == 0
                       else int(self.rng.integers(len(self.catalog))))
            action = self.local_swap(initial)
            if self.scores[action] > self.scores[best]:
                best = action
        return best


def greedy_full(score, n, budget, mode="at_most", rng=None, rho=0.0, target_size=None):
    """Textbook forward construction in the FULL allocation space (not table 1)."""
    z = np.zeros(n, dtype=np.int8)
    size = budget if target_size is None else target_size
    while z.sum() < size:
        choices = np.flatnonzero(z == 0)
        base = score(z)
        candidates = np.repeat(z[None], len(choices), axis=0)
        candidates[np.arange(len(choices)), choices] = 1
        gains = np.array([score(row) - base for row in candidates])
        if mode == "at_most" and target_size is None and gains.max() <= 1e-12:
            break
        rcl = np.flatnonzero(gains >= gains.max() - rho * np.ptp(gains) - 1e-12)
        j = int(rcl[0] if rng is None else rng.choice(rcl))
        z = candidates[j]
    return z


def local_swap_full(score, z, budget, mode="at_most", pair_moves=False):
    """Add/delete/swap ascent; optional two-node additions and deletions."""
    z = z.copy()
    while True:
        best, best_value = z, score(z)
        inside, outside = np.flatnonzero(z), np.flatnonzero(z == 0)
        moves = [((i,), (j,)) for i in inside for j in outside]
        if mode == "at_most":
            moves += [((i,), ()) for i in inside]
            if z.sum() < budget:
                moves += [((), (j,)) for j in outside]
            if pair_moves:
                moves += [(pair, ()) for pair in combinations(inside, 2)]
                if z.sum() + 2 <= budget:
                    moves += [((), pair) for pair in combinations(outside, 2)]
        for remove, add in moves:
            candidate = z.copy()
            candidate[list(remove)] = 0
            candidate[list(add)] = 1
            value = score(candidate)
            if value > best_value + 1e-12:
                best, best_value = candidate, value
        if np.array_equal(z, best):
            return z
        z = best


def grasp_full(score, n, budget, mode="at_most", restarts=16, rho=0.3, seed=0):
    rng = np.random.default_rng(seed)
    best = local_swap_full(score, greedy_full(score, n, budget, mode), budget, mode)
    for _ in range(restarts):
        size = budget if mode == "exact" else int(rng.integers(budget + 1))
        z = greedy_full(score, n, budget, mode, rng, rho, target_size=size)
        z = local_swap_full(score, z, budget, mode, pair_moves=True)
        if score(z) > score(best):
            best = z
    return best
