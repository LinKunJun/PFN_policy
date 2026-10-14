from functools import lru_cache
from time import perf_counter

import numpy as np


class PlanningLimit(RuntimeError):
    pass


class ExactDP:
    """Enumerate reachable layers, then perform finite-horizon backward induction."""
    def __init__(self, model, gamma=1.0, max_states=500_000, seconds=120.0):
        self.model, self.gamma = model, gamma
        self.max_states, self.seconds = max_states, seconds
        self.values, self.policy = {}, {}
        self.action_evaluations = 0

    def solve(self, initial):
        start = perf_counter()
        layers = [{initial}]
        count = 1
        while next(iter(layers[-1]))[0] > 0:
            layer = set()
            for key in sorted(layers[-1]):
                for a in np.flatnonzero(self.model.mask(key)):
                    layer.add(self.model.next(key, int(a)))
                if count + len(layer) > self.max_states or perf_counter() - start > self.seconds:
                    raise PlanningLimit("Exact DP did not finish within the configured resource limit.")
            layers.append(layer)
            count += len(layer)
        self.values = {key: 0.0 for key in layers[-1]}
        for layer in reversed(layers[:-1]):
            for key in sorted(layer):
                actions = np.flatnonzero(self.model.mask(key))
                q = [self.model.reward(key, int(a)) + self.gamma *
                     self.values[self.model.next(key, int(a))] for a in actions]
                index = int(np.argmax(q))
                self.values[key], self.policy[key] = float(q[index]), int(actions[index])
                self.action_evaluations += len(actions)
                if perf_counter() - start > self.seconds:
                    raise PlanningLimit("Exact DP backward induction exceeded its time limit.")
        self.states = count
        return self.values[initial]

    def act(self, obs):
        return self.policy[obs["tabular_state_key"]]


class RollingPlanner:
    def __init__(self, model, depth=2, gamma=1.0):
        self.model, self.depth, self.gamma = model, depth, gamma
        self.action_evaluations = 0

    def act(self, obs):
        @lru_cache(None)
        def lookahead(key, depth):
            if depth == 0 or key[0] == 0:
                return 0.0, -1
            actions = np.flatnonzero(self.model.mask(key))
            q = [self.model.reward(key, int(a)) + self.gamma *
                 lookahead(self.model.next(key, int(a)), depth - 1)[0] for a in actions]
            self.action_evaluations += len(actions)
            i = int(np.argmax(q))
            return float(q[i]), int(actions[i])
        key = obs["tabular_state_key"]
        return lookahead(key, min(self.depth, key[0]))[1]
