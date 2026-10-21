#from functools import lru_cache
#from time import perf_counter
#
#import numpy as np
#
#
#class PlanningLimit(RuntimeError):
#    pass
#
#
#class ExactDP:
#    """Enumerate reachable layers, then perform finite-horizon backward induction."""
#    def __init__(self, model, gamma=1.0, max_states=500_000, seconds=120.0):
#        self.model, self.gamma = model, gamma
#        self.max_states, self.seconds = max_states, seconds
#        self.values, self.policy = {}, {}
#        self.action_evaluations = 0
#
#    def solve(self, initial):
#        start = perf_counter()
#        layers = [{initial}]
#        count = 1
#        while next(iter(layers[-1]))[0] > 0:
#            layer = set()
#            for key in sorted(layers[-1]):
#                for a in np.flatnonzero(self.model.mask(key)):
#                    layer.add(self.model.next(key, int(a)))
#                if count + len(layer) > self.max_states or perf_counter() - start > self.seconds:
#                    raise PlanningLimit("Exact DP did not finish within the configured resource limit.")
#            layers.append(layer)
#            count += len(layer)
#        self.values = {key: 0.0 for key in layers[-1]}
#        for layer in reversed(layers[:-1]):
#            for key in sorted(layer):
#                actions = np.flatnonzero(self.model.mask(key))
#                q = [self.model.reward(key, int(a)) + self.gamma *
#                     self.values[self.model.next(key, int(a))] for a in actions]
#                index = int(np.argmax(q))
#                self.values[key], self.policy[key] = float(q[index]), int(actions[index])
#                self.action_evaluations += len(actions)
#                if perf_counter() - start > self.seconds:
#                    raise PlanningLimit("Exact DP backward induction exceeded its time limit.")
#        self.states = count
#        return self.values[initial]
#
#    def act(self, obs):
#        return self.policy[obs["tabular_state_key"]]
#
#
#class RollingPlanner:
#    def __init__(self, model, depth=2, gamma=1.0):
#        self.model, self.depth, self.gamma = model, depth, gamma
#        self.action_evaluations = 0
#
#    def act(self, obs):
#        @lru_cache(None)
#        def lookahead(key, depth):
#            if depth == 0 or key[0] == 0:
#                return 0.0, -1
#            actions = np.flatnonzero(self.model.mask(key))
#            q = [self.model.reward(key, int(a)) + self.gamma *
#                 lookahead(self.model.next(key, int(a)), depth - 1)[0] for a in actions]
#            self.action_evaluations += len(actions)
#            i = int(np.argmax(q))
#            return float(q[i]), int(actions[i])
#        key = obs["tabular_state_key"]
#        return lookahead(key, min(self.depth, key[0]))[1]
from math import isfinite, isnan
from operator import index
from time import perf_counter

import numpy as np

from ..env.core import Dynamics


class PlanningLimit(RuntimeError):
    pass


def _remaining(key):
    try:
        remaining = index(key[0])
    except (TypeError, IndexError) as error:
        raise ValueError("State remaining time must be a nonnegative integer.") from error
    if remaining < 0:
        raise ValueError("State remaining time must be a nonnegative integer.")
    return remaining


def _actions(model, key):
    actions = np.flatnonzero(model.mask(key))
    if not len(actions):
        raise ValueError("A nonterminal planning state has no valid action.")
    return actions


def _successor(model, key, action):
    successor = model.next(key, action)
    if _remaining(successor) != key[0] - 1:
        raise ValueError("Planning transitions must decrease remaining time by one.")
    return successor


def _best(q, actions):
    if not all(isfinite(value) for value in q):
        raise ValueError("Planning rewards and returns must be finite.")
    i = int(np.argmax(q))
    return float(q[i]), int(actions[i])


class ExactDP:
    """Reachable-layer finite-horizon DP, with a kernel for standard CD models.

    A successful solve replaces the previous solution. A failed solve publishes
    no partial solution. ``states`` counts (remaining, memory) pairs;
    ``action_evaluations`` counts Bellman/fallback scores, excluding enumeration.
    Time limits are cooperative checks, not preemption of model/NumPy calls.
    """

    def __init__(self, model, gamma=1.0, max_states=500_000, seconds=120.0):
        self.model, self.gamma = model, float(gamma)
        try:
            self.max_states = index(max_states)
        except TypeError as error:
            raise ValueError("max_states must be a nonnegative integer.") from error
        self.seconds = float(seconds)
        if self.max_states < 0 or isnan(self.seconds) or self.seconds < 0:
            raise ValueError("Planning resource limits must be nonnegative.")
        if not isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and nonnegative.")
        self.values, self.policy = {}, {}
        self.states = self.action_evaluations = 0

    def _check_time(self, start):
        if perf_counter() - start >= self.seconds:
            raise PlanningLimit("Exact DP exceeded its configured time limit.")

    def _check_states(self, count):
        if count > self.max_states:
            raise PlanningLimit("Exact DP exceeded its configured state limit.")

    def solve(self, initial):
        self.values, self.policy = {}, {}
        self.states = self.action_evaluations = 0
        start = perf_counter()
        remaining = _remaining(initial)
        self._check_states(1)
        self._check_time(start)
        if remaining == 0:
            self.values = {initial: 0.0}
            self.states = 1
            return 0.0

        task = getattr(self.model, "task", None)
        if getattr(task, "dynamics", None) == Dynamics.CD:
            # Only this concrete, unmodified model promises that the kernel's
            # mask, transition and static reward equal the public dynamics.
            from ..env.score import PlanningModel
            if type(self.model) is PlanningModel and all(
                getattr(getattr(self.model, name), "__func__", None)
                is getattr(PlanningModel, name) for name in ("mask", "next", "reward")
            ):
                return self._solve_cd(initial, start)
        return self._solve_general(initial, start)

    def _solve_cd(self, initial, start):
        task, gamma = self.model.task, self.gamma
        n, m = task.n, task.m
        remaining, memory = initial
        if len(memory) != n:
            raise ValueError("CD memory length must equal the number of nodes.")
        try:
            memory = tuple(index(value) for value in memory)
        except TypeError as error:
            raise ValueError("CD memory entries must be integers in {0, 1, 2}.") from error
        if any(value not in (0, 1, 2) for value in memory):
            raise ValueError("CD memory entries must be integers in {0, 1, 2}.")
        scores = np.asarray(self.model.base_scores, dtype=float)
        if scores.shape != (m,) or not np.isfinite(scores).all():
            raise ValueError("Expected one finite base score per catalog action.")
        base_scores = scores.tolist()
        self._check_time(start)

        busy_masks, set2_masks = [], []
        for a in range(m):
            set2 = 0
            for j, node in enumerate(np.flatnonzero(task.catalog[a])):
                set2 |= 2 << (2 * int(node))
                if j % 256 == 0:
                    self._check_time(start)
            busy_masks.append(set2 | (set2 >> 1))
            set2_masks.append(set2)
            self._check_time(start)
        mask_01 = ((1 << (2 * n)) - 1) // 3
        encoded = sum(value << (2 * i) for i, value in enumerate(memory))
        self._check_time(start)

        def decode(s):
            return tuple((s >> (2 * i)) & 3 for i in range(n))

        layers, count = [{encoded}], 1
        for _ in range(remaining):
            layer = set()
            for s in layers[-1]:
                decayed = (s >> 1) & mask_01
                for a in range(m):
                    if not (s & busy_masks[a]):
                        layer.add(decayed | set2_masks[a])
                        self._check_states(count + len(layer))
                    if a % 256 == 0:
                        self._check_time(start)
                self._check_time(start)
            layers.append(layer)
            count += len(layer)

        # Keep results local until the entire solve (including tuple decoding)
        # has passed its budget checks.
        layer_values = {s: 0.0 for s in layers[-1]}
        values, policy = {}, {}
        for s in layers[-1]:
            values[(0, decode(s))] = 0.0
            self._check_time(start)
        for step in range(len(layers) - 2, -1, -1):
            rem = remaining - step
            current = {}
            for s in layers[step]:
                decayed = (s >> 1) & mask_01
                best_q, best_a = -float("inf"), -1
                for a in range(m):
                    if not (s & busy_masks[a]):
                        successor = decayed | set2_masks[a]
                        q = base_scores[a] + gamma * layer_values[successor]
                        self.action_evaluations += 1
                        if best_a < 0 or q > best_q:
                            best_q, best_a = q, a
                    if a % 256 == 0:
                        self._check_time(start)
                if best_a < 0:
                    raise ValueError("A nonterminal planning state has no valid action.")
                if not isfinite(best_q):
                    raise ValueError("Planning rewards and returns must be finite.")
                current[s] = best_q
                key = (rem, decode(s))
                values[key], policy[key] = best_q, best_a
                self._check_time(start)
            layer_values = current
        self._check_time(start)
        self.values, self.policy, self.states = values, policy, count
        return values[initial]

    def _solve_general(self, initial, start):
        layers, count = [{initial}], 1
        for _ in range(_remaining(initial)):
            layer = set()
            for key in layers[-1]:
                for a in _actions(self.model, key):
                    layer.add(_successor(self.model, key, int(a)))
                    self._check_states(count + len(layer))
                    self._check_time(start)
                self._check_time(start)
            layers.append(layer)
            count += len(layer)
        values, policy = {}, {}
        for key in layers[-1]:
            values[key] = 0.0
            self._check_time(start)
        for layer in reversed(layers[:-1]):
            for key in layer:
                actions = _actions(self.model, key)
                q = []
                for a in actions:
                    action = int(a)
                    successor = _successor(self.model, key, action)
                    q.append(self.model.reward(key, action) + self.gamma * values[successor])
                    self.action_evaluations += 1
                    self._check_time(start)
                values[key], policy[key] = _best(q, actions)
                self._check_time(start)
        self._check_time(start)
        self.values, self.policy, self.states = values, policy, count
        return values[initial]

    def act(self, obs):
        key = obs["tabular_state_key"]
        if _remaining(key) == 0:
            raise ValueError("Cannot act on a terminal planning state.")
        mask = obs.get("valid_action_mask")
        action = self.policy.get(key)
        if action is not None and (mask is None or (action < len(mask) and mask[action])):
            return action
        if mask is None:
            mask = self.model.mask(key)
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise ValueError("A nonterminal planning state has no valid action.")
        # A fallback is explicitly myopic; solve() still raises on resource limits.
        q = [self.model.reward(key, int(a)) for a in valid]
        self.action_evaluations += len(valid)
        return _best(q, valid)[1]


class RollingPlanner:
    """Replan a depth-limited, zero-bootstrap objective at each decision.

    An explicit stack avoids Python recursion limits. Work/cache size can
    still grow rapidly with depth; no ExactDP resource budget applies here.
    Action evaluations accumulate across decisions in one evaluation episode.
    """

    def __init__(self, model, depth=2, gamma=1.0):
        try:
            self.depth = index(depth)
        except TypeError as error:
            raise ValueError("depth must be a positive integer.") from error
        if self.depth < 1:
            raise ValueError("depth must be a positive integer.")
        self.model, self.gamma = model, float(gamma)
        if not isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and nonnegative.")
        self.action_evaluations = 0

    def act(self, obs):
        key = obs["tabular_state_key"]
        remaining = _remaining(key)
        if remaining == 0:
            raise ValueError("Cannot act on a terminal planning state.")
        depth = min(self.depth, remaining)

        def lookahead(state, depth):
            if depth == 0 or state[0] == 0:
                return 0.0, -1
            actions = _actions(self.model, state)
            q = []
            for a in actions:
                action = int(a)
                reward = self.model.reward(state, action)
                successor = _successor(self.model, state, action)
                child_value = yield successor, depth - 1
                q.append(reward + self.gamma * child_value[0])
            self.action_evaluations += len(actions)
            return _best(q, actions)

        # Drive suspended lookaheads without recursive Python calls. Model calls
        # occur just as in the cached recursive algorithm;
        # no second forward enumeration or full edge table is needed.
        cache = {}
        requests = [(key, depth)]
        stack = [lookahead(key, depth)]
        response = None
        while stack:
            try:
                child = stack[-1].send(response)
            except StopIteration as completed:
                response = completed.value
                cache[requests.pop()] = response
                stack.pop()
            else:
                if child in cache:
                    response = cache[child]
                else:
                    requests.append(child)
                    stack.append(lookahead(*child))
                    response = None
        return response[1]
