from dataclasses import dataclass, field
import pickle

import numpy as np


@dataclass
class EmpiricalEntry:
    visits: int = 0
    reward_sum: float = 0.0
    # (next state, true terminal) -> [count, conditional reward sum, legal mask]
    outcomes: dict = field(default_factory=dict)

    def add(self, reward, next_key, done, next_mask):
        self.visits += 1
        self.reward_sum += reward
        key = (next_key, done)
        if key not in self.outcomes:
            self.outcomes[key] = [0, 0.0, next_mask.copy()]
        item = self.outcomes[key]
        item[0] += 1
        item[1] += reward

    def sample(self, rng):
        keys = list(self.outcomes)
        counts = np.array([self.outcomes[k][0] for k in keys], dtype=float)
        key = keys[int(rng.choice(len(keys), p=counts / self.visits))]
        count, total, mask = self.outcomes[key]
        return total / count, key[0], key[1], mask


class TabularQ:
    def __init__(self, m, seed=0, alpha=0.1, gamma=1.0,
                 epsilon_start=0.3, epsilon_end=0.05, epsilon_decay=2000,
                 dyna_steps=0):
        self.m, self.alpha, self.gamma = m, alpha, gamma
        self.epsilon_start, self.epsilon_end, self.epsilon_decay = epsilon_start, epsilon_end, epsilon_decay
        self.dyna_steps = dyna_steps
        self.action_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.model_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        self.tie_seed = int(seed)
        self.q, self.model, self.visited_pairs = {}, {}, []
        self.real_updates = self.model_updates = 0

    def values(self, key):
        # Evaluation of unseen states never mutates the training table.
        return self.q.get(key, np.zeros(self.m))

    def act(self, obs, explore=False, rng=None):
        valid = np.flatnonzero(obs["valid_action_mask"])
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.real_updates / self.epsilon_decay)
        if explore and self.action_rng.random() < epsilon:
            return int(self.action_rng.choice(valid))
        values = self.values(obs["tabular_state_key"])[valid]
        tied = valid[values == values.max()]
        if len(tied) == 1:
            return int(tied[0])
        if explore:
            chooser = self.action_rng
        elif rng is not None:
            chooser = rng
        else:
            # A standalone read-only query has a reproducible state-specific tie.
            # Full policy evaluation supplies its own independent random stream.
            remaining, memory = obs["tabular_state_key"]
            chooser = np.random.default_rng(np.random.SeedSequence([self.tie_seed, remaining, *memory]))
        return int(chooser.choice(tied))

    def _update(self, key, action, reward, next_key, done, next_mask):
        target = reward if done else reward + self.gamma * self.values(next_key)[next_mask].max()
        row = self.q.setdefault(key, np.zeros(self.m))
        row[action] += self.alpha * (target - row[action])

    def observe(self, obs, action, reward, next_obs, done):
        key, next_key = obs["tabular_state_key"], next_obs["tabular_state_key"]
        mask = next_obs["valid_action_mask"]
        self._update(key, action, reward, next_key, done, mask)
        self.real_updates += 1
        if self.dyna_steps == 0:
            return
        pair = (key, action)
        if pair not in self.model:
            self.model[pair] = EmpiricalEntry()
            self.visited_pairs.append(pair)
        self.model[pair].add(reward, next_key, done, mask)
        for _ in range(self.dyna_steps):
            pair = self.visited_pairs[int(self.model_rng.integers(len(self.visited_pairs)))]
            reward_sim, next_sim, done_sim, mask_sim = self.model[pair].sample(self.model_rng)
            self._update(*pair, reward_sim, next_sim, done_sim, mask_sim)
            self.model_updates += 1

    def save(self, path, binding, next_episode, driver_config):
        """Episode-boundary checkpoint, including separate RNG streams."""
        with open(path, "wb") as f:
            pickle.dump({"learner": self, "binding": binding, "next_episode": next_episode,
                         "driver_config": driver_config}, f)

    @staticmethod
    def load(path, binding):
        # Load only checkpoints produced by this project from a trusted source.
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload["binding"] != binding:
            raise ValueError("Checkpoint graph/catalog/node order/horizon/dynamics mismatch.")
        if not hasattr(payload["learner"], "tie_seed"):
            # Older checkpoints retain their training RNG; only read-only fallback ties use zero.
            payload["learner"].tie_seed = 0
        return payload["learner"], payload["next_episode"], payload["driver_config"]
