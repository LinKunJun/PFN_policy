import numpy as np


class Bandit:
    def __init__(self, m, method, seed, epsilon=0.1, sigma_reward=1.0, ucb_c=1.0,
                 frozen_scores=None):
        self.method, self.epsilon = method, epsilon
        self.sigma_reward, self.ucb_c = sigma_reward, ucb_c
        self.rng = np.random.default_rng(seed)
        self.counts, self.means = np.zeros(m, dtype=np.int64), np.zeros(m)
        self.frozen_scores = frozen_scores
        self.last_probability = 1.0

    def act(self, obs):
        valid = np.flatnonzero(obs["valid_action_mask"])
        if self.method == "Random":
            self.last_probability = 1 / len(valid)
            return int(self.rng.choice(valid))
        if self.method == "FrozenPFN":
            return int(valid[np.argmax(self.frozen_scores[valid])])
        if self.method == "UCB":
            unseen = valid[self.counts[valid] == 0]
            if len(unseen):
                return int(unseen[0])
            bonus = self.ucb_c * self.sigma_reward * np.sqrt(
                2 * np.log(max(2, int(self.counts.sum()) + 1)) / self.counts[valid])
            return int(valid[np.argmax(self.means[valid] + bonus)])
        if self.method == "epsilon-greedy":
            # Unvisited scores start at zero; ties use the lowest action index.
            greedy = int(valid[np.argmax(self.means[valid])])
            action = int(self.rng.choice(valid)) if self.rng.random() < self.epsilon else greedy
            self.last_probability = self.epsilon / len(valid) + (1 - self.epsilon) * (action == greedy)
            return action
        raise ValueError(f"Unknown bandit: {self.method}")

    def observe(self, obs, action, reward, next_obs, done):
        self.counts[action] += 1
        self.means[action] += (reward - self.means[action]) / self.counts[action]
