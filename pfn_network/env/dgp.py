from dataclasses import dataclass, replace
from math import comb

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


def make_catalog(n, budget, size, rng):
    """Random nested chains cover counts 0..B without consulting outcomes."""
    capacity = sum(comb(n, b) for b in range(budget + 1))
    if not 1 <= size <= capacity:
        raise ValueError("Requested catalog size exceeds the feasible space.")
    rows = {(0,) * n: None}
    while len(rows) < size:
        z = np.zeros(n, dtype=np.int8)
        for i in rng.permutation(n)[:budget]:
            z[i] = 1
            rows.setdefault(tuple(z.tolist()), None)
            if len(rows) == size:
                break
    return np.array(list(rows), dtype=np.int8)


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


@dataclass
class TaskBundle:
    public: PublicTask
    oracle: EnvModel
    task_id: str

    def env(self, seed, prediction=None, score_objective=False):
        source = ScoreReward(prediction) if score_objective else self.oracle
        return NetworkEnv(self.public, source, seed, prediction)

    def with_setting(self, dynamics, horizon):
        return replace(self, public=replace(self.public, dynamics=Dynamics(dynamics), horizon=horizon))


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
    catalog = make_catalog(n, prior.budget, prior.catalog_size, rng)
    public = PublicTask(adjacency, features, catalog, prior.budget,
                        prior.horizon if horizon is None else horizon,
                        Dynamics(dynamics), prior.fatigue_cost)
    coefficients = rng.normal(0, 0.5, (5, prior.feature_dim + 1))
    return TaskBundle(public, EnvModel(coefficients, prior.node_noise), f"{split}:{root_seed}:{index}")


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
