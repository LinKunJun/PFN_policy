from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from .core import exposures, fatigue_penalty, valid_mask, transition


def regression_features(features, degree, treatment, majority):
    x = np.column_stack((np.ones(len(features)), features, features ** 2,
                         degree / max(1, len(features) - 1)))
    group = (2 * treatment + majority).astype(int)
    return (np.eye(4)[group, :, None] * x[:, None, :]).reshape(len(x), -1)


def fit_factual_prediction(task, assignments, majority, outcomes, alpha=1.0):
    """Example upstream estimator, deliberately NOT called a static PFN."""
    degree = task.adjacency.sum(1)
    design = np.concatenate([regression_features(task.features, degree, z, s)
                             for z, s in zip(assignments, majority)])
    model = Ridge(alpha=alpha, fit_intercept=False).fit(design, outcomes.reshape(-1))
    mu = np.empty((task.n, 2, 2))
    for a in (0, 1):
        for s in (0, 1):
            mu[:, a, s] = model.predict(regression_features(
                task.features, degree, np.full(task.n, a), np.full(task.n, s)))
    mu[degree == 0, :, 1] = np.nan  # Unsupported query; never used by any allocation.
    return mu


@dataclass
class StaticScore:
    adjacency: np.ndarray
    prediction: np.ndarray

    def __call__(self, z):
        z = np.asarray(z, dtype=np.int8)
        _, majority = exposures(self.adjacency, z[None])
        return float(self.prediction[np.arange(len(z)), z, majority[0].astype(int)].mean())

    def catalog_values(self, catalog):
        _, majority = exposures(self.adjacency, catalog)
        values = self.prediction[np.arange(catalog.shape[1]), catalog, majority.astype(int)].mean(1)
        if not np.isfinite(values).all():
            raise ValueError("A used treatment/exposure prediction is missing or nonfinite.")
        return values


class PlanningModel:
    """Known public dynamics plus a specified base score; no environment access."""
    def __init__(self, task, base_scores, label="predicted_score"):
        self.task, self.base_scores, self.label = task, np.asarray(base_scores), label

    def mask(self, key):
        return valid_mask(self.task, key)

    def next(self, key, action):
        return transition(self.task, key, action)

    def reward(self, key, action):
        return float(self.base_scores[action] - fatigue_penalty(self.task, key))
