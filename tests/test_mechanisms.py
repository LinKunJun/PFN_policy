"""Analytical checks; run explicitly with pytest in the user's configured environment."""
from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch

from pfn_network.baselines.offline import greedy_full, grasp_full
from pfn_network.baselines.planning import ExactDP, RollingPlanner
from pfn_network.baselines.tabular import TabularQ
from pfn_network.env.core import (Dynamics, NetworkEnv, PublicTask, exposures,
                                  initial_key, transition, valid_mask)
from pfn_network.env.score import PlanningModel, StaticScore
from pfn_network.models import ModelConfig, PFNPolicy, FactualRecord
from pfn_network.train_eval.train import RolloutBuffer, finish_episode


class KnownRewards:
    def __init__(self, rewards, noise=0.0):
        self.rewards, self.noise = np.asarray(rewards), noise

    def prepare(self, task, exact):
        return self.rewards.copy()

    def sample_noise(self, rng, n):
        return float(rng.normal(0, self.noise))


def two_node_task(dynamics="CD", horizon=3):
    return PublicTask(np.array([[0, 1], [1, 0]], dtype=np.int8), np.zeros((2, 4), dtype=np.float32),
                      np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int8),
                      2, horizon, Dynamics(dynamics), 2.0)


def test_strict_majority_tie_and_isolate():
    a = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    exact, majority = exposures(a, np.array([[0, 1, 0, 1]]))
    assert exact[0, 0] == 0.5 and not majority[0, 0]
    assert exact[0, 3] == 0 and not majority[0, 3]


def test_cooldown_two_periods_and_invalid_action_before_noise():
    task = two_node_task(horizon=4)
    left = NetworkEnv(task, KnownRewards([0, 3, 3, 5], 0.2), 17)
    right = NetworkEnv(task, KnownRewards([0, 3, 3, 5], 0.2), 17)
    left.reset()
    right.reset()
    obs, r, *_ = left.step(1)
    assert r == right.step(1)[1]
    assert obs["tabular_state_key"][1] == (2, 0)
    with pytest.raises(ValueError):
        left.step(1)
    assert left.step(0)[1] == right.step(0)[1]
    assert not left._mask[1]
    obs, *_ = left.step(0)
    assert obs["valid_action_mask"][1]
    assert left.invalid_attempts == 1


def test_lookup_only_step_and_truth_boundary(monkeypatch):
    import pfn_network.env.core as core
    env = NetworkEnv(two_node_task(), KnownRewards([0, 3, 3, 5]), 3)
    obs, info = env.reset()
    assert "oracle_mu" not in obs and not info
    assert type(obs["task_budget"]) is int and obs["task_budget"] == env.task.budget
    assert all("reward" not in name and "mean" not in name for name in obs)
    def forbidden(*args):
        raise AssertionError("step recomputed exposures")
    monkeypatch.setattr(core, "exposures", forbidden)
    _, reward, done, truncated, info = env.step(1)
    assert isinstance(reward, float) and reward == 3 and not truncated and not info


def test_fc_separable_and_no_post_terminal_cost():
    task = replace(two_node_task("FC"), catalog=np.array([[0, 0], [1, 0]], dtype=np.int8))
    model = PlanningModel(task, [0, 0.8])
    exact = ExactDP(model)
    assert exact.solve(initial_key(task)) == pytest.approx(0.8)
    rolling = RollingPlanner(model, 2)
    key, actions = initial_key(task), []
    while key[0]:
        action = rolling.act({"tabular_state_key": key})
        actions.append(action)
        key = model.next(key, action)
    assert actions == [0, 0, 1]
    env = NetworkEnv(task, KnownRewards([0, 0.8]), 0)
    env.reset()
    assert sum(env.step(a)[1] for a in actions) == pytest.approx(0.8)


def test_cooldown_planning_hand_example():
    task = two_node_task()
    model = PlanningModel(task, [0, 3, 3, 5])
    dp = ExactDP(model)
    assert dp.solve(initial_key(task)) == pytest.approx(6)
    myopic = RollingPlanner(model, 1)
    key, total = initial_key(task), 0
    while key[0]:
        action = myopic.act({"tabular_state_key": key})
        total += model.reward(key, action)
        key = model.next(key, action)
    assert total == 5


def test_gaussian_reward_terminal_gae():
    buffer = RolloutBuffer(4, 0)
    buffer.rewards[:] = [2, 3, 100, 200]
    buffer.values[:] = [0.3, 0.4, 10, 20]
    buffer.done[:] = [False, True, False, True]
    finish_episode(buffer, 2, 4, 0.5, 1.0)
    finish_episode(buffer, 0, 2, 0.5, 1.0)
    np.testing.assert_allclose(buffer.targets, [3.5, 3, 200, 200])


def test_q_dyna_zero_and_read_only_evaluation(tmp_path):
    q = TabularQ(4, seed=8, dyna_steps=0)
    dyna = TabularQ(4, seed=8, dyna_steps=0)
    task = two_node_task()
    env = NetworkEnv(task, KnownRewards([0, 3, 3, 5]), 0)
    obs, _ = env.reset()
    while obs["tabular_state_key"][0]:
        action = q.act(obs, explore=True)
        assert action == dyna.act(obs, explore=True)
        next_obs, reward, done, _, _ = env.step(action)
        q.observe(obs, action, reward, next_obs, done)
        dyna.observe(obs, action, reward, next_obs, done)
        obs = next_obs
    assert q.q.keys() == dyna.q.keys()
    for key in q.q:
        np.testing.assert_array_equal(q.q[key], dyna.q[key])
    before = pickle.dumps(q)
    unseen = {"tabular_state_key": (8, (0, 0)), "valid_action_mask": np.ones(4, bool)}
    chosen = q.act(unseen)
    assert unseen["valid_action_mask"][chosen] and q.act(unseen) == chosen
    assert before == pickle.dumps(q)
    path = tmp_path / "q.pkl"
    q.save(path, task.fingerprint(), 1, {"root_seed": 1})
    loaded, episode, _ = TabularQ.load(path, task.fingerprint())
    assert episode == 1 and loaded.real_updates == 3


def test_dyna_factual_frequencies():
    from pfn_network.baselines.tabular import EmpiricalEntry
    entry = EmpiricalEntry()
    mask = np.array([True, False])
    entry.add(2.0, (1, (0,)), False, mask)
    entry.add(4.0, (1, (0,)), False, mask)
    entry.add(9.0, (0, (0,)), True, np.zeros(2, bool))
    assert entry.visits == 3 and entry.reward_sum / entry.visits == 5
    assert entry.outcomes[((1, (0,)), False)][:2] == [2, 6.0]


def test_majority_threshold_joint_gain():
    a = np.ones((3, 3), dtype=int) - np.eye(3, dtype=int)
    mu = np.array([[[0, 10], [-1, 10]]] * 3)
    score = StaticScore(a, mu)
    assert score(np.array([1, 1, 0])) == pytest.approx(8 / 3)
    assert not greedy_full(score, 3, 2).any()
    result = grasp_full(score, 3, 2, restarts=8, seed=0)
    assert score(result) == pytest.approx(8 / 3)


def test_joint_mask_probabilities_and_gradient():
    torch.manual_seed(4)
    task = two_node_task()
    env = NetworkEnv(task, KnownRewards([0, 3, 3, 5]), 1)
    obs, _ = env.reset()
    obs["valid_action_mask"] = np.array([True, False, False, False])
    model = PFNPolicy(ModelConfig(width=16, heads=2, transformer_layers=1)).eval()
    dist, value = model(obs)
    np.testing.assert_array_equal(dist.probs.detach().numpy(), [1, 0, 0, 0])
    loss = -dist.log_prob(torch.tensor(0)) + value.square()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_node_permutation_equivariance():
    torch.manual_seed(9)
    task = two_node_task("FC")
    env = NetworkEnv(task, KnownRewards([0, 3, 3, 5]), 0)
    obs, _ = env.reset()
    obs["pfn_state"][0, 0] = 0.7
    obs["pfn_state"][1, 0] = -0.3
    model = PFNPolicy(ModelConfig(width=16, heads=2, transformer_layers=1)).eval()
    perm = np.array([1, 0])
    permuted = dict(obs)
    for name in ["pfn_state"]:
        permuted[name] = obs[name][perm]
    for name in ["catalog", "catalog_exact_exposure", "catalog_majority_exposure"]:
        permuted[name] = obs[name][:, perm]
    permuted["edge_index"] = np.argsort(perm)[obs["edge_index"]]
    p, value = model(obs)
    p2, value2 = model(permuted)
    torch.testing.assert_close(p.probs, p2.probs, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(value, value2, atol=1e-6, rtol=1e-5)


def test_q_ties_use_external_rng_without_mutating_learner():
    q = TabularQ(4, seed=5)
    obs = {"tabular_state_key": (2, (0, 0)),
           "valid_action_mask": np.array([True, False, True, True])}
    before = pickle.dumps(q)
    rng = np.random.default_rng(16)
    actions = [q.act(obs, rng=rng) for _ in range(600)]
    assert set(actions) == {0, 2, 3}
    assert pickle.dumps(q) == before
    q.q[obs["tabular_state_key"]] = np.array([1.0, 10.0, 1.0 + 1e-10, 0.0])
    assert all(q.act(obs, rng=rng) == 2 for _ in range(20))


def test_sparse_gcn_matches_edge_index_forward_and_backward():
    GCNConv = pytest.importorskip("torch_geometric.nn").GCNConv
    torch.manual_seed(12)
    model = PFNPolicy(ModelConfig(width=8, heads=2))
    # Node 3 is isolated; both paths must add its self-loop.
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    original = GCNConv(8, 8)
    sparse = model.gnn[0]
    sparse.load_state_dict(original.state_dict())
    x = torch.randn(4, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    expected = original(x, edges)
    actual = sparse(y, model._normalized_adjacency(edges, 4))
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(y.grad, x.grad)
    for left, right in zip(sparse.parameters(), original.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


def test_prepared_rollout_matches_online_history():
    from pfn_network.train_eval.train import PPOConfig, collect_episode
    torch.manual_seed(9)
    model = PFNPolicy(ModelConfig(width=8, heads=2, context_length=4))
    env = NetworkEnv(two_node_task(), KnownRewards([0, 3, 3, 5]), 0)
    buffer = RolloutBuffer(6, 4)
    context = []
    for episode in range(2):
        buffer, context = collect_episode(model, env, episode, episode, context, None,
                                          PPOConfig(), deterministic=True, buffer=buffer)
    data = model.prepare_rollout(buffer)
    plans = model.prepare_minibatches(data, [np.arange(len(buffer))], 3)
    with torch.no_grad():
        for plan in plans[0]:
            dist, values = model.forward_prepared(data, plan)
            for local, row in enumerate(plan['ids'].tolist()):
                history = [buffer.records[i] for i in
                           buffer.history_indices[row, :buffer.history_lengths[row]]]
                expected_dist, expected_value = model(buffer.observations[row], history,
                                                       int(buffer.episodes[row]))
                torch.testing.assert_close(dist.logits[local], expected_dist.logits,
                                           atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(values[local], expected_value, atol=2e-5, rtol=2e-5)
