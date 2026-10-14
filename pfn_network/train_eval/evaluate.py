"""Independent policy evaluation, comparison tables, and plots."""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from ..runtime import action_generator as make_action_generator

from ..baselines.bandit import Bandit
from ..baselines.offline import CatalogSearch
from ..baselines.planning import ExactDP, PlanningLimit, RollingPlanner
from ..baselines.tabular import TabularQ
from ..env.core import exposures, fatigue_penalty, initial_key
from ..env.dgp import TaskPrior, sample_task, task_seed
from ..env.score import PlanningModel, StaticScore
from ..models import FactualRecord, ModelConfig, PFNPolicy
from .train import (PPOConfig, PPOTrainer, RolloutBuffer, collect_episode, episode_seed,
                    prediction_for, seed_everything, timed, write_json)


class Evaluator:
    """Truth lives here; no method passes these quantities to a policy."""
    def __init__(self, bundle):
        self.bundle = bundle
        exact, _ = exposures(bundle.public.adjacency, bundle.public.catalog)
        self.base_means = bundle.oracle.prepare(bundle.public, exact)

    def mean_reward(self, key, action):
        return float(self.base_means[action] - fatigue_penalty(self.bundle.public, key))


def evaluate_policy(bundle, act, seeds, gamma=1.0, context=(), episode=0, device="cpu"):
    """Each replicate receives a private context copy; evaluation never trains."""
    evaluator = Evaluator(bundle)
    rows = []
    for seed in seeds:
        env = bundle.env(int(seed))
        obs, _ = env.reset()
        local = list(context)
        observed = expected = decision_seconds = 0.0
        while obs["tabular_state_key"][0] > 0:
            with timed(device) as clock:
                action = act(obs, tuple(local), episode)
            decision_seconds += clock["seconds"]
            expected += gamma ** obs["episode_time"] * evaluator.mean_reward(obs["tabular_state_key"], action)
            next_obs, reward, done, _, _ = env.step(action)
            observed += gamma ** obs["episode_time"] * reward
            local.append(FactualRecord.from_step(obs, action, reward, next_obs, done, episode))
            obs = next_obs
        rows.append({"return": observed, "expected_return": expected,
                     "decision_seconds": decision_seconds,
                     "invalid_actions": env.invalid_attempts})
    return rows


def exact_planner(config, model):
    return ExactDP(model, config["ppo"]["gamma"], **config["planning"])


def calibrate_targets(config, output):
    """Evaluator-only validation reference. Test truth cannot set these targets."""
    prior, rows = TaskPrior(**config["prior"]), []
    gamma = config["ppo"]["gamma"]
    for dynamics in ("FC", "CD"):
        for index in range(config["meta"]["validation_tasks"]):
            bundle = sample_task(prior, config["seed"], "validation", index, dynamics)
            evaluator = Evaluator(bundle)
            model = PlanningModel(bundle.public, evaluator.base_means, "oracle_validation_only")
            dp = exact_planner(config, model)
            optimum = dp.solve(initial_key(bundle.public))
            values = []
            for replicate in range(config["evaluation"]["episodes"]):
                rng = np.random.default_rng(task_seed(config["seed"], "validation", index, 300 + replicate))
                key, total = initial_key(bundle.public), 0.0
                for t in range(prior.horizon):
                    a = int(rng.choice(np.flatnonzero(model.mask(key))))
                    total += gamma ** t * model.reward(key, a)
                    key = model.next(key, a)
                values.append(total)
            rows.append({"dynamics": dynamics, "task": index, "random_value": np.mean(values),
                         "oracle_optimum": optimum})
    frame = pd.DataFrame(rows)
    targets = {}
    for dynamics, group in frame.groupby("dynamics"):
        baseline, optimum = group[["random_value", "oracle_optimum"]].mean()
        targets[dynamics] = float(baseline + config["evaluation"]["target_fraction"] * (optimum - baseline))
    frame.to_csv(Path(output) / "validation_target_calibration.csv", index=False)
    write_json(Path(output) / "targets.json", {"targets": targets,
               "source": "validation tasks only; absolute return threshold",
               "prior": asdict(prior), "seed": config["seed"], "gamma": gamma})
    return targets


def offline_table(bundle, prediction, model, seed, device, restarts):
    task = bundle.public
    score = StaticScore(task.adjacency, prediction)
    scores = score.catalog_values(task.catalog)
    evaluator = Evaluator(bundle)
    true_majority = bundle.oracle.majority_means(task)
    majority_values = StaticScore(task.adjacency, true_majority).catalog_values(task.catalog)
    rows = []
    for method in ("Greedy", "Local Swap", "GRASP", "Amortized PFN (T=1)", "Catalog exact"):
        env = bundle.env(seed, prediction=prediction)
        obs, _ = env.reset()
        with timed(device if method.startswith("Amortized") else "cpu") as clock:
            if method.startswith("Amortized"):
                action = model.act(obs, deterministic=True)[0]
            elif method == "Catalog exact":
                action = int(np.argmax(scores))
            else:
                solver = CatalogSearch(task.catalog, scores, seed)
                if method == "Greedy":
                    action = solver.greedy()
                elif method == "Local Swap":
                    action = solver.local_swap(solver.greedy())
                else:
                    action = solver.grasp(restarts)
        _, reward, _, _, _ = env.step(action)
        rows.append({"method": method, "predicted_value": scores[action],
                     "V_mu": majority_values[action], "V_env": evaluator.base_means[action],
                     "observed_reward": reward, "predicted_gap": scores.max() - scores[action],
                     "decision_seconds": clock["seconds"], "action": action,
                     "action_space": "fixed catalog"})
    return rows


def bandit_table(bundle, prediction, config, seed):
    task, evaluator = bundle.public, Evaluator(bundle)
    means = evaluator.base_means
    scores = StaticScore(task.adjacency, prediction).catalog_values(task.catalog)
    rows, traces = [], []
    for method in ("Random", "UCB", "epsilon-greedy", "Frozen score"):
        env = bundle.env(seed)
        obs, _ = env.reset()
        agent = Bandit(task.m, "FrozenPFN" if method == "Frozen score" else method, seed,
                       sigma_reward=config["prior"]["node_noise"] / np.sqrt(task.n),
                       frozen_scores=scores, **config["bandit"])
        regret = reward_sum = seconds = 0.0
        while obs["tabular_state_key"][0] > 0:
            with timed() as clock:
                action = agent.act(obs)
            seconds += clock["seconds"]
            next_obs, reward, done, _, _ = env.step(action)
            agent.observe(obs, action, reward, next_obs, done)
            regret += float(means.max() - means[action])
            reward_sum += reward
            traces.append({"method": method, "round": obs["episode_time"] + 1,
                           "action": action, "reward": reward, "cumulative_reward": reward_sum,
                           "pseudo_regret": regret, "choice_probability": agent.last_probability,
                           "chosen_count": int(agent.counts[action]), "chosen_mean": agent.means[action]})
            obs = next_obs
        rows.append({"method": method, "pseudo_regret": regret, "cumulative_reward": reward_sum,
                     "decision_seconds_per_round": seconds / task.horizon})
    return rows, traces


def planning_table(bundle, prediction, config, seed):
    task, gamma = bundle.public, config["ppo"]["gamma"]
    scores = StaticScore(task.adjacency, prediction).catalog_values(task.catalog)
    evaluator = Evaluator(bundle)
    rows = []
    for method in ("Myopic", "Rolling Planning (d=2)", "Exact DP", "Oracle Exact DP"):
        source = evaluator.base_means if method == "Oracle Exact DP" else scores
        model = PlanningModel(task, source, "oracle" if method.startswith("Oracle") else "predicted_score")
        agent = exact_planner(config, model) if "Exact" in method else RollingPlanner(
            model, depth=1 if method == "Myopic" else 2, gamma=gamma)
        seconds = 0.0
        planning_start = perf_counter()
        try:
            if isinstance(agent, ExactDP):
                with timed() as clock:
                    agent.solve(initial_key(task))
                seconds += clock["seconds"]
            env = bundle.env(seed)
            obs, _ = env.reset()
            planned = actual = expected = 0.0
            while obs["tabular_state_key"][0] > 0:
                with timed() as clock:
                    action = agent.act(obs)
                seconds += clock["seconds"]
                weight = gamma ** obs["episode_time"]
                planned += weight * model.reward(obs["tabular_state_key"], action)
                expected += weight * evaluator.mean_reward(obs["tabular_state_key"], action)
                next_obs, reward, _, _, _ = env.step(action)
                actual += weight * reward
                obs = next_obs
            rows.append({"dynamics": str(task.dynamics), "method": method, "planned_value": planned,
                         "actual_return": actual, "actual_expected_return": expected,
                         "planning_seconds": seconds, "status": "complete", "model": model.label,
                         "states": getattr(agent, "states", np.nan),
                         "action_evaluations": agent.action_evaluations})
        except PlanningLimit as error:
            rows.append({"dynamics": str(task.dynamics), "method": method,
                         "planned_value": np.nan, "actual_return": np.nan, "actual_expected_return": np.nan,
                         "planning_seconds": perf_counter() - planning_start,
                         "status": str(error), "model": model.label})
    complete = {r["method"]: r for r in rows if r["status"] == "complete"}
    for row in rows:
        row["planned_gap"] = (complete["Exact DP"]["planned_value"] - row["planned_value"]
                              if "Exact DP" in complete and row["model"] == "predicted_score" else np.nan)
        row["actual_optimality_gap"] = (complete["Oracle Exact DP"]["actual_expected_return"] - row["actual_expected_return"]
                                       if "Oracle Exact DP" in complete else np.nan)
    return rows


def learning_table(bundle, pretrained, config, target, task_index, replicate, device, output):
    task, evaluation = bundle.public, config["evaluation"]
    gamma, root = config["ppo"]["gamma"], config["seed"]
    horizon = task.horizon
    total = evaluation["adaptation_interactions"]
    interval = evaluation["evaluate_every"]
    if total % horizon or interval % horizon or total % interval:
        raise ValueError("Adaptation/evaluation intervals must contain complete equal-sized episode batches.")
    evaluation_seeds = [episode_seed(root, "test", task_index, replicate, i, 500)
                        for i in range(evaluation["episodes"])]
    rows, curves, eval_traces = [], [], []
    for method in ("Q-learning", "Dyna-Q", "PPO (From Scratch)", "Amortized PFN"):
        seed = episode_seed(root, "test", task_index, replicate, 0, 400)
        seed_everything(seed)
        tabular = method in ("Q-learning", "Dyna-Q")
        context, seconds, next_episode, first_hit, invalid, eval_invalid = [], 0.0, 0, None, 0, 0
        if tabular:
            agent = TabularQ(task.m, seed, gamma=gamma,
                             dyna_steps=config["dyna_steps"] if method == "Dyna-Q" else 0,
                             **config["tabular"])
            trainer = None
        else:
            agent = PFNPolicy(ModelConfig(**config["model"])).to(device) if method == "PPO (From Scratch)" else pretrained
            agent.eval()
            trainer = PPOTrainer(agent, PPOConfig(**config["ppo"]), seed) if method == "PPO (From Scratch)" else None
            action_generator = (trainer.action_generator if trainer else
                                make_action_generator().manual_seed(seed))

        for interactions in range(0, total + 1, interval):
            eval_generator = make_action_generator().manual_seed(seed + 1)
            if tabular:
                eval_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
                act = lambda obs, history, ep: agent.act(obs, explore=False, rng=eval_rng)
            else:
                act = lambda obs, history, ep: agent.act(
                    obs, history, ep, deterministic=evaluation["deterministic_neural_eval"],
                    generator=eval_generator)[0]
            if tabular:
                stats = evaluate_policy(bundle, act, evaluation_seeds, gamma, context,
                                        next_episode, "cpu")
            else:
                with agent.cached_inference():
                    stats = evaluate_policy(bundle, act, evaluation_seeds, gamma, context,
                                            next_episode, device)
            mean_return = float(np.mean([r["return"] for r in stats]))
            eval_invalid += sum(r["invalid_actions"] for r in stats)
            if first_hit is None and mean_return >= target:
                first_hit = interactions
            row = {"method": method, "dynamics": str(task.dynamics), "interactions": interactions,
                   "long_term_return": mean_return,
                   "expected_return": float(np.mean([r["expected_return"] for r in stats])),
                   "decision_seconds_per_episode": float(np.mean([r["decision_seconds"] for r in stats])),
                   "training_seconds": seconds,
                   "model_updates": agent.model_updates if tabular else 0,
                   "gradient_steps": trainer.gradient_steps if trainer else 0,
                   "context_records": len(context), "target": target}
            curves.append(row)
            for eval_id, record in enumerate(stats):
                eval_traces.append({**record, "method": method, "dynamics": str(task.dynamics),
                                    "interactions": interactions, "evaluation_episode": eval_id})
            if interactions == total:
                rows.append({**row, "interactions_to_target": first_hit if first_hit is not None else np.nan,
                             "target_reached": first_hit is not None, "invalid_actions": invalid + eval_invalid,
                             "target_status": "reached" if first_hit is not None else "not reached",
                             "evaluation_interactions": len(stats) * horizon * (total // interval + 1),
                             "deployment_weight_updates": trainer.gradient_steps if trainer else 0})
                break
            with timed("cpu" if tabular else device) as clock:
                rollout = None if tabular else RolloutBuffer(interval, agent.config.context_length)
                for _ in range(interval // horizon):
                    noise_seed = episode_seed(root, "test", task_index, replicate, next_episode, 600)
                    env = bundle.env(noise_seed)
                    if tabular:
                        obs, _ = env.reset()
                        while obs["tabular_state_key"][0] > 0:
                            action = agent.act(obs, explore=True)
                            next_obs, reward, done, _, _ = env.step(action)
                            agent.observe(obs, action, reward, next_obs, done)
                            obs = next_obs
                    else:
                        steps, context = collect_episode(agent, env, noise_seed, next_episode, context,
                                                         action_generator, PPOConfig(**config["ppo"]), buffer=rollout)
                    invalid += env.invalid_attempts
                    next_episode += 1
                if trainer is not None:
                    trainer.update(rollout)
            seconds += clock["seconds"]
        name = method.replace(" ", "_").replace("(", "").replace(")", "")
        path = Path(output) / f"task{task_index}_seed{replicate}_{task.dynamics}_{name}"
        driver = {"root_seed": root, "task_index": task_index, "replicate": replicate,
                  "noise_stream": 600, "next_episode": next_episode, "context": context,
                  "binding": task.fingerprint()}
        if tabular:
            agent.save(path.with_suffix(".pkl"), task.fingerprint(), next_episode,
                       {k: v for k, v in driver.items() if k != "context"})
        elif trainer is not None:
            # Driver context contains factual NumPy arrays; keep separately from
            # the weights-only checkpoint so policy weight loading remains safe.
            import pickle
            with open(path.with_suffix(".context.pkl"), "wb") as f:
                pickle.dump(context, f)
            trainer.checkpoint(path.with_suffix(".pt"), {k: v for k, v in driver.items() if k != "context"})
    return rows, curves, eval_traces


def summarize(frame, groups, metrics):
    # Average replicate seeds within each task first; tasks are independent units.
    per_task = frame.groupby(groups + ["task"], dropna=False)[metrics].mean()
    aggregate = per_task.groupby(level=list(range(len(groups)))).agg(["mean", "std", "count"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def benchmark(config, output, static_model, sequential_model, targets, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    prior, settings = TaskPrior(**config["prior"]), config["evaluation"]
    tables = {k: [] for k in range(1, 5)}
    bandit_traces, learning_curves, eval_traces, setup = [], [], [], []
    for task_index in range(settings["test_tasks"]):
        with timed() as generation_time:
            base = sample_task(prior, config["seed"], "test", task_index)
        with timed() as estimation_time:
            prediction_path = (str(Path(config["prediction_dir"]) / f"task_{task_index}.npz")
                               if config.get("prediction_dir") else None)
            prediction = prediction_for(base, config["estimation_rounds"],
                                        task_seed(config["seed"], "test", task_index, 1), prediction_path)
        np.savez_compressed(output / f"task_{task_index}.npz", adjacency=base.public.adjacency,
                            features=base.public.features, catalog=base.public.catalog, mu_hat=prediction,
                            node_ids=np.arange(prior.n))
        with timed() as cache_time:
            StaticScore(base.public.adjacency, prediction).catalog_values(base.public.catalog)
            preparation_env = base.env(task_seed(config["seed"], "test", task_index, 9))
            preparation_env.reset()
        setup.append({"task": task_index, "task_id": base.task_id,
                      "generation_catalog_seconds": generation_time["seconds"],
                      "prediction_seconds": estimation_time["seconds"],
                      "score_and_env_cache_seconds": cache_time["seconds"],
                      "static_experimental_assignments": 0 if prediction_path else config["estimation_rounds"],
                      "prediction_source": "external" if prediction_path else "factual Ridge example",
                      "mean_degree": float(base.public.adjacency.sum(1).mean())})
        for replicate in settings["seeds"]:
            tags = {"task": task_index, "seed": replicate}
            noise_seed = episode_seed(config["seed"], "test", task_index, replicate, 0, 700)
            static = base.with_setting("NONE", 1)
            tables[1].extend([{**r, **tags} for r in offline_table(
                static, prediction, static_model, noise_seed, device, config["grasp_restarts"])])
            bandit = base.with_setting("NONE", settings["bandit_rounds"])
            rows, traces = bandit_table(bandit, prediction, config, noise_seed)
            tables[2].extend([{**r, **tags} for r in rows])
            bandit_traces.extend([{**r, **tags} for r in traces])
            for dynamics in ("FC", "CD"):
                dynamic = base.with_setting(dynamics, prior.horizon)
                tables[3].extend([{**r, **tags} for r in planning_table(dynamic, prediction, config, noise_seed)])
                rows, curves, traces = learning_table(dynamic, sequential_model, config, targets[dynamics],
                                                      task_index, replicate, device, checkpoints)
                tables[4].extend([{**r, **tags} for r in rows])
                learning_curves.extend([{**r, **tags} for r in curves])
                eval_traces.extend([{**r, **tags} for r in traces])
            # Save progress at the boundary of each independent replicate.
            for number, rows in tables.items():
                pd.DataFrame(rows).to_csv(output / f"table{number}_raw.csv", index=False)
            pd.DataFrame(bandit_traces).to_csv(output / "bandit_traces.csv", index=False)
            pd.DataFrame(learning_curves).to_csv(output / "learning_curves.csv", index=False)
            pd.DataFrame(eval_traces).to_csv(output / "evaluation_episodes.csv", index=False)
            print(f"Benchmark complete: task={task_index}, seed={replicate}", flush=True)
    pd.DataFrame(setup).to_csv(output / "preparation_costs.csv", index=False)
    specs = {
        1: (["method"], ["predicted_value", "V_env", "V_mu", "decision_seconds", "predicted_gap"]),
        2: (["method"], ["pseudo_regret", "decision_seconds_per_round", "cumulative_reward"]),
        3: (["dynamics", "method"], ["planned_value", "actual_return", "actual_expected_return", "planning_seconds"]),
        4: (["dynamics", "method"], ["long_term_return", "interactions_to_target", "target_reached",
                                       "invalid_actions", "model_updates", "training_seconds"])}
    for number, (groups, metrics) in specs.items():
        result = summarize(pd.DataFrame(tables[number]), groups, metrics)
        result.to_csv(output / f"table{number}_summary.csv", index=False)
        print(f"\n表格 {number}（先按任务内种子平均，再按独立任务汇总）")
        print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    # Paired differences use identical tasks and replicate seeds.
    frame = pd.DataFrame(tables[4])
    paired = frame.pivot(index=["dynamics", "task", "seed"], columns="method", values="long_term_return")
    for method in paired.columns:
        if method != "Amortized PFN":
            paired[f"PFN_minus_{method}"] = paired["Amortized PFN"] - paired[method]
    paired.to_csv(output / "paired_return_differences.csv")


def plot_curves(frame, x, y, title, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for method, group in frame.groupby("method"):
        per_task = group.groupby(["task", x])[y].mean().unstack(0)
        average, std = per_task.mean(1), per_task.std(1).fillna(0)
        ax.plot(average.index, average, label=method)
        ax.fill_between(average.index.to_numpy(), (average - std).to_numpy(),
                        (average + std).to_numpy(), alpha=0.15)
    ax.set(xlabel="Factual interactions", ylabel=y.replace("_", " "), title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_results(results):
    """Draw existing CSV results without rerunning any experiment."""
    results = Path(results)
    bandit = pd.read_csv(results / "bandit_traces.csv")
    plot_curves(bandit, "round", "pseudo_regret", "Catalog-restricted bandit regret",
                results / "bandit_regret.png")
    learning = pd.read_csv(results / "learning_curves.csv")
    for dynamics, group in learning.groupby("dynamics"):
        plot_curves(group, "interactions", "long_term_return", f"{dynamics}: frozen-policy evaluation",
                    results / f"learning_{dynamics}.png")
