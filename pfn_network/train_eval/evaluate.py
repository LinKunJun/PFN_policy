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


def _prepare_benchmark_task(config, output, task_index):
    prior = TaskPrior(**config["prior"])
    with timed() as generation_time:
        base = sample_task(prior, config["seed"], "test", task_index)
    with timed() as estimation_time:
        prediction_path = (str(Path(config["prediction_dir"]) / f"task_{task_index}.npz")
                           if config.get("prediction_dir") else None)
        prediction = prediction_for(base, config["estimation_rounds"],
                                    task_seed(config["seed"], "test", task_index, 1), prediction_path)
    # Both catalog variants must be recoverable: with_setting('CD') rebuilds
    # the CD-aware catalog, while FC/NONE share the non-CD exposure design.
    cd_catalog = base.with_setting('CD', prior.horizon).public.catalog
    np.savez_compressed(output / f"task_{task_index}.npz", adjacency=base.public.adjacency,
                        features=base.public.features, catalog=base.public.catalog,
                        catalog_cd=cd_catalog, mu_hat=prediction,
                        node_ids=np.arange(prior.n))
    with timed() as cache_time:
        StaticScore(base.public.adjacency, prediction).catalog_values(base.public.catalog)
        preparation_env = base.env(task_seed(config["seed"], "test", task_index, 9))
        preparation_env.reset()
    setup = {"task": task_index, "task_id": base.task_id,
                  "generation_catalog_seconds": generation_time["seconds"],
                  "prediction_seconds": estimation_time["seconds"],
                  "score_and_env_cache_seconds": cache_time["seconds"],
                  "static_experimental_assignments": 0 if prediction_path else config["estimation_rounds"],
                  "prediction_source": "external" if prediction_path else "factual Ridge example",
                  "mean_degree": float(base.public.adjacency.sum(1).mean())}
    return base, prediction, setup


def _evaluate_benchmark_job(config, checkpoints, static_model, sequential_model,
                            targets, device, base, prediction, task_index, replicate):
    prior, settings = TaskPrior(**config["prior"]), config["evaluation"]
    tables = {k: [] for k in range(1, 5)}
    bandit_traces, learning_curves, eval_traces = [], [], []
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
    return dict(tables=tables, bandit_traces=bandit_traces,
                learning_curves=learning_curves, eval_traces=eval_traces)


def _write_benchmark_progress(output, completed):
    """Only the parent writes aggregate CSVs, in original task/seed order."""
    tables = {k: [] for k in range(1, 5)}
    traces = {name: [] for name in ('bandit_traces', 'learning_curves', 'eval_traces')}
    for job_id in sorted(completed):
        result = completed[job_id]
        for number in tables:
            tables[number].extend(result['tables'][number])
        for name in traces:
            traces[name].extend(result[name])
    frames = {f'table{number}_raw.csv': pd.DataFrame(rows) for number, rows in tables.items()}
    for name, filename in [('bandit_traces', 'bandit_traces.csv'),
                           ('learning_curves', 'learning_curves.csv'),
                           ('eval_traces', 'evaluation_episodes.csv')]:
        frames[filename] = pd.DataFrame(traces[name])
    for filename, frame in frames.items():
        destination = Path(output) / filename
        temporary = destination.with_suffix('.csv.tmp')
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    return tables


def _summarize_benchmark(output, tables):
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



def _benchmark_serial(config, output, static_model, sequential_model, targets, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    completed, setup = {}, []
    settings = config['evaluation']
    for task_index in range(settings['test_tasks']):
        base, prediction, preparation = _prepare_benchmark_task(config, output, task_index)
        setup.append(preparation)
        for replicate in settings['seeds']:
            completed[len(completed)] = _evaluate_benchmark_job(
                config, checkpoints, static_model, sequential_model, targets, device,
                base, prediction, task_index, replicate)
            tables = _write_benchmark_progress(output, completed)
            print(f'Benchmark complete: task={task_index}, seed={replicate}', flush=True)
    pd.DataFrame(setup).to_csv(output / 'preparation_costs.csv', index=False)
    _summarize_benchmark(output, tables)


def _uniform_random_reference(bundle, gamma, seed, trajectories=3000):
    """Evaluator-side expected return of uniform legal actions (MC estimate).

    The sampled actions respect FC/CD state masks. Only the expected rewards
    are used, so no observation-noise sampling is mixed into the reference.
    No learned policy or third-party baseline is executed here.
    """
    from ..env.core import transition, valid_mask

    rng = np.random.default_rng(seed)
    task = bundle.public
    evaluator = Evaluator(bundle)
    total = 0.0
    for _ in range(trajectories):
        state, value, discount = initial_key(task), 0.0, 1.0
        while state[0] > 0:
            valid = np.flatnonzero(valid_mask(task, state))
            action = int(valid[rng.integers(len(valid))])
            value += discount * evaluator.mean_reward(state, action)
            discount *= gamma
            state = transition(task, state, action)
        total += value
    return total / trajectories


def _learning_normalization(results, learning):
    """Use per-test-task random and exact-oracle references for ALL methods.

    Random=0 and task-specific exact optimal return=1. References are used
    only to visualize completed evaluations, never to change model decisions.
    If an exact oracle is unavailable, fail instead of silently substituting
    an approximate method or computing a misleading normalized variance.
    """
    import json

    results = Path(results)
    config = json.loads((results / "config.json").read_text(encoding="utf-8"))
    planning = pd.read_csv(results / "table3_raw.csv")
    required = {"dynamics", "task", "method", "status", "planned_value"}
    if required - set(planning.columns):
        raise ValueError("table3_raw.csv lacks fields: " + str(sorted(required - set(planning.columns))))
    exact = planning.loc[(planning["method"] == "Oracle Exact DP") &
                         (planning["status"] == "complete"),
                         ["dynamics", "task", "planned_value"]].copy()
    if exact.empty:
        raise ValueError("No completed Oracle Exact DP references in table3_raw.csv")
    # Multiple seeds for the same task should produce exactly the same
    # noise-free optimal value; catch an inconsistent input before plotting.
    grouped = exact.groupby(["dynamics", "task"])["planned_value"]
    spread = grouped.max() - grouped.min()
    if (spread > 1e-5).any():
        raise ValueError("Oracle Exact DP values differ across seeds for the same task")
    oracle = grouped.mean().to_dict()

    prior = TaskPrior(**config["prior"])
    root = int(config["seed"])
    gamma = float(config["ppo"]["gamma"])
    expected = set(map(tuple, learning[["dynamics", "task"]].drop_duplicates().to_numpy()))
    missing = sorted(expected - set(oracle))
    if missing:
        raise ValueError("Cannot normalize without completed Oracle Exact DP for: " +
                         str(missing) + ". Check table3_raw.csv and planning limits.")

    references = []
    for dynamics, task_index in sorted(expected):
        task_index = int(task_index)
        bundle = sample_task(prior, root, "test", task_index, dynamics)
        # Benchmark writes task_<id>.npz. Check the regenerated public task to
        # prevent silently normalizing against a different environment prior.
        task_path = results / f"task_{task_index}.npz"
        if task_path.exists():
            with np.load(task_path) as saved:
                catalog_key = 'catalog_cd' if dynamics == 'CD' else 'catalog'
                if catalog_key not in saved.files:
                    raise ValueError(f'{task_path} is missing {catalog_key}; cannot safely '
                                     'reconstruct a dynamics-specific normalization baseline')
                for name, live in (("adjacency", bundle.public.adjacency),
                                   ("features", bundle.public.features),
                                   (catalog_key, bundle.public.catalog)):
                    if not np.array_equal(saved[name], live):
                        raise ValueError(f"{task_path} differs in {name}; cannot safely "
                                         "reconstruct the normalization baseline")
        baseline = _uniform_random_reference(
            bundle, gamma, task_seed(root, "test", task_index, 901))
        optimum = float(oracle[(dynamics, task_index)])
        if not np.isfinite(baseline) or not np.isfinite(optimum) or optimum - baseline <= 1e-6:
            raise ValueError(f"Invalid normalization range for {dynamics} task {task_index}: "
                             f"random={baseline}, oracle={optimum}")
        references.append({"dynamics": dynamics, "task": task_index,
                           "random_reference": baseline, "oracle_reference": optimum})
    return pd.DataFrame(references)


def plot_curves(frame, x, y, title, path, normalization=None):
    """Plot mean +/- sample SD across tasks (seed replicates averaged first).

    For FC/CD, first normalize each task's return with its own random and
    oracle reference, then compute across-task statistics. Do not pool raw
    task reward scales. The original unnormalized bandit chart is unchanged.
    """
    import matplotlib.pyplot as plt

    if normalization is not None:
        keys = ["dynamics", "task"]
        frame = frame.merge(normalization, on=keys, how="left", validate="many_to_one")
        if frame[["random_reference", "oracle_reference"]].isna().any().any():
            raise ValueError("Normalization references missing for learning curve rows")
        scale = frame["oracle_reference"] - frame["random_reference"]
        if (scale <= 1e-6).any():
            raise ValueError("Invalid oracle-random range in normalization references")
        # Same transform for every method in the same task; no clipping.
        frame = frame.assign(_normalized_return=(frame[y] - frame["random_reference"]) / scale)
        value = "_normalized_return"
    else:
        value = y

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for method, group in frame.groupby("method"):
        per_task = group.groupby(["task", x])[value].mean().unstack(0).sort_index()
        if normalization is not None and per_task.isna().any().any():
            raise ValueError(f"Incomplete task/interactions coverage for {method}")
        average, std = per_task.mean(axis=1), per_task.std(axis=1).fillna(0)
        ax.plot(average.index, average, label=method)
        ax.fill_between(average.index.to_numpy(), (average - std).to_numpy(),
                        (average + std).to_numpy(), alpha=0.15)
    if normalization is None:
        ax.set(xlabel="Factual interactions", ylabel=y.replace("_", " "), title=title)
    else:
        ax.set(xlabel="Factual interactions",
               ylabel="Normalized return (random=0, exact oracle=1)",
               title=title + " (task-normalized)")
        ax.axhline(0, color="0.4", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.axhline(1, color="0.4", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_results(results):
    """Redraw the original PNG filenames from saved results (no training)."""
    results = Path(results)
    bandit = pd.read_csv(results / "bandit_traces.csv")
    learning = pd.read_csv(results / "learning_curves.csv")
    normalization = _learning_normalization(results, learning)
    # This figure plots the independent bandit experiment, not the learned PFN.
    # Retain its original absolute pseudo-regret units and behavior.
    plot_curves(bandit, "round", "pseudo_regret", "Catalog-restricted bandit regret",
                results / "bandit_regret.png")
    for dynamics, group in learning.groupby("dynamics"):
        plot_curves(group, "interactions", "long_term_return",
                    f"{dynamics}: frozen-policy evaluation",
                    results / f"learning_{dynamics}.png", normalization=normalization)


# Single-file compatibility with the unchanged train_eval.run entry point.
import argparse
import io
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from ..runtime import resolve_device


def _read_evaluation_cli():
    main = sys.modules.get('__main__')
    spec = getattr(main, '__spec__', None)
    if (getattr(spec, 'name', None) != 'pfn_network.train_eval.run'
            or mp.current_process().name != 'MainProcess'):
        return None, None
    # Consume only the two new flags. All original/unknown flags remain for run.py.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    def positive(value):
        number = int(value)
        if number < 1:
            raise argparse.ArgumentTypeError('must be a positive integer')
        return number
    parser.add_argument('--eval-workers', type=positive)
    parser.add_argument('--eval-threads', type=positive)
    options, remaining = parser.parse_known_args(sys.argv[1:])
    if '-h' in remaining or '--help' in remaining:
        print('Additional evaluation options:')
        print('  --eval-workers N  One process per MUSA card (default: remaining visible cards; CPU: 1)')
        print('  --eval-threads N  CPU threads per evaluation process (parallel default: 1)')
    sys.argv[1:] = remaining
    return options.eval_workers, options.eval_threads


_CLI_WORKERS, _CLI_THREADS = _read_evaluation_cli()


def evaluation_devices(device, workers=None):
    if workers is not None and workers < 1:
        raise ValueError('--eval-workers must be positive')
    resolved = resolve_device(device)
    if resolved.type == 'cpu':
        return ['cpu'] * (workers if workers is not None else 1)
    start = resolved.index or 0
    available = torch.musa.device_count() - start
    workers = available if workers is None else workers
    if workers > available:
        raise ValueError('--eval-workers exceeds available MUSA devices from the selected device; use one worker per card')
    return [f'musa:{index}' for index in range(start, start + workers)]


def _model_payload(model):
    buffer = io.BytesIO()
    torch.save(dict(model_config=asdict(model.config),
                    model={key: value.detach().cpu() for key, value in model.state_dict().items()},
                    extra={}), buffer)
    return buffer.getvalue()


def benchmark(config, output, static_model, sequential_model, targets, device,
              *, eval_workers=None, eval_threads=None):
    """Preserve the original six arguments; parallelize independent task/seed jobs."""
    workers = eval_workers if eval_workers is not None else _CLI_WORKERS
    threads = eval_threads if eval_threads is not None else _CLI_THREADS
    devices = evaluation_devices(device, workers)
    settings = config['evaluation']
    total_jobs = settings['test_tasks'] * len(settings['seeds'])
    if settings['test_tasks'] < 1 or total_jobs < 1:
        raise ValueError('Evaluation needs positive test_tasks and nonempty seeds')
    if len(settings['seeds']) != len(set(settings['seeds'])):
        raise ValueError('Evaluation seeds must be unique to prevent checkpoint collisions')
    # Validate the requested card count first, then avoid starting idle workers.
    devices = devices[:total_jobs]
    workers = len(devices)
    if threads is not None and threads < 1:
        raise ValueError('--eval-threads must be positive')
    if workers == 1:
        if threads is None:
            return _benchmark_serial(config, output, static_model, sequential_model, targets, device)
        from threadpoolctl import threadpool_limits
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(threads)
            with threadpool_limits(limits=threads):
                return _benchmark_serial(config, output, static_model, sequential_model, targets, device)
        finally:
            torch.set_num_threads(previous)
    return benchmark_parallel(config, output, _model_payload(static_model),
                              _model_payload(sequential_model), targets, devices, threads or 1)


_WORKER = None


def _initialize_worker(config, static_payload, sequential_payload, targets, device, threads, checkpoints):
    global _WORKER
    # Spawned evaluators must never join an inherited training DDP process group.
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE'):
        os.environ.pop(key, None)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    limiter = threadpool_limits(limits=threads)
    device = str(resolve_device(device))
    static, _ = PFNPolicy.load(io.BytesIO(static_payload), device)
    sequential, _ = PFNPolicy.load(io.BytesIO(sequential_payload), device)
    static.requires_grad_(False)
    sequential.requires_grad_(False)
    _WORKER = dict(config=config, static=static, sequential=sequential, targets=targets,
                   device=device, checkpoints=Path(checkpoints), limiter=limiter)


def _run_job(base, prediction, task_index, replicate):
    w = _WORKER
    print(f"Benchmark start: task={task_index}, seed={replicate}, "
          f"device={w['device']}, pid={os.getpid()}", flush=True)
    result = _evaluate_benchmark_job(w['config'], w['checkpoints'], w['static'], w['sequential'],
                                     w['targets'], w['device'], base, prediction, task_index, replicate)
    return result, os.getpid()


def benchmark_parallel(config, output, static_payload, sequential_payload, targets, devices, threads=1):
    """Bound in-flight work to one job per worker; only parent writes table CSVs."""
    settings = config['evaluation']
    seeds = settings['seeds']
    if settings['test_tasks'] < 1 or not seeds:
        raise ValueError('Parallel evaluation needs positive test_tasks and nonempty seeds')
    if len(seeds) != len(set(seeds)):
        raise ValueError('Evaluation seeds must be unique to prevent checkpoint collisions')
    if threads < 1 or not devices:
        raise ValueError('Evaluation threads and number of devices must be positive')
    # Also validate direct API calls, not only the command-line entry point.
    if len(set(devices)) != len(devices) and devices != ['cpu'] * len(devices):
        raise ValueError('Each accelerator worker requires a distinct device')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    completed, preparation, tasks = {}, [], {}
    jobs = [(task, seed) for task in range(settings['test_tasks']) for seed in seeds]
    devices = devices[:len(jobs)]
    status = dict(mode='independent task/seed spawn workers', devices=devices,
                  threads_per_worker=threads, total_jobs=len(jobs), completed=[], status='running')

    def save_status():
        path = output / 'evaluation_parallel.json'
        temporary = path.with_suffix('.json.tmp')
        write_json(temporary, status)
        temporary.replace(path)

    save_status()
    try:
        for task in range(settings['test_tasks']):
            base, prediction, row = _prepare_benchmark_task(config, output, task)
            tasks[task] = (base, prediction)
            preparation.append(row)
        pd.DataFrame(preparation).to_csv(output / 'preparation_costs.csv', index=False)
        print(f'Parallel evaluation: {len(devices)} workers, {len(jobs)} task/seed jobs, '
              f'devices={devices}', flush=True)
        context = mp.get_context('spawn')
        pending = {}
        next_job = 0
        with ExitStack() as stack:
            pools = [stack.enter_context(ProcessPoolExecutor(
                max_workers=1, mp_context=context, initializer=_initialize_worker,
                initargs=(config, static_payload, sequential_payload,
                          targets, device, threads, str(checkpoints)))) for device in devices]

            def submit(worker, job_id):
                task, seed = jobs[job_id]
                base, prediction = tasks[task]
                future = pools[worker].submit(_run_job, base, prediction, task, seed)
                pending[future] = (worker, job_id)

            for worker in range(len(pools)):
                submit(worker, next_job)
                next_job += 1
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    worker, job_id = pending.pop(future)
                    task, seed = jobs[job_id]
                    try:
                        result, pid = future.result()
                    except Exception as error:
                        status['failed_job'] = dict(task=task, seed=seed, device=devices[worker])
                        raise RuntimeError(f'Evaluation failed: task={task}, seed={seed}, '
                                           f'device={devices[worker]}') from error
                    completed[job_id] = result
                    tables = _write_benchmark_progress(output, completed)
                    status['completed'].append(dict(task=task, seed=seed, device=devices[worker], pid=pid))
                    save_status()
                    print(f'Benchmark complete: task={task}, seed={seed}, device={devices[worker]} '
                          f'[{len(completed)}/{len(jobs)}]', flush=True)
                    if next_job < len(jobs):
                        submit(worker, next_job)
                        next_job += 1
        _summarize_benchmark(output, tables)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        save_status()
        raise
    status['status'] = 'complete'
    save_status()
