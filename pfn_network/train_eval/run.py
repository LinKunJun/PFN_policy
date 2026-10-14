import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from math import ceil
from pathlib import Path
import os
from ..runtime import runtime_info
from .. import distributed

import torch

from ..models import PFNPolicy
from .evaluate import benchmark, calibrate_targets, plot_results
from .train import seed_everything, train_meta, write_json


def resolve_config(config):
    """Resolve sequential defaults while retaining the static experiment settings."""
    config = deepcopy(config)
    config.setdefault("infra", {}).setdefault("env_workers", 8)
    config.setdefault("static_model", dict(config["model"]))
    config.setdefault("static_ppo", dict(config["ppo"]))
    for key in ("model", "static_model"):
        config[key].pop("reward_auxiliary", None)
        config[key].pop("base_reward_history", None)
    for key in ("ppo", "static_ppo"):
        config[key].pop("reward_aux_coef", None)
    if "sequential_meta" not in config:
        original = config["meta"]
        sequential = dict(original)
        length = config["model"]["context_length"] if config["model"]["history"] else 0
        episodes = max(original["episodes_per_task"], ceil(length / config["prior"]["horizon"]) + 2)
        ratio = original["episodes_per_task"] / episodes
        sequential.update(episodes_per_task=episodes,
                          updates=max(1, ceil(original["updates"] * ratio)),
                          validate_every=max(1, ceil(original["validate_every"] * ratio)))
        # Other budgets round up to a complete update, never split an episode.
        config["sequential_meta"] = sequential
    sequential = config["sequential_meta"]
    sequential.setdefault("validation_protocol", "deployment")
    sequential.setdefault("validation_adaptation_interactions", config["evaluation"]["evaluate_every"])
    sequential.setdefault("validation_episodes", 4)
    return config


def load_model(path, objective, config, device):
    model, extra = PFNPolicy.load(path, device)
    if extra.get("objective") != objective:
        raise ValueError(f"Expected a {objective} checkpoint, not an untrained or differently supervised policy.")
    if extra.get("binding", {}).get("root_seed") != config["seed"]:
        raise ValueError("Use the same experiment root seed to preserve disjoint latent-task splits.")
    if model.config.feature_dim != config["prior"]["feature_dim"]:
        raise ValueError("Checkpoint feature dimension does not match the task prior.")
    model.requires_grad_(False)
    return model, extra


def main():
    parser = argparse.ArgumentParser(description="Network interference research pipeline")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--stage", choices=["all", "train", "calibrate", "evaluate", "plot"], default="all")
    parser.add_argument("--objective", choices=["static", "sequential", "both"], default="both")
    parser.add_argument("--output")
    parser.add_argument("--results", type=Path, help="Existing result directory for --stage plot")
    parser.add_argument("--device", default="musa")
    parser.add_argument("--env-workers", type=int, help="CPU simulator processes per rank")
    parser.add_argument("--static-checkpoint")
    parser.add_argument("--sequential-checkpoint")
    parser.add_argument("--targets")
    parser.add_argument("--resume", help="Resume one meta-training objective; use --objective static/sequential")
    args = parser.parse_args()
    if args.stage == "plot":
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            parser.error("Plotting requires a single process")
        if args.results is None:
            parser.error("--stage plot requires --results")
        plot_results(args.results)
        return
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and args.stage not in ("train", "all"):
        parser.error("Use a single process for calibration, evaluation or plotting")
    if world > 1 and not args.output:
        parser.error("Distributed training requires a shared explicit --output path")
    args.device = str(distributed.initialize(args.device))
    primary = distributed.is_main()
    config = resolve_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.env_workers is not None:
        if args.env_workers < 0:
            parser.error("--env-workers must be non-negative")
        config["infra"]["env_workers"] = args.env_workers
    if config["prior"]["feature_dim"] != config["model"]["feature_dim"]:
        raise ValueError("Task/model feature dimensions must match.")
    output = Path(args.output or ("runs/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")))
    if primary:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", config)
    distributed.barrier()
    runtime = dict(runtime_info(args.device), rank=distributed.rank(), world_size=world,
                   distributed_backend="mccl" if world > 1 and args.device.startswith("musa") else
                                        "gloo" if world > 1 else None,
                   parallelism="synchronous DDP" if world > 1 else "single process")
    write_json(output / f"runtime_rank{distributed.rank()}.json", runtime)
    if primary:
        write_json(output / "runtime.json", runtime)
    static_path, sequential_path = args.static_checkpoint, args.sequential_checkpoint
    if args.stage in ("all", "train"):
        objectives = ["static", "sequential"] if args.objective == "both" else [args.objective]
        if args.resume and len(objectives) != 1:
            raise ValueError("--resume requires one --objective.")
        for objective in objectives:
            seed_everything(config["seed"] + (objective == "static"))
            train_meta(config, output, objective, args.device, args.resume)
            if objective == "static":
                static_path = output / "static_best.pt"
            else:
                sequential_path = output / "sequential_best.pt"
    # All training collectives finish before rank 0's independent benchmark.
    distributed.barrier()
    distributed.finish()
    if not primary:
        return
    targets = None
    if args.targets:
        payload = json.loads(Path(args.targets).read_text(encoding="utf-8"))
        if payload["seed"] != config["seed"] or payload["prior"] != config["prior"] or payload["gamma"] != config["ppo"]["gamma"]:
            raise ValueError("Target calibration must match this validation prior and discount.")
        targets = payload["targets"]
        write_json(output / "targets.json", payload)
    elif args.stage in ("all", "calibrate"):
        targets = calibrate_targets(config, output)
    if args.stage in ("all", "evaluate"):
        if static_path is None or sequential_path is None or targets is None:
            raise ValueError("Evaluation requires static/sequential trained checkpoints and validation targets.")
        static, static_extra = load_model(static_path, "static", config, args.device)
        sequential, sequential_extra = load_model(sequential_path, "sequential", config, args.device)
        write_json(output / "checkpoint_provenance.json", {"static": static_extra, "sequential": sequential_extra,
                   "static_checkpoint": str(static_path), "sequential_checkpoint": str(sequential_path)})
        benchmark(config, output, static, sequential, targets, args.device)
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        distributed.finish()
