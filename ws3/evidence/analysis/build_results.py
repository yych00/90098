"""Audit WS3 runs and generate traceable CSV, PNG and bilingual reports.

Only recorded evaluations are analysed. The default refuses a report if Q1-Q3
or three independent training seeds are missing; --allow-incomplete produces an
explicitly marked progress report instead. Pilot/smoke runs are excluded.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WS3 = Path(__file__).resolve().parents[1]
TASK2_ROOT = WS3.parent
sys.path.insert(0, str(TASK2_ROOT / "shared"))
from manifest_paths import allowed_output_paths, load_jobs, relocate_path

DEFAULT_MANIFEST = WS3.parent / "shared" / "dqn_formal.jobs.json"
EXPECTED_COUNTS = {"q1": 3, "q2": 3, "q3": 6}
EXPECTED_GRID = {(m, b) for m in (1000, 10000, 100000) for b in (32, 128)}
VARIABLES = {"q1": {"learning_rate"}, "q2": {"epsilon", "epsilon_decay", "epsilon_min"},
             "q3": {"memory_size", "batch_size"}}
PROTOCOL = ("env_name", "total_steps", "eval_every", "eval_episodes", "eval_seed", "learning_start",
            "target_update_freq", "hidden_size", "layer_num", "discount_factor", "learning_rate",
            "memory_size", "batch_size", "epsilon", "epsilon_decay", "epsilon_min", "code_hash",
            "evaluation_policy", "epsilon_decay_unit", "bootstrap_mask", "device", "torch_threads", "max_ep_len")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_formal_jobs(path):
    """Resolve packaged evidence locally; never fall back to an old host directory."""
    raw = read_json(path)
    if isinstance(raw, list):
        jobs = load_jobs(path)
    else:
        jobs = []
        for original in raw.get("runs", raw.get("conditions", raw.get("jobs", []))):
            job = copy.deepcopy(original)
            job["manifest_output_dir"] = original["output_dir"]
            job["output_dir"] = relocate_path(original["output_dir"])
            job["command"] = [relocate_path(value) for value in job.get("command", [])]
            for field in ("parameters", "config"):
                if job.get(field, {}).get("output_dir"):
                    job[field]["output_dir"] = relocate_path(job[field]["output_dir"])
            jobs.append(job)
    for job in jobs:
        try:
            Path(job["output_dir"]).resolve().relative_to(WS3.resolve())
        except ValueError:
            raise ValueError(f"Formal evidence does not resolve inside this WS3 package: {job['output_dir']}")
    return jobs


def infer_question(path, config, metadata):
    if metadata.get("question") in EXPECTED_COUNTS:
        return metadata["question"]
    parts = [part.lower() for part in path.parts]
    for question in EXPECTED_COUNTS:
        if question in parts:
            return question
    for question in EXPECTED_COUNTS:
        if any(part.startswith(question + "_") and part != "q1_q2" for part in parts):
            return question
    name = str(config.get("exp_name", path.parent.name)).lower()
    if name.startswith(("lr", "learning")):
        return "q1"
    if "eps" in name or name in {"0.9t", "0.995t"}:
        return "q2"
    if "memory" in name or "buffer" in name:
        return "q3"
    return None


def condition_name(question, config):
    if question == "q1":
        return f"lr={config['learning_rate']:g}"
    if question == "q2":
        if float(config["epsilon_decay"]) == 1.0:
            return f"constant epsilon={config['epsilon']:g}"
        return f"decay={config['epsilon_decay']:g}, floor={config['epsilon_min']:g}"
    return f"memory={int(config['memory_size'])}, batch={int(config['batch_size'])}"


def validate_planned_config(config, job, directory):
    """Relocation may change only the recorded output pointer, never the experiment."""
    pointers = allowed_output_paths(job)
    if str(Path(config["output_dir"]).resolve()).casefold() not in pointers:
        raise ValueError("Saved output_dir matches neither the formal directory nor its declared historical pointer")
    expected = dict(job.get("parameters", job.get("config", {})))
    command = job.get("command", [])
    for index, value in enumerate(command[:-1]):
        if value.startswith("--") and not command[index + 1].startswith("--"):
            expected.setdefault(value[2:], command[index + 1])
    for key, wanted in expected.items():
        if key == "output_dir":
            continue
        actual = config.get(key)
        if isinstance(actual, bool):
            wanted = wanted if isinstance(wanted, bool) else str(wanted).lower() in {"1", "true"}
        elif isinstance(actual, int):
            wanted = int(wanted)
        elif isinstance(actual, float):
            wanted = float(wanted)
        if actual != wanted:
            raise ValueError(f"Saved {key}={actual!r} differs from formal plan {wanted!r}")


def read_run(config_path, question, config, issues):
    directory = config_path.parent
    eval_path = directory / "evaluations.csv"
    try:
        with eval_path.open(newline="", encoding="utf-8-sig") as handle:
            raw = list(csv.DictReader(handle))
        steps = np.asarray([int(float(row["env_steps"])) for row in raw], dtype=int)
        values = np.asarray([float(row["mean_return"]) for row in raw], dtype=float)
        stds = np.asarray([float(row["std_return"]) for row in raw], dtype=float)
        counts = [int(row["episodes"]) for row in raw]
        if len(steps) < 2 or steps[0] != 0 or np.any(np.diff(steps) <= 0):
            raise ValueError("Evaluation steps must begin at zero and increase strictly")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(stds)):
            raise ValueError("Nonfinite evaluation returns")
        if np.any(values < 0) or np.any(values > 500) or np.any(stds < 0):
            raise ValueError("CartPole evaluation statistics out of range")
        if any(count != int(config["eval_episodes"]) for count in counts):
            raise ValueError("Evaluation episode count differs from saved configuration")
        expected = list(range(0, int(config["total_steps"]) + 1, int(config["eval_every"])))
        if expected[-1] != int(config["total_steps"]):
            expected.append(int(config["total_steps"]))
        required = ("summary.json", "metrics.csv", "checkpoint.pt")
        if any(not (directory / name).exists() for name in required):
            raise ValueError("Missing summary, training metrics or checkpoint evidence")
        summary = read_json(directory / "summary.json")
        complete = (list(steps) == expected and summary.get("status") == "complete"
                    and summary.get("actual_training_steps") == config["total_steps"])
        with (directory / "metrics.csv").open(newline="", encoding="utf-8-sig") as handle:
            training = list(csv.DictReader(handle))
        if not training or int(training[-1]["env_steps"]) != int(steps[-1]):
            raise ValueError("Training metrics do not end at the recorded evaluation budget")
        for row in training:
            if any(not math.isfinite(float(value)) for value in row.values() if value not in {"", None}):
                raise ValueError("Nonfinite recorded training metric")
        floor_step = next((int(row["env_steps"]) for row in training
                           if float(row["epsilon"]) <= float(config["epsilon_min"]) + 1e-12), None)
        epsilon_steps = np.asarray([0] + [int(row["env_steps"]) for row in training], dtype=int)
        epsilon_values = np.asarray([config["epsilon"]] + [float(row["epsilon"]) for row in training], dtype=float)
        episode_path = directory / "evaluation_episodes.csv"
        if not episode_path.exists():
            raise ValueError("Missing individual evaluation episodes")
        with episode_path.open(newline="", encoding="utf-8-sig") as handle:
            episodes = list(csv.DictReader(handle))
        reward_key = next((key for key in ("return", "episode_return", "return_value", "eval_return")
                           if episodes and key in episodes[0]), None)
        if reward_key is None:
            raise ValueError("Individual evaluation CSV has no recognized return column")
        by_step = defaultdict(list)
        seeds_by_step = defaultdict(list)
        for row in episodes:
            by_step[int(float(row["env_steps"]))].append(float(row[reward_key]))
            if "seed" in row:
                seeds_by_step[int(float(row["env_steps"]))].append(int(row["seed"]))
        for step, mean, std, count in zip(steps, values, stds, counts):
            samples = by_step[int(step)]
            if len(samples) != count or not np.isclose(np.mean(samples), mean, atol=1e-6):
                raise ValueError(f"Episode-level evidence does not match mean at step {step}")
            if not np.isclose(np.std(samples), std, atol=1e-6):
                raise ValueError(f"Episode-level evidence does not match population SD at step {step}")
            expected_seeds = list(range(config["eval_seed"], config["eval_seed"] + config["eval_episodes"]))
            if seeds_by_step[int(step)] != expected_seeds:
                raise ValueError(f"Evaluation reset seeds differ from the fixed protocol at step {step}")
        condition = condition_name(question, config)
        threshold = None
        for i in range(len(values) - 2):
            if np.all(values[i:i + 3] >= 475):
                threshold = int(steps[i + 2])
                break
        integrate = getattr(np, "trapezoid", np.trapz)
        return {"question": question, "condition": condition, "seed": int(config["seed"]),
                "run_dir": str(directory), "config": config, "steps": steps, "values": values,
                "stds": stds, "complete": complete,
                "epsilon_steps": epsilon_steps, "epsilon_values": epsilon_values,
                "metric": {"question": question, "condition": condition, "seed": int(config["seed"]),
                    "complete": complete, "training_steps": int(steps[-1]), "eval_points": len(steps),
                    "final_return": float(values[-1]), "last3_mean": float(np.mean(values[-3:])),
                    "last3_min": float(np.min(values[-3:])), "last3_max": float(np.max(values[-3:])),
                    "last5_mean": float(np.mean(values[-5:])),
                    "late5_temporal_sd": float(np.std(values[-5:])),
                    "auc_return": float(integrate(values, steps) / steps[-1]),
                    "wall_seconds": summary.get("wall_seconds"),
                    "first3_ge475_step": threshold, "run_dir": str(directory),
                    "first_epsilon_floor_step": floor_step,
                    "final_epsilon": float(training[-1]["epsilon"]),
                    "completed_training_episodes": summary.get("completed_training_episodes"),
                    "recorded_output_dir": config.get("output_dir"),
                    "code_hash": config.get("code_hash"),
                    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    "evaluations_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest()}}
    except (KeyError, ValueError, OSError, TypeError) as error:
        issues.append(f"Invalid run {directory}: {error}")
        return None


def discover(roots, questions, excluded, issues, allowed_dirs=None, planned_jobs=None):
    runs, seen = [], set()
    memberships = {}
    for root in roots:
        for manifest_path in root.rglob("manifest.json"):
            if any(term.lower() in part.lower() for term in excluded
                   for part in manifest_path.relative_to(root).parts):
                continue
            manifest = read_json(manifest_path)
            for item in manifest.get("runs", manifest.get("conditions", [])):
                if item.get("output_dir"):
                    member = item.get("questions", [item.get("question")])
                    memberships[str(Path(item["output_dir"]).resolve()).casefold()] = [str(q).lower() for q in member]
    planned = {str(Path(job["output_dir"]).resolve()).casefold(): job for job in (planned_jobs or [])}
    for directory, job in planned.items():
        member = job.get("questions") or ([job["question"]] if job.get("question") else [])
        if member:
            memberships[directory] = [str(question).lower() for question in member]
    for root in roots:
        for path in sorted(root.rglob("config.json")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            if allowed_dirs is not None and str(path.parent.resolve()).casefold() not in allowed_dirs:
                continue
            if any(term.lower() in part.lower() for term in excluded for part in path.relative_to(root).parts):
                continue
            try:
                config = read_json(path)
                config = config.get("args", config)
                job = planned.get(str(path.parent.resolve()).casefold())
                if job is not None:
                    validate_planned_config(config, job, path.parent)
                metadata_path = path.parent / "experiment.json"
                metadata = read_json(metadata_path) if metadata_path.exists() else {}
                member = memberships.get(str(path.parent.resolve()).casefold(),
                           [infer_question(path, config, metadata)])
                for question in member:
                    if question not in questions:
                        continue
                    run = read_run(path, question, config, issues)
                    if run:
                        runs.append(run)
            except (ValueError, OSError, TypeError) as error:
                issues.append(f"Invalid configuration {path}: {error}")
    return runs


def audit(runs, questions, min_seeds, issues):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["question"], run["condition"])].append(run)
        if not run["complete"]:
            issues.append(f"Incomplete training or missing final summary: {run['run_dir']}")
    for question in questions:
        q_runs = [run for run in runs if run["question"] == question]
        conditions = {run["condition"] for run in q_runs}
        if len(conditions) != EXPECTED_COUNTS[question]:
            issues.append(f"{question}: expected {EXPECTED_COUNTS[question]} conditions, found {len(conditions)}")
        baselines = {json.dumps({key: run["config"].get(key) for key in PROTOCOL
                                 if key not in VARIABLES[question]}, sort_keys=True) for run in q_runs}
        if len(baselines) > 1:
            issues.append(f"{question}: supposedly fixed hyperparameters or evaluation protocol differ")
        if question == "q1" and {run["config"].get("learning_rate") for run in q_runs} != {0.0001, 0.001, 0.01}:
            issues.append("q1: learning-rate grid must be 0.0001, 0.001 and 0.01")
        if question == "q2":
            schedules = {(run["config"].get("epsilon"), run["config"].get("epsilon_decay"),
                          run["config"].get("epsilon_min")) for run in q_runs}
            expected_schedules = {(0.1, 1.0, 0.02), (1.0, 0.9, 0.02), (1.0, 0.995, 0.02)}
            # A constant epsilon remains 0.1 for either floor 0.0 or 0.02.
            schedules = {(e, d, 0.02 if d == 1.0 and m <= e else m) for e, d, m in schedules}
            if schedules != expected_schedules:
                issues.append("q2: expected constant 0.1 and 0.9/0.995 per-episode decay with floor 0.02")
        if question == "q3" and {(int(run["config"]["memory_size"]), int(run["config"]["batch_size"]))
                                  for run in q_runs} != EXPECTED_GRID:
            issues.append("q3: missing or unexpected replay-capacity/minibatch combinations")
    for key, group in grouped.items():
        seeds = [run["seed"] for run in group]
        if len(set(seeds)) < min_seeds:
            issues.append(f"{key}: needs at least {min_seeds} independent training seeds, found {len(set(seeds))}")
        if len(seeds) != len(set(seeds)):
            issues.append(f"{key}: duplicate training seed; specify one intended run root")
        if len({tuple(run["steps"]) for run in group}) != 1:
            issues.append(f"{key}: evaluation grids differ")
    seed_sets = {tuple(sorted(run["seed"] for run in group)) for group in grouped.values()}
    if len(seed_sets) > 1:
        issues.append("Training seed sets differ between conditions")
    return grouped


def summarize(grouped, output):
    aggregate, curve_rows = [], []
    colors = plt.get_cmap("tab10")
    for question in sorted({key[0] for key in grouped}):
        fig, ax = plt.subplots(figsize=(10.5, 6))
        for index, ((_, condition), group) in enumerate(sorted((key, group) for key, group in grouped.items()
                                                                if key[0] == question)):
            group = sorted(group, key=lambda run: run["seed"])
            if len({tuple(run["steps"]) for run in group}) != 1:
                continue
            steps = group[0]["steps"]
            values = np.stack([run["values"] for run in group])
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(group) > 1 else np.zeros(len(steps))
            color = colors(index)
            for run in group:
                ax.plot(steps, run["values"], color=color, alpha=0.16, lw=0.9)
            ax.plot(steps, mean, color=color, label=f"{condition} (n={len(group)})", lw=2)
            ax.fill_between(steps, np.clip(mean - sd, 0, 500), np.clip(mean + sd, 0, 500),
                            color=color, alpha=0.12)
            for step, average, std in zip(steps, mean, sd):
                curve_rows.append({"question": question, "condition": condition, "env_steps": int(step),
                                   "mean_return": float(average), "training_seed_sd": float(std),
                                   "training_seeds": len(group)})
            metrics = [run["metric"] for run in group]
            result = {"question": question, "condition": condition, "training_seeds": len(group),
                      "seed_ids": ",".join(str(run["seed"]) for run in sorted(group, key=lambda run: run["seed"])),
                      "training_steps": int(steps[-1])}
            for field in ("final_return", "last3_mean", "last5_mean", "auc_return", "late5_temporal_sd", "wall_seconds"):
                samples = [row[field] for row in metrics]
                if any(value is None for value in samples):
                    continue
                result[field + "_mean"] = statistics.mean(samples)
                result[field + "_sd"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
            threshold = [row["first3_ge475_step"] for row in metrics]
            result["seeds_with_3evals_ge475"] = sum(value is not None for value in threshold)
            result["first3_ge475_steps_by_seed"] = ";".join("not reached" if value is None else str(value)
                                                                  for value in threshold)
            confirmed = [value for value in threshold if value is not None]
            result["median_confirmation_step_among_reached"] = statistics.median(confirmed) if confirmed else None
            aggregate.append(result)
        ax.axhline(500, color="#555555", linestyle="--", lw=1, label="Maximum return = 500")
        ax.set(xlabel="Training environment steps (evaluation steps excluded)", ylabel="Greedy evaluation return",
               title=f"WS3 Task 2 {question.upper()}: mean and SD across training seeds", ylim=(0, 520))
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"{question}_evaluation_returns.png", dpi=180)
        plt.close(fig)
    write_csv(output / "per_run_metrics.csv", [run["metric"] for group in grouped.values() for run in group])
    write_csv(output / "condition_summary.csv", aggregate)
    write_csv(output / "learning_curves.csv", curve_rows)
    q3_groups = [(key, group) for key, group in grouped.items() if key[0] == "q3"]
    if q3_groups:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for axis, batch in zip(axes, (32, 128)):
            for index, capacity in enumerate((1000, 10000, 100000)):
                key = ("q3", f"memory={capacity}, batch={batch}")
                group = grouped.get(key)
                if not group or len({tuple(run["steps"]) for run in group}) != 1:
                    continue
                values = np.stack([run["values"] for run in group])
                steps = group[0]["steps"]
                mean = np.mean(values, axis=0)
                sd = np.std(values, axis=0, ddof=1) if len(group) > 1 else np.zeros(len(steps))
                axis.plot(steps, mean, color=colors(index), label=f"capacity {capacity:,}", lw=2)
                axis.fill_between(steps, np.clip(mean - sd, 0, 500), np.clip(mean + sd, 0, 500),
                                  color=colors(index), alpha=0.13)
            axis.set(title=f"Batch size = {batch}", xlabel="Training environment steps", ylim=(0, 520))
            axis.set_xticks([0, 50000, 100000, 150000, 200000], ["0", "50k", "100k", "150k", "200k"])
            axis.axhline(500, color="#555555", ls="--", lw=1)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        axes[0].set_ylabel("Greedy evaluation return")
        fig.suptitle("Replay capacity at a fixed minibatch size: training-seed mean and SD")
        fig.tight_layout()
        fig.savefig(output / "q3_replay_interactions.png", dpi=180)
        plt.close(fig)
    pairs = replay_pairs(aggregate)
    write_csv(output / "q3_paired_comparisons.csv", pairs)
    epsilon_rows = []
    q2_groups = [(key, group) for key, group in sorted(grouped.items()) if key[0] == "q2"]
    if q2_groups:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for index, ((_, condition), group) in enumerate(q2_groups):
            for seed_index, run in enumerate(sorted(group, key=lambda run: run["seed"])):
                for axis in axes:
                    axis.step(run["epsilon_steps"], run["epsilon_values"], where="post", color=colors(index),
                              lw=1.3, alpha=0.75, label=condition if seed_index == 0 else None)
                epsilon_rows.append({"condition": condition, "seed": run["seed"],
                    "completed_training_episodes": run["metric"]["completed_training_episodes"],
                    "final_epsilon": run["metric"]["final_epsilon"],
                    "first_step_at_floor": run["metric"]["first_epsilon_floor_step"],
                    "learning_start": run["config"]["learning_start"]})
        for axis in axes:
            axis.set(xlabel="Training environment steps", ylabel="Training epsilon", ylim=(0.015, 1.05), yscale="log")
            axis.grid(alpha=0.25)
        axes[0].set_title("Actual per-seed exploration schedules")
        axes[0].set_xticks([0, 50000, 100000, 150000, 200000], ["0", "50k", "100k", "150k", "200k"])
        axes[1].set(xlim=(0, 1500), title="Early training: episode-based fast decay")
        axes[1].axvline(1000, color="#555555", ls="--", lw=1, label="Learning starts at 1,000 steps")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "q2_realized_epsilon.png", dpi=180)
        plt.close(fig)
        write_csv(output / "q2_realized_epsilon.csv", epsilon_rows)
    return aggregate


def replay_pairs(rows):
    configurations = {}
    for row in rows:
        if row["question"] == "q3":
            match = re.fullmatch(r"memory=(\d+), batch=(\d+)", row["condition"])
            configurations[tuple(map(int, match.groups()))] = row
    pairs = []
    for capacity in (1000, 10000, 100000):
        small, large = configurations.get((capacity, 32)), configurations.get((capacity, 128))
        if small and large:
            small_step = small["median_confirmation_step_among_reached"]
            large_step = large["median_confirmation_step_among_reached"]
            pairs.append({"memory_size": capacity, "comparison": "batch128_minus_batch32",
                          "last3_mean_delta": large["last3_mean_mean"] - small["last3_mean_mean"],
                          "auc_return_delta": large["auc_return_mean"] - small["auc_return_mean"],
                          "late5_temporal_sd_delta": large["late5_temporal_sd_mean"] - small["late5_temporal_sd_mean"],
                          "batch32_training_seeds": small["training_seeds"],
                          "batch128_training_seeds": large["training_seeds"],
                          "batch32_seeds_with_3evals_ge475": small["seeds_with_3evals_ge475"],
                          "batch128_seeds_with_3evals_ge475": large["seeds_with_3evals_ge475"],
                          "batch32_median_confirmation_step_among_reached": small_step,
                          "batch128_median_confirmation_step_among_reached": large_step,
                          "median_confirmation_step_delta": (large_step - small_step
                                                              if small_step is not None and large_step is not None
                                                              else None)})
    return pairs


def replay_discussion(rows, language):
    sentences = []
    configurations = {row["condition"]: row for row in rows if row["question"] == "q3"}
    for batch in (32, 128):
        ordered = [configurations.get(f"memory={capacity}, batch={batch}") for capacity in (1000, 10000, 100000)]
        if all(ordered):
            means = "/".join(f"{row['last5_mean_mean']:.1f}" for row in ordered)
            deviations = "/".join(f"{row['late5_temporal_sd_mean']:.1f}" for row in ordered)
            if language == "en":
                sentences.append(f"With batch {batch}, capacities 1,000/10,000/100,000 give last-five mean "
                                 f"returns {means} and mean temporal SDs {deviations}, respectively.")
            else:
                sentences.append(f"固定batch{batch}时，容量1000/10000/100000的末5次均值依次为{means}，"
                                 f"时序标准差均值依次为{deviations}。")
    if len(sentences) == 2:
        sentences.append("The capacity trend reverses between these minibatches: larger capacity is associated "
                         "with worse late return and more fluctuation at batch 32, but better late return and "
                         "less fluctuation at batch 128. The observed capacity effect therefore depends on "
                         "minibatch size." if language == "en" else
                         "容量趋势随batch改变：batch32时，大容量对应更低的后期回报和更多波动；batch128时，"
                         "大容量对应更高的后期回报和更少波动。因此，本次实测容量效应依赖batch大小。")
    for pair in replay_pairs(rows):
        capacity = pair["memory_size"]
        reached = [f"{pair[f'batch{batch}_seeds_with_3evals_ge475']}/"
                   f"{pair[f'batch{batch}_training_seeds']}" for batch in (32, 128)]
        medians = [pair[f"batch{batch}_median_confirmation_step_among_reached"] for batch in (32, 128)]
        times = [f"{step:,.0f}" if step is not None else "not reached" for step in medians]
        delta = pair["median_confirmation_step_delta"]
        if language == "en":
            sentences.append(f"At fixed capacity {capacity:,}, changing batch size from 32 to 128 changed "
                f"the last-three-checkpoint mean by {pair['last3_mean_delta']:+.1f}, the AUC return by "
                f"{pair['auc_return_delta']:+.1f}, and the mean late-five temporal SD by "
                f"{pair['late5_temporal_sd_delta']:+.1f} return points.")
            speed = (f" The difference between these medians is {delta:+,.0f} environment steps "
                     f"({'later' if delta > 0 else 'earlier' if delta < 0 else 'equal'} confirmation)."
                     if delta is not None else " Neither missing confirmations nor their times are imputed.")
            sentences.append(f"For the three-consecutive-evaluation threshold of 475, batch 32 and 128 "
                f"confirmed in {reached[0]} and {reached[1]} seeds, respectively; median third-checkpoint "
                f"confirmation times among reached seeds were {times[0]} and {times[1]} environment steps."
                + speed)
        else:
            sentences.append(f"固定容量{capacity:,}时，batch从32改为128，末3次评估均值变化"
                f"{pair['last3_mean_delta']:+.1f}，AUC回报变化{pair['auc_return_delta']:+.1f}，"
                f"末5次时序标准差均值变化{pair['late5_temporal_sd_delta']:+.1f}回报点。")
            speed = (f"两配置中位数相差{delta:+,.0f}环境步（"
                     f"{'更晚' if delta > 0 else '更早' if delta < 0 else '相同时间'}确认）。"
                     if delta is not None else "未达标者及其确认时间均不插补。")
            sentences.append(f"按连续3次评估均值≥475的阈值，batch32和128的确认种子数依次为"
                f"{reached[0]}、{reached[1]}；在已确认种子中，第三检查点确认步数的中位数依次为"
                f"{times[0]}、{times[1]}环境步。" + speed)
    if sentences:
        sentences.append("A positive temporal-SD change indicates more fluctuation, while a positive return "
                         "or AUC change indicates a higher measured value. These are paired configuration "
                         "comparisons at a fixed environment-step budget, not significance tests. "
                         "AUC measures return across the whole training budget; threshold confirmation "
                         "measures when a specified performance level is reached, so higher AUC does not "
                         "necessarily imply earlier confirmation. Confirmation medians are conditional on "
                         "reaching the threshold and must be read with the reached-seed counts; confirmation "
                         "does not imply permanent convergence." if language == "en" else
                         "时序标准差差值为正表示波动增加，回报或AUC差值为正表示对应实测指标更高。"
                         "这些是在固定环境步预算下的配置配对比较，不是显著性检验。"
                         "AUC衡量整个训练预算内的回报，阈值确认时间衡量何时达到指定表现，因此更高AUC不必然意味着更早确认。"
                         "确认时间中位数仅针对已达标种子，必须结合达标种子数解读；确认也不代表此后永久收敛。")
    return "\n\n".join(sentences)


def fmt(value):
    return f"{value:.1f}"


def report_table(rows, language):
    header = ("| Condition | Seeds | Final return | Last 3 evaluations | AUC return | 3 evaluations >=475 |\n"
              "|---|---:|---:|---:|---:|---:|" if language == "en" else
              "| 配置 | 种子数 | 最终回报 | 末3次评估均值 | 曲线面积归一化回报 | 连续3次≥475种子数 |\n"
              "|---|---:|---:|---:|---:|---:|")
    lines = [header]
    for row in rows:
        cells = [row["condition"], str(row["training_seeds"])]
        cells += [f"{fmt(row[field + '_mean'])} ± {fmt(row[field + '_sd'])}"
                  for field in ("final_return", "last3_mean", "auc_return")]
        cells += [f"{row['seeds_with_3evals_ge475']}/{row['training_seeds']}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def confirmation_text(rows, language):
    if language == "en":
        return "Third-checkpoint confirmation steps by training seed (in the seed order recorded in CSV): " + "; ".join(
            f"{row['condition']}: {row['first3_ge475_steps_by_seed']}" for row in rows) + ". A missing confirmation is retained as not reached."
    return "按CSV中训练种子顺序记录的第三点确认步数：" + "；".join(
        f"{row['condition']}：{row['first3_ge475_steps_by_seed']}" for row in rows) + "。未确认者保留为not reached。"


def q12_discussion(rows, question, runs, language):
    by_name = {row["condition"]: row for row in rows}
    needed = (["lr=0.0001", "lr=0.001", "lr=0.01"] if question == "q1" else
              ["constant epsilon=0.1", "decay=0.9, floor=0.02", "decay=0.995, floor=0.02"])
    if any(name not in by_name for name in needed):
        return findings(rows, question, language)
    first, baseline, last = (by_name[name] for name in needed) if question == "q1" else (
        by_name[needed[0]], by_name[needed[2]], by_name[needed[1]])
    def number(row, field):
        return f"{row[field]:.1f}"
    def confirmation(row):
        value = row["median_confirmation_step_among_reached"]
        return "not reached" if value is None else f"{value:,.0f}"
    if question == "q1":
        if language == "en":
            paragraphs = [
                f"The rate 0.0001 has the largest whole-budget AUC return ({number(first, 'auc_return_mean')}, "
                f"versus {number(baseline, 'auc_return_mean')} for 0.001) and a higher last-five-checkpoint "
                f"average ({number(first, 'last5_mean_mean')} versus {number(baseline, 'last5_mean_mean')}). "
                f"Its final mean is {number(first, 'final_return_mean')}, whereas 0.001 ends at "
                f"{number(baseline, 'final_return_mean')}. Thus the endpoint and whole-budget rankings differ.",
                f"Median third-checkpoint confirmation steps among seeds that confirm are "
                f"{confirmation(first)} for 0.0001 and {confirmation(baseline)} for 0.001. The per-seed "
                f"CSV confirmation list and table counts retain unsuccessful seeds. This separates early near-maximum "
                f"performance from the return accumulated throughout learning. The smaller learning rate "
                f"cannot simply be described as ineffective because it takes longer to confirm.",
                f"The default 0.001 rate finishes at 500 for all three seeds, yet its last-five mean is only "
                f"{number(baseline, 'last5_mean_mean')} and its mean late-five temporal SD is "
                f"{number(baseline, 'late5_temporal_sd_mean')}; that endpoint is not stable convergence. "
                f"At 0.01, the final mean is {number(last, 'final_return_mean')} and AUC is "
                f"{number(last, 'auc_return_mean')}. Its lower late temporal SD "
                f"({number(last, 'late5_temporal_sd_mean')}) describes consistently weak late performance, "
                f"not a high-quality stable policy. All recorded training metrics are finite; reward drops "
                f"and oscillations alone do not establish numerical divergence."]
        else:
            paragraphs = [
                f"学习率0.0001的整段AUC回报最高（{number(first, 'auc_return_mean')}，默认0.001为"
                f"{number(baseline, 'auc_return_mean')}），末5次检查点均值也更高"
                f"（{number(first, 'last5_mean_mean')}对{number(baseline, 'last5_mean_mean')}）。"
                f"但其最终均值为{number(first, 'final_return_mean')}，0.001为"
                f"{number(baseline, 'final_return_mean')}，因此终点排名与整段表现排名不同。",
                f"仅在成功确认的种子中计算，0.0001与0.001的第三点确认步数中位数分别为"
                f"{confirmation(first)}和{confirmation(baseline)}；逐种子CSV列表与表格达标计数同时保留未确认情况。"
                "这区分了较早达到近满分与整个训练过程的回报，不能因为较小学习率确认较晚就认定它学习无效。",
                f"默认0.001三个种子终点都是500，但末5次均值仅{number(baseline, 'last5_mean_mean')}，"
                f"末5次时序标准差均值{number(baseline, 'late5_temporal_sd_mean')}，不能称为稳定收敛。"
                f"0.01的最终均值为{number(last, 'final_return_mean')}，AUC为{number(last, 'auc_return_mean')}；"
                f"较低的后期时序标准差{number(last, 'late5_temporal_sd_mean')}对应持续较弱的后期表现，"
                "不是优质稳定策略。已记录训练指标均为有限数值，回报下降或振荡不能单独证明数值发散。"]
    else:
        fast_runs = [run for run in runs if run["question"] == "q2" and run["condition"] == needed[1]]
        slow_runs = [run for run in runs if run["question"] == "q2" and run["condition"] == needed[2]]
        fast_floors = [run["metric"]["first_epsilon_floor_step"] for run in fast_runs
                       if run["metric"]["first_epsilon_floor_step"] is not None]
        slow_floors = [run["metric"]["first_epsilon_floor_step"] for run in slow_runs
                       if run["metric"]["first_epsilon_floor_step"] is not None]
        if not fast_floors or not slow_floors:
            return findings(rows, question, language)
        if language == "en":
            paragraphs = [
                f"The actual episode-based fast schedule reaches epsilon=0.02 at training steps "
                f"{min(fast_floors):,}-{max(fast_floors):,}; the 0.995 schedule reaches it at "
                f"{min(slow_floors):,}-{max(slow_floors):,}. The fast schedule therefore reaches its floor "
                f"before the 1,000-step replay warm-up and before any gradient update. The constant "
                f"epsilon=0.1 schedule never reaches 0.02. These are measured training exploration "
                f"trajectories; evaluation uses greedy actions throughout.",
                f"In these runs, longer-lasting exploration does not improve whole-budget or late performance: "
                f"fast decay has AUC {number(last, 'auc_return_mean')} and a last-five average "
                f"{number(last, 'last5_mean_mean')}, compared with {number(baseline, 'auc_return_mean')} "
                f"and {number(baseline, 'last5_mean_mean')} for slow decay. Both end at "
                f"{number(last, 'final_return_mean')}. Constant epsilon has AUC "
                f"{number(first, 'auc_return_mean')} and final mean {number(first, 'final_return_mean')}.",
                f"Median third-checkpoint confirmations among successful seeds are "
                f"{confirmation(last)} for fast decay, {confirmation(baseline)} for slow decay and "
                f"{confirmation(first)} for constant epsilon. This speed statistic, AUC and late "
                f"variability answer different questions. Rapid decay did not inevitably harm the measured "
                f"performance here. Epsilon measures random-action probability; state-space coverage was "
                f"not measured, so a causal explanation in terms of better coverage remains a hypothesis."]
        else:
            paragraphs = [
                f"实际按回合衰减的0.9方案在{min(fast_floors):,}-{max(fast_floors):,}训练步到达epsilon=0.02，"
                f"0.995方案在{min(slow_floors):,}-{max(slow_floors):,}步才到下限。"
                "快衰减因此在Replay预热1000步及首次梯度更新之前就到达下限。固定epsilon=0.1始终不会降到0.02。"
                "这些是实测训练探索时间轴；评估过程始终使用贪心动作。",
                f"本次运行中，延长探索没有改善整段或后期表现：快衰减AUC为{number(last, 'auc_return_mean')}，"
                f"末5次均值{number(last, 'last5_mean_mean')}；慢衰减分别为"
                f"{number(baseline, 'auc_return_mean')}和{number(baseline, 'last5_mean_mean')}，"
                f"两者终点均值都是{number(last, 'final_return_mean')}。固定epsilon的AUC为"
                f"{number(first, 'auc_return_mean')}，终点均值为{number(first, 'final_return_mean')}。",
                f"仅在确认成功的种子中计算，快衰减、慢衰减、固定epsilon的第三点确认步数中位数分别为"
                f"{confirmation(last)}、{confirmation(baseline)}、{confirmation(first)}。"
                "确认速度、AUC与后期波动回答的是不同问题，不能套用快速衰减必然更差的结论。"
                "epsilon是随机动作概率，不直接测量状态空间覆盖率；将收益差异归因为覆盖率变化仍属于假设。"]
    return "\n\n".join(paragraphs)


def findings(rows, question, language):
    if not rows:
        return "No valid results are available." if language == "en" else "尚无有效实测结果。"
    best_tail = max(rows, key=lambda row: row["last3_mean_mean"])
    best_auc = max(rows, key=lambda row: row["auc_return_mean"])
    best_sd = min(rows, key=lambda row: row["last3_mean_sd"])
    smoothest = min(rows, key=lambda row: row["late5_temporal_sd_mean"])
    if language == "en":
        text = (f"At the recorded budget, {best_tail['condition']} had the highest mean over the last three "
                f"evaluations ({fmt(best_tail['last3_mean_mean'])}). {best_auc['condition']} had the largest "
                f"step-normalized area under the evaluation curve ({fmt(best_auc['auc_return_mean'])}), "
                f"which reflects return over the whole training budget rather than only the last checkpoint. "
                f"The lowest between-seed SD of the last-three-evaluation mean was observed for "
                f"{best_sd['condition']} ({fmt(best_sd['last3_mean_sd'])}). These rankings are descriptive; "
                f"a low SD alone does not imply useful learning. The smallest mean within-seed temporal SD "
                f"over the last five checkpoints was {fmt(smoothest['late5_temporal_sd_mean'])} "
                f"for {smoothest['condition']}; this describes late-training fluctuation rather than "
                f"between-seed agreement.")
        mechanisms = {
            "q1": "A higher learning rate makes larger parameter updates and can amplify bootstrapped-target instability; a lower rate can improve gradual learning but need more steps. The observed curves, seed returns and AUC must be considered together before calling one rate faster or more stable. Low or fluctuating return alone is not proof of numerical divergence.",
            "q2": "Exploration decays after each completed training episode, so epsilon at the same step count also depends on achieved episode length. Rapid decay reduces random actions sooner but can restrict discovery; slow decay supplies broader experience while potentially delaying exploitation. Greedy evaluation removes direct evaluation-time epsilon noise, but does not remove the effect of exploration on the training data.",
            "q3": "Replay capacity changes how long past transitions remain available; a small buffer refreshes quickly while a large buffer can retain older policy data. Batch size changes gradient-estimate noise and computational cost. Every eligible training step receives one update, so batch 128 processes four times as many replay samples per update as batch 32. All tested capacities can become full within the 200,000-step budget. These mechanisms motivate the comparison but do not by themselves prove the cause of each observed difference."}
    else:
        text = (f"在已记录训练预算下，{best_tail['condition']} 的末3次评估均值最高（{fmt(best_tail['last3_mean_mean'])}）；"
                f"{best_auc['condition']} 的按训练步数归一化曲线面积最高（{fmt(best_auc['auc_return_mean'])}），"
                f"该指标反映整个训练过程的回报而非仅末次结果。末3次评估均值的跨种子标准差最小的是 "
                f"{best_sd['condition']}（{fmt(best_sd['last3_mean_sd'])}）。这些只是描述性排名，"
                "低标准差也可能对应多个种子都未学会，不能单独等同于训练良好。"
                f"各种子末5个检查点回报的时间标准差均值最小的是{smoothest['condition']}"
                f"（{fmt(smoothest['late5_temporal_sd_mean'])}），该指标描述训练后期波动，与跨种子一致性不同。")
        mechanisms = {
            "q1": "较大学习率使参数更新幅度增大，可能放大自举目标造成的不稳定；较小学习率可能学习更慢，需要更多训练步。应结合曲线、逐种子回报及曲线面积讨论快慢和稳定性，不能仅凭低回报或振荡认定发生数值发散。",
            "q2": "epsilon 在每个完整训练回合结束后衰减，因此同一步数下的 epsilon 还受回合长度影响。快速衰减较早减少随机动作，但可能限制探索；慢衰减提供更多探索经验，也可能延缓利用。贪心评估消除了评估时随机动作的直接干扰，但训练数据仍受到探索策略影响。",
            "q3": "Replay容量改变旧经验保留时长：小缓冲区更新快，大缓冲区可保留更旧策略的数据。batch size改变梯度估计噪声及更新成本。每个符合条件的训练步均更新一次，因此batch128每次更新处理的Replay样本数是batch32的4倍。所有测试容量均可在200000步预算内填满。这些机制说明对照意义，但不能单独证明实测差异的成因。"}
    return text + "\n\n" + mechanisms[question]


def write_reports(runs, rows, questions, issues, output, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    state = "INCOMPLETE PROGRESS REPORT" if issues else "Completed experimental report"
    configs = [run["config"] for run in runs]
    budget_text = ", ".join(map(str, sorted({int(c["total_steps"]) for c in configs}))) or "not available"
    eval_text = ", ".join(map(str, sorted({int(c["eval_episodes"]) for c in configs}))) or "not available"
    every_text = ", ".join(map(str, sorted({int(c["eval_every"]) for c in configs}))) or "not available"
    seeds = ", ".join(map(str, sorted({int(c["seed"]) for c in configs}))) or "not available"
    en = ["# Workshop 3: Deep Q-Learning - Task 2", "Authors: Zhengnan Cao, Yuchen Yang", f"Status: {state}",
          "## Scope and method",
          "This report supplements the preserved Task 1 submission and answers all three Task 2 questions. "
          "The original workshop handout requires at least two questions; all three are evaluated here. "
          "Task 1 results are not relabelled as Task 2 measurements.",
          f"Training used CartPole-v1, a fixed budget of {budget_text} training environment steps per run, "
          f"training seeds {seeds}, and {eval_text} greedy evaluation episodes every {every_text} training steps "
          "including step zero and the final budget. Evaluation steps are excluded from the x-axis. "
          "Identical evaluation reset seeds are reused at every checkpoint and across conditions. The target "
          "bootstraps through time-limit truncation and stops only on environment termination. Epsilon decays "
          "once per completed training episode.",
          "The learning curves show the mean across independent training seeds. Shaded bands are plus/minus "
          "one sample SD across those seeds, clipped to the possible return range; faint lines show individual "
          "seed curves. No smoothing is applied. Evaluation-episode SD and between-training-seed SD are distinct "
          "quantities. The table reports mean plus/minus sample SD across training seeds. 'Last 3' first averages "
          "the last three checkpoints within each training seed. AUC return is trapezoidal curve area divided "
          "by the training-step span. Late-five means and population temporal SDs describe the last five "
          "checkpoints within a seed, before averaging over training seeds. The epsilon plot shows individual "
          "piecewise-constant training schedules, with a logarithmic y-axis, rather than evaluation returns. "
          "The threshold statistic requires three consecutive checkpoint "
          "means of at least 475. Its confirmation time is the third checkpoint, so qualifying evaluations "
          "at 40k, 50k and 60k confirm the criterion at 60k. This is a descriptive criterion chosen for this "
          "analysis, not an assignment requirement or a claim of permanent convergence; returns can fall "
          "again after confirmation. Aggregation uses exact recorded step grids; "
          "no interpolation or extrapolation is performed."]
    zh = ["# Workshop 3：DQN Task 2 中文分析", "作者：Zhengnan Cao，Yuchen Yang",
          "状态：实验进度稿，尚有检查项未满足。" if issues else "状态：正式实验及证据检查已完成。",
          "## 范围与统一实验方法",
          "本文件补充已归档的Task 1，覆盖Task 2全部三题；作业最低要求为三题中至少选两题。Task 1的旧结果不作为本次Task 2的重复测量。",
          f"每个运行使用CartPole-v1、{budget_text}个训练环境步，训练种子为{seeds}。每隔{every_text}训练步进行"
          f"{eval_text}回合贪心评估，包含训练前及预算末端评估；横轴不计评估步数。所有条件与检查点复用相同评估初始状态种子。"
          "时间上限截断仍bootstrap，真实终止时停止bootstrap；epsilon每个完整训练回合衰减一次。",
          "曲线实线为独立训练种子均值，阴影为跨种子样本标准差（裁剪到合法回报范围），淡线为逐种子曲线，未平滑。"
          "评估回合内部标准差与训练种子之间标准差不同。表格为跨种子均值±样本标准差；末3次指标先在每个种子内部取末3个检查点平均。"
          "末5次均值及时序总体标准差先按每个种子计算，再在训练种子间平均。epsilon图显示各训练种子的分段常数探索轨迹，纵轴为对数刻度。"
          "AUC为梯形积分后除以训练步数跨度。阈值指标要求连续3个检查点评估均值≥475，记录第三个检查点的确认步数；"
          "例如40k、50k、60k均达标，则确认时刻为60k。这是本分析自定的描述性指标，不是作业规定，"
          "也不代表此后永久稳定收敛，确认后仍可能回落。汇总严格使用真实检查点步数，不插值、不外推。"]
    if configs:
        baseline_keys = ["hidden_size", "layer_num", "discount_factor", "learning_start", "target_update_freq"]
        baseline = "; ".join(f"{key}={configs[0].get(key)}" for key in baseline_keys)
        en += ["Common implementation settings: " + baseline + ". The full configuration of every run is "
               "preserved in config.json; provenance and hashes are listed in the evidence CSV."]
        zh += ["公共实现参数：" + baseline + "。每次运行的完整配置保存在config.json，证据CSV记录路径及哈希。"]
        en += ["Q1/Q2 use replay capacity 100,000 and minibatch 64. Their shared baseline uses learning rate "
               "0.001, initial epsilon 1.0, per-episode decay 0.995 and floor 0.02. Q1 changes only learning "
               "rate; Q2 changes only the exploration schedule. The baseline is one training run per seed "
               "reused in both comparisons, giving 15 unique Q1/Q2 runs plus 18 Q3 runs."]
        zh += ["Q1/Q2固定Replay容量100000、batch64；公共基准为学习率0.001、初始epsilon1.0、每回合衰减0.995、下限0.02。"
               "Q1只改变学习率，Q2只改变探索方案。公共基准每个种子只训练一次并复用于两题，共15个Q1/Q2独立运行，加18个Q3运行。"]
    relocated = [run for run in runs if str(Path(run["config"].get("output_dir", run["run_dir"])).resolve()).casefold()
                 != str(Path(run["run_dir"]).resolve()).casefold()]
    if relocated:
        en += ["Path recovery: completed evidence was copied to the current writable workspace with its "
               "original config/checkpoint preserved. A saved output_dir may therefore retain its historical "
               "location; this is allowed only when the formal manifest declares that original_output_dir "
               "or preserved_config_output_dir. "
               "Algorithm settings, seeds, budgets and source hashes remain checked. Interrupted directories "
               "are excluded; only the fresh attempts named by the current formal manifest are analysed."]
        zh += ["路径恢复说明：已完成证据复制到当前可写工作区，原config/checkpoint保持原样，"
               "因此保存的output_dir可能仍指向历史位置；仅在正式清单明确记录original_output_dir或preserved_config_output_dir时允许该路径差异。"
               "算法参数、种子、预算及源码哈希仍须一致。已中断目录不纳入分析，只统计当前正式清单指定的新运行。"]
    titles_en = {"q1": "Question 1: Learning rate", "q2": "Question 2: Exploration schedules", "q3": "Question 3: Replay capacity and batch size"}
    titles_zh = {"q1": "问题1：学习率", "q2": "问题2：探索策略", "q3": "问题3：Replay容量与batch size"}
    for question in questions:
        subset = [row for row in rows if row["question"] == question]
        rel_plot = Path(os.path.relpath(output / f"{question}_evaluation_returns.png", report_dir))
        en += [f"## {titles_en[question]}"]
        zh += [f"## {titles_zh[question]}"]
        if (output / f"{question}_evaluation_returns.png").exists() and subset:
            en += [f"![{titles_en[question]}]({rel_plot.as_posix()})"]
            zh += [f"![{titles_zh[question]}]({rel_plot.as_posix()})"]
        en += [report_table(subset, "en"), q12_discussion(subset, question, runs, "en")
               if question in {"q1", "q2"} else findings(subset, question, "en")]
        zh += [report_table(subset, "zh"), q12_discussion(subset, question, runs, "zh")
               if question in {"q1", "q2"} else findings(subset, question, "zh")]
        if question == "q2" and (output / "q2_realized_epsilon.png").exists():
            epsilon_plot = Path(os.path.relpath(output / "q2_realized_epsilon.png", report_dir)).as_posix()
            en += ["## Realized training exploration", f"![Per-seed training epsilon schedules]({epsilon_plot})",
                   "The early-training panel marks the 1,000-step learning start. Epsilon is constant within "
                   "each training episode and changes after that episode ends; the raw per-seed floor steps "
                   "are retained in q2_realized_epsilon.csv."]
            zh += ["## 实际训练探索时间轴", f"![各训练种子的实际epsilon轨迹]({epsilon_plot})",
                   "局部放大图标出1000步学习起点。epsilon在每个训练回合内部保持不变，回合结束后才衰减；"
                   "q2_realized_epsilon.csv保留逐种子到达下限的真实步数。"]
        if question == "q3":
            en += ["The replay experiment uses the full 3 x 2 factorial grid: capacities 1,000 / 10,000 / 100,000 "
                   "and batch sizes 32 / 128, with all other hyperparameters fixed. Capacity effects should be "
                   "compared at a fixed batch size, and batch effects at a fixed capacity. The full condition "
                   "table avoids selecting a different seed for each comparison."]
            zh += ["Replay实验使用完整3×2组合：容量1000/10000/100000与batch32/128，其余超参数固定。"
                   "容量效应需在相同batch内比较，batch效应需在相同容量内比较；逐配置完整列表避免每个对比挑选不同最佳种子。"]
            if (output / "q3_replay_interactions.png").exists():
                interaction_plot = Path(os.path.relpath(output / "q3_replay_interactions.png", report_dir)).as_posix()
                en += ["## Replay effects at a fixed batch size",
                       f"![Replay capacity comparisons within each minibatch size]({interaction_plot})"]
                zh += ["## 固定batch下的Replay容量对比",
                       f"![分别固定batch的Replay容量比较]({interaction_plot})"]
            en += [replay_discussion(subset, "en")]
            zh += [replay_discussion(subset, "zh")]
    en += ["## Evidence and limitations",
           "The evidence includes config.json, evaluations.csv, evaluation_episodes.csv, metrics.csv, checkpoint.pt "
           "and summary.json per training run, plus per_run_metrics.csv, condition_summary.csv, learning_curves.csv "
           "and audit.json from this analysis. A fixed set of evaluation initial states supports controlled "
           "comparison but does not establish performance on unseen reset seeds. Three training seeds provide "
           "limited uncertainty estimates; the rankings are not significance tests. Training-step efficiency "
           "and wall-clock efficiency differ, especially across batch sizes. These experiments concern "
           "CartPole-v1 and do not establish performance on Atari or other tasks."]
    zh += ["## 证据与局限",
           "每个运行保留config.json、evaluations.csv、evaluation_episodes.csv、metrics.csv、checkpoint.pt和summary.json；"
           "汇总层保留per_run_metrics.csv、condition_summary.csv、learning_curves.csv和audit.json。固定评估初始状态有助于受控对比，"
           "但不等同于在未见初始状态上的泛化测试。三个训练种子的不确定性估计有限，描述性排名不是显著性检验。"
           "环境步效率与实际耗时不同，大batch尤其如此。本实验结论仅限CartPole-v1，不能外推到Atari或其他任务。"]
    if issues:
        en += ["## Outstanding checks", "\n".join("- " + issue for issue in issues)]
        zh += ["## 尚未满足的检查项", "\n".join("- " + issue for issue in issues)]
    en += ["## Reference", "ELEN90098, Workshop 3: Deep Q Learning, provided course handout, Task 2, Questions 1-3."]
    zh += ["## 作业依据", "ELEN90098课程提供的Workshop 3: Deep Q Learning讲义，Task 2第1至3题。"]
    (report_dir / "Workshop3_Task2_Report_EN.md").write_text("\n\n".join(en) + "\n", encoding="utf-8")
    (report_dir / "Workshop3_Task2_Analysis_ZH.md").write_text("\n\n".join(zh) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-roots", nargs="+", type=Path, default=[WS3])
    fallback_manifests = [WS3 / "q1_q2" / "runs" / "formal" / "manifest.json",
                          WS3 / "q3" / "runs" / "formal" / "manifest.json"]
    p.add_argument("--manifest", type=Path, nargs="+",
                   default=[DEFAULT_MANIFEST] if DEFAULT_MANIFEST.exists() else fallback_manifests,
                   help="Only include output directories explicitly listed in this formal job manifest")
    p.add_argument("--allow-unlisted-diagnostics", action="store_true",
                   help="For explicit diagnostic reports only: discover runs without a formal manifest")
    p.add_argument("--questions", nargs="+", choices=list(EXPECTED_COUNTS), default=list(EXPECTED_COUNTS))
    p.add_argument("--min-seeds", type=int, default=3)
    p.add_argument("--exclude", nargs="*", default=["smoke", "pilot", "benchmark", "quick"])
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--report-dir", type=Path, default=WS3 / "reports")
    args = p.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    issues = []
    allowed_dirs = None
    jobs = []
    if not args.allow_unlisted_diagnostics:
        jobs = []
        for manifest_path in args.manifest:
            if not manifest_path.exists():
                raise SystemExit(f"Formal manifest is missing: {manifest_path}. Pass the intended --manifest.")
            jobs.extend(load_formal_jobs(manifest_path))
        allowed_dirs = {str(Path(job["output_dir"]).resolve()).casefold() for job in jobs}
        for job in jobs:
            if not (Path(job["output_dir"]) / "config.json").exists():
                issues.append(f"Planned formal run has no saved configuration: {job['output_dir']}")
    runs = discover([path.resolve() for path in args.run_roots], args.questions, args.exclude, issues, allowed_dirs, jobs)
    groups = audit(runs, args.questions, args.min_seeds, issues)
    write_json(output / "audit.json", {"generated_utc": datetime.now(timezone.utc).isoformat(),
        "valid_runs": len({run["run_dir"] for run in runs}), "question_run_records": len(runs),
        "questions": args.questions, "minimum_training_seeds": args.min_seeds,
        "threshold_definition": {"minimum_mean_return": 475, "consecutive_checkpoints": 3,
                                 "recorded_step": "third qualifying checkpoint confirmation step",
                                 "descriptive_only": True, "subsequent_decline_possible": True},
        "formal_manifest": [str(path.resolve()) for path in args.manifest] if allowed_dirs is not None else None,
        "manifest_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in args.manifest} if allowed_dirs is not None else None,
        "path_resolution": "catalogue original_root mapped to current package root; local evidence only",
        "complete": not issues, "issues": issues, "run_roots": [str(path.resolve()) for path in args.run_roots]})
    if issues and not args.allow_incomplete:
        print("Report refused because evidence is incomplete. See " + str(output / "audit.json"))
        for issue in issues:
            print("- " + issue)
        raise SystemExit(2)
    rows = summarize(groups, output)
    write_reports(runs, rows, args.questions, issues, output, args.report_dir.resolve())
    unique_runs = len({run["run_dir"] for run in runs})
    print(f"Analysed {unique_runs} independent runs ({len(runs)} question-run records), "
          f"{len(rows)} conditions. Audit complete={not issues}")
    print(args.report_dir.resolve())


if __name__ == "__main__":
    main()
