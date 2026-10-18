"""Audit completed experiment evidence without running new training."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Policy_Gradient"))
TASK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK_ROOT / "shared"))
from run_policy_gradient import source_metadata
from analyze_results import require_v2_config, require_fresh_v2_jobs, CORRECTED_V2_HASH
from manifest_paths import load_jobs, allowed_output_paths


def rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    jobs = load_jobs(root / "jobs.json")
    require_fresh_v2_jobs(jobs)
    original_audit_path = root / "evidence_audit.json"
    is_relocated = any(str(Path(job["output_dir"]).resolve()).casefold() !=
                       str(Path(job.get("manifest_output_dir", job["output_dir"])).resolve()).casefold() for job in jobs)
    checked, pending, failures, initial_returns_by_seed, path_provenance = [], [], [], {}, []
    for job in jobs:
        output = Path(job["output_dir"])
        assert output.resolve().is_relative_to(TASK_ROOT), "Refusing to read a host run outside the current package root"
        if not (output / "summary.json").exists():
            pending.append(job["id"])
            continue
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "completed":
            if summary["status"] == "failed":
                error = summary.get("error", "")
                numerical = "FloatingPointError" in error or ("Expected parameter" in error and "invalid values" in error)
                failures.append({"id": job["id"], "error": error, "actual_training_steps": summary["actual_training_steps"],
                                 "classification": "numerical_failure" if numerical else "requires_resolution"})
            else:
                pending.append(job["id"])
            continue
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        require_v2_config(config)
        # Relocation must preserve completed configs/checkpoints byte-for-byte.
        allowed_paths = allowed_output_paths(job)
        assert str(Path(config["output_dir"]).resolve()).casefold() in allowed_paths, (job["id"], "unexplained recorded output path")
        path_provenance.append({"id": job["id"], "formal_directory": str(output),
                                "recorded_output_dir": config["output_dir"], "attempt": 1, "run_origin": "fresh_corrected_v2"})
        assert config["code_hash"] == source_metadata()["code_hash"] == CORRECTED_V2_HASH, job["id"]
        for field in ("env_name", "seed", "batch_steps", "learning_rate", "discount_gamma", "hidden_size", "layer_num",
                      "eval_seed", "eval_episodes", "eval_every", "total_steps"):
            expected = job["command"][job["command"].index("--" + field) + 1]
            assert str(config[field]) == expected or (field != "env_name" and float(config[field]) == float(expected)), (job["id"], field)
        for flag in ("use_reward_to_go", "use_baseline"):
            assert config[flag] == ("--" + flag in job["command"]), (job["id"], flag)
        assert config["normalize_advantage"] is False and config["gae_lambda"] == 1 and config["max_grad_norm"] == 0, job["id"]
        evaluations = rows(output / "evaluations.csv")
        episodes = rows(output / "evaluation_episodes.csv")
        metrics = rows(output / "metrics.csv")
        groups = defaultdict(list)
        for episode in episodes:
            groups[int(episode["env_steps"])].append(episode)
        x = [int(row["env_steps"]) for row in evaluations]
        assert x[0] == 0 and all(a < b for a, b in zip(x, x[1:])), job["id"]
        assert x[-1] == summary["actual_training_steps"] >= config["total_steps"], job["id"]
        assert x[-1] < config["total_steps"] + config["batch_steps"] + config["max_episode_steps"], job["id"]
        assert len(episodes) == len(evaluations) * config["eval_episodes"], job["id"]
        for evaluation in evaluations:
            group = groups[int(evaluation["env_steps"])]
            assert len(group) == config["eval_episodes"], job["id"]
            assert [int(row["seed"]) for row in group] == config["eval_environment_seeds"], job["id"]
            returns = [float(row["return"]) for row in group]
            assert np.isfinite(returns).all() and min(returns) >= 0 and max(returns) <= config["max_episode_steps"], job["id"]
            assert np.isclose(np.mean(returns), float(evaluation["mean_return"])), job["id"]
            assert np.isclose(np.std(returns), float(evaluation["std_return"])), job["id"]
        initial_returns = [float(row["return"]) for row in groups[0]]
        seed = config["seed"]
        if seed in initial_returns_by_seed:
            assert initial_returns == initial_returns_by_seed[seed], (job["id"], "initial-policy control differs across configurations")
        else:
            initial_returns_by_seed[seed] = initial_returns
        assert len(metrics) == summary["updates"], job["id"]
        total = 0
        for index, metric in enumerate(metrics, 1):
            total += int(metric["batch_steps"])
            assert int(metric["update"]) == index and int(metric["env_steps"]) == total, job["id"]
            for field in ("actor_loss", "gradient_norm", "advantage_mean", "advantage_std"):
                assert np.isfinite(float(metric[field])), (job["id"], field)
        assert total == summary["actual_training_steps"], job["id"]
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert checkpoint["env_steps"] == total and checkpoint["updates"] == len(metrics), job["id"]
        assert checkpoint["version"] == "task2-pg-2.0.0", job["id"]
        expected_baseline_episodes = sum(int(row["batch_episodes"]) for row in metrics) if config["use_baseline"] else 0
        assert checkpoint["baseline_episode_count"] == expected_baseline_episodes, (job["id"], "baseline history episode count")
        assert np.isfinite(checkpoint["baseline_return_sum"]), job["id"]
        if not config["use_baseline"]:
            assert checkpoint["baseline_return_sum"] == 0, job["id"]
        assert checkpoint["config"]["code_hash"] == config["code_hash"], job["id"]
        assert checkpoint["config"]["output_dir"] == config["output_dir"], (job["id"], "checkpoint/config path history differs")
        assert all(torch.isfinite(value).all() for value in checkpoint["actor_state_dict"].values()), job["id"]
        if config["tensorboard"]:
            assert list((output / "tensorboard").glob("events.out.tfevents.*")), job["id"]
            events = EventAccumulator(str(output / "tensorboard"), size_guidance={"scalars": 0}).Reload()
            for tag, field in (("Eval Returns", "mean_return"), ("Eval Returns Std", "std_return")):
                scalars = events.Scalars(tag)
                assert [record.step for record in scalars] == x, (job["id"], tag, "step sequence differs")
                # TensorBoard serializes scalars as float32 while the CSV retains doubles.
                assert np.allclose([record.value for record in scalars],
                                   [float(row[field]) for row in evaluations], atol=1e-4, rtol=1e-6), (job["id"], tag, "values differ")
        checked.append(job["id"])
    unresolved = any(row["classification"] == "requires_resolution" for row in failures)
    result = {"status": "requires_resolution" if unresolved else "partial" if pending else "passed_with_numerical_failures" if failures else "passed",
              "completed_runs_audited": len(checked), "expected_runs": len(jobs),
              "pending": pending, "failed_runs": failures,
              "path_provenance": path_provenance,
              "formal_manifest_provenance": "45 fresh corrected-v2 runs; no historical PG reuse or v1 recovery-prefix comparison",
              "fresh_v2_jobs": len(jobs), "completed_fresh_v2_runs": len(checked), "reused_historical_pg_runs": 0,
              "estimator_version": "corrected-v2", "checks": ["corrected_v2_trainer_code_hash_and_estimator_contract", "checkpoint_baseline_history_episode_count", "actual_configuration_matches_job_manifest", "no_extensions_in_core_comparison",
                         "recorded_output_path_matches_current_or_relocated_v2_catalogue_path",
                         "identical_initial_policy_evaluations_for_paired_training_seeds", "strictly_increasing_actual_steps", "budget_and_overshoot_bound",
                         "all_eval_episode_counts_and_fixed_seeds", "all_eval_means_and_standard_deviations",
                         "finite_bounded_evaluation_returns", "cumulative_batch_steps_and_update_count", "finite_training_metrics",
                         "checkpoint_step_count_config_hash_finite_actor", "tensorboard_event_presence",
                         "tensorboard_eval_scalar_steps_and_float32_values_match_all_csv_evaluations"]}
    target = root / ("evidence_audit_recheck.json" if is_relocated else "evidence_audit.json")
    result["audit_output"] = str(target)
    if is_relocated:
        result["original_audit_preserved"] = str(original_audit_path) if original_audit_path.exists() else None
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"pending", "path_provenance"}}))
    if pending and not args.allow_partial:
        raise SystemExit("Unfinished runs remain")
    if unresolved:
        raise SystemExit("Non-numerical training failures require resolution")


if __name__ == "__main__":
    main()
