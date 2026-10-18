"""Independent structural and numeric audit of completed experiment manifests.

Checks episode-level source data against plotted evaluation statistics, finite
losses, checkpoint step counts and complete budget accounting. Does not train.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from manifest_paths import load_jobs


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(job):
    output = Path(job["output_dir"])
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] in {"complete", "completed", "success"}, summary
    evaluations = read_csv(output / "evaluations.csv")
    episodes = read_csv(output / "evaluation_episodes.csv")
    metrics = read_csv(output / "metrics.csv")
    steps = [int(row["env_steps"]) for row in evaluations]
    assert steps[0] == 0 and all(b > a for a, b in zip(steps, steps[1:])), steps
    actual = int(summary["actual_training_steps"])
    assert steps[-1] == actual and actual >= config["total_steps"]
    if "batch_steps" in config:
        assert actual < config["total_steps"] + config["batch_steps"] + config.get("max_episode_steps", 1000)
    else:
        assert actual == config["total_steps"]
    assert len(episodes) == len(evaluations) * config["eval_episodes"]
    for row in evaluations:
        step = int(row["env_steps"])
        detailed = [item for item in episodes if int(item["env_steps"]) == step]
        assert len(detailed) == config["eval_episodes"]
        assert [int(item["seed"]) for item in detailed] == list(range(config["eval_seed"], config["eval_seed"] + config["eval_episodes"]))
        values = np.array([float(item["return"]) for item in detailed])
        assert np.isfinite(values).all()
        assert np.isclose(values.mean(), float(row["mean_return"]), atol=1e-8)
        assert np.isclose(values.std(), float(row["std_return"]), atol=1e-8)
        if config["env_name"] in {"CartPole-v1", "InvertedPendulum-v5"}:
            assert np.all((values >= 0) & (values <= (500 if config["env_name"] == "CartPole-v1" else 1000)))
    assert metrics and int(metrics[-1]["env_steps"]) == actual
    for row in metrics:
        for name, value in row.items():
            if value not in {"", None}:
                assert np.isfinite(float(value)), (name, value)
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint.get("env_steps", checkpoint.get("training_steps")) == actual
    state = checkpoint.get("actor_state_dict", checkpoint.get("q_net"))
    assert state and all(torch.isfinite(value).all().item() for value in state.values())
    assert checkpoint["config"]["code_hash"] == config["code_hash"]
    return {"id": job["id"], "status": "passed", "training_steps": actual,
            "evaluation_points": len(evaluations), "evaluation_episodes": len(episodes),
            "code_hash": config["code_hash"],
            "checkpoint_sha256": hashlib.sha256((output / "checkpoint.pt").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    jobs = []
    for manifest in args.manifests:
        jobs.extend(load_jobs(manifest))
    reports = []
    for job in jobs:
        try:
            reports.append(validate(job))
        except Exception as error:
            reports.append({"id": job["id"], "status": "failed", "error": repr(error)})
    result = {"runs": len(reports), "passed": sum(row["status"] == "passed" for row in reports), "results": reports}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}))
    for report in reports:
        if report["status"] == "failed":
            print(json.dumps(report))
    return 0 if result["passed"] == result["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
