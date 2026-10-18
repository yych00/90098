"""Summarize completed pilot runs and select by the last three evaluation means."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPECTED_PG_HASH = "54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341"
sys.path.insert(0, str(ROOT.parents[1] / "shared"))
from validate_runs import validate
from manifest_paths import load_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["cartpole_pilot", "halfcheetah_pilot"])
    args = parser.parse_args()
    manifest = load_jobs(ROOT / (args.phase + ".jobs.json"))
    records = []
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    configs = {}
    audit = []
    for job in manifest:
        audit.append(validate(job))
        path = Path(job["output_dir"])
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] not in {"complete", "completed"}:
            raise ValueError(f"Incomplete pilot: {job['id']}")
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("code_hash") != EXPECTED_PG_HASH or config.get("estimator_version") != "corrected-v2":
            raise ValueError(f"Pilot is not the frozen corrected-v2 estimator: {job['id']}")
        if config["seed"] != 0:
            raise ValueError(f"Pilot selection requires training seed 0: {job['id']}")
        configs[job["id"]] = config
        with (path / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        steps = np.array([float(row["env_steps"]) for row in rows])
        means = np.array([float(row["mean_return"]) for row in rows])
        label = job["id"].replace(args.phase + "_", "").removesuffix("_s0")
        axis.plot(steps, means, label=label, linewidth=1.6)
        records.append({"run": job["id"], "output_dir": str(path), "training_steps": int(steps[-1]),
                        "initial_mean_return": float(means[0]), "final_mean_return": float(means[-1]),
                        "last3_mean_return": float(means[-3:].mean()), "best_mean_return": float(means.max()),
                        "wall_seconds": summary["wall_seconds"]})
    records.sort(key=lambda row: (-row["last3_mean_return"], row["run"]))
    target = ROOT / (args.phase + "_summary.csv")
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    winner = records[0]
    config = configs[winner["run"]]
    parameters = {name: config[name] for name in ["learning_rate", "batch_steps", "layer_num", "hidden_size", "discount_gamma", "use_reward_to_go",
                  "use_baseline", "normalize_advantage", "gae_lambda", "baseline_learning_rate", "max_grad_norm"]}
    selection = {"selection_rule": "highest mean of final three fixed-seed evaluation means; seed-0 pilot only",
                 "estimator_version": "corrected-v2", "code_hash": EXPECTED_PG_HASH,
                 "pilot": winner, "parameters": parameters,
                 "candidate_ranking": records,
                 "raw_data_audit": audit,
                 "caution": "pilot selection is exploratory; final three-seed comparison is reported separately"}
    selection_path = ROOT / (args.phase.replace("_pilot", "") + "_selection.json")
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    axis.set(xlabel="Training environment steps", ylabel="Mean evaluation episode return",
             title=args.phase.replace("_", " ").title() + " (one training seed, ten evaluation episodes)")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    figure.savefig(ROOT / (args.phase + ".png"), dpi=180)
    plt.close(figure)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
