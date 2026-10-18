"""Generate explicit WS4 Q3/Q4 experiment manifests; does not start training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "Policy_Gradient" / "run_policy_gradient.py"
EXPECTED_PG_HASH = "54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341"


def job(identifier, environment, steps, seed=0, **parameters):
    output = ROOT / "q3_q4" / "runs" / identifier
    command = [sys.executable, "-u", str(TRAINER), "--env_name", environment,
               "--output_dir", str(output), "--total_steps", str(steps), "--seed", str(seed),
               "--eval_seed", "10000", "--eval_every", "10000", "--eval_episodes", "10",
               "--device", "cpu"]
    options = {"discount_gamma": 0.99, "hidden_size": 64, **parameters}
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                command.append("--" + key)
        else:
            command.extend(["--" + key, str(value)])
    return {"id": identifier, "command": command, "cwd": str(ROOT), "output_dir": str(output),
            "estimator_version": "corrected-v2", "expected_code_hash": EXPECTED_PG_HASH}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["cartpole_pilot", "halfcheetah_pilot", "cartpole_final", "halfcheetah_final"])
    parser.add_argument("--selection", type=Path, help="JSON {parameters: {...}} selected after inspecting pilot results")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    jobs = []
    if args.phase == "cartpole_pilot":
        configurations = {
            "rtg_lr1e3_b1k": dict(use_reward_to_go=True, learning_rate=1e-3, batch_steps=1000),
            "rtg_lr3e3_b2k": dict(use_reward_to_go=True, learning_rate=3e-3, batch_steps=2000),
            "baseline_lr1e3_b2k": dict(use_baseline=True, learning_rate=1e-3, batch_steps=2000),
        }
        jobs = [job("cartpole_pilot_" + name + "_s0", "CartPole-v1", args.steps or 100000,
                    layer_num=3, **parameters) for name, parameters in configurations.items()]
    elif args.phase == "halfcheetah_pilot":
        configurations = {
            "rtg_raw": dict(use_reward_to_go=True, learning_rate=3e-4),
            "rtg_normalized": dict(use_reward_to_go=True, normalize_advantage=True, learning_rate=1e-3),
            "gae_normalized": dict(gae_lambda=.95, normalize_advantage=True, learning_rate=3e-4,
                                   baseline_learning_rate=1e-3),
        }
        jobs = [job("halfcheetah_pilot_" + name + "_s0", "HalfCheetah-v5", args.steps or 200000,
                    layer_num=2, batch_steps=5000, **parameters) for name, parameters in configurations.items()]
    else:
        if args.selection is None:
            parser.error("final experiments require an explicit --selection JSON")
        selected = json.loads(args.selection.read_text(encoding="utf-8"))
        if selected.get("code_hash") != EXPECTED_PG_HASH or selected.get("estimator_version") != "corrected-v2":
            parser.error("final selection must come from audited corrected-v2 pilots")
        cheetah = args.phase.startswith("halfcheetah")
        jobs = [job(args.phase + "_s" + str(seed), "HalfCheetah-v5" if cheetah else "CartPole-v1",
                    args.steps or (1000000 if cheetah else 200000), seed=seed,
                    **selected["parameters"]) for seed in [0, 1, 2]]
    target = ROOT / "q3_q4" / (args.phase + ".jobs.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(jobs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)
    print("jobs:", len(jobs))


if __name__ == "__main__":
    main()
