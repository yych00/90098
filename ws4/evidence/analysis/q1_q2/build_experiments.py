"""Materialize 15 distinct configurations / 45 three-seed jobs for WS4 Q1-Q2."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--total_steps", type=int, default=500000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    trainer = root.parent / "Policy_Gradient" / "run_policy_gradient.py"
    configs = {}
    for method in ("vanilla", "rtg", "baseline"):
        for learning_rate in (1e-4, 1e-3, 1e-2):
            configs[(method, learning_rate, 1000)] = ["Q1"]
        for batch_steps in (1000, 2000, 5000):
            configs.setdefault((method, 1e-3, batch_steps), []).append("Q2")
    jobs, grid = [], []
    for (method, lr, batch), questions in configs.items():
        config_id = f"{method}_lr{lr:.0e}_b{batch}"
        grid.append({"config_id": config_id, "method": method, "learning_rate": lr, "batch_steps": batch, "questions": questions})
        for seed in (0, 1, 2):
            identifier = f"{config_id}_s{seed}"
            output = root / "runs" / identifier
            command = [args.python, str(trainer), "--env_name", "InvertedPendulum-v5", "--output_dir", str(output),
                       "--total_steps", str(args.total_steps), "--batch_steps", str(batch), "--learning_rate", str(lr),
                       "--discount_gamma", "0.99", "--hidden_size", "64", "--layer_num", "2", "--seed", str(seed),
                       "--eval_seed", "10000", "--eval_every", "10000", "--eval_episodes", "10", "--device", "cpu",
                       "--torch_threads", "1", "--tensorboard", "--exp_name", identifier]
            if method == "rtg":
                command.append("--use_reward_to_go")
            elif method == "baseline":
                command.append("--use_baseline")
            jobs.append({"id": identifier, "command": command, "cwd": str(trainer.parent), "output_dir": str(output),
                         "estimator_version": "corrected-v2", "attempt": 1,
                         "recovery_status": "fresh_corrected_v2",
                         "expected_code_hash": "54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341"})
    (root / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "design.json").write_text(json.dumps({
        "environment": "InvertedPendulum-v5", "total_steps": args.total_steps, "training_seeds": [0, 1, 2],
        "eval_seed": 10000, "eval_episodes": 10, "eval_every": 10000, "discount_gamma": 0.99,
        "network": "64 x 2 ReLU", "gradient_clipping": "disabled", "advantage_normalization": False,
        "estimator_version": "corrected-v2",
        "baseline_definition": "mean discounted trajectory return from previous batches only; initial zero",
        "rtg_discount_convention": "sum_{k=t}^{T-1} gamma**k * reward[k]",
        "difference_from_task1": "common gamma changed from 0.9 to 0.99; new isolated fixed-seed stochastic evaluation",
        "configurations": grid, "unique_configurations": len(grid), "jobs": len(jobs),
        "shared_results": "lr=1e-3, batch=1000 is reused by Q1 and Q2 for each of three methods"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(jobs)} jobs, {len(grid)} distinct configurations: {root / 'jobs.json'}")


if __name__ == "__main__":
    main()
