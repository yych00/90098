"""Reload a completed checkpoint, reproduce its fixed-seed final evaluations.

Optionally render a fixed-seed PG video. This is evaluation, never training.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import hashlib
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["pg", "dqn"])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    historical_hashes = {"pg": "80dce14ad64559fa127592a64d682aa19a8dd67ea2aa797dd8ce4c08be7b2b6a",
                         "dqn": "a4350a035f2b565c8d0e0b6343fe3c254acf54c5b7cecb581d60ff97e0bcd8dd"}
    trainer_root = ROOT / "shared/historical_v1" if config["code_hash"] == historical_hashes[args.kind] else ROOT
    env = gym.make(config["env_name"])
    try:
        if args.kind == "pg":
            sys.path.insert(0, str(trainer_root / "ws4" / "Policy_Gradient"))
            runner = importlib.import_module("run_policy_gradient")
            assert runner.source_metadata()["code_hash"] == config["code_hash"], "Current source differs from training snapshot"
            agent = runner.PGAgent(env, device="cpu", hidden_size=config["hidden_size"], layer_num=config["layer_num"],
                gamma=config["discount_gamma"], use_reward_to_go=config["use_reward_to_go"],
                use_baseline=config["use_baseline"], normalize_advantage=config["normalize_advantage"],
                actor_lr=config["learning_rate"], critic_lr=config["baseline_learning_rate"],
                gae_lambda=config["gae_lambda"], max_grad_norm=config["max_grad_norm"])
            agent.actor.load_state_dict(checkpoint["actor_state_dict"])
            if agent.critic is not None:
                agent.critic.load_state_dict(checkpoint["critic_state_dict"])
            observed = runner.evaluate(env, agent, config["eval_seed"], config["eval_episodes"])
            returns = [row["return"] for row in observed]
        else:
            sys.path.insert(0, str(trainer_root / "ws3" / "DQN"))
            runner = importlib.import_module("run_dqn")
            assert runner.code_hash() == config["code_hash"], "Current source differs from training snapshot"
            agent = runner.DQNAgent(env, device="cpu", hidden_size=config["hidden_size"], layer_num=config["layer_num"],
                gamma=config["discount_factor"], epsilon=config["epsilon"], epsilon_min=config["epsilon_min"],
                epsilon_decay=config["epsilon_decay"], memory_size=config["memory_size"], batch_size=config["batch_size"],
                lr=config["learning_rate"], target_update_freq=config["target_update_freq"], seed=config["seed"])
            agent.q_net.load_state_dict(checkpoint["q_net"])
            seeds = list(range(config["eval_seed"], config["eval_seed"] + config["eval_episodes"]))
            returns = runner.evaluate(agent, env, seeds)
        actual = checkpoint.get("env_steps", checkpoint.get("training_steps"))
        with (args.run_dir / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as stream:
            expected = [float(row["return"]) for row in csv.DictReader(stream) if int(row["env_steps"]) == actual]
        assert len(expected) == len(returns) and np.allclose(expected, returns, atol=1e-9, rtol=0)
        report = {"status": "passed", "kind": args.kind, "run_dir": str(args.run_dir),
                  "checkpoint_sha256": hashlib.sha256((args.run_dir / "checkpoint.pt").read_bytes()).hexdigest(),
                  "training_steps": actual, "code_hash": config["code_hash"],
                  "expected_returns": expected, "reloaded_returns": returns,
                  "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}
        if args.video is not None:
            if args.kind != "pg":
                parser.error("video rendering currently supports PG only")
            args.video.parent.mkdir(parents=True, exist_ok=True)
            runner.record_video(config["env_name"], agent, args.video, config["eval_seed"])
            report.update(video=str(args.video), video_seed=config["eval_seed"], video_episode_return=float(returns[0]),
                          video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report))
    finally:
        env.close()


if __name__ == "__main__":
    main()
