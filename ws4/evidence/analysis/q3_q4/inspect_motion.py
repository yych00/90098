"""Measure the fixed video episode without changing the trained controller."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import gymnasium as gym
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "Policy_Gradient"))
import run_policy_gradient as runner


def main():
    torch.set_num_threads(1)
    checkpoint_path = ROOT / "runs/halfcheetah_final_s0/checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("code_hash") != runner.source_metadata()["code_hash"] or config.get("estimator_version") != "corrected-v2":
        raise ValueError("Motion inspection requires this workspace's corrected-v2 checkpoint")
    env = gym.make(config["env_name"])
    try:
        agent = runner.PGAgent(env, device="cpu", hidden_size=config["hidden_size"], layer_num=config["layer_num"],
                    gamma=config["discount_gamma"], use_reward_to_go=config["use_reward_to_go"],
                    use_baseline=config["use_baseline"], normalize_advantage=config["normalize_advantage"],
                    actor_lr=config["learning_rate"], critic_lr=config["baseline_learning_rate"],
                    gae_lambda=config["gae_lambda"], max_grad_norm=config["max_grad_norm"])
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        model = env.unwrapped.model
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        positions = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index): int(model.jnt_qposadr[index]) for index in range(model.njnt)}
        total, rows = 0.0, []
        with runner.isolated_rng(config["eval_seed"], "cpu"):
            state, _ = env.reset(seed=config["eval_seed"])
            done = False
            index = 0
            while True:
                pitch = float(env.unwrapped.data.qpos[positions["rooty"]])
                rows.append({"step": index, "time_seconds": index * env.unwrapped.dt,
                             "root_x_m": float(env.unwrapped.data.qpos[positions["rootx"]]),
                             "torso_z_m": float(env.unwrapped.data.xpos[torso, 2]),
                             "root_pitch_rad": pitch, "wrapped_pitch_rad": (pitch + np.pi) % (2 * np.pi) - np.pi,
                             "cumulative_return": total})
                if done:
                    break
                action = agent.sample_action(state)
                state, reward, terminated, truncated, _ = env.step(np.clip(action, env.action_space.low, env.action_space.high))
                total += float(reward)
                index += 1
                done = terminated or truncated
        # Validate against the freshly reproduced run itself; a source-only
        # package does not contain the original checkpoint-reload QA record.
        with (ROOT / "runs/halfcheetah_final_s0/evaluation_episodes.csv").open(newline="", encoding="utf-8") as stream:
            expected_rows = [row for row in csv.DictReader(stream)
                             if int(row["env_steps"]) == checkpoint["env_steps"]
                             and int(row["seed"]) == config["eval_seed"]]
        assert len(expected_rows) == 1, "Missing final fixed-seed episode in run evidence"
        expected = float(expected_rows[0]["return"])
        assert np.isclose(total, expected, rtol=0, atol=1e-9)
        with (ROOT / "representative_seed0_motion.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        pitch = np.array([row["wrapped_pitch_rad"] for row in rows])
        summary = {"training_seed": 0, "evaluation_seed": config["eval_seed"], "steps": index,
                   "code_hash": config["code_hash"], "estimator_version": config["estimator_version"],
                   "checkpoint_env_steps": checkpoint["env_steps"],
                   "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                   "simulated_seconds": rows[-1]["time_seconds"], "episode_return": total,
                   "root_x_start_m": rows[0]["root_x_m"], "root_x_end_m": rows[-1]["root_x_m"],
                   "forward_displacement_m": rows[-1]["root_x_m"] - rows[0]["root_x_m"],
                   "max_abs_wrapped_pitch_rad": float(np.abs(pitch).max()),
                   "fraction_abs_pitch_over_45_degrees": float((np.abs(pitch) > np.pi / 4).mean()),
                   "torso_height_min_m": min(row["torso_z_m"] for row in rows),
                   "torso_height_max_m": max(row["torso_z_m"] for row in rows),
                   "pitch_threshold_note": "45 degrees is a descriptive posture statistic, not an environment termination/solved rule"}
        (ROOT / "motion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))
    finally:
        env.close()


if __name__ == "__main__":
    main()
