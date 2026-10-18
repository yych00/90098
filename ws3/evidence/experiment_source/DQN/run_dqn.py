"""Reproducible Task 2 runner, with an exact training environment-step budget.

Based on ws3/DQN; network, loss, optimiser and target-update semantics preserved.
Independent greedy evaluation uses the same reset seeds at every checkpoint.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path

import gymnasium as gym
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from dqn.agent import DQNAgent

VERSION = "ws3-task2-dqn-v1"
SOURCE_FILES = ("run_dqn.py", "dqn/agent.py", "dqn/network.py")


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def code_hash():
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


@contextlib.contextmanager
def preserve_rng():
    """Evaluation cannot alter training's Python, NumPy, Torch or CUDA RNG."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate(agent, env, seeds):
    returns = []
    was_training = agent.q_net.training
    with preserve_rng():
        agent.q_net.eval()
        try:
            for seed in seeds:
                observation, _ = env.reset(seed=seed)
                total, done = 0.0, False
                while not done:
                    observation, reward, terminated, truncated, _ = env.step(agent.greedy_action(observation))
                    total += float(reward)
                    done = terminated or truncated
                returns.append(total)
        finally:
            agent.q_net.train(was_training)
    return returns


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env_name", default="CartPole-v1")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_seed", type=int, default=10000)
    p.add_argument("--total_steps", type=int, default=200000)
    p.add_argument("--eval_every", type=int, default=10000)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch_threads", type=int, default=1)
    p.add_argument("--memory_size", type=int, default=100000)
    p.add_argument("--learning_start", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--target_update_freq", type=int, default=500)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--layer_num", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--discount_factor", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon_min", type=float, default=0.02)
    p.add_argument("--epsilon_decay", type=float, default=0.995)
    p.add_argument("--max_ep_len", type=int, default=None)
    p.add_argument("--exp_name", default=None)
    p.add_argument("--render", action="store_true")
    p.add_argument("--log_video", action="store_true")
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                   help="Write TensorBoard scalars when the installed runtime supports it.")
    p.add_argument("--episodes", type=int, default=None,
                   help="Legacy argument recorded only; total_steps controls the training budget.")
    return p


def train(args):
    if min(args.total_steps, args.eval_every, args.eval_episodes, args.batch_size,
           args.memory_size, args.target_update_freq, args.torch_threads) <= 0:
        raise ValueError("Step budgets, sizes, frequency and threads must be positive")
    if args.memory_size < max(args.batch_size, args.learning_start):
        raise ValueError("Replay capacity must support batch_size and learning_start")
    if not 0 <= args.epsilon_min <= args.epsilon <= 1 or not 0 < args.epsilon_decay <= 1:
        raise ValueError("Invalid epsilon schedule")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    config = {**vars(args), "output_dir": str(output), "code_version": VERSION,
              "code_hash": code_hash(), "python_version": platform.python_version(),
              "torch_version": torch.__version__, "gymnasium_version": gym.__version__,
              "numpy_version": np.__version__, "evaluation_policy": "greedy",
              "evaluation_seeds": seeds, "epsilon_decay_unit": "completed_training_episode",
              "target_update_unit": "gradient_update", "bootstrap_mask": "terminated_only",
              "original_source": "ws3/DQN", "replay_sampling": "uniform_without_replacement"}
    write_json(output / "config.json", config)
    started = time.perf_counter()
    total_steps = episodes = 0
    summary = {"status": "running", "actual_training_steps": 0, "code_hash": config["code_hash"]}
    write_json(output / "summary.json", summary)
    env = eval_env = writer = None
    try:
        if args.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir=str(output / "tensorboard"))
            except ImportError:
                print("TensorBoard is unavailable; CSV evidence remains complete.", flush=True)
        kwargs = {} if args.max_ep_len is None else {"max_episode_steps": args.max_ep_len}
        env = gym.make(args.env_name, render_mode="human" if args.render else None, **kwargs)
        eval_env = gym.make(args.env_name, **kwargs)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("DQN requires a discrete action space")
        env.action_space.seed(args.seed)
        agent = DQNAgent(env, device=args.device, hidden_size=args.hidden_size,
                         layer_num=args.layer_num, gamma=args.discount_factor,
                         epsilon=args.epsilon, epsilon_min=args.epsilon_min,
                         epsilon_decay=args.epsilon_decay, memory_size=args.memory_size,
                         batch_size=args.batch_size, lr=args.learning_rate,
                         target_update_freq=args.target_update_freq, seed=args.seed)
        last_update, eval_records = {}, []
        with (output / "evaluations.csv").open("w", newline="", encoding="utf-8") as ef, \
             (output / "evaluation_episodes.csv").open("w", newline="", encoding="utf-8") as epf, \
             (output / "metrics.csv").open("w", newline="", encoding="utf-8") as mf:
            ew = csv.DictWriter(ef, fieldnames=["env_steps", "mean_return", "std_return", "episodes"])
            epw = csv.DictWriter(epf, fieldnames=["env_steps", "episode", "seed", "return"])
            mw = csv.DictWriter(mf, fieldnames=["env_steps", "episode", "episode_return", "episode_steps",
                                                "completed", "terminated", "truncated", "epsilon",
                                                "updates", "critic_loss", "q_values", "target_values", "gradient"])
            for csv_writer in (ew, epw, mw):
                csv_writer.writeheader()

            def log_eval():
                values = evaluate(agent, eval_env, seeds)
                row = {"env_steps": total_steps, "mean_return": float(np.mean(values)),
                       "std_return": float(np.std(values)), "episodes": len(values)}
                ew.writerow(row)
                for i, (seed, value) in enumerate(zip(seeds, values)):
                    epw.writerow({"env_steps": total_steps, "episode": i, "seed": seed, "return": value})
                ef.flush()
                epf.flush()
                eval_records.append(row)
                if writer is not None:
                    writer.add_scalar("Evaluation/mean_return", row["mean_return"], total_steps)
                    writer.add_scalar("Evaluation/std_return", row["std_return"], total_steps)
                    writer.add_scalar("Training/epsilon", agent.epsilon, total_steps)
                print(f"step={total_steps} eval={row['mean_return']:.1f} eps={agent.epsilon:.4f}", flush=True)

            log_eval()
            state, _ = env.reset(seed=args.seed)
            episode_reward, episode_steps = 0.0, 0
            while total_steps < args.total_steps:
                action = agent.sample_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.replay_buffer.add(state, action, reward, next_state, float(terminated))
                state = next_state
                total_steps += 1
                episode_steps += 1
                episode_reward += float(reward)
                if len(agent.replay_buffer) >= max(args.learning_start, args.batch_size):
                    last_update = agent.update(agent.replay_buffer.sample(args.batch_size))
                completed = terminated or truncated
                if completed or total_steps == args.total_steps:
                    if completed:
                        episodes += 1
                        agent.epsilon_step()
                    mw.writerow({"env_steps": total_steps, "episode": episodes if completed else episodes + 1,
                                 "episode_return": episode_reward, "episode_steps": episode_steps,
                                 "completed": int(completed), "terminated": int(terminated),
                                 "truncated": int(truncated), "epsilon": agent.epsilon,
                                 "updates": agent.total_update_steps, **last_update})
                    mf.flush()
                    if writer is not None:
                        writer.add_scalar("Training/episode_return", episode_reward, total_steps)
                        for key, value in last_update.items():
                            writer.add_scalar("Training/" + key, value, total_steps)
                    if completed and total_steps < args.total_steps:
                        state, _ = env.reset()
                        episode_reward, episode_steps = 0.0, 0
                if total_steps % args.eval_every == 0 or total_steps == args.total_steps:
                    log_eval()
        torch.save({"q_net": agent.q_net.state_dict(), "target_net": agent.target_net.state_dict(),
                    "optimizer": agent.optimizer.state_dict(), "config": config,
                    "training_steps": total_steps, "episodes": episodes,
                    "updates": agent.total_update_steps, "epsilon": agent.epsilon,
                    "purpose": "evaluation checkpoint; replay is not included for training resume"}, output / "checkpoint.pt")
        if args.log_video:
            import imageio.v2 as imageio
            video_env = gym.make(args.env_name, render_mode="rgb_array", **kwargs)
            try:
                state, _ = video_env.reset(seed=seeds[0])
                with imageio.get_writer(str(output / "final_policy.mp4"), fps=50) as writer:
                    writer.append_data(video_env.render())
                    done = False
                    while not done:
                        state, _, terminated, truncated, _ = video_env.step(agent.greedy_action(state))
                        writer.append_data(video_env.render())
                        done = terminated or truncated
            finally:
                video_env.close()
        elapsed = time.perf_counter() - started
        summary.update(status="complete", actual_training_steps=total_steps,
                       completed_training_episodes=episodes, updates=agent.total_update_steps,
                       final_epsilon=agent.epsilon, wall_seconds=elapsed,
                       training_steps_per_wall_second=total_steps / elapsed,
                       final_mean_return=eval_records[-1]["mean_return"],
                       final_std_return=eval_records[-1]["std_return"],
                       best_mean_return=max(row["mean_return"] for row in eval_records),
                       first_steps_mean_return_ge_475=next((row["env_steps"] for row in eval_records if row["mean_return"] >= 475), None),
                       evaluation_checkpoints=len(eval_records), tensorboard_enabled=writer is not None)
        write_json(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException as error:
        summary.update(status="failed", actual_training_steps=total_steps,
                       wall_seconds=time.perf_counter() - started,
                       error=repr(error), traceback=traceback.format_exc())
        write_json(output / "summary.json", summary)
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    train(parser().parse_args())
