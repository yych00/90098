import gymnasium as gym
import numpy as np
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import argparse
import time
import contextlib
import csv
import hashlib
import json
import random
import traceback
from pathlib import Path

from dqn.agent import DQNAgent
from utils.util import collect_trajectories
from utils.logger import Logger


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hash():
    digest = hashlib.sha256()
    for name in ("run_dqn.py", "dqn/agent.py", "dqn/network.py", "utils/util.py", "utils/logger.py"):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


@contextlib.contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
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
    was_training = agent.q_net.training
    with preserve_rng():
        agent.q_net.eval()
        try:
            batch = collect_trajectories(env, agent, len(seeds), render=False,
                                         greedy=True, seeds=seeds)
            return [float(np.sum(rewards)) for rewards in batch["rews_list"]]
        finally:
            agent.q_net.train(was_training)


def train(args):
    defaults = vars(build_parser().parse_args(["--env_name", args.env_name]))
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.device = str(args.device)
    if min(args.episodes, args.eval_every, args.eval_episodes, args.batch_size,
           args.memory_size, args.target_update_freq, args.torch_threads) <= 0:
        raise ValueError("Episode counts, frequencies, sizes and threads must be positive")
    if args.total_steps is not None and args.total_steps <= 0:
        raise ValueError("total_steps must be positive when specified")
    if args.learning_start < 0 or args.memory_size < max(args.learning_start, args.batch_size):
        raise ValueError("Replay capacity must support learning_start and batch_size")
    if not 0 <= args.epsilon_min <= args.epsilon <= 1 or not 0 < args.epsilon_decay <= 1:
        raise ValueError("Invalid epsilon schedule")
    if not 0 <= args.discount_factor <= 1 or args.learning_rate <= 0:
        raise ValueError("Invalid discount factor or learning rate")
    if args.max_ep_len is not None and args.max_ep_len <= 0:
        raise ValueError("max_ep_len must be positive")
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    log_dir = "logs/"
    log_dir = os.path.join(log_dir, args.env_name)
    mode = "_standard" if not args.exp_name else ""
    log_name = "DQN"
    if args.exp_name:
        log_name = log_name + "_" + args.exp_name
    log_name = log_name + mode + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = Path(args.output_dir or os.path.join(log_dir, log_name)).resolve()
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError("Preserving nonempty output directory: " + str(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "output_dir": str(log_dir), "code_hash": source_hash(),
              "source_version": "comment-preserving-submission-v3", "evaluation_policy": "greedy",
              "epsilon_decay_unit": "completed_training_episode", "bootstrap_mask": "terminated_only",
              "target_update_unit": "gradient_update", "gradient_norm_clip": 10.0,
              "replay_sampling": "uniform_without_replacement", "evaluation_rng_isolated": True,
              "evaluation_seeds": list(range(args.eval_seed, args.eval_seed + args.eval_episodes))}
    write_json(log_dir / "config.json", config)
    summary = {"status": "running", "actual_training_steps": 0, "code_hash": config["code_hash"]}
    write_json(log_dir / "summary.json", summary)
    started = time.perf_counter()
    logger = env = eval_env = None
    total_envsteps = completed_episodes = 0
    try:
        logger = Logger(log_dir=str(log_dir / "tensorboard")) if args.tensorboard else None
        render_mode = "human" if args.render else None
        kwargs = {} if args.max_ep_len is None else {"max_episode_steps": args.max_ep_len}
        env = gym.make(args.env_name, render_mode=render_mode, **kwargs)
        eval_env = gym.make(args.env_name, **kwargs)
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        assert discrete, "DQN only supports discrete action space"
        env.action_space.seed(args.seed)

        agent = DQNAgent(env,
                         batch_size=args.batch_size, device=args.device,
                         hidden_size=args.hidden_size, layer_num=args.layer_num,
                         gamma=args.discount_factor, epsilon=args.epsilon,
                         epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay,
                         lr=args.learning_rate, memory_size=args.memory_size,
                         target_update_freq=args.target_update_freq, seed=args.seed)
        returns = []
        update_info = None
        evaluations = []
        with (log_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as metric_file, \
             (log_dir / "evaluations.csv").open("w", newline="", encoding="utf-8") as evaluation_file, \
             (log_dir / "evaluation_episodes.csv").open("w", newline="", encoding="utf-8") as episode_file:
            metric_writer = csv.DictWriter(metric_file, fieldnames=["env_steps", "episode", "episode_return",
                "episode_steps", "completed", "terminated", "truncated", "epsilon", "updates",
                "critic_loss", "q_values", "target_values", "gradient"])
            evaluation_writer = csv.DictWriter(evaluation_file, fieldnames=["env_steps", "mean_return", "std_return", "episodes"])
            episode_writer = csv.DictWriter(episode_file, fieldnames=["env_steps", "episode", "seed", "return"])
            for writer in (metric_writer, evaluation_writer, episode_writer):
                writer.writeheader()

            def log_evaluation():
                values = evaluate(agent, eval_env, config["evaluation_seeds"])
                row = {"env_steps": total_envsteps, "mean_return": float(np.mean(values)),
                       "std_return": float(np.std(values)), "episodes": len(values)}
                evaluations.append(row)
                evaluation_writer.writerow(row)
                for index, (seed, value) in enumerate(zip(config["evaluation_seeds"], values)):
                    episode_writer.writerow({"env_steps": total_envsteps, "episode": index, "seed": seed, "return": value})
                evaluation_file.flush()
                episode_file.flush()
                if logger is not None:
                    logger.log_scalar(row["mean_return"], "Eval Returns", total_envsteps)
                print(f"step={total_envsteps} eval={row['mean_return']:.2f} epsilon={agent.epsilon:.4f}", flush=True)

            log_evaluation()
            episode_limit = args.episodes if args.total_steps is None else args.total_steps
            for episode in range(episode_limit):
                state, _ = env.reset(seed=args.seed if episode == 0 else None)
                episode_reward = 0.0
                episode_step = 0
                done = False
                while not done:

                    ### (TODO) Get the action from the agent with agent.sample_action(...)
                    action = agent.sample_action(state)

                    ### (TODO) Implement the action
                    step_result = env.step(action)

                    next_state, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)

                    ### Add the tuple to the replay buffer
                    agent.replay_buffer.add(state, action, reward, next_state, float(terminated))

                    state = next_state
                    episode_reward += float(reward)
                    episode_step += 1
                    total_envsteps += 1

                    if len(agent.replay_buffer) >= max(args.learning_start, args.batch_size):
                        ### (TODO) Sample args.batch_size from the agent's replay buffer
                        batch = agent.replay_buffer.sample(args.batch_size)

                        ### (TODO) Update the agent with the batch
                        update_info = agent.update(batch)

                        ### Log data
                        if logger is not None:
                            logger.log_scalar(update_info['critic_loss'], "Network Loss", total_envsteps)
                            logger.log_scalar(update_info['q_values'], "Q Values", total_envsteps)
                            logger.log_scalar(update_info['target_values'], "Target Q Values", total_envsteps)

                    exhausted = args.total_steps is not None and total_envsteps >= args.total_steps
                    if done or exhausted:
                        if done:
                            completed_episodes += 1
                            agent.epsilon_step()
                        row = {"env_steps": total_envsteps, "episode": completed_episodes if done else completed_episodes + 1,
                               "episode_return": episode_reward, "episode_steps": episode_step,
                               "completed": int(done), "terminated": int(terminated), "truncated": int(truncated),
                               "epsilon": agent.epsilon, "updates": agent.total_update_steps}
                        row.update({key: update_info[key] for key in ("critic_loss", "q_values", "target_values", "gradient")} if update_info else {})
                        metric_writer.writerow(row)
                        metric_file.flush()
                    if total_envsteps % args.eval_every == 0 or exhausted:
                        log_evaluation()
                    if exhausted:
                        break
                returns.append(episode_reward)
                if logger is not None:
                    logger.log_scalar(agent.epsilon, "_Epsilon", episode)
                if args.render:
                    time.sleep(0.05)
                if exhausted:
                    break
            if evaluations[-1]["env_steps"] != total_envsteps:
                log_evaluation()

        torch.save({"q_net": agent.q_net.state_dict(), "target_net": agent.target_net.state_dict(),
                    "optimizer": agent.optimizer.state_dict(), "config": config,
                    "training_steps": total_envsteps, "episodes": completed_episodes,
                    "updates": agent.total_update_steps, "epsilon": agent.epsilon,
                    "purpose": "evaluation checkpoint; replay is not included for exact resume"}, log_dir / "checkpoint.pt")

        if args.log_video:
            import imageio.v2 as imageio
            video_env = gym.make(args.env_name, render_mode="rgb_array", **kwargs)
            try:
                with preserve_rng():
                    final_batch = collect_trajectories(video_env, agent, 1, render=True,
                                                        greedy=True, seeds=[args.eval_seed])
                video = torch.tensor(final_batch['images']).permute(0,1,4,2,3)  # [N,T,H,W,C] → [N,T,C,H,W]
                print("Logging video")
                if logger is not None:
                    logger.log_video(video, "Final Video", step=total_envsteps)
                with imageio.get_writer(str(log_dir / "final_policy.mp4"), fps=video_env.metadata.get("render_fps", 50)) as video_writer:
                    for frame in final_batch["images"][0]:
                        video_writer.append_data(frame)
            finally:
                video_env.close()

        elapsed = time.perf_counter() - started
        summary.update(status="complete", actual_training_steps=total_envsteps,
                       completed_training_episodes=completed_episodes, updates=agent.total_update_steps,
                       final_epsilon=agent.epsilon, wall_seconds=elapsed,
                       training_steps_per_wall_second=total_envsteps / max(elapsed, 1e-9),
                       final_mean_return=evaluations[-1]["mean_return"], final_std_return=evaluations[-1]["std_return"],
                       best_mean_return=max(row["mean_return"] for row in evaluations),
                       evaluation_checkpoints=len(evaluations), tensorboard_enabled=logger is not None)
        return summary
    except BaseException as error:
        summary.update(status="failed", actual_training_steps=total_envsteps,
                       error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        write_json(log_dir / "summary.json", summary)
        if logger is not None:
            logger.close()
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Deep Q Learning")
    parser.add_argument("--env_name", type=str, required=True, help="gym environment")
    parser.add_argument("--episodes", type=int, default=1000, help="training episodes when total_steps is omitted")
    parser.add_argument("--memory_size", type=int, default=100000, help="replay buffer size")
    parser.add_argument("--learning_start", type=int, default=1000, help="learning start")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--log_video", action="store_true")
    parser.add_argument("--target_update_freq", type=int, default=500, help="gradient updates between hard target copies")
    parser.add_argument("--hidden_size", type=int, default=64, help="neural network hidden layer size")
    parser.add_argument("--layer_num", type=int, default=3, help="neural network layer number")
    parser.add_argument("--render", action="store_true", help="whether to render")
    parser.add_argument("--max_ep_len", type=int, help="max length for each episode")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0, help="beginning epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.02, help="minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="epsilon decay weight")
    parser.add_argument("--exp_name", default=None, help="add extra info to log")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--total_steps", type=int, default=None, help="exact environment-step budget; overrides episodes")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=False)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(args.device)
    # device = 'cpu'
    args.device = str(device)
    train(args)
