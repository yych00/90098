import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import time
import csv
import json
import random
from pathlib import Path

from policy_gradient.agent import PGAgent
from utils.util import collect_trajectories, isolated_rng
from utils.logger import Logger


def train(args):
    defaults = dict(output_dir=None, seed=0, eval_seed=10000, eval_episodes=10,
                    torch_threads=1, max_grad_norm=0.0)
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log_dir = "logs/"
    log_dir = os.path.join(log_dir, args.env_name)    
    log_name = "PG" 
    if args.exp_name:
        log_name = log_name + "_" + args.exp_name

    log_name = log_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    log_dir = os.path.join(log_dir, log_name)
    if args.output_dir is not None:
        log_dir = args.output_dir
    log_dir = Path(log_dir)
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError("Preserving an existing run directory: " + str(log_dir))
    if not (os.path.exists(log_dir)):
        os.makedirs(log_dir)
    logger = Logger(log_dir=log_dir)
    config = {**vars(args), "implementation": "teacher-template-completed-v3",
              "baseline_definition": "mean discounted trajectory returns from previous batches only; initial zero",
              "rtg_discount_convention": "sum_{k=t}^{T-1} gamma**k * reward[k]",
              "checkpoint_purpose": "policy inspection; not exact training resume"}
    _write_json(log_dir / "config.json", config)
    
    render_mode = None
    if args.render:
        render_mode="human"
    
    env = gym.make(args.env_name, render_mode=render_mode)
    eval_env = gym.make(args.env_name)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    try:
        agent = PGAgent(env, device=args.device,
                        hidden_size=args.hidden_size, layer_num=args.layer_num,
                        gamma=args.discount_gamma, 
                        use_reward_to_go=args.use_reward_to_go,
                        use_baseline=args.use_baseline,
                        normalize_advantage=args.normalize_advantage,
                        actor_lr=args.learning_rate,
                        critic_lr=args.baseline_learning_rate, 
                        gae_lambda=args.gae_lambda, max_grad_norm=args.max_grad_norm)

        total_envsteps = 0
        update_idx = 0
        next_eval = args.eval_every
        next_video = args.video_every
        initial = collect_trajectories(eval_env, agent, args.eval_episodes,
                                       seeds=list(range(args.eval_seed, args.eval_seed + args.eval_episodes)))
        initial_returns = [float(np.sum(traj)) for traj in initial["rews_list"]]
        logger.log_scalar(np.mean(initial_returns), "Eval Returns", 0)
        logger.log_scalar(np.std(initial_returns), "Eval Returns Std", 0)
        _save_evaluation(log_dir, 0, initial_returns, [len(traj) for traj in initial["rews_list"]], args.eval_seed)

        while total_envsteps < args.total_steps:

            batch = {'states':[], 'next_states':[], 'actions':[], 'rewards':[], 'terminateds':[], 'dones':[]}
            ep_returns = []
            steps_this_batch = 0
            clipped_actions = 0

            while steps_this_batch < args.batch_steps:
            
                states, next_states, actions, rewards, terminateds, dones = [],[],[],[],[],[]

                state, _ = env.reset()
                episode_reward = 0
                episode_step = 0
                done = False

                while not done:

                    ### (TODO) Get the action from the policy with the most recent state
                    action = agent.sample_action(state)

                    ### (TODO) use that action to take a step in the environment
                    env_action = action if agent.discrete else np.clip(action, env.action_space.low, env.action_space.high)
                    if not agent.discrete:
                        clipped_actions += int(np.any(action != env_action))
                    step_result = env.step(env_action)

                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated

                    # Record result of taking that action
                    states.append(state)
                    next_states.append(next_state)
                    actions.append(action)                        
                    rewards.append(reward)
                    terminateds.append(terminated)
                    dones.append(done)

                    state = next_state
                    episode_reward += reward
                    episode_step += 1
                    total_envsteps += 1
                
                
                batch["states"].append(states)
                batch["next_states"].append(next_states)
                batch["actions"].append(actions)
                batch["rewards"].append(rewards)
                batch["terminateds"].append(terminateds)
                batch["dones"].append(dones)
            
                ep_returns.append(sum(rewards))
                steps_this_batch += len(rewards)

                if args.render:
                    time.sleep(0.05)
            

            for k in batch:
                batch[k] = np.concatenate(batch[k])

        
            ### (TODO) update the agent using the sampled batch
            update_info = agent.update(batch)


            update_idx += 1
            _append_csv(log_dir / "metrics.csv", {"update": update_idx, "env_steps": total_envsteps,
                        "batch_steps": steps_this_batch, "batch_episodes": len(ep_returns),
                        **update_info, "clipped_action_fraction": clipped_actions / steps_this_batch})

            ### Logging
            logstep = total_envsteps
        
            logger.log_scalar(update_info["actor_loss"], "Training/Policy Loss", logstep)
            if update_info["critic_loss"] is not None:
                logger.log_scalar(update_info["critic_loss"], "Training/Critic Loss", logstep)

            logger.log_scalar(update_info["mean_log_pi"], "Training/Mean Log Pi", logstep)
            logger.log_scalar(update_info["advantage_mean"], "Training/Advantages Mean", logstep)
            logger.log_scalar(update_info["advantage_std"], "Training/Advantages Std", logstep)

            train_return = np.mean(ep_returns)
            logger.log_scalar(train_return, "Training/Train Returns", logstep)
            logger.log_scalar(len(ep_returns), "Training/Episodes per Batch", logstep)

            if total_envsteps >= next_eval or total_envsteps >= args.total_steps:
                while next_eval <= total_envsteps:
                    next_eval += args.eval_every                   
                eval_batch = collect_trajectories(eval_env, agent, batch_size=args.eval_episodes, render=False,
                                                   seeds=list(range(args.eval_seed, args.eval_seed + args.eval_episodes)))
                # eval_return = np.sum(eval_batch['rewards'])
                # logger.log_scalar(eval_return, "Eval Returns", logstep)
            
                eval_returns = [np.sum(traj) for traj in eval_batch['rews_list']]
                _save_evaluation(log_dir, logstep, eval_returns,
                                 [len(traj) for traj in eval_batch["rews_list"]], args.eval_seed)
            
                logger.log_scalar(np.mean(eval_returns), "Eval Returns", logstep)
                logger.log_scalar(np.std(eval_returns), "Eval Returns Std", logstep)
                print(f"Update {update_idx} | steps {total_envsteps}/{args.total_steps} | "
                    f"eval {np.mean(eval_returns):.1f} | actor loss {update_info['actor_loss']:.4f}")
        
            if args.log_video and total_envsteps >= next_video:
                while next_video <= total_envsteps:
                    next_video += args.video_every           
                render_env = gym.make(args.env_name,render_mode="rgb_array")
                final_batch = collect_trajectories(render_env, agent, 1, render=True, seeds=[args.eval_seed])
                video = torch.tensor(final_batch['images']).permute(0,1,4,2,3)  # [N,T,H,W,C] → [N,T,C,H,W]
                print("Logging video")
                logger.log_video(video, "Video", step=total_envsteps, fps=render_env.metadata.get("render_fps", 30))
                render_env.close()

        torch.save({"config": config, "env_steps": total_envsteps, "updates": update_idx,
                    "actor_state_dict": agent.actor.state_dict(),
                    "critic_state_dict": agent.critic.state_dict() if agent.critic is not None else None,
                    "baseline_return_sum": agent.baseline_return_sum,
                    "baseline_episode_count": agent.baseline_episode_count}, log_dir / "checkpoint.pt")
        _write_json(log_dir / "summary.json", {"status": "completed", "requested_training_steps": args.total_steps,
                    "actual_training_steps": total_envsteps, "updates": update_idx,
                    "training_seeds": args.seed, "evaluation_episodes_per_checkpoint": args.eval_episodes})
    finally:
        env.close()
        eval_env.close()
        logger.close()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def _append_csv(path, row):
    path = Path(path)
    first = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if first:
            writer.writeheader()
        writer.writerow(row)


def _save_evaluation(log_dir, step, returns, lengths, seed):
    _append_csv(log_dir / "evaluations.csv", {"env_steps": step, "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)), "episodes": len(returns)})
    for episode, (value, length) in enumerate(zip(returns, lengths)):
        _append_csv(log_dir / "evaluation_episodes.csv", {"env_steps": step, "episode": episode,
                    "seed": seed + episode, "return": float(value), "length": length})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Policy Gradient")
    parser.add_argument("--env_name", type=str, required=True, help="gym environment") 
    parser.add_argument("--total_steps", type=int, default=500_000, help="total env steps to train for")
    parser.add_argument("--batch_steps", type=int, default=1000, help="min env steps per update")
    parser.add_argument("--eval_every", type=int, default=10_000, help="env steps between evals")
    parser.add_argument("--video_every", type=int, default=100_000, help="env steps between videos")

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--discount_gamma", type=float, default=0.9)
    
    parser.add_argument("--layer_num", type=int, default=2, help="neural network layer number")
    parser.add_argument("--hidden_size", type=int, default=64, help="neural network hidden layer size")

    parser.add_argument("--use_reward_to_go", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    
    parser.add_argument("--normalize_advantage", action="store_true")
    parser.add_argument("--gae_lambda", type=float, default=1.0) 
    parser.add_argument("--baseline_learning_rate", type=float, default=1e-3)
    parser.add_argument("--render", action="store_true", help="whether to render")      
    parser.add_argument("--log_video", action="store_true")
    parser.add_argument("--exp_name", default=None, help="add extra info (parameters) to log")
    parser.add_argument("--output_dir", default=None, help="empty output directory; default: relative logs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    args = parser.parse_args()
    if min(args.total_steps, args.batch_steps, args.eval_every, args.video_every, args.eval_episodes, args.torch_threads) <= 0:
        parser.error("step counts, evaluation episodes and thread count must be positive")
    if not 0 <= args.discount_gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("gamma and GAE lambda must be in [0, 1]")
    if args.learning_rate <= 0 or args.baseline_learning_rate <= 0 or args.max_grad_norm < 0:
        parser.error("learning rates must be positive and gradient clipping nonnegative")
    if args.use_reward_to_go and (args.use_baseline or args.gae_lambda < 1):
        parser.error("RTG, prior-batch baseline and GAE are separate estimators")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print('########################')
    print(f"Device: {device}")
    args.device = device

    train(args)
