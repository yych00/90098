from collections import OrderedDict
import numpy as np
import copy
import torch.nn as nn
import torch
import gymnasium as gym
import cv2
from typing import Dict, Tuple, List

############################################
############################################


def sample_trajectory(
    env: gym.Env, policy: nn.Module, max_length: int, render: bool = False
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob, _ = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    true_terminations, truncations = [], []
    steps = 0
    while True:
        # render an image
        if render:
            img = env.render()
            if isinstance(img, list):
                img = img[0]
            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        # TODO use the most recent ob and the policy to decide what to do
        with torch.no_grad():
            if hasattr(policy, "sample_action"):
                ac = policy.sample_action(ob)
            else:
                device = next(policy.parameters()).device
                state = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                sample = policy(state).sample()
                ac = int(sample.item()) if isinstance(env.action_space, gym.spaces.Discrete) else sample.squeeze(0).cpu().numpy()

        # TODO: use that action to take a step in the environment
        env_action = ac if isinstance(env.action_space, gym.spaces.Discrete) else np.clip(
            ac, env.action_space.low, env.action_space.high
        )
        next_ob, rew, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated

        # TODO rollout can end due to done, or due to max_length
        steps += 1
        rollout_done: bool = done or (max_length is not None and steps >= max_length)

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)
        true_terminations.append(terminated)
        truncations.append(truncated or (rollout_done and not done))

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
        "terminated": np.array(true_terminations, dtype=np.float32),
        "truncated": np.array(truncations, dtype=np.float32),
    }


def sample_trajectories(
    env: gym.Env,
    policy: nn.Module,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += get_traj_length(traj)
    return trajs, timesteps_this_batch


def sample_n_trajectories(
    env: gym.Env, policy: nn.Module, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])
