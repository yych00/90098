import numpy as np
import torch

def collect_trajectories(env, agent, batch_size, render=False):
    batch = {'states':[], 'actions':[], 'log_probs':[], 'dists':[], 'rews_list':[], \
             'next_state':[], 'dones': [], 'images': []}
    for _ in range(batch_size):
        obs, info = env.reset() 
        if isinstance(obs, tuple):  # gym v0.26+ return (state, info)
            obs = obs[0]
        traj_rewards = []
        image_obs = []
        done = False
        
        while not done:
            with torch.no_grad():  
                # Use the sample_action interface shared by SACAgent and PGAgent.
                action = agent.sample_action(obs)
                action = np.clip(action, env.action_space.low, env.action_space.high)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
            else:
                next_obs, reward, terminated, info = step_result
                truncated = False
            # next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated

            if render:
                img = env.render()  # [H,W,C]
                if isinstance(img, list):
                    img = img[0]
                image_obs.append(img)

            batch['states'].append(obs)
            batch['actions'].append(action)
            # batch['log_probs'].append(logp)
            # batch['dists'].append(dist)
            batch['dones'].append(done)
            traj_rewards.append(reward)
            obs = next_obs

        batch['rews_list'].append(traj_rewards)
        if render:
            batch['images'].append(np.array(image_obs, dtype=np.uint8))

    batch['rewards'] = np.concatenate(batch['rews_list'])
    batch['states'] = np.stack(batch['states'])
    batch['actions'] = np.stack(batch['actions'])
    batch['next_state'] = np.array(next_obs)  # [state_dim], last state

    if render:
        max_T = max(frames.shape[0] for frames in batch['images'])
        padded = []
        for frames in batch['images']:
            arr = np.array(frames)
            T = arr.shape[0]
            if T < max_T:
                pad = np.repeat(arr[-1][None], max_T - T, axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            padded.append(arr)
        batch['images'] = np.stack(padded)  # [N,T,H,W,C]
    return batch

