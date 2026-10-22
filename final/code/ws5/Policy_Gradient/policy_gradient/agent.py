import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from policy_gradient.network import GaussianPolicy, CategoricalPolicy, ValueNet
from gymnasium.spaces import Discrete

class PGAgent:
    def __init__(self, env, device='cpu',
                 hidden_size=64, layer_num=2,
                 use_reward_to_go=False, use_baseline=False,
                 gamma=0.99, 
                 gae_lambda=1.0, 
                 actor_lr=1e-3, critic_lr=1e-3, 
                 normalize_advantage=False, max_grad_norm=0.0):
        
        self.device = device
        self.discrete = isinstance(env.action_space, Discrete)
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.use_reward_to_go = use_reward_to_go 
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_baseline = use_baseline
        self.normalize_advantage = normalize_advantage
        self.max_grad_norm = max_grad_norm
        # Baseline uses earlier batches only, never the current trajectory's reward.
        self.baseline_return_sum = 0.0
        self.baseline_episode_count = 0

        assert not (use_reward_to_go and use_baseline), \
            "cannot apply reward-to-go and baseline at the same time"
        assert not (use_reward_to_go and (gae_lambda < 1.0)), \
            "cannot apply reward-to-go and gae at the same time"
        
        if self.discrete:
            self.actor = CategoricalPolicy(self.state_dim, self.act_dim, hidden_size, layer_num)
        else:
            self.actor = GaussianPolicy(self.state_dim, self.act_dim, hidden_size, layer_num)
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high

        # value network if needed
        if (gae_lambda < 1.0):
            self.critic = ValueNet(self.state_dim, hidden_size, layer_num)
            self.critic_loss = nn.MSELoss()
        else:
            self.critic = None
        
        self.actor.to(self.device)

        if self.critic is not None:
            self.critic.to(self.device)

        self.optimizer_policy = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_value = optim.Adam(self.critic.parameters(), lr=critic_lr) if self.critic else None


    @torch.no_grad()
    def sample_action(self, state_np):
        
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)        


        ### (TODO) Forward the actor network to get a distribution of action: actor.forward(...)
        action_distribution = self.actor(state)

        ### (TODO) Sample the distribution        
        action = action_distribution.sample()


        # Convert action to numpy
        if self.discrete:
            action = int(action.item())
        else:
            # Preserve the action dimension and the raw sample used for log_prob.
            action = action.squeeze(0).cpu().numpy()
        
        return action

    def generalized_advantage_estimation(self, state_np, next_state_np, rewards, terminateds, dones):
        ### (TODO - W11) - Generalized Advantage Estimation
        states = torch.tensor(np.asarray(state_np), dtype=torch.float32, device=self.device)
        next   = torch.tensor(np.asarray(next_state_np), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values      = self.critic(states).reshape(-1).cpu().numpy()
            next_values = self.critic(next).reshape(-1).cpu().numpy()

        rewards     = np.asarray(rewards, dtype=np.float32)
        terminateds = np.asarray(terminateds, dtype=np.float32)
        dones       = np.asarray(dones, dtype=np.float32)

        deltas = rewards + self.gamma * (1.0 - terminateds) * next_values - values
        advantages = np.zeros_like(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * running
            advantages[t] = running
        returns = advantages + values


        return returns, advantages



    def update(self, batch):
        rewards, states, actions, dones = \
            batch['rewards'], batch['states'], batch['actions'], batch['dones']

        ### (TODO) Split rewards into trajectories using done flags
        ### A trajectory ends when done==True 
        ### The achieved reward_trajectories is a list of numpy arrays, each array is a trajectory of rewards
        reward_trajectories = []
        start = 0
        for i, done in enumerate(dones):
            if done:
                reward_trajectories.append(np.asarray(rewards[start:i + 1], dtype=np.float32))
                start = i + 1
        if start != len(rewards) or not reward_trajectories:
            raise ValueError("A batch must contain complete, nonempty trajectories")
        
        ### (TODO) Calculate returns and advantages
        if self.critic is None:
            ### self._discounted_reward_to_go and self._discounted_return calculate returns for a single trajectory
            ### returns and advantages must be concatenated into vector
            
            if not self.use_reward_to_go:
                ### (TODO) If no variance reduction technique is used, calculate the returns
                returns = np.concatenate([
                    self._discounted_return(traj) for traj in reward_trajectories
                ])

            else:
                ### (TODO) If reward-to-go is used, calculate the returns
                returns = np.concatenate([
                    self._discounted_reward_to_go(traj) for traj in reward_trajectories
                ])

            if (not self.use_baseline) or self.use_reward_to_go:        
                ### (TODO) For the original Policy Gradient or reward-to-go, advantages = returns
                advantages = returns.copy()

            else:
                ### (TODO) If baseline is selected, calculate the baseline and the advantages                
                baseline = (self.baseline_return_sum / self.baseline_episode_count
                            if self.baseline_episode_count else 0.0)
                advantages = returns - baseline
                # Update history after forming advantages; single-trajectory batches
                # retain a reward signal instead of subtracting themselves away.
                self.baseline_return_sum += sum(float(self._discounted_return(traj)[0])
                                                for traj in reward_trajectories)
                self.baseline_episode_count += len(reward_trajectories)


        ### (TODO - W11) Genaralized Advantage Estimation
        else:
            returns, advantages = self.generalized_advantage_estimation(
                states, batch['next_states'], rewards, batch['terminateds'], dones
            )

        ### (TODO - W11) Normalize the advantages of the batch to have a mean of 0 and a standard deviation of 1
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            actions, dtype=torch.long if self.discrete else torch.float32,
            device=self.device
        )
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)


        ### (TODO) Compute log probabilities of selected actions, 
        ### Sum across the action dimension so each sample has a single log-prob value.
        distribution = self.actor(states)
        if self.discrete:
            log_pi = distribution.log_prob(actions.reshape(-1))
        else:
            log_pi = distribution.log_prob(actions.reshape(-1, self.act_dim)).sum(dim=-1)
        if log_pi.shape != advantages.shape:
            raise ValueError("log_pi and advantages must both have shape (batch_steps,)")

        ### (TODO) Compute policy gradient loss: -E[log_pi * advantage]
        actor_loss = -(log_pi * advantages.detach()).mean()

        
        self.optimizer_policy.zero_grad()
        actor_loss.backward()
        # Report the pre-clipping gradient norm; it is not gradient variance.
        gradient_norm = torch.linalg.vector_norm(torch.stack([
            parameter.grad.detach().norm() for parameter in self.actor.parameters()
            if parameter.grad is not None
        ]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite policy gradient")
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optimizer_policy.step()

        critic_loss = None
        if self.critic:
            prediction = self.critic(states).reshape(-1)
            critic_loss = self.critic_loss(prediction, returns)
            self.optimizer_value.zero_grad()
            critic_loss.backward()
            self.optimizer_value.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item() if critic_loss is not None else None,
            "mean_log_pi": log_pi.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std(unbiased=False).item(),
            "gradient_norm": gradient_norm.item()
        }
    
    def _discounted_return(self, traj_reward):
        ### (TODO) Compute the return for each trajectory, extend to the size of rewards
        # In PG, we use the same discounted return for the entire trajectory.
        rewards = np.asarray(traj_reward, dtype=np.float64)
        total = np.sum((self.gamma ** np.arange(len(rewards))) * rewards)
        returns = np.full(len(rewards), total, dtype=np.float32)
        return returns
    
    
    def _discounted_reward_to_go(self, traj_reward):
        ### (TODO) Compute the return for each step in the trajectory
        # With reward-to-go, the return is computed for each step using only the future timesteps
        # Absolute episode-origin discount matches the full-return objective:
        # sum_{k=t}^{T-1} gamma**k * r_k = gamma**t * relative_RTG_t.
        returns = np.empty(len(traj_reward), dtype=np.float32)
        running = 0.0
        for t in range(len(traj_reward) - 1, -1, -1):
            running = float(traj_reward[t]) + self.gamma * running
            returns[t] = running
        returns *= self.gamma ** np.arange(len(traj_reward))
        return returns
