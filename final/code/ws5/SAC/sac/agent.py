import torch
import torch.nn.functional as F
# import torch.nn as nn
import torch.optim as optim
from sac.network import ActorGaussianPolicy, QNet
import random
from collections import deque
import numpy as np

class SACAgent:
    def __init__(self, env,
                 device='cpu', hidden_size=64, layer_num=2,
                 actor_lr=3e-4, critic_lr=1e-4,
                 gamma=0.99, tau=0.005, temperature=0.2, 
                 memory_size=1e6, optimized=True, fused_optimizer=False, capturable=False):
        self.env = env
        self.device = torch.device(device)
        self.optimized = optimized
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.gamma = gamma

        self.tau = tau
        self.init_temperature = temperature
        
        # Action-space bounds
        self.act_low  = torch.as_tensor(env.action_space.low,  dtype=torch.float32, device=self.device)
        self.act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)
        self.scale = (self.act_high - self.act_low) / 2.0
        self.loc   = (self.act_high + self.act_low) / 2.0

        # ----- Networks -----
        self.actor = ActorGaussianPolicy(
            self.state_dim, self.act_dim,
            hidden_size=hidden_size, layer_num=layer_num, validate_args=not optimized
        ).to(self.device)
        
        self.critic1 = QNet(self.state_dim,self.act_dim, hidden_size, layer_num).to(self.device)
        self.critic2 = QNet(self.state_dim,self.act_dim, hidden_size, layer_num).to(self.device)
        self.target_critic1 = QNet(self.state_dim,self.act_dim, hidden_size, layer_num).to(self.device)
        self.target_critic2 = QNet(self.state_dim,self.act_dim, hidden_size, layer_num).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        # self.target_critic1.eval()
        # self.target_critic2.eval()

        # Store the loss callable; calling F.mse_loss() without inputs would fail.
        self.critic_loss_function = F.mse_loss

        # ----- Optimizers -----
        # foreach/fused changes the optimizer implementation, not Adam settings or updates per step.
        # Keep the reference path to compare numerics with identical data, randomness and initial weights.
        options = {'fused': True} if fused_optimizer else {'foreach': optimized}
        options['capturable'] = capturable
        if fused_optimizer and self.device.type != 'cuda':
            raise ValueError('fused_optimizer requires CUDA')
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, **options)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr, **options)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr, **options)

        # ----- Temperature auto-tuning -----
        self.log_temperature = torch.tensor(float(np.log(temperature)), dtype=torch.float32,
                                            requires_grad=True, device=self.device)
        self.temperature_optimizer = optim.Adam([self.log_temperature], lr=3e-4,
                                                capturable=capturable)
        self.target_entropy = -float(self.act_dim)

        self.replay_buffer = ReplayBuffer(int(memory_size), self.state_dim, self.act_dim,
                                          self.device, packed=optimized)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    # ----- scaling helpers -----
    def _scale_to_env(self, action):     # action in [-1,1]
        return action * self.scale + self.loc
    
    
    @torch.no_grad()
    def sample_action(self, state):
        
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)

        ### (TODO) Forward the actor network to get a distribution of action
        ### The actor network returns a TransformedDistribution over [-1,1]
        # Environment action sampling does not need an autograd graph.
        action_distribution = self.actor(state)

        ### (TODO) Sample the distribution to get an action
        # Normal.sample uses torch.normal, whose scale checks prevent CUDA Graph capture.
        # Under no_grad, rsample draws the same Gaussian distribution using a capturable normal transform.
        action = action_distribution.rsample() if self.optimized else action_distribution.sample()

        ### Scale the action to match the env bound
        action_env = self._scale_to_env(action)
        
        action_env = action_env.detach().squeeze(0).cpu().numpy()
        
        return action_env


    def update(self, batch, return_tensors=False):
        
        states, actions_env, rewards, next_states, dones = \
            batch['states'], batch['actions'], batch['rewards'], batch['next_states'], batch['dones']
                
        # -----------------------
        # Critic update 
        # -----------------------

        ### (TODO) Compute target value for the critic
        #  a) get next-action distribution at next_states
        #  b) rsample next actions (on [-1,1]) and compute their log-probs
        #       NOTE: rsample() enables gradients to flow through the stochastic sampling step.
        #       Although we're in no_grad here (for critic target), the same logic will be used
        #       in the actor update below, where rsample() is critical for the policy gradient
        #       to be computed via the reparameterization trick.
        #  c) scale next actions to env range for Q evaluation
        #  d) evaluate target Q using the two target critics and take the minimum
        #  e) assemble the soft target: 
        #       NOTE: use rewards, dones, gamma, next_q, temperature, and next_log_pi.
        with torch.no_grad():            
            next_action_distribution = self.actor(next_states)
            next_actions = next_action_distribution.rsample()
            # Independent already sums action dimensions; add only a column dimension to avoid [B]/[B,1] broadcasting.
            next_log_pi = next_action_distribution.log_prob(next_actions).unsqueeze(-1)
            
            assert next_log_pi.ndim == 2 and next_log_pi.shape[1] == 1, \
            f"log_pi shape should be [batch_size,1], got {next_log_pi.shape}"

            next_actions_env = self._scale_to_env(next_actions)
            
            next_q1 = self.target_critic1(next_states, next_actions_env)
            next_q2 = self.target_critic2(next_states, next_actions_env)
            next_q = torch.minimum(next_q1, next_q2)

            target_q = rewards + self.gamma * (1.0 - dones) * (next_q - self.temperature * next_log_pi)

        # (TODO) Evaluate current Q using the two critics with actions_env from replay
        q1 = self.critic1(states, actions_env)
        q2 = self.critic2(states, actions_env)

        # (TODO) Compute losses (MSE) for the critic networks (self.critic_loss_function)
        critic1_loss = self.critic_loss_function(q1, target_q)
        critic2_loss = self.critic_loss_function(q2, target_q)


        # Critic network update
        
        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        # The Q networks share no parameters, so one backward on their summed losses is equivalent.
        if self.optimized:
            (critic1_loss + critic2_loss).backward()
        else:
            critic1_loss.backward()
            critic2_loss.backward()
        # critic1_grad = torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=10.0)
        self.critic1_optimizer.step()

        # critic2_grad = torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=10.0)
        self.critic2_optimizer.step()


        # -----------------------
        # Actor update 
        # -----------------------
        
        ### (TODO) policy improvement step
        #  a) get distribution at 'states'
        #  b) rsample new actions and log-probs (on [-1,1])
        #       NOTE: Here, using rsample() is essential — it allows gradients to propagate
        #       through the action sampling.
        #  c) scale actions to env range for Q evaluation
        #  d) get the minimum between Q value estimates with the new actions from the two networks
        #  e) compute actor loss: E[ temp * log_pi - Q ]

        action_distributions = self.actor(states)
        new_actions = action_distributions.rsample()
        log_pi = action_distributions.log_prob(new_actions).unsqueeze(-1)

        new_actions_env = self._scale_to_env(new_actions)
        
        # The actor step needs dQ/da but not Q-parameter gradients; do not wrap Q in no_grad.
        critics = list(self.critic1.parameters()) + list(self.critic2.parameters())
        if self.optimized:
            for parameter in critics:
                parameter.requires_grad_(False)
        q1_new = self.critic1(states, new_actions_env)
        q2_new = self.critic2(states, new_actions_env)
        q_value = torch.minimum(q1_new, q2_new)

        assert log_pi.ndim == 2 and log_pi.shape[1] == 1, \
        f"log_pi shape should be [batch_size,1], got {log_pi.shape}"
        assert q_value.ndim == 2 and q_value.shape[1] == 1, \
        f"q_value shape should be [batch_size,1], got {q_value.shape}"
        
        actor_loss = (self.temperature.detach() * log_pi - q_value).mean()

        # Actor network update
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        # actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()
        if self.optimized:
            for parameter in critics:
                parameter.requires_grad_(True)

        # -----------------------
        # Temperature update
        # -----------------------
        temperature_loss = -(self.log_temperature * (log_pi + self.target_entropy).detach()).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()

        # -----------------------
        # Target critic soft update
        # -----------------------
        with torch.no_grad():
            if self.optimized:
                # Multi-tensor operations combine target-update Python loops and small CUDA kernels.
                # Keep tau and update frequency unchanged; compare roundoff against the reference path.
                targets = list(self.target_critic1.parameters()) + list(self.target_critic2.parameters())
                sources = list(self.critic1.parameters()) + list(self.critic2.parameters())
                torch._foreach_mul_(targets, 1-self.tau)
                torch._foreach_add_(targets, sources, alpha=self.tau)
            else:
                for target, source in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
                for target, source in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)

        metrics = {
            'critic_loss1': critic1_loss.detach(), 'critic_loss2': critic2_loss.detach(),
            'actor_loss': actor_loss.detach(), 'temperature': self.temperature.detach(),
            'entropy': -log_pi.detach().mean(), 'target_value': target_q.mean().detach(),
            'q_value': q_value.mean().detach(),
        }
        # Accumulate log windows on the device and copy once, avoiding seven .item() calls per step.
        return metrics if return_tensors else {key: value.item() for key, value in metrics.items()}
    
    

class ReplayBuffer:
    def __init__(self, size=int(1e6), state_dim=17, act_dim=6, device='cpu', packed=False):
        if int(size) != size or size <= 0:
            raise ValueError('Replay capacity must be a positive integer')
        size = int(size)
        self.size, self.ptr, self.n, self.device = size, 0, 0, device
        self.packed = packed
        self.widths = [state_dim, act_dim, 1, state_dim, 1]
        # Store each transition contiguously for one indexing operation and device copy instead of five.
        # Column views preserve the original fields, sampling distribution, capacity and storage precision.
        if packed:
            self.storage = np.zeros((size, sum(self.widths)), np.float32)
            cuts = np.cumsum(self.widths)[:-1]
            self.s, self.a, self.r, self.s2, self.d = np.split(self.storage, cuts, axis=1)
        else:
            self.s  = np.zeros((size, state_dim), np.float32)
            self.a  = np.zeros((size, act_dim), np.float32)
            self.r  = np.zeros((size, 1), np.float32)
            self.s2 = np.zeros((size, state_dim), np.float32)
            self.d  = np.zeros((size, 1), np.float32)

    def add(self, s, a, r, s_, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s_, d
        self.ptr = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, bs):
        if self.n == 0 or bs <= 0:
            raise ValueError('Cannot sample an empty replay or nonpositive batch')
        idx = np.random.randint(0, self.n, bs)
        if self.packed:
            rows = torch.as_tensor(self.storage[idx], device=self.device)
            return dict(zip(['states','actions','rewards','next_states','dones'],
                            rows.split(self.widths, dim=1)))
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return {"states": t(self.s), "actions": t(self.a), "rewards": t(self.r),
                "next_states": t(self.s2), "dones": t(self.d)}

    def __len__(self): return self.n
