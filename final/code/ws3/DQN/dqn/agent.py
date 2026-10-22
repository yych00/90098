import torch
import torch.nn as nn
import torch.optim as optim

import random
from collections import deque
import numpy as np

from dqn.network import QNet


class DQNAgent:
    def __init__(self, env, device='cpu',
                 hidden_size=64, layer_num=3, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.0,
                 epsilon_decay=0.995, memory_size=10000, batch_size=64,
                 lr=1e-3, target_update_freq=10, seed=0):
        self.env = env
        self.device = device
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.action_rng = random.Random(seed)
        self.replay_buffer = ReplayBuffer(int(memory_size), device=device, seed=seed + 1, state_dim=self.state_dim)
        
        self.total_update_steps = 0

        self.target_update_freq = target_update_freq

        self.q_net = QNet(self.state_dim, self.act_dim, hidden_size, layer_num).to(device)
        self.target_net = QNet(self.state_dim, self.act_dim, hidden_size, layer_num).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.loss_function = nn.MSELoss()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def sample_action(self, state_np):
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(self.device)        
        
        ### (TODO) get the action from the critic using an epsilon-greedy strategy
        if self.action_rng.random() < self.epsilon:
            action = self.action_rng.randrange(self.act_dim)
        else:
            with torch.no_grad():
                q_values = self.q_net(state)
                action = torch.argmax(q_values, dim=1).item()
        
        return action
    
    def greedy_action(self, state_np):
        with torch.no_grad():
            state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(state).argmax(dim=1).item())

    def update(self, batch):
        
        states, actions, rewards, next_states, dones = batch
        actions = actions.long().unsqueeze(1).to(self.device)  # [batch_size, 1]

        ### (TODO) Compute target values (r + gamma*Q(s',a';w-))
        ### 1. Compute Q(s', : ;w-) for all next actions from the target network
        ### 2. Select the best action a' and get Q(s',a';w-)
        ### 3. Compute target values  
        ### Note that, if the step is the termination (done), there is no next state, i.e. Q(s',a';w-) = 0
        with torch.no_grad():
            next_qa_values = self.target_net(next_states)
            
            next_action = torch.argmax(next_qa_values, dim=1, keepdim=True)

            next_q_values = next_qa_values.gather(1, next_action)

            target_values = rewards + self.gamma * next_q_values * (1.0 - dones)


        ### (TODO) Compute Q(s,a;w_i)
        ### 1. Compute Q(s, : ;w_i) for all actions from the Q network
        ### 2. Gather the value for action taken
        qa_values = self.q_net(states)
        q_values = qa_values.gather(1, actions)

        
        ### (TODO) Compute loss, with the given MSE self.loss_function
        loss = self.loss_function(q_values, target_values)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DQN loss")

        ### Update the critic network wrt computed loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.total_update_steps += 1

        ### (TODO) Update the target network every self.target_update_freq with self.update_target_net()
        if self.total_update_steps % self.target_update_freq == 0:
            self.update_target_net()

        return {        
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "gradient": grad_norm.item(),
            "epsilon": self.epsilon,
            # "td_error": (q_values - target_values).detach().cpu().numpy(),
        }

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def epsilon_step(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class ReplayBuffer:
    def __init__(self, size=int(1e6), device='cpu', seed=0, state_dim=None):
        self.capacity = int(size)
        if self.capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.device = device
        self.rng = random.Random(seed)
        self.position = self.length = 0
        self.states = self.next_states = None
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.dones = np.empty(self.capacity, dtype=np.float32)
        if state_dim is not None:
            self._allocate_states(state_dim)

    def _allocate_states(self, state_dim):
        self.states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)

    def add(self, s, a, r, s_, done):
        if self.states is None:
            self._allocate_states(np.asarray(s).shape[0])
        index = self.position
        self.states[index], self.actions[index], self.rewards[index] = s, a, r
        self.next_states[index], self.dones[index] = s_, done
        self.position = (index + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.sample(range(self.length), batch_size)
        s, a, r, s_, d = (array[indices] for array in
                          (self.states, self.actions, self.rewards, self.next_states, self.dones))
        
        # Stacking as a single numpy array before converting to tensor is much faster
        return (
            torch.as_tensor(np.stack(s), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.stack(a), dtype=torch.int64, device=self.device),
            torch.as_tensor(np.stack(r), dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.as_tensor(np.stack(s_), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.stack(d), dtype=torch.float32, device=self.device).unsqueeze(1)
        )

    def __len__(self):
        return self.length
