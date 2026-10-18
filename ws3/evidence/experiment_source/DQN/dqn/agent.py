"""Workshop DQN: Adam, MSE, gradient clipping, uniform replay and hard targets."""
import random
import numpy as np
import torch
from torch import nn
from dqn.network import QNet


class ReplayBuffer:
    """Circular arrays; uniform sampling without replacement, as in the original."""
    def __init__(self, size, state_dim, device="cpu", seed=0):
        self.capacity, self.device = int(size), device
        self.states = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.terminated = np.empty((self.capacity, 1), dtype=np.float32)
        self.position = self.length = 0
        self.rng = random.Random(seed)

    def add(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.terminated[i] = next_state, terminated
        self.position = (i + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.sample(range(self.length), batch_size)
        return tuple(torch.as_tensor(array[indices], device=self.device)
                     for array in (self.states, self.actions, self.rewards,
                                   self.next_states, self.terminated))

    def __len__(self):
        return self.length


class DQNAgent:
    def __init__(self, env, device="cpu", hidden_size=64, layer_num=3,
                 gamma=0.99, epsilon=1.0, epsilon_min=0.02,
                 epsilon_decay=0.995, memory_size=100000, batch_size=64,
                 lr=1e-3, target_update_freq=500, seed=0):
        self.device, self.gamma = device, gamma
        self.state_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.epsilon, self.epsilon_min = epsilon, epsilon_min
        self.epsilon_decay, self.batch_size = epsilon_decay, batch_size
        self.action_rng = random.Random(seed)
        self.replay_buffer = ReplayBuffer(memory_size, self.state_dim, device, seed + 1)
        self.q_net = QNet(self.state_dim, self.act_dim, hidden_size, layer_num).to(device)
        self.target_net = QNet(self.state_dim, self.act_dim, hidden_size, layer_num).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()
        self.total_update_steps, self.target_update_freq = 0, target_update_freq

    def greedy_action(self, state):
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(tensor).argmax(dim=1).item())

    def sample_action(self, state):
        if self.action_rng.random() < self.epsilon:
            return self.action_rng.randrange(self.act_dim)
        return self.greedy_action(state)

    def update(self, batch):
        states, actions, rewards, next_states, terminated = batch
        with torch.no_grad():
            target_values = rewards + self.gamma * self.target_net(next_states).max(1, keepdim=True).values * (1.0 - terminated)
        q_values = self.q_net(states).gather(1, actions.long().unsqueeze(1))
        loss = self.loss_function(q_values, target_values)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DQN loss")
        self.optimizer.zero_grad()
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.total_update_steps += 1
        if self.total_update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return {"critic_loss": loss.item(), "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(), "gradient": gradient.item()}

    def epsilon_step(self):
        # t in epsilon(t) counts COMPLETED training episodes, not environment steps.
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
