import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np


# Actor network for continuous actions
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, layer_num=2,
                 state_dependent_std=False):
        super().__init__()
        layers = []
        prev = state_dim
        for _ in range(layer_num):
            layers += [nn.Linear(prev, hidden_size), nn.ReLU()]
            prev = hidden_size
        layers += [nn.Linear(prev, action_dim)]
        self.fc_mean = nn.Sequential(*layers)
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.log_std = nn.Linear(hidden_size, action_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mean = self.fc_mean(x)
        if self.state_dependent_std:
            std = self.log_std(x).clamp(-20, 20).exp()
        else:
            std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample_action(self, state):
        state = torch.FloatTensor(state)
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        # action = action.clamp(-1,1)
        logp = dist.log_prob(action).sum(axis=-1)
        return action, logp, dist


# Actor network for discrete actions
class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, layer_num=2):
        super().__init__()
        layers = []
        prev = state_dim
        for _ in range(layer_num):
            layers += [nn.Linear(prev, hidden_size), nn.ReLU()]
            prev = hidden_size
        layers += [nn.Linear(prev, action_dim)]
        self.fc_logits = nn.Sequential(*layers)
        # self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        logits = self.fc_logits(x)
        return logits

    def sample_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        # action = action.clamp(-1,1)
        logp = dist.log_prob(action).sum(axis=-1)

        return action, logp, dist
    

# Critic Network 
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_size=128, layer_num=2):
        super().__init__()
        layers = []
        prev = state_dim
        for _ in range(layer_num):
            layers += [nn.Linear(prev, hidden_size), nn.ReLU()]
            prev = hidden_size
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state).squeeze(-1)


# Q Net
class QNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, layer_num):
        super().__init__()
        layers = []
        input_dim = state_dim
        for i in range(layer_num):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, act_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.net(state)



