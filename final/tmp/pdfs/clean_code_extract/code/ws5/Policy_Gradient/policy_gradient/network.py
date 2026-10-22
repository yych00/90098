import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import numpy as np

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, layer_num=2):
        super().__init__()
        layers = []
        prev = state_dim
        for _ in range(layer_num):
            layers += [nn.Linear(prev, hidden_size), nn.ReLU()]
            prev = hidden_size
        layers += [nn.Linear(prev, action_dim)]
        self.fc_mean = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        
        mean = self.fc_mean(state)
        std = self.log_std.exp().expand_as(mean)

        distribution = Normal(mean, std)
        return distribution


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

    def forward(self, state):
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=next(self.parameters()).device
        )
        logits = self.fc_logits(state_tensor)
        distribution = Categorical(logits=logits)

        return distribution
    

# Value Network (for baseline/GAE)
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_size=64, layer_num=2):
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


