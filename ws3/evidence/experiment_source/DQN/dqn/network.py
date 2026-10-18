"""The workshop's 3 x 64 ReLU Q-network (architecture unchanged)."""
from torch import nn


class QNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size=64, layer_num=3):
        super().__init__()
        layers = []
        for layer in range(layer_num):
            layers.extend([nn.Linear(state_dim if layer == 0 else hidden_size,
                                     hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)
