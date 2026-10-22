import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform


class ActorGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        layer_num: int = 2,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        validate_args: bool = True
    ):
        super().__init__()
        layers = []
        in_features = state_dim
        for _ in range(layer_num):
            layers += [nn.Linear(in_features, hidden_size), nn.ReLU()]
            in_features = hidden_size
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(in_features, action_dim)
        self.log_std_head = nn.Linear(in_features, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.validate_args = validate_args


    def forward(self, state: torch.Tensor) -> TransformedDistribution:

        if state.dim() == 1:
            state = state.unsqueeze(0)

        h = self.backbone(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # Repeated GPU parameter checks convert valid.all() to a Python boolean and synchronize.
        # clamp followed by exp ensures positive std; the optimized path disables repeated distribution checks.
        # The runner checks NaN/Inf every log window; the reference path retains the original checks.
        base = Independent(Normal(mean, std, validate_args=self.validate_args), 1,
                           validate_args=self.validate_args)

        transforms = [TanhTransform(cache_size=1)]

        return TransformedDistribution(base, transforms, validate_args=self.validate_args)

    
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


# SAC Q Net
class QNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, layer_num):
        super().__init__()
        layers = []
        input_dim = state_dim + act_dim
        for i in range(layer_num):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

