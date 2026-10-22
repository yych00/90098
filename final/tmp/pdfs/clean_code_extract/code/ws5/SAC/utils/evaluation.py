"""Fixed-seed evaluation with reusable environments, batched inference and independent RNG streams."""
import time
import copy
from types import SimpleNamespace
import numpy as np
import torch


class EvaluationPool:
    def __init__(self, gym, env_name, episodes=10, seed=10000):
        self.envs = [gym.make(env_name) for _ in range(episodes)]
        self.seed = seed

    @torch.no_grad()
    def evaluate(self, agent, batch_size=10, device=None):
        start = time.perf_counter()
        if device is not None and torch.device(device)!=agent.device:
            # Small evaluation batches can use the CPU to avoid per-step GPU dispatch and synchronization.
            # deepcopy avoids random initialization and training RNG consumption; copying counts as evaluation time.
            scale=agent.scale.to(device)
            loc=agent.loc.to(device)
            agent=SimpleNamespace(actor=copy.deepcopy(agent.actor).to(device),
                                  device=torch.device(device),act_dim=agent.act_dim,
                                  _scale_to_env=lambda a:a*scale+loc)
        returns = np.zeros(len(self.envs), dtype=np.float64)
        lengths = np.zeros(len(self.envs), dtype=np.int64)
        for first in range(0, len(self.envs), batch_size):
            ids = list(range(first, min(first + batch_size, len(self.envs))))
            observations = {i: self.envs[i].reset(seed=self.seed+i)[0] for i in ids}
            # Each episode has an independent noise stream, unaffected by the evaluation batch size.
            # Evaluation neither consumes training randomness nor requires saving and restoring training RNG.
            noise_rng = {i: np.random.default_rng(self.seed+i) for i in ids}
            active = ids[:]
            while active:
                states = torch.as_tensor(np.stack([observations[i] for i in active]),
                                         dtype=torch.float32, device=agent.device)
                distribution = agent.actor(states)
                normal = distribution.base_dist.base_dist
                noise = np.stack([noise_rng[i].standard_normal(agent.act_dim) for i in active])
                noise = torch.as_tensor(noise, dtype=torch.float32, device=agent.device)
                actions = agent._scale_to_env(torch.tanh(normal.loc + normal.scale * noise)).cpu().numpy()
                next_active = []
                for row, i in enumerate(active):
                    obs, reward, terminated, truncated, _ = self.envs[i].step(actions[row])
                    observations[i] = obs
                    returns[i] += reward
                    lengths[i] += 1
                    if not (terminated or truncated):
                        next_active.append(i)
                active = next_active
        return {'returns': returns.tolist(), 'lengths': lengths.tolist(),
                'mean_return': float(returns.mean()), 'std_return': float(returns.std()),
                'env_steps': int(lengths.sum()), 'seconds': time.perf_counter()-start}

    def close(self):
        for env in self.envs:
            env.close()
