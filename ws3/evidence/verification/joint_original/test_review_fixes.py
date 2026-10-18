"""Regression checks for corrected PG semantics, including an exact expected gradient.

No formal runs are trained or modified. Run with the RL Python environment.
"""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ws4/Policy_Gradient"))
from Policy_Gradient.agent import PGAgent


class CorrectedPolicyGradient(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(100)
        torch.set_num_threads(1)
        self.env = gym.make("CartPole-v1")

    def tearDown(self):
        self.env.close()

    @staticmethod
    def batch(reward):
        return {"states": np.zeros((4, 4)), "next_states": np.zeros((4, 4)),
                "actions": np.array([0, 0, 0, 0]), "rewards": np.full(4, reward),
                "terminateds": np.array([False, False, False, True]),
                "dones": np.array([False, False, False, True])}

    def test_single_trajectory_keeps_reward_gradient(self):
        agent = PGAgent(self.env, use_baseline=True)
        before = [p.detach().clone() for p in agent.actor.parameters()]
        result = agent.update(self.batch(1.0))
        expected_return = sum(agent.gamma ** np.arange(4))
        self.assertAlmostEqual(result["advantage_mean"], expected_return, places=5)
        self.assertGreater(result["gradient_norm"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())))
        result = agent.update(self.batch(2.0))
        # Current batch is worth 2R, but baseline remains previous batch's R.
        self.assertAlmostEqual(result["advantage_mean"], expected_return, places=5)
        self.assertEqual(agent.baseline_episode_count, 2)

    def test_discount_clock_resets_per_trajectory(self):
        agent = PGAgent(self.env, use_reward_to_go=True, gamma=0.5)
        np.testing.assert_allclose(agent._discounted_reward_to_go([1, 2, 3]), [2.75, 1.75, .75])
        np.testing.assert_allclose(agent._discounted_reward_to_go([4, 6]), [7, 3])

    def test_exact_expected_rtg_gradient_matches_full_return(self):
        # Two Bernoulli decisions with opposite score signs. Relative RTG
        # reverses the gradient here; corrected RTG must match full return.
        agent = PGAgent(self.env, use_reward_to_go=True, gamma=.99)
        gradients = {"full": 0., "rtg": 0., "relative_rtg": 0.}
        for a0, a1 in itertools.product([0, 1], repeat=2):
            rewards = [.995 * a0, float(a1)]
            score = np.array([a0 - .5, .5 - a1])
            full = agent._discounted_return(rewards)
            rtg = agent._discounted_reward_to_go(rewards)
            relative = rtg / (agent.gamma ** np.arange(2))
            for name, weights in [("full", full), ("rtg", rtg), ("relative_rtg", relative)]:
                gradients[name] += .25 * np.dot(score, weights) / 2
        self.assertAlmostEqual(gradients["full"], .000625, places=7)
        self.assertAlmostEqual(gradients["rtg"], gradients["full"], places=7)
        self.assertLess(gradients["relative_rtg"], 0)

    def test_formal_entrypoint_uses_declared_training_sources(self):
        subprocess.run([sys.executable, str(ROOT / "shared/run_all.py"), "--dry-run"], check=True, cwd=ROOT)
        jobs = json.loads((ROOT / "shared/relocated_formal.jobs.json").read_text(encoding="utf-8"))
        self.assertEqual(len(jobs), 84)
        counts = {"dqn": 0, "pg": 0}
        for job in jobs:
            trainers = [Path(arg) for arg in job["command"] if arg.endswith(("run_dqn.py", "run_policy_gradient.py"))]
            self.assertEqual(len(trainers), 1)
            if trainers[0].name == "run_dqn.py":
                self.assertTrue(trainers[0].is_relative_to(ROOT / "shared/historical_v1/ws3/DQN"))
                counts["dqn"] += 1
            else:
                self.assertEqual(trainers[0], ROOT / "ws4/Policy_Gradient/run_policy_gradient.py")
                counts["pg"] += 1
            self.assertTrue(trainers[0].exists())
        self.assertEqual(counts, {"dqn": 33, "pg": 51})


if __name__ == "__main__":
    unittest.main()
