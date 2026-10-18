"""Small semantic checks for the experiment runner; no claim of learning success."""
import copy
import json
from pathlib import Path
import random
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Policy_Gradient"))
from run_policy_gradient import evaluate, collect_batch, source_metadata
from Policy_Gradient.agent import PGAgent
from analyze_results import CORRECTED_V2_HASH
from gradient_variance import gradient_vector


def main():
    torch.set_num_threads(1)
    assert source_metadata()["code_hash"] == CORRECTED_V2_HASH
    env = gym.make("CartPole-v1")
    torch.manual_seed(10)
    agent = PGAgent(env)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    results = evaluate(env, agent, 10000, 3)
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2])
    assert results == evaluate(env, agent, 10000, 3)
    rewards = np.array([1., 2., 3.])
    gamma = agent.gamma
    assert np.allclose(agent._discounted_return(rewards), [1 + 2 * gamma + 3 * gamma ** 2] * 3)
    assert np.allclose(agent._discounted_reward_to_go(rewards),
                       [1 + 2 * gamma + 3 * gamma ** 2, gamma * (2 + 3 * gamma), 3 * gamma ** 2])
    env.reset(seed=8)
    batch, returns, lengths, clipped = collect_batch(env, agent, 100)
    assert len(batch["rewards"]) >= 100 and batch["dones"][-1]
    assert sum(lengths) == len(batch["rewards"])
    assert all(len(batch[key]) == len(batch["rewards"]) for key in batch)
    assert np.isfinite(agent.update(batch)["actor_loss"])
    # A one-trajectory batch must not use its own return as its baseline.
    baseline = PGAgent(env, use_baseline=True, actor_lr=0)
    single = {"states": np.zeros((3, 4)), "next_states": np.zeros((3, 4)),
              "actions": np.array([0, 1, 0]), "rewards": rewards,
              "terminateds": np.array([False, False, True]), "dones": np.array([False, False, True])}
    expected_return = float(baseline._discounted_return(rewards)[0])
    first = baseline.update(single)
    assert np.isclose(first["advantage_mean"], expected_return) and first["gradient_norm"] > 0
    assert baseline.baseline_episode_count == 1 and np.isclose(baseline.baseline_return_sum, expected_return)
    second_batch = {**single, "rewards": rewards * 2}
    second = baseline.update(second_batch)
    assert np.isclose(second["advantage_mean"], expected_return)
    assert baseline.baseline_episode_count == 2 and np.isclose(baseline.baseline_return_sum, expected_return * 3)
    # Compare the diagnostic with actual trainer gradients, using the same
    # fixed prior calibration history rather than a same-batch sample mean.
    prior_return_sum, prior_episode_count = 7.5, 3
    maximum_error = {}
    initial_state = copy.deepcopy(agent.actor.state_dict())
    for method in ("vanilla", "rtg", "baseline"):
        candidate = PGAgent(env, use_reward_to_go=method == "rtg", use_baseline=method == "baseline", actor_lr=0)
        candidate.actor.load_state_dict(initial_state)
        if method == "baseline":
            candidate.baseline_return_sum = prior_return_sum
            candidate.baseline_episode_count = prior_episode_count
        expected = gradient_vector(candidate, single, method, prior_return_sum / prior_episode_count)
        candidate.update(single)
        actual = torch.cat([parameter.grad.reshape(-1) for parameter in candidate.actor.parameters()]).detach().numpy()
        assert np.allclose(expected, actual, atol=1e-7, rtol=1e-6), method
        maximum_error[method] = float(np.max(np.abs(expected - actual)))
    env.close()
    # A multi-action Gaussian must yield one log probability per state.
    continuous = gym.make("HalfCheetah-v5")
    agent = PGAgent(continuous)
    sample = {"states": np.zeros((2, 17)), "next_states": np.zeros((2, 17)),
              "actions": np.zeros((2, 6)), "rewards": np.array([1., 2.]),
              "terminateds": np.array([False, False]), "dones": np.array([False, True])}
    assert np.isfinite(agent.update(sample)["actor_loss"])
    gaussian_state = copy.deepcopy(agent.actor.state_dict())
    for method in ("vanilla", "rtg", "baseline"):
        candidate = PGAgent(continuous, use_reward_to_go=method == "rtg", use_baseline=method == "baseline", actor_lr=0)
        candidate.actor.load_state_dict(gaussian_state)
        if method == "baseline":
            candidate.baseline_return_sum = prior_return_sum
            candidate.baseline_episode_count = prior_episode_count
        expected = gradient_vector(candidate, sample, method, prior_return_sum / prior_episode_count)
        candidate.update(sample)
        actual = torch.cat([parameter.grad.reshape(-1) for parameter in candidate.actor.parameters()]).detach().numpy()
        assert np.allclose(expected, actual, atol=1e-7, rtol=1e-6), method
        maximum_error["gaussian_" + method] = float(np.max(np.abs(expected - actual)))
    continuous.close()
    result = {"status": "passed", "estimator_version": "corrected-v2", "code_hash": CORRECTED_V2_HASH,
              "diagnostic_gradient_max_abs_error": maximum_error,
              "checks": ["evaluation_rng_isolation", "fixed_seed_evaluation_repeatability",
              "discounted_return_semantics", "absolute_episode_origin_discounted_rtg", "complete_batch_alignment",
              "single_trajectory_baseline_retains_reward_signal", "baseline_uses_only_prior_batches",
              "diagnostic_gradient_equals_trainer_for_all_core_methods", "categorical_update", "multidimensional_gaussian_update"]}
    (Path(__file__).resolve().parent / "contract_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
