# Workshop 4: Policy Gradient

Authors: Zhengnan Cao and Yuchen Yang

## Files

| File | Purpose |
|---|---|
| `Policy_Gradient/run_policy_gradient.py` | Training CLI, batch collection, evaluation and output |
| `Policy_Gradient/policy_gradient/agent.py` | Vanilla PG, reward-to-go, prior-batch baseline and optional GAE |
| `Policy_Gradient/policy_gradient/network.py` | Gaussian/categorical actors and value network |
| `Policy_Gradient/utils/util.py` | Rollouts and isolated evaluation randomness |
| `Policy_Gradient/utils/logger.py` | TensorBoard logger |
| `Policy_Gradient/utils/log_utils.py` | Inactive legacy teaching reference |

## Installation

Use the shared environment in `../README.md`, or install the local `requirements.txt` with Python 3.10. The shared environment includes the CartPole rendering extra as well as MuJoCo.

Run all following commands from `ws4`:

```powershell
python Policy_Gradient/run_policy_gradient.py --help
```

## InvertedPendulum baseline

```powershell
python Policy_Gradient/run_policy_gradient.py --env_name InvertedPendulum-v5 --total_steps 500000 --batch_steps 1000 --learning_rate 0.001 --discount_gamma 0.99 --hidden_size 64 --layer_num 2 --use_reward_to_go --seed 0 --device cpu --output_dir runs/pg_rtg_ip_seed0
```

For Vanilla PG, omit `--use_reward_to_go`. For the prior-batch baseline, replace it with `--use_baseline`. The two flags are mutually exclusive. Repeat each report condition with training seeds 0, 1 and 2 in separate directories.

## Report experiments

| Report section | Configuration |
|---|---|
| Task 1 Q1 | InvertedPendulum state, action and reward definition |
| Task 1 Q2 | Compare Vanilla PG, RTG and baseline at learning rate 0.001, batch 1000 |
| Task 2 Q1 | Learning rates 0.0001, 0.001, 0.01 at batch 1000, for all three estimators |
| Task 2 Q2 | Batch steps 1000, 2000, 5000 at learning rate 0.001, for all three estimators |
| Task 2 Q3 | CartPole PG against WS3 DQN |
| Task 2 Q4 | HalfCheetah PG controller and motion observations |

Q1/Q2 use InvertedPendulum-v5, 500000 requested steps, gamma 0.99 and a 64-by-2 network. Keep advantage normalization, GAE and gradient clipping disabled in these core comparisons. The CLI gamma default is 0.9, so explicitly pass `--discount_gamma 0.99` for report settings.

CartPole selected controller:

```powershell
python Policy_Gradient/run_policy_gradient.py --env_name CartPole-v1 --total_steps 200000 --batch_steps 2000 --learning_rate 0.003 --discount_gamma 0.99 --hidden_size 64 --layer_num 3 --use_reward_to_go --seed 0 --device cpu --output_dir runs/pg_cartpole_seed0
```

HalfCheetah selected controller:

```powershell
python Policy_Gradient/run_policy_gradient.py --env_name HalfCheetah-v5 --total_steps 1000000 --batch_steps 5000 --learning_rate 0.001 --discount_gamma 0.99 --hidden_size 64 --layer_num 2 --use_reward_to_go --normalize_advantage --seed 0 --device cpu --output_dir runs/pg_halfcheetah_seed0
```

## Estimator and evaluation conventions

Vanilla PG repeats the episode-origin discounted return at every step. RTG uses `sum(gamma**k * reward[k], k=t..T-1)`, retaining the outer episode-origin discount factor. The baseline subtracts the mean discounted trajectory return from previous batches only, initially zero; it is not a fitted critic or current-batch mean.

Continuous-action policies retain the raw Gaussian sample for log probabilities and clip only the environment action. CartPole uses a categorical actor. The optional GAE branch uses a value network for `gae_lambda < 1`; WS5 documents that extension separately.

Evaluation is stochastic in a separate environment with fixed environment and policy-sampling seeds 10000-10009. Python, NumPy and PyTorch training random states are restored afterwards. Defaults are ten episodes and an evaluation interval of 10000 training steps.

## Outputs and step budgets

Use a unique `--output_dir` for every run. Outputs include `config.json`, `metrics.csv`, `evaluations.csv`, `evaluation_episodes.csv`, `summary.json`, `checkpoint.pt` and TensorBoard events. Complete episodes and the final batch may exceed `--total_steps`. Checkpoints are for inspection, not exact training resumption.

## Quick integration check

```powershell
python Policy_Gradient/run_policy_gradient.py --env_name InvertedPendulum-v5 --total_steps 100 --batch_steps 50 --eval_every 100 --eval_episodes 1 --discount_gamma 0.99 --use_reward_to_go --seed 0 --device cpu --output_dir runs/pg_smoke_seed0
```

This check does not reproduce the full experiments. See `../VERIFICATION.md`. The source is the existing template-preserving submission port; archived report measurements remain separate from this package.
