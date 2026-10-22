# Workshop 3: Deep Q-Learning

Authors: Zhengnan Cao and Yuchen Yang

## Files

| File | Purpose |
|---|---|
| `DQN/run_dqn.py` | Training CLI, isolated greedy evaluation, CSVs and checkpoint output |
| `DQN/dqn/agent.py` | DQN updates, replay, epsilon-greedy actions and target network |
| `DQN/dqn/network.py` | Q-network |
| `DQN/utils/util.py` | Trajectory collection and random-state isolation |
| `DQN/utils/logger.py` | TensorBoard logging |

## Installation

Use the shared environment described in `../README.md`, or install `requirements.txt` from this `ws3` directory using Python 3.10:

```powershell
python -m pip install -r requirements.txt
python DQN/run_dqn.py --help
```

## Baseline training

Run all commands below from `ws3`:

```powershell
python DQN/run_dqn.py --env_name CartPole-v1 --device cpu --seed 0 --total_steps 200000 --learning_rate 0.001 --discount_factor 0.99 --memory_size 100000 --batch_size 64 --learning_start 1000 --target_update_freq 500 --hidden_size 64 --layer_num 3 --epsilon 1.0 --epsilon_decay 0.995 --epsilon_min 0.02 --eval_every 10000 --eval_episodes 10 --tensorboard --output_dir runs/dqn_baseline_seed0
```

Repeat with training seeds 0, 1 and 2 and distinct output directories for the report protocol. `--total_steps` gives an exact environment-step budget and overrides `--episodes`. If omitted, episode-based training defaults to 1000 episodes. Leave `--max_ep_len` unset for the report's CartPole time limit.

## Report experiments

| Report section | Settings varied from the baseline |
|---|---|
| Task 1 Q1 | CartPole state, action and reward definition |
| Task 1 Q2 | DQN baseline learning curve |
| Task 2 Q1 | `--learning_rate`: 0.0001, 0.001, 0.01 |
| Task 2 Q2 | Fixed epsilon: `--epsilon 0.1 --epsilon_decay 1.0`; decaying epsilon: initial 1.0, decay 0.9 or 0.995, floor 0.02 |
| Task 2 Q3 | Replay capacity: 1000, 10000, 100000; batch size: 32 or 128 |

Each listed value is a separate run. The shared reference uses replay capacity 100000 and batch size 64. The report contains 33 archived 200000-step runs; the command above specifies the shared reference, not the entire experiment sweep.

## Learning and evaluation details

The default network has three 64-unit ReLU hidden layers. Training uses Adam, mean squared TD error and gradient-norm clipping at 10. Replay sampling is uniform without replacement. Training begins after at least `max(learning_start, batch_size)` replay entries, and the target network is copied every 500 gradient updates by default.

True termination suppresses bootstrapping; time-limit truncation ends an episode but retains the bootstrap value. Epsilon decays after a completed training episode. A final partial episode at the step budget does not trigger an extra decay.

Greedy evaluation uses a separate environment and ten fixed reset seeds, 10000-10009, at step zero, every 10000 training steps and at the final checkpoint. Evaluation preserves training random streams and does not change epsilon.

## Outputs

Each output directory contains `config.json`, `metrics.csv`, `evaluations.csv`, `evaluation_episodes.csv`, `summary.json` and `checkpoint.pt`. `--tensorboard` adds events under `tensorboard/`; `--log_video` adds a final policy video. The checkpoint does not contain replay or the complete environment state.

## Quick integration check

```powershell
python DQN/run_dqn.py --env_name CartPole-v1 --device cpu --seed 0 --total_steps 200 --eval_every 100 --eval_episodes 2 --max_ep_len 25 --learning_start 8 --batch_size 8 --output_dir runs/dqn_smoke_seed0
```

This intentionally changes the budget, evaluation count, warm-up and episode limit. It verifies execution, not report performance. See `../VERIFICATION.md` for the delivered package checks. The source retains completed TODO prompts from the teaching template; it is a submission port rather than the exact archived experimental trainer.
