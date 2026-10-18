# Workshop 3: Deep Q-Learning

Authors: Zhengnan Cao, Yuchen Yang

This folder contains the original Python directory layout, classes, function
interfaces and retained teaching comments, including the TODO prompts. The
implementations underneath those prompts are complete. Extra optional arguments
and small helpers provide reproducible training and independent evaluation.
Machine-specific launchers, logs, checkpoints and caches are not included.

## Run

Use Python 3.10 and install the recorded CPU-compatible dependencies from this
folder. A suitable existing PyTorch installation can also be used.

```sh
python -m pip install -r requirements.txt
python DQN/run_dqn.py --help
python DQN/run_dqn.py --env_name CartPole-v1 --device cpu --seed 0 --total_steps 200000 --tensorboard --output_dir runs/baseline_seed0
```

Paths are relative to the current working directory. No module requires a
machine-specific directory or an external copy of this project. Existing
nonempty output directories are rejected. For a quick integration check, use
`--total_steps 200 --eval_every 100 --max_ep_len 25 --learning_start 8 --batch_size 8`.
Those short-run settings are only checks, not the report's formal experiment.

`--total_steps` imposes an exact training environment-step budget and takes
precedence over `--episodes`. If it is omitted, the original episode-count mode
is retained, with 1000 episodes by default. `--max_ep_len` changes the training
and evaluation time limit; leave it unset for the report's CartPole-v1 limit.
`--tensorboard` enables scalar events, `--log_video` records the final greedy
policy, and `--render` displays training. These options are not required for
headless CPU training.

## Algorithm and evaluation

- The Q-network uses three 64-unit ReLU hidden layers by default, Adam at 0.001,
  mean squared TD error and gradient-norm clipping at 10.
- Replay is uniform without replacement. Learning starts when replay contains
  at least `max(learning_start, batch_size)` transitions. The target network is
  copied every `target_update_freq` gradient updates, with default frequency 500.
- The TD mask uses true termination only. Time-limit truncation ends a training
  episode but retains the bootstrap value.
- Epsilon is multiplied by its decay once after each completed training
  episode and bounded by its floor. A final partial episode at a step budget
  does not trigger decay. The defaults are 1.0, 0.995 and 0.02.
- Greedy evaluation uses a separate environment, ten episodes and fixed reset
  seeds 10000 through 10009, initially, every 10000 training steps and at the
  final checkpoint. Evaluation preserves training random streams, does not
  decay epsilon and does not count toward training steps.
- CPU training defaults to one PyTorch thread. Action selection and replay
  sampling use separately seeded random generators.

Each output directory contains configuration, training and evaluation CSVs, a
summary and an evaluation checkpoint. A checkpoint does not contain replay or
the complete environment state, so it is not an exact training-resume snapshot.

## Report parameter provenance

The report's WS3 Task 2 measurements use 33 independent 200000-step runs with
training seeds 0, 1 and 2. Their archived trainer hash is
`a4350a035f2b565c8d0e0b6343fe3c254acf54c5b7cecb581d60ff97e0bcd8dd`.
The measured results, model files and audit records remain separate from this
submission source folder. This comment-preserving port is a different source
snapshot; it does not relabel the saved experiments as newly trained runs.

The CLI exposes the report's controlled settings:

| Comparison | Arguments varied |
|---|---|
| Learning rate | `--learning_rate 0.0001`, `0.001`, `0.01` |
| Exploration | `--epsilon 0.1 --epsilon_decay 1.0`, or initial epsilon 1 with decay 0.9/0.995 and floor 0.02 |
| Replay and minibatch | `--memory_size 1000/10000/100000` and `--batch_size 32/128` |

The shared reference uses replay capacity 100000 and batch size 64. Across these
comparisons, gamma is 0.99, warm-up is 1000, target frequency is 500, and the
network is 64 units by three hidden layers. Use distinct output directories for
each configuration and seed. Slash-separated values in the table indicate
separate runs, not literal CLI argument syntax.

## Template provenance

Comment and interface fidelity was checked against the available early `DQN`
source tree, whose five Python files agree with the earlier submitted source
folder. An unmodified original teacher `DQN.zip` was not available for this
check. No claim is made that this implementation is byte-identical to an
unavailable starter archive. Offline numerical and short-run checks are kept
outside this submission folder.
