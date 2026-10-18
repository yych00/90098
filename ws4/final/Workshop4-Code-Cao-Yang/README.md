# Workshop 4 — Policy Gradient

Authors: Zhengnan Cao and Yuchen Yang

The `Policy_Gradient/` folder follows the supplied workshop template, including its lowercase `policy_gradient/` package, original classes, functions, comments and TODO annotations. The code below each TODO is completed. The original training loop is retained.

## Setup and run

Use Python 3.10 or newer (checked with Python 3.10.21). Install the recorded dependencies, then run commands from `Policy_Gradient/`:

```sh
python -m pip install -r requirements.txt
cd Policy_Gradient
python run_policy_gradient.py --env_name InvertedPendulum-v5 --discount_gamma 0.99 --use_reward_to_go --device cpu
```

Run the installation command from the submission root, before changing directory. MuJoCo is used through Gymnasium's `mujoco` extra. Rendering and video are optional.

The report's selected CartPole configuration is:

```sh
python run_policy_gradient.py --env_name CartPole-v1 --total_steps 200000 --batch_steps 2000 --learning_rate 0.003 --discount_gamma 0.99 --hidden_size 64 --layer_num 3 --use_reward_to_go --seed 0 --device cpu
```

The report's selected HalfCheetah configuration is:

```sh
python run_policy_gradient.py --env_name HalfCheetah-v5 --total_steps 1000000 --batch_steps 5000 --learning_rate 0.001 --discount_gamma 0.99 --hidden_size 64 --layer_num 2 --use_reward_to_go --normalize_advantage --seed 0 --device cpu
```

Use training seeds 0, 1 and 2 separately for the three-seed comparisons. Q1/Q2 use `InvertedPendulum-v5`, 500000 requested steps, 64x2 networks and gamma 0.99. At batch 1000, compare learning rates 0.0001, 0.001 and 0.01. At learning rate 0.001, compare batch sizes 1000, 2000 and 5000. For each setting, use no estimator flag for Vanilla, `--use_reward_to_go` for RTG, or `--use_baseline` for the prior-batch baseline. Core Q1/Q2 comparisons leave normalization, GAE and gradient clipping disabled.

The template's command-line gamma default remains 0.9; the report configurations explicitly set 0.99. `--output_dir` optionally selects an empty output directory. Otherwise runs are written under relative `logs/`. Existing nonempty run directories are preserved and rejected. Outputs include scalar events, evaluation CSVs, configuration, summary and a policy-inspection checkpoint. Evaluation steps are excluded from the training budget. Complete episodes and the final batch may overshoot the requested budget. Checkpoints do not support exact training resumption.

## Estimators and action handling

- Vanilla repeats the full episode-origin discounted return at every trajectory step.
- RTG uses `sum(gamma**k * reward[k], k=t..T-1)`, including the outer `gamma**t` relative to conventional RTG. Discount powers restart at each episode.
- `--use_baseline` subtracts the mean discounted trajectory return from **previous batches only**, initially zero. Current trajectories are added to history after their advantages are formed. It does not subtract a batch's own mean or fit a value network.
- Gaussian actions are kept as raw samples for the policy loss. Only the action passed to the environment is clipped. Log probabilities are summed over action components. CartPole uses a categorical actor and integer actions.
- `--normalize_advantage`, `--gae_lambda 0.95`, `--baseline_learning_rate` and `--max_grad_norm` expose optional extensions. GAE uses local TD advantages without outer episode-origin weighting. True termination suppresses value bootstrap; time-limit truncation retains it, while both end the episode recursion.
- Evaluation uses ten episodes by default, fixed environment and policy-sampling seeds 10000–10009, a separate environment, and restored Python/NumPy/PyTorch RNG streams. `--eval_seed`, `--eval_episodes` and `--eval_every` are optional controls. `--log_video` logs a fixed-seed video without advancing training RNG.

The numerical learning core is adapted from the verified corrected-v2 implementation. The submission layout and runner preserve the teacher's interfaces; the archived experimental trainer and measurements remain separate artifacts.
