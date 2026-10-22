# Workshop 5: Actor-Critic and Soft Actor-Critic

Authors: Zhengnan Cao and Yuchen Yang

## Scope and report status

The existing report uses 17 completed formal conditions: four AC/GAE conditions requesting 500000 training steps each and thirteen SAC conditions at 1000000 steps each. All use preselected training seed 0. These are single-seed comparisons; temporal fluctuations are not variation across independent training seeds.

This organized source package does not rerun those experiments. It replaces the obsolete planning README with instructions for the completed implementations. Raw measurements and frozen original sources remain in the original evidence archive.

## Files

| File or directory | Purpose |
|---|---|
| `Policy_Gradient/run_policy_gradient.py` | AC/GAE training entry point |
| `Policy_Gradient/policy_gradient/agent.py` | Actor, critic, GAE recursion and policy/value updates |
| `SAC/run_sac.py` | SAC training, replay collection, evaluation, metrics and checkpoints |
| `SAC/sac/agent.py` | Twin critics, policy, temperature tuning and replay |
| `SAC/sac/network.py` | Squashed Gaussian actor and Q networks |
| `SAC/sac/cuda_update.py` | Optional CUDA Graph update and policy inference |
| `SAC/utils/evaluation.py` | Batched fixed-seed stochastic evaluation |
| `experiment_commands.json` | All 17 original formal commands with portable paths |

The `utils/log_utils.py` files are inactive legacy teaching references. Active training and evaluation do not import them. Other utilities retain the teaching package layout.

## Installation and working directory

Use Python 3.10 and the shared environment documented in `../README.md`. The local `requirements.txt` retains the original dependency pins. All commands below run from `ws5`:

```powershell
python Policy_Gradient/run_policy_gradient.py --help
python SAC/run_sac.py --help
```

## AC/GAE training

The Task 1 HalfCheetah configuration is:

```powershell
python Policy_Gradient/run_policy_gradient.py --env_name HalfCheetah-v5 --total_steps 500000 --batch_steps 1000 --learning_rate 0.001 --baseline_learning_rate 0.001 --discount_gamma 0.99 --hidden_size 64 --layer_num 2 --gae_lambda 0.99 --normalize_advantage --max_grad_norm 0 --eval_every 10000 --eval_episodes 10 --seed 0 --device cpu --output_dir runs/ac_hc_lambda099__seed0
```

For Task 2 Q1, use `--env_name InvertedPendulum-v5` and run GAE lambda values 0.90, 0.95 and 0.99 separately. Keep the other AC settings unchanged, including advantage normalization. Change the output directory for each condition.

True termination removes the value bootstrap; time-limit truncation retains it. Both end the GAE recursion at episode boundaries. `--gae_lambda 1.0` selects the inherited no-critic branch; it is not an executable AC endpoint experiment in this implementation.

## SAC training

Portable CPU baseline:

```powershell
python SAC/run_sac.py --env_name HalfCheetah-v5 --total_steps 1000000 --memory_size 1000000 --batch_size 256 --learning_start 5000 --actor_learning_rate 0.0001 --critic_learning_rate 0.0001 --discount_gamma 0.99 --temperature 0.2 --tau 0.005 --hidden_size 256 --layer_num 2 --seed 0 --device cpu --eval_device cpu --eval_every 10000 --eval_episodes 10 --eval_batch_size 10 --torch_threads 1 --log_every 1000 --output_dir runs/sac_cpu_baseline_seed0
```

The formal SAC experiments used CUDA, with `--fused_optimizer --cuda_graph --cuda_policy_graph`, and CPU evaluation. To use that profile, change `--device cpu` to `--device cuda` and add those three flags. Use a new output directory. A CPU run is a portable alternative, not a promise of bitwise equality with the recorded CUDA trajectories.

CUDA Graph updates require CUDA, the fused optimizer and the optimized path. Policy graph capture additionally requires `--cuda_graph`. The optional `--reference` flag selects the completed unoptimized numerical path and cannot be combined with graph updates.

## Report experiments

All SAC factors change individually from the shared baseline above:

| Report section | Conditions |
|---|---|
| Task 1 Q1 | HalfCheetah environment and stochastic policies |
| Task 1 Q2 | HalfCheetah AC/GAE with lambda 0.99 |
| Task 1 Q3 | HalfCheetah SAC baseline |
| Task 2 Q1 | InvertedPendulum AC/GAE lambda 0.90, 0.95, 0.99 |
| Task 2 Q2 | SAC initial temperature 0.01, 0.2, 1.0; automatic tuning remains active |
| Task 2 Q3 | Actor/critic learning-rate pairs: (1e-4, 1e-4), (1e-2, 1e-4), (1e-4, 1e-2), (1e-5, 1e-4), (1e-4, 1e-5) |
| Task 2 Q4 | Replay capacity 100000, 500000, 1000000; batch size 32, 256, 512; learning start 1000, 5000, 10000 |

The shared SAC baseline is reused across comparisons. `experiment_commands.json` contains argument arrays for all 17 distinct formal conditions, with paths relative to this directory. Its `original_run_status` describes the archived completed run, not a new run performed by this package.

To execute one selected archived configuration, inspect the list and run:

```powershell
python -c "import json,subprocess,sys; jobs=json.load(open('experiment_commands.json')); job=next(j for j in jobs if j['condition_id']=='ac_ip_lambda095'); subprocess.run([sys.executable]+job['command'][1:],check=True)"
```

This command starts a full training run. GPU SAC entries require the CUDA profile described above. Existing nonempty output directories are rejected.

## Evaluation, logging and outputs

PG/GAE and SAC evaluation is stochastic with ten fixed environment and policy-noise seeds, 10000-10009, every 10000 training steps. Evaluation does not consume the training random streams. SAC's per-episode noise streams are independent of evaluation batching; numerical batching differences can still change long simulator trajectories.

SAC performs one update per eligible environment transition. `--log_every 1000` averages metrics over counted update windows; it does not reduce the update frequency. These window means are not per-update losses. Use the recorded update count when aggregating a final partial window.

Outputs include `config.json`, `metrics.csv`, `evaluations.csv`, `evaluation_episodes.csv`, `summary.json`, `checkpoint.pt` and TensorBoard events; SAC also saves `episodes.csv`. GAE may exceed the requested budget to complete episodes and the final batch. Neither checkpoint includes the full state required for exact training resumption.

## Quick integration checks

```powershell
python Policy_Gradient/run_policy_gradient.py --env_name InvertedPendulum-v5 --total_steps 100 --batch_steps 50 --eval_every 100 --eval_episodes 1 --discount_gamma 0.99 --gae_lambda 0.95 --normalize_advantage --seed 0 --device cpu --output_dir runs/gae_smoke_seed0
python SAC/run_sac.py --env_name HalfCheetah-v5 --total_steps 16 --learning_start 8 --batch_size 8 --memory_size 64 --eval_every 16 --eval_episodes 1 --eval_batch_size 1 --hidden_size 16 --layer_num 1 --log_every 8 --seed 0 --device cpu --eval_device cpu --output_dir runs/sac_smoke_seed0
```

These small checks verify real environment interaction, updates and saved outputs. They do not establish learning quality or repeat the report's experiments. See `../VERIFICATION.md` for the checks actually performed on the packaged source.
