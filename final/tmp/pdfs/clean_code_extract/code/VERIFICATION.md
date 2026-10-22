# Package Verification

Verification date: 22 October 2026 (Australia/Sydney).

## Source and documentation

- All 25 Python files passed in-memory compilation without creating bytecode caches.
- Executable ASTs match the original final code packages after excluding docstrings. Seven WS5 files contain English translations of comments; two of those also have translated module docstrings. No algorithm, argument interface or training behavior was changed.
- All delivered code, documentation, filenames and portable command records were checked for Chinese characters; none were found.
- The original 28 source/dependency files have source and delivered SHA-256 values in `SOURCE_MANIFEST.json`. The shared requirements add both Gymnasium environment extras while retaining the recorded version pins.
- All four entry points passed `--help` after clean ZIP extraction. Extracted files matched their packaged SHA-256 values.

## Real-environment integration checks

The checks used the existing Python 3.10.21 environment with PyTorch 2.5.1+cu121, Gymnasium 1.3.0, NumPy 2.2.6 and MuJoCo 3.15.0. All four runs used the CPU. Commands correspond to the quick checks in each workshop README, with output directories redirected outside the source package.

| Entry point | Environment | Requested steps | Actual steps | Updates | Result |
|---|---|---:|---:|---:|---|
| WS3 DQN | CartPole-v1 | 200 | 200 | 193 | Completed |
| WS4 PG/RTG | InvertedPendulum-v5 | 100 | 117 | 2 | Completed |
| WS5 AC/GAE | InvertedPendulum-v5 | 100 | 118 | 2 | Completed |
| WS5 SAC | HalfCheetah-v5 | 16 | 16 | 9 | Completed |

The PG/GAE overshoot follows the documented complete-episode/batch rule. Each run produced its configuration, metrics, evaluation CSVs, summary and checkpoint. Numeric CSV fields were checked for NaN/Inf. These checks verify environment stepping, learning updates, evaluation and output creation; their returns are not evidence of learned policy quality.

## Boundaries

The report's full training experiments were not rerun. A clean dependency installation, fresh CUDA Graph execution, policy-video playback and manual TensorBoard browser viewing were not performed for this packaging task. The ZIP excludes all check outputs, training runs, model weights and bytecode caches. Local preparation records remain outside the package under `tmp/pdfs/` beside the combined report.
