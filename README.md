# 90098 - Reinforcement Learning Workshops

Complete cleaned ELEN90098 Workshop 3 and Workshop 4 directories: final reports, completed submission code and experimental evidence.

Authors: Zhengnan Cao and Yuchen Yang.

## Reports and code

| Workshop | English report | Chinese report | Submission code |
|---|---|---|---|
| WS3: Deep Q-Learning | [PDF](ws3/final/Workshop3_Final_Report_EN.pdf) / [Markdown](ws3/final/Workshop3_Final_Report_EN.md) | [PDF](ws3/final/Workshop3_Final_Report_ZH.pdf) / [Markdown](ws3/final/Workshop3_Final_Report_ZH.md) | [DQN](ws3/final/Workshop3-Code-Cao-Yang/) |
| WS4: Policy Gradient | [PDF](ws4/final/Workshop4_Final_Report_EN.pdf) / [Markdown](ws4/final/Workshop4_Final_Report_EN.md) | [PDF](ws4/final/Workshop4_Final_Report_ZH.pdf) / [Markdown](ws4/final/Workshop4_Final_Report_ZH.md) | [Policy Gradient](ws4/final/Workshop4-Code-Cao-Yang/) |

Each report combines Task 1 and Task 2. Experimental questions explicitly identify the question, submitted plots/results and discussion answers. Teaching comments and TODO annotations remain in the code, with the implementations filled in. Submission folders contain English READMEs and dependency lists; no code ZIP is required to browse this repository.

## Topics

- **WS3:** CartPole DQN, learning rate, epsilon-greedy exploration schedules, replay capacity and minibatch size.
- **WS4:** InvertedPendulum Vanilla PG, reward-to-go and a previous-batch historical-return baseline; learning rates and batch steps; CartPole PG versus DQN; a tuned HalfCheetah controller.

Reports use the verified corrected experiment set: 33 DQN formal runs, 51 corrected PG formal runs and six PG selection pilots. Corrected RTG uses episode-origin discount powers; the baseline is formed from earlier batches only. Results include seed variability and unsuccessful configurations. In particular, the HalfCheetah results do not establish uniformly reliable control.

## Running the code

Use Python 3.10 or newer. Follow the setup and command-line examples in the corresponding code README:

- [WS3 setup and commands](ws3/final/Workshop3-Code-Cao-Yang/README.md)
- [WS4 setup and commands](ws4/final/Workshop4-Code-Cao-Yang/README.md)

The recorded checks used Python 3.10.21 and PyTorch 2.5.1. MuJoCo is required for InvertedPendulum and HalfCheetah.

## Snapshot contents

Both workshop directories are included in full. Each contains `final/` and `evidence/`. The evidence includes original assignment materials, frozen experiment source, run configurations, metrics, evaluations, checkpoints, result analysis, available media, integrity manifests and verification records. No training or PDF regeneration was performed during publication preparation.

Historical records retain their original machine paths as provenance. Use the relative paths in each `evidence/run_manifest.json` to locate archived runs. Historical analysis scripts may need their paths adapted before rerunning. Operational preparation records describe the pre-upload state and remain historical snapshots.

From the repository root, verify the archived delivery without training:

```sh
python -B ws3/evidence/verification/verify_clean_delivery.py
```

`PUBLICATION_MANIFEST.json` records the size and SHA256 of every published payload file.
