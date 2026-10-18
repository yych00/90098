# Corrected-v2 formal experiment workspace, 2026-10-17

This workspace is independent of its historical-v1 parent. Do not overwrite or
relabel the parent's raw data, reports or deliveries. Formal results in this
workspace comprise **33 preserved WS3 DQN runs and 51 newly trained corrected-v2
WS4 PG runs**. The six PG selection pilots also use corrected-v2. Completion is
established by this workspace's audits, not by this document or a parent audit.

## Training sources and estimator corrections

The 33 DQN runs retain their original config/checkpoint bytes and trainer hash
`a4350a035f2b565c8d0e0b6343fe3c254acf54c5b7cecb581d60ff97e0bcd8dd`.
Their exact source is `shared/historical_v1/ws3/DQN`. The active DQN hash
`5f1759dfab898740afa191ee17b39ca408d5ff52b4bba3c2de009fa683e56c96`
includes only the separate video-writer repair; it is not substituted for the
source that generated the preserved runs.
`shared/dqn_reuse_inventory.json` records byte-identical copying and SHA256 for
all 198 core evidence files. Packaging checks every inventory hash again;
`preserved_config_output_dir` explicitly preserves each saved config's origin.

All 51 PG formal runs and six selection pilots must declare estimator version
`corrected-v2`, runner version `task2-pg-2.0.0`, and trainer hash
`54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341`.
Their source is the active `ws4/Policy_Gradient`, never archived v1 PG.
Baseline uses the mean discounted trajectory return from earlier batches only,
initially zero, updating history after current advantages are formed. This
avoids the structural zero reward signal caused by subtracting a single
trajectory's own return. It does not guarantee nonzero advantages or lower
variance on every batch. RTG includes outer `gamma**t`, resetting time at each
trajectory, to match Vanilla's episode-origin discount convention. GAE,
normalization and gradient clipping remain explicitly labelled extensions.

New selection, analysis, plots, motion observations and bilingual WS4 reports
must come from this workspace's corrected-v2 runs. Historical v1 rankings,
selection JSON, gradient diagnostic and motion observations cannot establish
corrected-v2 performance. Models are evaluation checkpoints, not exact training
resume snapshots; PG baseline history is saved, but full environment/RNG state
is not. DQN checkpoints omit replay.

## Reproduction and release guards

`shared/run_all.py` requires an explicitly sealed corrected-v2 catalogue, then
relocates all 84 jobs. Only DQN is redirected to its archived trainer. Every PG
job must target the active corrected-v2 entrypoint; legacy/unknown trainers are
rejected. `--dry-run` checks the catalogue without training.

`shared/check_source_integrity.py` audits the two formal source roots separately,
checks the current repaired sources separately, requires a 33/51 formal split,
and checks all complete configs against the requested parameters. It can report
incomplete runs during execution with `--allow-incomplete`; such a report never
passes final delivery.

Packaging requires this workspace's complete 84-run raw audit and source audit,
a corrected-v2 WS4 analysis audit, and matching hashes plus all-page visual QA
for all four PDFs. It compares each config/checkpoint with the audit, verifies
both formal source hashes, and requires corrected-v2 selection pilots. Old
recovery/portability records are optional context, not release approval. Output
must use a directory whose path includes `corrected_v2`, preserving v1 delivery.
The source package includes both selection JSONs and `motion_summary.json`.

WS3 retains its manifest-selection repair, separate video writer, and explicit
batch-size comparisons of AUC and the third consecutive >=475 checkpoint with
success counts. These changes do not justify replacing the original DQN runs.
