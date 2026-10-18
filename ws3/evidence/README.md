# Workshop 3 evidence

`run_manifest.json` maps the retained runs to current **relative** paths and records core SHA256 values. Configs, checkpoints and CSVs were copied without modification; their old absolute output paths remain historical provenance. Read the manifest rather than following those old paths. WS3 stores the 33 DQN formal runs once; WS4 stores 51 corrected PG formal runs and six selection pilots. The DQN comparison in the WS4 report refers to the corresponding WS3 runs, not duplicate copies.

`assignment/` holds the original handout and, for WS4, its original template ZIP. `experiment_source/` is the exact source that produced the retained results; it is distinct from the comment-preserving source in `../final/`. The shared recorded runtime is at `../../ws3/evidence/experiment_source/requirements-recorded.txt`.

`analysis/` contains result tables, local gradient diagnostics, media and scalar-event exports. `verification/` keeps the final QA and original joint experiment audits. Original analysis/test scripts are retained as historical authoring sources; their original workspace paths have not been silently rewritten, and they may require adaptation to rerun. They are not submission entrypoints. Use the final code README to run the submitted implementations.

The original shared audit records live once in `../../ws3/evidence/verification/joint_original/`. All cleanup copy mappings are listed in `verification/cleanup/relocation_manifest.json`. Run `python -B ../../ws3/evidence/verification/verify_clean_delivery.py` from this directory for portable hash verification. No training is performed by that check.
