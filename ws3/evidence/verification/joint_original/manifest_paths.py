"""Resolve declared catalogue paths inside a relocated evidence/source package.

Only the recorded package-root prefix is mapped. Historical config pointers stay
unchanged, and are accepted only through the manifest's explicit provenance.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import json
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _previous_root():
    catalog = ROOT / "shared/experiment_catalog.json"
    if not catalog.exists():
        return None
    return json.loads(catalog.read_text(encoding="utf-8"))["original_root"]


def relocate_path(value):
    candidate = Path(value)
    if candidate.is_absolute() and candidate.resolve().is_relative_to(ROOT):
        return value  # Idempotent, including a package extracted beneath the old root.
    previous = _previous_root()
    if previous is None:
        return value
    if str(value).casefold() == previous.casefold():
        return str(ROOT)
    prefix = previous.rstrip("/\\") + "\\"
    if str(value).casefold().startswith(prefix.casefold()):
        return str(ROOT.joinpath(*PureWindowsPath(str(value)[len(prefix):]).parts))
    return value


def load_jobs(path):
    jobs = json.loads(Path(path).read_text(encoding="utf-8"))
    assert isinstance(jobs, list), path
    mapped = []
    for original in jobs:
        job = copy.deepcopy(original)
        job["manifest_output_dir"] = original["output_dir"]
        job["output_dir"] = relocate_path(original["output_dir"])
        job["cwd"] = relocate_path(original["cwd"])
        job["command"] = [relocate_path(value) for value in original["command"]]
        assert Path(job["output_dir"]).resolve().is_relative_to(ROOT), (job["id"], "output outside package")
        assert Path(job["cwd"]).resolve().is_relative_to(ROOT), (job["id"], "cwd outside package")
        if "parameters" in job and "output_dir" in job["parameters"]:
            job["parameters"]["output_dir"] = relocate_path(job["parameters"]["output_dir"])
        mapped.append(job)
    return mapped


def allowed_output_paths(job):
    values = {job["output_dir"], job.get("manifest_output_dir", job["output_dir"])}
    if job.get("original_output_dir"):
        values.add(job["original_output_dir"])
    if job.get("preserved_config_output_dir"):
        values.add(job["preserved_config_output_dir"])
    return {str(Path(value).resolve()).casefold() for value in values}
