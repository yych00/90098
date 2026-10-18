"""Export exact raw Q3/Q4 CSV curves and the fixed MP4 preview to TensorBoard.

This is explicitly post-training visualization, not an additional training run.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
EXPECTED_PG_HASH = "54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341"
sys.path.insert(0, str(ROOT.parents[1] / "shared"))
from manifest_paths import load_jobs


def main():
    checks = []
    for phase in ["cartpole_final", "halfcheetah_final"]:
        jobs = load_jobs(ROOT / (phase + ".jobs.json"))
        for job in jobs:
            run = Path(job["output_dir"])
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            if config.get("estimator_version") != "corrected-v2" or config.get("code_hash") != EXPECTED_PG_HASH:
                raise ValueError(f"Refusing an old estimator export: {job['id']}")
            destination = run / "tensorboard_export"
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError("Preserving existing TensorBoard export: " + str(destination))
            with (run / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            writer = SummaryWriter(str(destination))
            try:
                for row in rows:
                    writer.add_scalar("Eval Returns", float(row["mean_return"]), int(row["env_steps"]))
                    writer.add_scalar("Eval Returns Std", float(row["std_return"]), int(row["env_steps"]))
                if job["id"] == "halfcheetah_final_s0":
                    reader = imageio.get_reader(str(ROOT / "representative_seed0.mp4"))
                    frames = [np.asarray(Image.fromarray(frame).resize((160, 160)))
                              for index, frame in enumerate(reader) if index % 2 == 0]
                    reader.close()
                    video = np.stack(frames).transpose(0, 3, 1, 2)[None]
                    writer.add_video("Final Video", video, global_step=int(rows[-1]["env_steps"]), fps=10)
                writer.flush()
            finally:
                writer.close()
            accumulator = EventAccumulator(str(destination), size_guidance={"scalars": 0, "images": 0}).Reload()
            scalars = accumulator.Scalars("Eval Returns")
            assert len(scalars) == len(rows)
            assert [event.step for event in scalars] == [int(row["env_steps"]) for row in rows]
            assert np.allclose([event.value for event in scalars], [float(row["mean_return"]) for row in rows], atol=1e-4)
            if job["id"] == "halfcheetah_final_s0":
                assert "Final Video" in accumulator.Tags()["images"]
            checks.append({"id": job["id"], "code_hash": config["code_hash"],
                           "checkpoint_sha256": hashlib.sha256((run / "checkpoint.pt").read_bytes()).hexdigest(),
                           "evaluations_csv_sha256": hashlib.sha256((run / "evaluations.csv").read_bytes()).hexdigest(),
                           "video_sha256": (hashlib.sha256((ROOT / "representative_seed0.mp4").read_bytes()).hexdigest()
                                            if job["id"] == "halfcheetah_final_s0" else None),
                           "scalar_points_verified": len(rows), "tags": accumulator.Tags()})
    record = {"status": "passed", "source": "post-training export from raw CSV and representative MP4",
              "video_preview": "same fixed episode, resized to160x160 and sampled every2 frames at10fps",
              "runs": checks}
    (ROOT / "tensorboard_export_check.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
