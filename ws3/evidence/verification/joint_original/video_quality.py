"""Decode every exported video frame and make a time-labelled QA contact sheet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Checkpoint used to render this fixed-episode video")
    parser.add_argument("--render-record", required=True, type=Path,
                        help="Passed reload_checkpoint.py record produced while rendering this MP4")
    args = parser.parse_args()
    render = json.loads(args.render_record.read_text(encoding="utf-8"))
    config = json.loads((args.checkpoint.parent / "config.json").read_text(encoding="utf-8"))
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    video_sha = hashlib.sha256(args.video.read_bytes()).hexdigest()
    expected_hash = "54d3600cf459dd85ee20b8af00f1aacf47ed022eb0714807e99e1f4ba6794341"
    if (render.get("status") != "passed" or render.get("kind") != "pg"
            or render.get("checkpoint_sha256") != checkpoint_sha
            or render.get("video_sha256") != video_sha
            or render.get("code_hash") != config.get("code_hash") or config.get("code_hash") != expected_hash
            or render.get("video_seed") != config.get("eval_seed")
            or config.get("seed") != 0 or config.get("env_name") != "HalfCheetah-v5"):
        raise ValueError("Render record does not bind this MP4 to the corrected-v2 seed-0 checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(args.video))
    meta = reader.get_meta_data()
    fps = float(meta["fps"])
    length = int(round(meta["duration"] * fps))
    wanted = set(int(value) for value in np.linspace(0, max(0, length - 1), 6))
    selected, frame_count = [], 0
    minimum, maximum = 255, 0
    for index, frame in enumerate(reader):
        assert frame.ndim == 3 and frame.shape[2] == 3 and np.isfinite(frame).all()
        minimum, maximum = min(minimum, int(frame.min())), max(maximum, int(frame.max()))
        if index in wanted:
            selected.append((index, Image.fromarray(frame)))
        frame_count += 1
    reader.close()
    assert frame_count > 1 and len(selected) >= 5 and minimum < maximum
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    canvas = Image.new("RGB", (1020, 760), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), f"HalfCheetah-v5 | training seed 0 | fixed evaluation seed {config['eval_seed']}",
              font=font, fill="#18334a")
    for index, (frame_number, picture) in enumerate(selected):
        picture.thumbnail((330, 320))
        left = 10 + (index % 3) * 340
        top = 50 + (index // 3) * 350
        canvas.paste(picture, (left + (330 - picture.width) // 2, top))
        draw.text((left + 5, top + 322), f"t={frame_number / fps:.2f}s", font=font, fill="#333333")
    sheet = args.output_dir / "representative_seed0_frames.png"
    canvas.save(sheet)
    report = {"status": "decoded_all_frames", "video": str(args.video), "frames": frame_count,
              "fps": fps, "duration_seconds": frame_count / fps, "size": list(meta["size"]),
              "sha256": video_sha, "contact_sheet": str(sheet),
              "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": checkpoint_sha,
              "code_hash": render["code_hash"], "video_seed": render["video_seed"],
              "render_record_sha256": hashlib.sha256(args.render_record.read_bytes()).hexdigest(),
              "contact_sheet_sha256": hashlib.sha256(sheet.read_bytes()).hexdigest(),
              "visual_behavior_review": "requires separate human/model inspection; decoding alone does not prove control quality"}
    (args.output_dir / "video_decode_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
