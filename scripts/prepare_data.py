#!/usr/bin/env python3
"""Create the local ``data/`` tree from a user-provided orbit MP4.

The ``data/`` directory is gitignored. Run this after cloning to reproduce the
layout used by the SfM → annotate → YOLO6D pipeline:

  data/<video>.mp4
  data/frames/          # downscaled (~1600), ~2 fps  — default SfM (run1)
  data/frames_native/   # native resolution, same sampling — HQ path (run2)
  data/extract_meta.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACT = ROOT / "scripts" / "extract_frames.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build data/ from an orbit MP4 (prompts if --video omitted)."
    )
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to the phone orbit MP4. If omitted, you will be prompted.",
    )
    p.add_argument(
        "--data_dir",
        type=Path,
        default=ROOT / "data",
        help="Output data directory (default: <repo>/data).",
    )
    p.add_argument("--fps", type=float, default=2.0, help="Sampling rate for frames.")
    p.add_argument(
        "--max_side",
        type=int,
        default=1600,
        help="Longest side for data/frames (0 = native). Default matches run1.",
    )
    p.add_argument(
        "--skip_native",
        action="store_true",
        help="Do not also extract data/frames_native.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing frames directories / copied video.",
    )
    return p.parse_args()


def prompt_video() -> Path:
    print("请提供绕拍静物的 MP4 视频路径（手机环绕拍摄）。")
    print("示例: /path/to/VID_20260725_165829.mp4")
    raw = input("视频路径: ").strip().strip("'\"")
    if not raw:
        raise SystemExit("未提供视频路径，已退出。")
    return Path(raw).expanduser()


def run_extract(video: Path, out_dir: Path, *, fps: float, max_side: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EXTRACT),
        "--video",
        str(video),
        "--out_dir",
        str(out_dir),
        "--fps",
        str(fps),
        "--max_side",
        str(max_side),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    # extract_frames prints meta JSON; also rewrite a stable summary beside frames
    meta_path = out_dir / "extract_meta.json"
    # Prefer reading stdout meta from a second light probe via OpenCV sizes
    n = len(list(out_dir.glob("frame_*")))
    return {"out_dir": str(out_dir), "n_frames": n, "fps": fps, "max_side": max_side}


def main() -> None:
    args = parse_args()
    video_in = args.video or prompt_video()
    video_in = video_in.resolve()
    if not video_in.is_file():
        raise SystemExit(f"视频不存在: {video_in}")
    if video_in.suffix.lower() != ".mp4":
        print(f"警告: 扩展名是 {video_in.suffix!r}，脚本按 MP4 流程处理。", flush=True)

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    dest_video = data_dir / video_in.name
    if dest_video.resolve() != video_in:
        if dest_video.exists() and not args.force:
            print(f"保留已有: {dest_video}")
        else:
            print(f"复制视频 → {dest_video}")
            shutil.copy2(video_in, dest_video)
    else:
        print(f"视频已在 data 内: {dest_video}")

    frames = data_dir / "frames"
    frames_native = data_dir / "frames_native"
    if frames.exists() and any(frames.iterdir()) and not args.force:
        raise SystemExit(f"{frames} 非空。加 --force 覆盖，或先删掉该目录。")
    if frames.exists() and args.force:
        shutil.rmtree(frames)
    if frames_native.exists() and args.force:
        shutil.rmtree(frames_native)

    info_ds = run_extract(dest_video, frames, fps=args.fps, max_side=args.max_side)
    info_native = None
    if not args.skip_native:
        info_native = run_extract(dest_video, frames_native, fps=args.fps, max_side=0)

    summary = {
        "video": str(dest_video),
        "frames": info_ds,
        "frames_native": info_native,
        "note": "data/ is gitignored; regenerate with scripts/prepare_data.py",
    }
    summary_path = data_dir / "extract_meta.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nDone → {data_dir}")
    print("下一步: bash scripts/run_sfm.sh")


if __name__ == "__main__":
    main()
