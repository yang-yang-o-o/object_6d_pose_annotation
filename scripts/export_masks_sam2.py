#!/usr/bin/env python3
"""
Export YOLO6D masks with Ultralytics SAM2 **temporal** propagation + denoise.

Uses interactive dumps under ``annotator/masks/{stem}.png`` as memory seeds
(``SAM2DynamicInteractivePredictor``), propagates along time order, and cleans
each mask (largest CC, fill holes, light morphological close).

Example:
  unset VIRTUAL_ENV
  export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  uv run python scripts/export_masks_sam2.py --run outputs/run1
  uv run python scripts/export_masks_sam2.py --run outputs/run1 --targets yolo6d
  uv run python scripts/export_masks_sam2.py --run outputs/run1 --max_frames 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "pose_annotator"))

from export_bridge import ensure_yolo6d_mask_preview, write_mask_png  # noqa: E402
from segment import clean_binary_mask, propagate_masks_temporal  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "outputs/run1")
    p.add_argument(
        "--targets",
        nargs="+",
        default=["yolo6d", "yolo6d_full"],
        choices=["yolo6d", "yolo6d_full"],
    )
    p.add_argument("--max_frames", type=int, default=0, help="0 = all")
    p.add_argument(
        "--clean_interactive",
        action="store_true",
        default=True,
        help="Rewrite annotator/masks/*.png with denoise (default on)",
    )
    p.add_argument("--no_clean_interactive", action="store_false", dest="clean_interactive")
    return p.parse_args()


def clean_interactive_dumps(mask_dir: Path) -> dict:
    if not mask_dir.is_dir():
        return {"n": 0}
    n = 0
    stats = []
    for p in sorted(mask_dir.glob("*.png")):
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        before = cv2.connectedComponents((raw > 127).astype(np.uint8), 8)[0] - 1
        cleaned = clean_binary_mask(raw)
        after = cv2.connectedComponents((cleaned > 127).astype(np.uint8), 8)[0] - 1
        cv2.imwrite(str(p), cleaned)
        # also refresh last_mask to latest cleaned file name-wise: skip
        stats.append({"stem": p.stem, "components": [before, after]})
        n += 1
    return {"n": n, "files": stats}


def load_seed_map(interactive_dir: Path) -> dict[str, np.ndarray]:
    seeds: dict[str, np.ndarray] = {}
    if not interactive_dir.is_dir():
        return seeds
    for p in interactive_dir.glob("*.png"):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        seeds[p.stem] = clean_binary_mask(m)
    return seeds


def export_yolo6d_temporal(run: Path, scene: dict, *, max_frames: int = 0) -> dict:
    yolo = run / "yolo6d"
    if not yolo.is_dir():
        return {"target": "yolo6d", "n": 0, "skipped": "missing yolo6d/"}
    interactive = run / "annotator" / "masks"
    seeds = load_seed_map(interactive)
    frames = []
    for fr in scene["frames"]:
        stem = fr["stem"]
        if not (yolo / "labels" / f"{stem}.txt").exists():
            continue
        img = Path(fr["image"])
        rgb = yolo / "rgb" / Path(fr["image"]).name
        image = rgb if rgb.exists() else img
        frames.append(
            {
                "stem": stem,
                "image": image,
                "seed_mask": seeds.get(stem),
            }
        )
    if max_frames > 0:
        frames = frames[:max_frames]
    if not any(f["seed_mask"] is not None for f in frames):
        return {
            "target": "yolo6d",
            "n": 0,
            "error": "no interactive seeds in annotator/masks/",
        }

    # Ensure first seed appears early: if first frames have no seed, start from first seed index
    first_seed = next(i for i, f in enumerate(frames) if f["seed_mask"] is not None)
    frames = frames[first_seed:]

    print(f"[yolo6d] temporal propagate {len(frames)} frames, seeds={sum(f['seed_mask'] is not None for f in frames)}")
    masks = propagate_masks_temporal(frames)
    mask_dir = yolo / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stem, m in tqdm(masks.items(), desc="write yolo6d masks"):
        write_mask_png(mask_dir / f"{stem}.png", m)
    preview = None
    try:
        preview = ensure_yolo6d_mask_preview(yolo, force=True)
    except Exception as e:
        print(f"[preview] yolo6d preview_mask.mp4 failed: {e}")
    return {
        "target": "yolo6d",
        "n": len(masks),
        "n_seeds": sum(f["seed_mask"] is not None for f in frames),
        "method": "sam2_temporal+clean",
        "out": str(mask_dir),
        "preview_mask": str(preview) if preview else None,
    }


def export_yolo6d_full_temporal(run: Path, scene: dict, *, max_frames: int = 0) -> dict:
    full = run / "yolo6d_full"
    rgb_dir = full / "rgb"
    pose_dir = full / "poses_w2c"
    if not pose_dir.is_dir():
        return {"target": "yolo6d_full", "n": 0, "skipped": "missing poses_w2c/"}

    step = 15
    extract_meta_path = ROOT / "data" / "extract_meta.json"
    if extract_meta_path.exists():
        step = int(json.loads(extract_meta_path.read_text(encoding="utf-8")).get("step", step))
    meta_path = full / "localize_meta.json"
    if meta_path.exists():
        step = int(json.loads(meta_path.read_text(encoding="utf-8")).get("keyframe_step", step))

    seeds_kf = load_seed_map(run / "annotator" / "masks")
    # Map keyframe stem frame_000037 → video index 37*step
    seed_by_video: dict[int, np.ndarray] = {}
    for stem, m in seeds_kf.items():
        try:
            ki = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        seed_by_video[ki * step] = m

    poses = sorted(pose_dir.glob("*.txt"))
    frames = []
    for pose_path in poses:
        stem = pose_path.stem  # full_000555
        try:
            vid = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        img = rgb_dir / f"{stem}.jpg"
        if not img.exists():
            img = rgb_dir / f"{stem}.png"
        frames.append(
            {
                "stem": stem,
                "image": img,
                "seed_mask": seed_by_video.get(vid),
                "vid": vid,
            }
        )
    frames.sort(key=lambda x: x["vid"])
    if max_frames > 0:
        frames = frames[:max_frames]
    if not any(f["seed_mask"] is not None for f in frames):
        return {
            "target": "yolo6d_full",
            "n": 0,
            "error": "no interactive seeds mapped onto full_* frames",
        }
    first_seed = next(i for i, f in enumerate(frames) if f["seed_mask"] is not None)
    frames = frames[first_seed:]

    print(
        f"[yolo6d_full] temporal propagate {len(frames)} frames, "
        f"seeds={sum(f['seed_mask'] is not None for f in frames)}, step={step}"
    )
    masks = propagate_masks_temporal(frames)
    mask_dir = full / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for stem, m in tqdm(masks.items(), desc="write yolo6d_full masks"):
        write_mask_png(mask_dir / f"{stem}.png", m)
    preview = None
    try:
        preview = ensure_yolo6d_mask_preview(full, force=True)
    except Exception as e:
        print(f"[preview] yolo6d_full preview_mask.mp4 failed: {e}")
    return {
        "target": "yolo6d_full",
        "n": len(masks),
        "n_seeds": sum(f["seed_mask"] is not None for f in frames),
        "method": "sam2_temporal+clean",
        "keyframe_step": step,
        "out": str(mask_dir),
        "preview_mask": str(preview) if preview else None,
    }


def main():
    args = parse_args()
    run = args.run.resolve()
    ann = run / "annotator"
    scene = json.loads((ann / "scene.json").read_text(encoding="utf-8"))
    interactive = ann / "masks"

    if args.clean_interactive:
        info = clean_interactive_dumps(interactive)
        print(f"Cleaned interactive dumps: {info['n']}")
        for row in info.get("files", []):
            b, a = row["components"]
            if b != a:
                print(f"  {row['stem']}: components {b} → {a}")

    reports = []
    if "yolo6d" in args.targets:
        reports.append(export_yolo6d_temporal(run, scene, max_frames=args.max_frames))
    if "yolo6d_full" in args.targets:
        reports.append(export_yolo6d_full_temporal(run, scene, max_frames=args.max_frames))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
