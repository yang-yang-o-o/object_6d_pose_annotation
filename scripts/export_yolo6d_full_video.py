#!/usr/bin/env python3
"""
Dense YOLO6D labels for *all* video frames from keyframe SfM + static object frame.

Background:
  SfM only ran on sampled keyframes (e.g. ~2 fps → 74 frames). The object is static
  in world coordinates (object_frame.json). Camera poses for intermediate frames are
  obtained by interpolating keyframe extrinsics (SLERP + lerp), then projecting the
  same object box.

Example:
  python scripts/export_yolo6d_full_video.py --run outputs/run1

Outputs:
  outputs/run1/yolo6d_full/
    rgb/frame_XXXXXX.jpg   # resized to SfM K resolution
    labels/frame_XXXXXX.txt
    train.txt / test.txt
    poses_w2c/frame_XXXXXX.txt
    densify_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "pose_annotator"))
from export_bridge import obj_to_world, project, write_box_ply  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=ROOT / "outputs/run1")
    p.add_argument("--video", type=Path, default=None,
                   help="Default: data/extract_meta.json → video")
    p.add_argument("--step", type=int, default=None,
                   help="Keyframe stride in source video. Default: extract_meta.step")
    p.add_argument("--every_n", type=int, default=1,
                   help="Export every N-th video frame (1 = all frames)")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Default: <run>/yolo6d_full")
    p.add_argument("--max_frames", type=int, default=0, help="0 = no limit (debug)")
    p.add_argument("--jpeg_quality", type=int, default=92)
    return p.parse_args()


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    R = c2w[:3, :3]
    c = c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ c
    return w2c


def interpolate_w2c(w2c_a: np.ndarray, w2c_b: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate camera pose in c2w (center lerp + rotation SLERP)."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 1e-12:
        return w2c_a.copy()
    if alpha >= 1.0 - 1e-12:
        return w2c_b.copy()
    Ca = w2c_to_c2w(w2c_a)
    Bb = w2c_to_c2w(w2c_b)
    ca, cb = Aa[:3, 3], Bb[:3, 3]
    c = (1.0 - alpha) * ca + alpha * cb
    key_times = [0.0, 1.0]
    key_rots = Rotation.from_matrix([Aa[:3, :3], Bb[:3, :3]])
    slerp = Slerp(key_times, key_rots)
    R = slerp([alpha]).as_matrix()[0]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3] = c
    return c2w_to_w2c(c2w)


def label_line(class_id: int, uv: np.ndarray, W: int, H: int) -> str:
    uvn = uv.copy()
    uvn[:, 0] /= W
    uvn[:, 1] /= H
    xs, ys = uvn[:, 0], uvn[:, 1]
    vals = [float(class_id)]
    for i in range(9):
        vals.extend([float(xs[i]), float(ys[i])])
    vals.extend([float(xs.max() - xs.min()), float(ys.max() - ys.min())])
    return " ".join(f"{v:.6f}" for v in vals)


def main():
    args = parse_args()
    run = args.run.resolve()
    ann = run / "annotator"
    scene = json.loads((ann / "scene.json").read_text(encoding="utf-8"))
    frame_obj = json.loads((ann / "object_frame.json").read_text(encoding="utf-8"))

    meta_path = ROOT / "data" / "extract_meta.json"
    extract_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    video = Path(args.video) if args.video else Path(extract_meta.get("video", ROOT / "data" / "VID_20260725_165829.mp4"))
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    step = int(args.step if args.step is not None else extract_meta.get("step", 15))
    every_n = max(1, int(args.every_n))

    K = np.asarray(scene["K"], dtype=np.float64)
    W, H = int(scene["width"]), int(scene["height"])
    key_w2c = [np.asarray(fr["w2c"], dtype=np.float64) for fr in scene["frames"]]
    n_kf = len(key_w2c)
    if n_kf < 2:
        raise SystemExit("Need ≥2 keyframes to interpolate")

    _, _, pts9_w = obj_to_world(frame_obj)
    class_id = int(frame_obj.get("class_id", 0))

    out = args.out_dir or (run / "yolo6d_full")
    out = out.resolve()
    rgb_dir = out / "rgb"
    lab_dir = out / "labels"
    pose_dir = out / "poses_w2c"
    for d in (rgb_dir, lab_dir, pose_dir):
        d.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # Keyframe k ↔ source video index k * step (same as extract_frames.py)
    max_src_needed = (n_kf - 1) * step
    if max_src_needed >= n_video:
        print(f"WARN: last keyframe maps to src {max_src_needed} but video has {n_video} frames")

    train_list = []
    n_ok = 0
    n_skip = 0
    v = 0
    pbar = tqdm(total=n_video, desc="dense export")
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if v % every_n != 0:
            v += 1
            pbar.update(1)
            continue

        # interpolate between surrounding keyframes
        k0 = int(np.clip(v // step, 0, n_kf - 1))
        k1 = min(k0 + 1, n_kf - 1)
        t0 = k0 * step
        t1 = k1 * step
        if k1 == k0 or t1 == t0:
            alpha = 0.0
        else:
            alpha = (v - t0) / float(t1 - t0)
            # beyond last keyframe: hold last pose
            if v > (n_kf - 1) * step:
                alpha = 1.0
                k0 = k1 = n_kf - 1
        w2c = interpolate_w2c(key_w2c[k0], key_w2c[k1], alpha)

        uv, z = project(K, w2c, pts9_w)
        if z[0] <= 0:
            n_skip += 1
            v += 1
            pbar.update(1)
            continue

        # resize to SfM calibration size
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        stem = f"frame_{v:06d}"
        cv2.imwrite(
            str(rgb_dir / f"{stem}.jpg"),
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
        )
        (lab_dir / f"{stem}.txt").write_text(
            label_line(class_id, uv, W, H) + "\n", encoding="utf-8"
        )
        np.savetxt(pose_dir / f"{stem}.txt", w2c)
        train_list.append(f"rgb/{stem}.jpg")
        n_ok += 1

        v += 1
        pbar.update(1)
        if args.max_frames > 0 and n_ok >= args.max_frames:
            break
    pbar.close()
    cap.release()

    (out / "train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8")
    (out / "test.txt").write_text("\n".join(train_list[::10]) + "\n", encoding="utf-8")
    (out / "object_frame.json").write_text(
        json.dumps(frame_obj, indent=2), encoding="utf-8"
    )
    write_box_ply(out / "object.ply", frame_obj["size"])

    densify_meta = {
        "method": "keyframe_pose_interpolation",
        "note": "Object is static in world; camera poses between SfM keyframes are interpolated (SLERP). "
                "For higher accuracy on all frames, run image localization against the SfM model instead.",
        "video": str(video.resolve()),
        "src_size": [src_w, src_h],
        "src_fps": src_fps,
        "n_video_frames": n_video,
        "n_keyframes": n_kf,
        "keyframe_step": step,
        "every_n": every_n,
        "export_size": [W, H],
        "n_labels": n_ok,
        "n_skipped_behind": n_skip,
        "run": str(run),
        "object_frame": str(ann / "object_frame.json"),
    }
    (out / "densify_meta.json").write_text(json.dumps(densify_meta, indent=2), encoding="utf-8")
    print(json.dumps(densify_meta, indent=2))
    print(f"\nWrote {n_ok} labels → {out}")


if __name__ == "__main__":
    main()
