#!/usr/bin/env python3
"""
Localize every video frame against the run's SfM model (scheme 2), then export YOLO6D.

Pipeline:
  1) Extract video frames at SfM resolution (same W×H / intrinsics family)
  2) SuperPoint features on queries
  3) Temporal retrieval: each query ↔ nearest N keyframes
  4) LightGlue match query↔keyframes
  5) hloc absolute pose (PnP+refine) vs COLMAP model
  6) Apply metric_scale from annotator; project static object_frame → YOLO6D

Example:
  python scripts/localize_full_video.py --run outputs/run1

Outputs:
  outputs/run1/yolo6d_full/
    rgb/ full_XXXXXX.jpg
    labels/ full_XXXXXX.txt
    poses_w2c/ full_XXXXXX.txt   # metric
    localize_meta.json
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
sys.path.insert(0, str(ROOT / "third_party" / "Hierarchical-Localization"))
sys.path.insert(0, str(ROOT / "tools" / "pose_annotator"))

from export_bridge import obj_to_world, project, write_box_ply  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=ROOT / "outputs/run1")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--step", type=int, default=None, help="Keyframe stride in source video")
    p.add_argument("--every_n", type=int, default=1, help="Localize every N-th video frame")
    p.add_argument("--n_ref", type=int, default=5, help="Nearest keyframes to match per query")
    p.add_argument("--max_frames", type=int, default=0, help="0 = all (debug limit)")
    p.add_argument("--skip_extract", action="store_true", help="Reuse existing loc_query/ + features")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--jpeg_quality", type=int, default=92)
    return p.parse_args()


def camera_to_K(camera) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model.name if hasattr(camera.model, "name") else str(camera.model)
    if "SIMPLE_PINHOLE" in model or "SIMPLE_RADIAL" in model:
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    else:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def rigid3d_to_w2c(T) -> np.ndarray:
    """pycolmap Rigid3d cam_from_world → 4x4 w2c."""
    w2c = np.eye(4, dtype=np.float64)
    # rotation.matrix() is R such that Xc = R Xw + t
    if hasattr(T, "matrix"):
        M = np.asarray(T.matrix(), dtype=np.float64)
        w2c[:3, :4] = M[:3, :4]
    else:
        R = np.asarray(T.rotation.matrix(), dtype=np.float64)
        t = np.asarray(T.translation, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
    return w2c


def parse_pose_results(path: Path) -> dict[str, np.ndarray]:
    """Parse hloc write_poses text: name qw qx qy qz tx ty tz → w2c 4x4."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        name = parts[0]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        # COLMAP qvec is wxyz; scipy / matrix from hamilton
        # R from q: same as pycolmap / COLMAP
        w, x, y, z = qw, qx, qy, qz
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = [tx, ty, tz]
        out[name] = w2c
    return out


def extract_query_frames(
    video: Path,
    out_dir: Path,
    W: int,
    H: int,
    every_n: int,
    max_frames: int,
    jpeg_quality: int,
) -> list[tuple[int, str]]:
    """Returns list of (video_index, image_name)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mapping = []
    v = 0
    pbar = tqdm(total=n_total, desc="extract query frames")
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if v % every_n == 0:
            if img.shape[1] != W or img.shape[0] != H:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            name = f"full_{v:06d}.jpg"
            cv2.imwrite(
                str(out_dir / name),
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            mapping.append((v, name))
            if max_frames > 0 and len(mapping) >= max_frames:
                v += 1
                pbar.update(1)
                break
        v += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return mapping


def nearest_keyframes(v: int, step: int, n_kf: int, n_ref: int) -> list[int]:
    """Keyframe indices (0..n_kf-1) nearest to video frame v."""
    center = int(np.clip(round(v / float(step)), 0, n_kf - 1))
    # gather by distance
    idxs = list(range(n_kf))
    idxs.sort(key=lambda k: abs(k * step - v))
    return sorted(idxs[:n_ref])


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
    sfm_dir = run / "sfm"
    if not sfm_dir.exists():
        raise SystemExit(f"Missing SfM model: {sfm_dir}")

    scene = json.loads((ann / "scene.json").read_text(encoding="utf-8"))
    frame_obj = json.loads((ann / "object_frame.json").read_text(encoding="utf-8"))
    scale_info = {}
    if (ann / "scale.json").exists():
        scale_info = json.loads((ann / "scale.json").read_text(encoding="utf-8"))
    metric_scale = float(
        scale_info.get("metric_scale_cumulative")
        or scene.get("metric_scale")
        or frame_obj.get("metric_scale")
        or 1.0
    )

    extract_meta_path = ROOT / "data" / "extract_meta.json"
    extract_meta = (
        json.loads(extract_meta_path.read_text(encoding="utf-8"))
        if extract_meta_path.exists()
        else {}
    )
    video = Path(args.video) if args.video else Path(
        extract_meta.get("video", ROOT / "data" / "VID_20260725_165829.mp4")
    )
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    step = int(args.step if args.step is not None else extract_meta.get("step", 15))

    import pycolmap
    from hloc import extract_features, localize_sfm, match_features

    rec = pycolmap.Reconstruction(str(sfm_dir))
    cam0 = next(iter(rec.cameras.values()))
    W, H = int(cam0.width), int(cam0.height)
    K = camera_to_K(cam0)
    # Prefer annotator K if present (should match)
    if "K" in scene:
        K = np.asarray(scene["K"], dtype=np.float64)
        W, H = int(scene["width"]), int(scene["height"])

    kf_names = [rec.images[i].name for i in sorted(rec.images.keys(), key=lambda i: rec.images[i].name)]
    # Ensure sorted by frame index
    kf_names = sorted(kf_names, key=lambda n: int(Path(n).stem.split("_")[-1]))
    n_kf = len(kf_names)
    print(f"SfM keyframes: {n_kf}  camera {W}x{H}  metric_scale={metric_scale:.6g}")

    loc_dir = run / "localize"
    query_dir = loc_dir / "query_images"
    loc_dir.mkdir(parents=True, exist_ok=True)

    map_path = loc_dir / "query_index.json"
    if args.skip_extract and map_path.exists():
        mapping = [(int(a), b) for a, b in json.loads(map_path.read_text())]
        print(f"Reuse {len(mapping)} query frames from {query_dir}")
    else:
        mapping = extract_query_frames(
            video,
            query_dir,
            W,
            H,
            every_n=max(1, args.every_n),
            max_frames=args.max_frames,
            jpeg_quality=args.jpeg_quality,
        )
        map_path.write_text(json.dumps(mapping), encoding="utf-8")
        print(f"Extracted {len(mapping)} query frames → {query_dir}")

    query_names = [name for _, name in mapping]

    # --- features ---
    feature_conf = extract_features.confs["superpoint_max"]
    feature_conf["model"]["max_keypoints"] = 4096
    feature_conf["preprocessing"]["resize_max"] = 1600
    feature_conf["output"] = "feats-query-superpoint-n4096-rmax1600"
    feats_ref = run / "feats-superpoint-n4096-rmax1600.h5"
    if not feats_ref.exists():
        raise SystemExit(f"Missing reference features: {feats_ref}")

    print("\n[1/4] Extract query SuperPoint features …", flush=True)
    feats_q = extract_features.main(
        feature_conf,
        query_dir,
        loc_dir,
        as_half=True,
        overwrite=False,
    )

    # --- temporal pairs + retrieval file ---
    pairs_path = loc_dir / "pairs-loc.txt"
    retrieval_path = loc_dir / "retrieval.txt"
    with open(pairs_path, "w", encoding="utf-8") as fp, open(
        retrieval_path, "w", encoding="utf-8"
    ) as fr:
        for v, qname in mapping:
            refs = nearest_keyframes(v, step, n_kf, args.n_ref)
            for ki in refs:
                rname = kf_names[ki]
                fp.write(f"{qname} {rname}\n")
                fr.write(f"{qname} {rname}\n")
    print(f"[1/4] Pairs: {sum(1 for _ in open(pairs_path))} (n_ref={args.n_ref})")

    # --- match ---
    print("\n[2/4] LightGlue matching query↔keyframes …", flush=True)
    matcher_conf = match_features.confs["superpoint+lightglue"]
    matches_path = loc_dir / "matches-loc.h5"
    match_features.main(
        matcher_conf,
        pairs_path,
        features=feats_q,
        matches=matches_path,
        features_ref=feats_ref,
        overwrite=False,
    )

    # --- query list with intrinsics ---
    model_name = cam0.model.name if hasattr(cam0.model, "name") else str(cam0.model)
    # pycolmap may print CameraModelId.OPENCV
    if "OPENCV" in model_name:
        model_name = "OPENCV"
    elif "SIMPLE_RADIAL" in model_name:
        model_name = "SIMPLE_RADIAL"
    params = " ".join(str(float(x)) for x in cam0.params)
    queries_path = loc_dir / "queries_with_intrinsics.txt"
    with open(queries_path, "w", encoding="utf-8") as f:
        for name in query_names:
            f.write(f"{name} {model_name} {W} {H} {params}\n")

    # --- localize ---
    print("\n[3/4] Absolute pose localization …", flush=True)
    results_path = loc_dir / "poses.txt"
    localize_sfm.main(
        reference_sfm=sfm_dir,
        queries=queries_path,
        retrieval=retrieval_path,
        features=feats_q,
        matches=matches_path,
        results=results_path,
        ransac_thresh=12,
        covisibility_clustering=False,
    )

    poses_sfm = parse_pose_results(results_path)
    print(f"Localized {len(poses_sfm)} / {len(query_names)}")

    # Keyframe exact poses from reconstruction (SfM units) as fallback / exact hits
    kf_w2c_sfm = {}
    for im in rec.images.values():
        kf_w2c_sfm[im.name] = rigid3d_to_w2c(im.cam_from_world())

    _, _, pts9_w = obj_to_world(frame_obj)  # metric
    class_id = int(frame_obj.get("class_id", 0))

    out = (args.out_dir or (run / "yolo6d_full")).resolve()
    rgb_dir = out / "rgb"
    lab_dir = out / "labels"
    pose_dir = out / "poses_w2c"
    for d in (rgb_dir, lab_dir, pose_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("\n[4/4] Export YOLO6D labels …", flush=True)
    train_list = []
    n_ok = 0
    n_fail = 0
    n_from_kf = 0
    failed = []

    for v, qname in tqdm(mapping, desc="export"):
        # Prefer exact keyframe pose when video index lands on a keyframe
        k_exact = v // step if (v % step == 0 and (v // step) < n_kf) else None
        w2c_sfm = None
        src = "loc"
        if k_exact is not None:
            w2c_sfm = kf_w2c_sfm[kf_names[k_exact]]
            src = "keyframe"
            n_from_kf += 1
        elif qname in poses_sfm:
            w2c_sfm = poses_sfm[qname]
        else:
            # fallback: nearest keyframe pose
            ki = int(np.clip(round(v / float(step)), 0, n_kf - 1))
            w2c_sfm = kf_w2c_sfm[kf_names[ki]]
            src = "fallback_kf"
            n_fail += 1
            failed.append({"frame": v, "name": qname, "fallback": kf_names[ki]})

        w2c = w2c_sfm.copy()
        w2c[:3, 3] *= metric_scale

        uv, z = project(K, w2c, pts9_w)
        if z[0] <= 0:
            n_fail += 1
            failed.append({"frame": v, "name": qname, "reason": "behind_camera", "src": src})
            continue

        # copy/link rgb
        src_img = query_dir / qname
        dst_img = rgb_dir / qname
        if not dst_img.exists():
            import shutil

            shutil.copy2(src_img, dst_img)

        stem = Path(qname).stem
        (lab_dir / f"{stem}.txt").write_text(
            label_line(class_id, uv, W, H) + "\n", encoding="utf-8"
        )
        np.savetxt(pose_dir / f"{stem}.txt", w2c)
        train_list.append(f"rgb/{qname}")
        n_ok += 1

    (out / "train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8")
    (out / "test.txt").write_text("\n".join(train_list[::10]) + "\n", encoding="utf-8")
    (out / "object_frame.json").write_text(
        json.dumps(frame_obj, indent=2), encoding="utf-8"
    )
    write_box_ply(out / "object.ply", frame_obj["size"])

    meta = {
        "method": "hloc_localize_against_sfm",
        "video": str(video.resolve()),
        "sfm_dir": str(sfm_dir),
        "n_keyframes": n_kf,
        "keyframe_step": step,
        "every_n": args.every_n,
        "n_ref": args.n_ref,
        "metric_scale": metric_scale,
        "export_size": [W, H],
        "n_queries": len(mapping),
        "n_labels": n_ok,
        "n_from_keyframe_exact": n_from_kf,
        "n_failed_or_fallback": n_fail,
        "localized_raw": len(poses_sfm),
        "failed": failed[:50],
        "out_dir": str(out),
    }
    (out / "localize_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (loc_dir / "localize_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "failed"}, indent=2))
    print(f"\nDone → {out}  ({n_ok} labels)")


if __name__ == "__main__":
    main()
