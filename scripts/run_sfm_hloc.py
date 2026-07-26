#!/usr/bin/env python3
"""
High-quality object SfM using hloc-style deep features (SuperPoint + LightGlue)
and COLMAP triangulation/BA via pycolmap.

Inspired by OnePose / OnePose++ (deep matching + COLMAP), but works without
ARKit: poses come from SfM; metric scale is applied later.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--matcher", choices=["superpoint+lightglue", "disk+lightglue", "superpoint+superglue"],
                   default="superpoint+lightglue")
    p.add_argument("--max_keypoints", type=int, default=4096,
                   help="SuperPoint max keypoints.")
    p.add_argument("--resize_max", type=int, default=1600,
                   help="Feature extraction longest-side (keypoints scaled back to full image).")
    p.add_argument("--n_match_neighbors", type=int, default=40,
                   help="Each image matches to this many sequential neighbors (orbit-friendly).")
    p.add_argument("--exhaustive", action="store_true",
                   help="Exhaustive matching (slower, better for short sequences).")
    return p.parse_args()


def build_pairs(image_names: list[str], n_neighbors: int, exhaustive: bool):
    n = len(image_names)
    pairs = []
    if exhaustive:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((image_names[i], image_names[j]))
    else:
        # Sequential + wrap-around for closed orbits
        for i in range(n):
            for d in range(1, n_neighbors + 1):
                j = (i + d) % n
                a, b = sorted([image_names[i], image_names[j]])
                pairs.append((a, b))
        pairs = sorted(set(pairs))
    return pairs


def main():
    args = parse_args()
    # Ensure local hloc is importable if vendored
    repo_root = Path(__file__).resolve().parents[1]
    hloc_root = repo_root / "third_party" / "Hierarchical-Localization"
    if hloc_root.is_dir():
        sys.path.insert(0, str(hloc_root))

    from hloc import extract_features, match_features, reconstruction
    import pycolmap

    images = args.images.resolve()
    outputs = args.outputs.resolve()
    outputs.mkdir(parents=True, exist_ok=True)

    image_names = sorted(
        [p.name for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if len(image_names) < 3:
        raise SystemExit(f"Need >=3 images, found {len(image_names)} in {images}")

    print(f"Images: {len(image_names)} in {images}", flush=True)
    print(f"Config: matcher={args.matcher} max_keypoints={args.max_keypoints} "
          f"resize_max={args.resize_max}", flush=True)

    if args.matcher == "superpoint+lightglue":
        feature_conf = extract_features.confs["superpoint_max"]
        feature_conf["model"]["max_keypoints"] = args.max_keypoints
        feature_conf["preprocessing"]["resize_max"] = args.resize_max
        feature_conf["output"] = f"feats-superpoint-n{args.max_keypoints}-rmax{args.resize_max}"
        matcher_conf = match_features.confs["superpoint+lightglue"]
    elif args.matcher == "disk+lightglue":
        feature_conf = extract_features.confs["disk"]
        feature_conf["preprocessing"]["resize_max"] = args.resize_max
        matcher_conf = match_features.confs["disk+lightglue"]
    else:
        feature_conf = extract_features.confs["superpoint_aachen"]
        feature_conf["model"]["max_keypoints"] = args.max_keypoints
        feature_conf["preprocessing"]["resize_max"] = args.resize_max
        matcher_conf = match_features.confs["superglue"]

    sfm_pairs = outputs / "pairs-sfm.txt"
    if args.exhaustive or len(image_names) <= 80:
        # Exhaustive is fine for ~70-80 frames
        pairs = [(image_names[i], image_names[j])
                 for i in range(len(image_names))
                 for j in range(i + 1, len(image_names))]
        print(f"Exhaustive pairs: {len(pairs)}", flush=True)
    else:
        pairs = build_pairs(image_names, args.n_match_neighbors, exhaustive=False)
        print(f"Sequential pairs: {len(pairs)} (neighbors={args.n_match_neighbors})", flush=True)

    with open(sfm_pairs, "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")

    print(f"\n[1/3] Feature extraction → {feature_conf['output']}", flush=True)
    t0 = __import__("time").time()
    feature_path = extract_features.main(
        feature_conf, images, outputs, as_half=True
    )
    print(f"[1/3] Features done in {__import__('time').time()-t0:.1f}s → {feature_path}", flush=True)

    print(f"\n[2/3] Matching {len(pairs)} pairs ({args.matcher})", flush=True)
    t0 = __import__("time").time()
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )
    print(f"[2/3] Matching done in {__import__('time').time()-t0:.1f}s → {match_path}", flush=True)

    sfm_dir = outputs / "sfm"
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)

    print("\n[3/3] COLMAP reconstruction / BA ...", flush=True)
    t0 = __import__("time").time()
    # Prefer OPENCV camera model for phone videos (radial distortion)
    image_options = pycolmap.ImageReaderOptions()
    # Older/newer pycolmap API differences
    try:
        mapper_options = {
            "min_model_size": 10,
            "min_num_matches": 15,
        }
        model = reconstruction.main(
            sfm_dir,
            images,
            sfm_pairs,
            feature_path,
            match_path,
            camera_mode=pycolmap.CameraMode.SINGLE,
            image_options={"camera_model": "OPENCV"},
            mapper_options=mapper_options,
        )
    except TypeError:
        model = reconstruction.main(
            sfm_dir,
            images,
            sfm_pairs,
            feature_path,
            match_path,
            camera_mode=pycolmap.CameraMode.SINGLE,
        )
    print(f"[3/3] Reconstruction done in {__import__('time').time()-t0:.1f}s", flush=True)

    if model is None:
        raise SystemExit("SfM failed: no reconstruction")

    # pycolmap Reconstruction may be returned directly or written to disk
    if isinstance(model, dict):
        # multiple models — pick largest
        best_id = max(model.keys(), key=lambda k: model[k].num_reg_images())
        rec = model[best_id]
        print(f"Selected model {best_id}: {rec.num_reg_images()} images, {rec.num_points3D()} points")
    else:
        rec = model
        print(f"Reconstruction: {rec.num_reg_images()} images, {rec.num_points3D()} points")

    summary = {
        "n_images_input": len(image_names),
        "n_images_registered": int(rec.num_reg_images()),
        "n_points3D": int(rec.num_points3D()),
        "matcher": args.matcher,
        "sfm_dir": str(sfm_dir),
    }
    (outputs / "sfm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
