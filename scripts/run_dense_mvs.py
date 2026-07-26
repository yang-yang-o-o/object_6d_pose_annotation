#!/usr/bin/env python3
"""Dense multi-view stereo (COLMAP MVS) on top of an existing sparse SfM model."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd):
    print(">>", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(cmd)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--sfm_dir", type=Path, required=True)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--max_image_size", type=int, default=1600)
    p.add_argument("--skip_stereo", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ws = args.workspace
    sparse = ws / "sparse" / "0"
    dense = ws / "dense"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True, exist_ok=True)

    copied = False
    for name in ("cameras.bin", "images.bin", "points3D.bin",
                 "cameras.txt", "images.txt", "points3D.txt"):
        src = args.sfm_dir / name
        if src.exists():
            shutil.copy2(src, sparse / name)
            copied = True
    if not copied:
        raise SystemExit(f"No COLMAP model files found in {args.sfm_dir}")

    if dense.exists():
        shutil.rmtree(dense)
    dense.mkdir(parents=True, exist_ok=True)

    run([
        "colmap", "image_undistorter",
        "--image_path", str(args.images),
        "--input_path", str(sparse),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
        "--max_image_size", str(args.max_image_size),
    ])
    if args.skip_stereo:
        print("Undistorted only; skipped patch_match_stereo")
        return

    # CPU COLMAP (apt package has no CUDA) — still OK, just slower
    run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ])
    run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense / "fused.ply"),
    ])
    print(f"Dense point cloud: {dense / 'fused.ply'}")


if __name__ == "__main__":
    main()
