#!/usr/bin/env python3
"""
Compute metric scale from two 3D points in the sparse reconstruction.

Example:
  # After inspecting sparse.ply in MeshLab / CloudCompare, pick two points
  # that correspond to a known real length (e.g. object height = 0.185 m):
  python scripts/apply_scale.py \\
    --sfm_dir outputs/run1/sfm \\
    --p1 0.12 0.05 -0.30 \\
    --p2 0.12 0.22 -0.30 \\
    --real_length_m 0.185 \\
    --out_dir outputs/run1/metric
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sfm_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--p1", type=float, nargs=3, required=True, help="3D point A in SfM coords")
    p.add_argument("--p2", type=float, nargs=3, required=True, help="3D point B in SfM coords")
    p.add_argument("--real_length_m", type=float, required=True,
                   help="Real-world distance between p1 and p2 in meters")
    return p.parse_args()


def main():
    args = parse_args()
    p1 = np.asarray(args.p1, dtype=np.float64)
    p2 = np.asarray(args.p2, dtype=np.float64)
    d_sfm = float(np.linalg.norm(p2 - p1))
    if d_sfm < 1e-9:
        raise SystemExit("p1 and p2 are identical")
    scale = args.real_length_m / d_sfm
    info = {
        "p1": p1.tolist(),
        "p2": p2.tolist(),
        "sfm_length": d_sfm,
        "real_length_m": args.real_length_m,
        "scale": scale,
    }
    print(json.dumps(info, indent=2))

    # Delegate to export_poses with scale
    import subprocess
    import sys

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "export_poses.py"),
        "--sfm_dir", str(args.sfm_dir),
        "--out_dir", str(args.out_dir),
        "--scale", str(scale),
    ]
    subprocess.check_call(cmd)
    (args.out_dir / "scale.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
