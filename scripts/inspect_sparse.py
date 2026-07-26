#!/usr/bin/env python3
"""Print sparse cloud AABB / diameter to help pick scale reference points."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_ply_xyz(path: Path):
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue
            x, y, z, *_ = line.split()
            pts.append((float(x), float(y), float(z)))
    return np.asarray(pts, dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=Path, required=True)
    args = p.parse_args()
    xyz = load_ply_xyz(args.ply)
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    c = 0.5 * (mn + mx)
    print(f"n_points={len(xyz)}")
    print(f"min={mn}")
    print(f"max={mx}")
    print(f"center={c}")
    print(f"extent (max-min)={mx - mn}")
    print(f"bbox diagonal={np.linalg.norm(mx - mn):.6f} (SfM units)")
    print("Open sparse.ply in MeshLab, pick two endpoints of a known real length,")
    print("then run scripts/apply_scale.py with those coords and --real_length_m.")


if __name__ == "__main__":
    main()
