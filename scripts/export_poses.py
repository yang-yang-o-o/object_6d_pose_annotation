#!/usr/bin/env python3
"""
Export COLMAP/hloc SfM results:
  - camera intrinsics K
  - per-frame c2w / w2c (world == object until you redefine the frame)
  - colored sparse PLY
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap


def camera_to_K(camera: pycolmap.Camera) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model.name if hasattr(camera.model, "name") else str(camera.model)
    if "SIMPLE_PINHOLE" in model or "SIMPLE_RADIAL" in model:
        f, cx, cy = params[0], params[1], params[2]
        fx = fy = f
    else:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def image_w2c(im) -> np.ndarray:
    # pycolmap>=3.9/4: cam_from_world() -> Rigid3d
    T = np.asarray(im.cam_from_world().matrix(), dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :4] = T[:3, :4]
    return w2c


def export_ply(points: np.ndarray, colors: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sfm_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Metric scale: multiplies translations and 3D points.")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "poses_c2w").mkdir(exist_ok=True)
    (out / "poses_w2c").mkdir(exist_ok=True)
    (out / "obj_in_cam").mkdir(exist_ok=True)

    rec = pycolmap.Reconstruction(str(args.sfm_dir))
    s = float(args.scale)

    cam0 = next(iter(rec.cameras.values()))
    K = camera_to_K(cam0)
    np.savetxt(out / "K.txt", K)
    (out / "camera.json").write_text(
        json.dumps(
            {
                "model": cam0.model.name if hasattr(cam0.model, "name") else str(cam0.model),
                "width": int(cam0.width),
                "height": int(cam0.height),
                "params": list(map(float, cam0.params)),
                "K": K.tolist(),
                "scale": s,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = []
    for iid in sorted(rec.images.keys(), key=lambda i: rec.images[i].name):
        im = rec.images[iid]
        w2c = image_w2c(im)
        w2c[:3, 3] *= s
        c2w = np.linalg.inv(w2c)
        stem = Path(im.name).stem
        np.savetxt(out / "poses_c2w" / f"{stem}.txt", c2w)
        np.savetxt(out / "poses_w2c" / f"{stem}.txt", w2c)
        # Until a custom object frame is defined: world == object
        np.savetxt(out / "obj_in_cam" / f"{stem}.txt", w2c)
        manifest.append({"image": im.name, "image_id": int(iid), "stem": stem})

    pts, cols = [], []
    for _, p in rec.points3D.items():
        pts.append(np.asarray(p.xyz, dtype=np.float64) * s)
        cols.append(np.asarray(p.color, dtype=np.uint8))
    export_ply(np.asarray(pts), np.asarray(cols), out / "sparse.ply")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(manifest)} poses -> {out}")
    print(f"Registered: {rec.num_reg_images()}, points: {rec.num_points3D()}, scale={s}")


if __name__ == "__main__":
    main()
