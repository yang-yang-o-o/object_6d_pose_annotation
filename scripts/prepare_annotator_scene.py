#!/usr/bin/env python3
"""
Pack a run's export into a browser-friendly annotator scene:
  - downsampled colored points (binary)
  - camera K + w2c poses
  - image list pointing at data/frames
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


_TYPE_MAP = {
    "float": ("<f4", 4),
    "float32": ("<f4", 4),
    "double": ("<f8", 8),
    "float64": ("<f8", 8),
    "uchar": ("u1", 1),
    "uint8": ("u1", 1),
    "char": ("i1", 1),
    "int": ("<i4", 4),
    "int32": ("<i4", 4),
    "uint": ("<u4", 4),
    "ushort": ("<u2", 2),
    "short": ("<i2", 2),
}


def read_ply_xyzrgb(path: Path, max_points: int = 250_000):
    with open(path, "rb") as f:
        n = None
        fmt = "ascii"
        props = []  # (name, ply_type)
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("format"):
                fmt = "binary" if "binary" in line else "ascii"
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.startswith("property"):
                parts = line.split()
                # property <type> <name>  OR property list ...
                if parts[1] == "list":
                    raise SystemExit(f"Unsupported list property in {path}")
                props.append((parts[-1], parts[1]))
            if line == "end_header":
                break
        if n is None or not props:
            raise SystemExit(f"Bad PLY: {path}")

        names = [p[0] for p in props]
        if not all(c in names for c in ("x", "y", "z")):
            raise SystemExit(f"PLY missing xyz: {path}")

        if fmt == "ascii":
            pts, cols = [], []
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
            has_rgb = all(c in names for c in ("red", "green", "blue"))
            ir = names.index("red") if has_rgb else -1
            ig = names.index("green") if has_rgb else -1
            ib = names.index("blue") if has_rgb else -1
            for _ in range(n):
                parts = f.readline().decode().split()
                pts.append([float(parts[ix]), float(parts[iy]), float(parts[iz])])
                if has_rgb:
                    cols.append([int(parts[ir]), int(parts[ig]), int(parts[ib])])
                else:
                    cols.append([200, 200, 200])
            xyz = np.asarray(pts, dtype=np.float32)
            rgb = np.asarray(cols, dtype=np.uint8)
        else:
            dtype_fields = []
            for i, (name, ply_t) in enumerate(props):
                if ply_t not in _TYPE_MAP:
                    raise SystemExit(f"Unsupported PLY type {ply_t} for {name}")
                np_t, _ = _TYPE_MAP[ply_t]
                dtype_fields.append((f"f{i}_{name}", np_t))
            dtype = np.dtype(dtype_fields)
            data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
            xyz = np.stack(
                [data[f"f{names.index('x')}_x"],
                 data[f"f{names.index('y')}_y"],
                 data[f"f{names.index('z')}_z"]],
                1,
            ).astype(np.float32)
            if all(c in names for c in ("red", "green", "blue")):
                rgb = np.stack(
                    [data[f"f{names.index('red')}_red"],
                     data[f"f{names.index('green')}_green"],
                     data[f"f{names.index('blue')}_blue"]],
                    1,
                ).astype(np.uint8)
            else:
                rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)

    ok = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[ok], rgb[ok]
    if len(xyz) == 0:
        raise SystemExit(f"No finite points in {path}")
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) == 0:
        raise SystemExit(f"All points filtered as outliers in {path}")

    if len(xyz) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("outputs/run1"))
    p.add_argument("--images", type=Path, default=Path("data/frames"))
    p.add_argument("--ply", type=Path, default=None,
                   help="Default: dense_mvs.ply if present else sparse.ply")
    p.add_argument("--max_points", type=int, default=200_000)
    return p.parse_args()


def main():
    args = parse_args()
    export = args.run / "export"
    out = args.run / "annotator"
    out.mkdir(parents=True, exist_ok=True)

    ply = args.ply
    if ply is None:
        cand = export / "dense_mvs.ply"
        ply = cand if cand.exists() else export / "sparse.ply"
    if not ply.exists():
        raise SystemExit(f"No PLY found: {ply}")

    xyz, rgb = read_ply_xyzrgb(ply, max_points=args.max_points)
    center = xyz.mean(axis=0)
    extent = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    if not np.isfinite(extent) or extent < 1e-6:
        raise SystemExit(f"Bad extent {extent}")

    pts_path = out / "points.bin"
    with open(pts_path, "wb") as f:
        f.write(struct.pack("<I", len(xyz)))
        f.write(xyz.astype("<f4").tobytes())
        f.write(rgb.astype("u1").tobytes())

    K = np.loadtxt(export / "K.txt")
    cam_json = json.loads((export / "camera.json").read_text())
    manifest = json.loads((export / "manifest.json").read_text())
    frames = []
    for item in manifest:
        stem = item["stem"]
        w2c = np.loadtxt(export / "poses_w2c" / f"{stem}.txt")
        img_name = item["image"]
        img_path = args.images / img_name
        if not img_path.exists():
            img_path = args.images / f"{stem}.jpg"
        frames.append({
            "stem": stem,
            "image": str(img_path.resolve()),
            "image_rel": f"frames/{img_path.name}",
            "w2c": w2c.tolist(),
        })

    scene = {
        "run": str(args.run.resolve()),
        "ply_source": str(ply.resolve()),
        "n_points": int(len(xyz)),
        "center": center.tolist(),
        "extent": extent,
        "K": K.tolist(),
        "width": int(cam_json.get("width", 1600)),
        "height": int(cam_json.get("height", 900)),
        "points_url": "points.bin",
        "frames": frames,
        "object_frame_default": {
            "center": center.tolist(),
            "euler_deg": [0.0, 0.0, 0.0],
            "size": [extent * 0.15, extent * 0.15, extent * 0.2],
            "class_id": 0,
            "class_name": "object",
        },
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    print(f"Wrote {out}/scene.json + points.bin ({len(xyz)} pts)")
    print(f"center={center.tolist()} extent={extent:.4f}")
    print(f"Frames: {len(frames)}  images: {args.images}")


if __name__ == "__main__":
    main()
