#!/usr/bin/env python3
"""
Dense colored point cloud for scale picking / visualization.

apt COLMAP has no CUDA, so patch_match_stereo cannot run. Instead:
  1) Depth Anything V2 predicts per-image relative depth
  2) Align each depth map to COLMAP sparse points (scale+shift in SfM units)
  3) Back-project and voxel-fuse into one dense PLY (same frame as sparse.ply)

Usage (undistorted MVS workspace preferred):
  python scripts/dense_from_depth.py \\
    --sfm_dir outputs/run1/mvs/dense/sparse \\
    --images outputs/run1/mvs/dense/images \\
    --out_ply outputs/run1/export/dense.ply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap
import torch
from PIL import Image


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
    T = np.asarray(im.cam_from_world().matrix(), dtype=np.float64)
    w2c = np.eye(4)
    w2c[:3, :4] = T[:3, :4]
    return w2c


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sfm_dir", type=Path, required=True,
                   help="COLMAP model (prefer undistorted mvs/dense/sparse)")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out_ply", type=Path, required=True)
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Base-hf",
                   help="HF depth model id")
    p.add_argument("--stride", type=int, default=2,
                   help="Pixel stride when back-projecting (1=densest, slower)")
    p.add_argument("--voxel", type=float, default=0.0,
                   help="Voxel size in SfM units for fusion; 0=auto from sparse")
    p.add_argument("--min_track_len", type=int, default=3,
                   help="Min track length when sampling sparse depth for alignment")
    p.add_argument("--max_rel_err", type=float, default=0.15,
                   help="Drop depth pixels whose multi-view check fails (if enabled)")
    p.add_argument("--every_n", type=int, default=1,
                   help="Use every N-th registered image (1=all)")
    return p.parse_args()


def load_depth_pipe(model_id: str, device: str):
    from transformers import pipeline
    return pipeline(
        task="depth-estimation",
        model=model_id,
        device=0 if device.startswith("cuda") else -1,
    )


@torch.inference_mode()
def predict_depth(pipe, rgb: Image.Image) -> np.ndarray:
    out = pipe(rgb)
    depth = out["depth"]
    if isinstance(depth, Image.Image):
        d = np.asarray(depth, dtype=np.float64)
        if d.ndim == 3:
            d = d[..., 0]
        # HF Depth-Anything returns an 8-bit visualization sometimes; prefer tensor if present
        if "predicted_depth" in out:
            d = np.asarray(out["predicted_depth"], dtype=np.float64)
            if d.ndim == 3:
                d = d[0]
        return d
    return np.asarray(depth, dtype=np.float64)


def predict_depth_tensor(pipe, rgb: Image.Image, target_hw: tuple[int, int]) -> np.ndarray:
    """Return relative depth map resized to (H, W). Larger = farther for DA-V2."""
    out = pipe(rgb)
    if "predicted_depth" in out and out["predicted_depth"] is not None:
        d = out["predicted_depth"]
        if hasattr(d, "detach"):
            d = d.detach().float().cpu().numpy()
        d = np.asarray(d, dtype=np.float64)
        if d.ndim == 3:
            d = d.squeeze(0)
    else:
        depth_img = out["depth"]
        d = np.asarray(depth_img, dtype=np.float64)
        if d.ndim == 3:
            d = d.mean(axis=2)
    h, w = target_hw
    if d.shape != (h, w):
        d_img = Image.fromarray(d.astype(np.float32), mode="F")
        d = np.asarray(d_img.resize((w, h), Image.BILINEAR), dtype=np.float64)
    return d


def align_depth_to_sparse(
    depth_rel: np.ndarray,
    uvs: np.ndarray,
    z_cam: np.ndarray,
) -> tuple[float, float, float]:
    """
    Depth Anything V2 predicted_depth is larger for nearer surfaces (corr with 1/z).
    Fit: 1/z = a * d + b, then z = 1 / (a*d + b).
    Returns (a, b, inlier_ratio).
    """
    h, w = depth_rel.shape
    ui = np.clip(np.round(uvs[:, 0]).astype(int), 0, w - 1)
    vi = np.clip(np.round(uvs[:, 1]).astype(int), 0, h - 1)
    d = depth_rel[vi, ui]
    z = z_cam
    valid = (d > 1e-6) & np.isfinite(d) & (z > 1e-6) & np.isfinite(z)
    d, z = d[valid], z[valid]
    if len(d) < 8:
        raise RuntimeError(f"Too few sparse depth samples: {len(d)}")

    inv_z = 1.0 / z
    rng = np.random.default_rng(0)
    n = len(d)
    slopes = []
    for _ in range(min(2000, n * 20)):
        i, j = rng.integers(0, n, size=2)
        if abs(d[i] - d[j]) < 1e-9:
            continue
        slopes.append((inv_z[i] - inv_z[j]) / (d[i] - d[j]))
    if not slopes:
        raise RuntimeError("Failed to estimate depth scale")
    a = float(np.median(slopes))
    b = float(np.median(inv_z - a * d))
    resid = inv_z - (a * d + b)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-12
    inliers = np.abs(resid - med) < 4.0 * 1.4826 * mad
    if inliers.sum() >= 8:
        A = np.stack([d[inliers], np.ones(int(inliers.sum()))], axis=1)
        sol, _, _, _ = np.linalg.lstsq(A, inv_z[inliers], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        pred_z = 1.0 / np.maximum(a * d + b, 1e-8)
        rel = np.abs(pred_z - z) / z
        inliers = rel < 0.1
    return a, b, float(inliers.mean())


def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float):
    if voxel <= 0 or len(pts) == 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    # hash
    span = keys.max(axis=0) - keys.min(axis=0) + 1
    flat = keys[:, 0] + span[0] * (keys[:, 1] + span[1] * keys[:, 2])
    order = np.argsort(flat)
    flat_s = flat[order]
    pts_s = pts[order]
    cols_s = cols[order]
    uniq = np.ones(len(flat_s), dtype=bool)
    uniq[1:] = flat_s[1:] != flat_s[:-1]
    idx = order[uniq]  # noqa: keep first of each voxel in sorted space
    # average within voxel
    # simpler: take first
    return pts_s[uniq], cols_s[uniq]


def write_ply_binary(path: Path, pts: np.ndarray, cols: np.ndarray):
    """MeshLab-friendly binary PLY: include alpha so each vertex is 16 bytes aligned."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(pts)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar alpha\n"
        "end_header\n"
    ).encode("ascii")
    data = np.empty(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
    ])
    data["x"] = pts[:, 0]
    data["y"] = pts[:, 1]
    data["z"] = pts[:, 2]
    data["r"] = cols[:, 0]
    data["g"] = cols[:, 1]
    data["b"] = cols[:, 2]
    data["a"] = 255
    assert data.itemsize == 16, data.itemsize
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading depth model: {args.model}")
    pipe = load_depth_pipe(args.model, device)

    rec = pycolmap.Reconstruction(str(args.sfm_dir))
    # sparse points by image for alignment
    pts3d = {pid: np.asarray(p.xyz, dtype=np.float64) for pid, p in rec.points3D.items()}
    track_ok = {
        pid for pid, p in rec.points3D.items()
        if len(p.track.elements) >= args.min_track_len
    }

    # auto voxel from sparse bbox diagonal
    all_xyz = np.stack(list(pts3d.values()), axis=0)
    diag = float(np.linalg.norm(all_xyz.max(0) - all_xyz.min(0)))
    voxel = args.voxel if args.voxel > 0 else max(diag / 1500.0, 1e-4)
    print(f"Sparse points={len(pts3d)}, bbox_diag={diag:.4f}, voxel={voxel:.6f}")

    image_ids = sorted(rec.images.keys(), key=lambda i: rec.images[i].name)
    image_ids = image_ids[:: max(1, args.every_n)]

    acc_pts = []
    acc_cols = []
    for k, iid in enumerate(image_ids):
        im = rec.images[iid]
        path = args.images / im.name
        if not path.exists():
            # try basename only
            path = args.images / Path(im.name).name
        if not path.exists():
            print(f"[skip] missing image {im.name}")
            continue

        cam = rec.cameras[im.camera_id]
        K = camera_to_K(cam)
        w2c = image_w2c(im)
        R, t = w2c[:3, :3], w2c[:3, 3]

        rgb = Image.open(path).convert("RGB")
        w, h = rgb.size
        if (w, h) != (int(cam.width), int(cam.height)):
            rgb = rgb.resize((int(cam.width), int(cam.height)), Image.BILINEAR)
            w, h = rgb.size

        depth_rel = predict_depth_tensor(pipe, rgb, (h, w))

        # collect sparse correspondences visible in this image
        uvs, zs = [], []
        # pycolmap Image has .points2D
        for p2d in im.points2D:
            if not p2d.has_point3D():
                continue
            pid = p2d.point3D_id
            if pid not in track_ok:
                continue
            xy = np.asarray(p2d.xy, dtype=np.float64)
            X = pts3d[pid]
            Xc = R @ X + t
            if Xc[2] <= 1e-6:
                continue
            uvs.append(xy)
            zs.append(Xc[2])
        if len(uvs) < 8:
            print(f"[skip] {im.name}: only {len(uvs)} sparse samples")
            continue
        uvs = np.asarray(uvs)
        zs = np.asarray(zs)
        try:
            a, b, inl = align_depth_to_sparse(depth_rel, uvs, zs)
        except RuntimeError as e:
            print(f"[skip] {im.name}: {e}")
            continue
        if not np.isfinite(a) or abs(a) < 1e-12:
            print(f"[skip] {im.name}: bad scale a={a}")
            continue

        # DA-V2: 1/z = a*d + b
        inv_map = a * depth_rel + b
        z_map = np.where(inv_map > 1e-6, 1.0 / inv_map, np.nan)
        z_lo = float(np.percentile(zs, 1)) * 0.85
        z_hi = float(np.percentile(zs, 99)) * 1.15
        z_ok = np.isfinite(z_map) & (z_map >= z_lo) & (z_map <= z_hi)

        stride = max(1, args.stride)
        ys = np.arange(0, h, stride)
        xs = np.arange(0, w, stride)
        uu, vv = np.meshgrid(xs, ys)
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)
        mask = z_ok[vv, uu]
        uu, vv = uu[mask], vv[mask]
        z = z_map[vv, uu]
        # back-project
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x_cam = (uu - cx) / fx * z
        y_cam = (vv - cy) / fy * z
        Xc = np.stack([x_cam, y_cam, z], axis=1)
        # cam -> world
        c2w = np.linalg.inv(w2c)
        Rw, tw = c2w[:3, :3], c2w[:3, 3]
        Xw = (Rw @ Xc.T).T + tw

        # drop outliers far from sparse scene bbox
        pad = 0.25 * (all_xyz.max(0) - all_xyz.min(0))
        lo, hi = all_xyz.min(0) - pad, all_xyz.max(0) + pad
        keep = np.all((Xw >= lo) & (Xw <= hi), axis=1)
        Xw, uu, vv = Xw[keep], uu[keep], vv[keep]
        if len(Xw) < 1000:
            print(f"[skip] {im.name}: too few in-bbox pts ({len(Xw)})")
            continue

        rgb_np = np.asarray(rgb, dtype=np.uint8)
        cols = rgb_np[vv, uu]

        acc_pts.append(Xw.astype(np.float32))
        acc_cols.append(cols)
        print(
            f"[{k+1}/{len(image_ids)}] {im.name}: samples={len(uvs)} "
            f"inl={inl:.2f} a={a:.4g} b={b:.4g} z=[{z_lo:.2f},{z_hi:.2f}] pts+={len(Xw)}",
            flush=True,
        )

        # incremental voxel fuse every few frames to limit RAM
        if (k + 1) % 8 == 0 or (k + 1) == len(image_ids):
            pts = np.concatenate(acc_pts, axis=0)
            cols = np.concatenate(acc_cols, axis=0)
            pts, cols = voxel_downsample(pts, cols, voxel)
            acc_pts = [pts]
            acc_cols = [cols]
            print(f"  fused vertices so far: {len(pts)}", flush=True)

    if not acc_pts:
        raise SystemExit("No points generated")
    pts = np.concatenate(acc_pts, axis=0)
    cols = np.concatenate(acc_cols, axis=0)
    pts, cols = voxel_downsample(pts, cols, voxel)
    write_ply_binary(args.out_ply, pts, cols)
    print(f"Wrote {len(pts)} points -> {args.out_ply}")


if __name__ == "__main__":
    main()
