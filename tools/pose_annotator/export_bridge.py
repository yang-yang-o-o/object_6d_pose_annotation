"""Shared export: object_frame.json + SfM poses → YOLO6D labels."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


def euler_xyz_deg_to_R(euler_deg):
    rx, ry, rz = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_wxyz_to_R(q):
    """Unit quaternion (w, x, y, z) → rotation matrix (same as Three.js)."""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm([w, x, y, z])
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def object_corners_local(size):
    sx, sy, sz = [0.5 * float(v) for v in size]
    # singleshotpose-style 8 corners (+ center handled separately)
    corners = []
    for x in (-sx, sx):
        for y in (-sy, sy):
            for z in (-sz, sz):
                corners.append([x, y, z])
    return np.asarray(corners, dtype=np.float64)


def obj_to_world(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns R_wo (obj→world), t_wo, and 9 points in world (center+8corners)."""
    if frame.get("quaternion_wxyz") is not None:
        R = quat_wxyz_to_R(frame["quaternion_wxyz"])
    else:
        R = euler_xyz_deg_to_R(frame["euler_deg"])
    t = np.asarray(frame["center"], dtype=np.float64)
    local = object_corners_local(frame["size"])
    world_corners = (R @ local.T).T + t
    center = t.copy()
    pts9 = np.vstack([center[None, :], world_corners])
    return R, t, pts9


def project(K, w2c, Xw):
    Xc = (w2c[:3, :3] @ Xw.T + w2c[:3, 3:4]).T
    z = Xc[:, 2:3]
    uv = (K @ (Xc / np.maximum(z, 1e-8)).T).T
    return uv[:, :2], z[:, 0]


def render_obb_mask(
    K: np.ndarray,
    w2c: np.ndarray,
    corners_w: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Binary uint8 mask HxW in {0, 255}: filled 2D silhouette of the oriented box.

    Fallback when SAM2 is unavailable. Prefer ``resolve_frame_mask`` / SAM2.
    """
    import cv2

    uv, z = project(K, w2c, np.asarray(corners_w, dtype=np.float64))
    pts = uv[z > 1e-6]
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(pts.astype(np.float32))
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
    return mask


def projected_obb_bbox_xyxy(
    K: np.ndarray,
    w2c: np.ndarray,
    corners_w: np.ndarray,
    width: int,
    height: int,
    *,
    pad_ratio: float = 0.03,
) -> list[float] | None:
    """Axis-aligned bbox of visible OBB corners, optionally padded. None if <2 pts."""
    uv, z = project(K, w2c, np.asarray(corners_w, dtype=np.float64))
    pts = uv[z > 1e-6]
    if len(pts) < 2:
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw, bh = float(x2 - x1), float(y2 - y1)
    pad = pad_ratio * max(bw, bh, 1.0)
    x1 = float(np.clip(x1 - pad, 0, width - 1))
    y1 = float(np.clip(y1 - pad, 0, height - 1))
    x2 = float(np.clip(x2 + pad, 0, width - 1))
    y2 = float(np.clip(y2 + pad, 0, height - 1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    """Write grayscale PNG mask (0/255)."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255
    if not cv2.imwrite(str(path), m):
        raise RuntimeError(f"Failed to write mask: {path}")


def load_mask_u8(path: Path, width: int, height: int) -> np.ndarray | None:
    """Load PNG mask → HxW uint8 {0,255}, resized if needed."""
    import cv2

    path = Path(path)
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[0] != height or m.shape[1] != width:
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return ((m > 127).astype(np.uint8) * 255)


def resolve_frame_mask(
    *,
    stem: str,
    image_path: Path,
    K: np.ndarray,
    w2c: np.ndarray,
    corners_w: np.ndarray,
    width: int,
    height: int,
    interactive_mask_dir: Path | None = None,
    interactive_stems: list[str] | None = None,
    prefer_sam2: bool = True,
) -> tuple[np.ndarray, str]:
    """
    Resolve a YOLO6D-style mask for one frame.

    Priority:
      1) interactive SAM2 dump under ``annotator/masks/{stem}.png``
         (also accepts alternate stems, e.g. keyframe stem for full_* frames)
      2) Ultralytics SAM2 with projected-OBB box prompt
      3) OBB convex-hull silhouette (fallback)

    Returns (mask_u8_0_255, source_tag).
    """
    candidates: list[Path] = []
    if interactive_mask_dir is not None:
        interactive_mask_dir = Path(interactive_mask_dir)
        stems = [stem] + list(interactive_stems or [])
        seen = set()
        for s in stems:
            if not s or s in seen:
                continue
            seen.add(s)
            candidates.append(interactive_mask_dir / f"{s}.png")
    for p in candidates:
        m = load_mask_u8(p, width, height)
        if m is not None and int((m > 0).sum()) > 0:
            from segment import clean_binary_mask

            return clean_binary_mask(m), f"interactive:{p.name}"

    if prefer_sam2:
        bbox = projected_obb_bbox_xyxy(K, w2c, corners_w, width, height)
        if bbox is not None and Path(image_path).exists():
            try:
                from segment import clean_binary_mask, segment_sam2_bbox

                fg, _info = segment_sam2_bbox(Path(image_path), bbox)
                if fg.shape[0] != height or fg.shape[1] != width:
                    import cv2

                    fg = cv2.resize(
                        fg.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                return clean_binary_mask(fg.astype(np.uint8) * 255), "sam2_bbox"
            except Exception as e:
                print(f"[mask] SAM2 failed for {stem}: {e}")

    return render_obb_mask(K, w2c, corners_w, width, height), "obb_hull"


def _unique_snap_dir(run_dir: Path, prefix: str = "yolo6d_snap") -> Path:
    """Return a non-existing snapshot directory path under run_dir."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = run_dir / f"{prefix}_{stamp}"
    if not candidate.exists():
        return candidate
    for i in range(1, 1000):
        candidate = run_dir / f"{prefix}_{stamp}_{i:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate snapshot dir under {run_dir}")


def prepare_yolo6d_outdir(run_dir: Path, *, snapshot: bool = True) -> tuple[Path, Path | None]:
    """
    Prepare export root. If snapshot and ``yolo6d`` already exists (non-empty),
    rename it to ``yolo6d_snap_YYYYMMDD_HHMMSS`` and create a fresh ``yolo6d``.

    Returns (yolo_root, snap_path_or_None).
    """
    run_dir = Path(run_dir)
    yolo_root = run_dir / "yolo6d"
    snap_path = None
    if snapshot and yolo_root.exists():
        has_content = any(yolo_root.iterdir())
        if has_content:
            # Ensure preview exists on the dir that is about to become a snap
            try:
                ensure_yolo6d_preview(yolo_root)
                ensure_yolo6d_mask_preview(yolo_root)
            except Exception as e:
                print(f"[export] preview before snapshot failed: {e}")
            snap_path = _unique_snap_dir(run_dir, prefix="yolo6d_snap")
            yolo_root.rename(snap_path)
    yolo_root.mkdir(parents=True, exist_ok=True)
    return yolo_root, snap_path


# Corner edges matching object_corners_local order (x,y,z nested)
_BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),  # z
    (0, 2), (1, 3), (4, 6), (5, 7),  # y
    (0, 4), (1, 5), (2, 6), (3, 7),  # x
]


def _finalize_preview_mp4(tmp_path: Path, out_path: Path) -> Path | None:
    """Convert tmp mp4v → H.264 when ffmpeg is available; else keep mp4v."""
    import shutil
    import subprocess

    tmp_path = Path(tmp_path)
    out_path = Path(out_path)
    if not tmp_path.exists():
        return None
    if shutil.which("ffmpeg"):
        h264 = out_path.with_name(out_path.stem + "_h264.mp4")
        r = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(tmp_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-preset",
                "fast",
                str(h264),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and h264.exists():
            tmp_path.unlink(missing_ok=True)
            if out_path.exists():
                out_path.unlink()
            h264.rename(out_path)
        else:
            if out_path.exists():
                out_path.unlink()
            tmp_path.rename(out_path)
            h264.unlink(missing_ok=True)
    else:
        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)
    return out_path if out_path.exists() else None


def _iter_rgb_images(yolo_root: Path) -> list[Path]:
    rgb_dir = Path(yolo_root) / "rgb"
    if not rgb_dir.is_dir():
        return []
    imgs = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    return sorted({p.resolve(): p for p in imgs}.values(), key=lambda p: p.name)


def _resolve_preview_rgb_pairs(yolo_root: Path) -> list[tuple[Path, Path]]:
    """
    Return (rgb_path, mask_path) pairs for preview_mask.

    Prefer ``rgb/``; if empty (broken keyframe symlinks), fall back to
    ``../annotator/scene.json`` image paths matched by mask stems.
    """
    yolo_root = Path(yolo_root)
    mask_dir = yolo_root / "mask"
    if not mask_dir.is_dir():
        return []

    pairs: list[tuple[Path, Path]] = []
    imgs = _iter_rgb_images(yolo_root)
    if imgs:
        for img in imgs:
            m = mask_dir / f"{img.stem}.png"
            if m.exists():
                pairs.append((img, m))
        if pairs:
            return pairs

    # Fallback: scene.json frames
    scene_path = yolo_root.parent / "annotator" / "scene.json"
    if not scene_path.exists():
        return []
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    by_stem = {fr["stem"]: Path(fr["image"]) for fr in scene.get("frames", [])}
    for m in sorted(mask_dir.glob("*.png")):
        img = by_stem.get(m.stem)
        if img is not None and img.exists():
            pairs.append((img, m))
    return pairs


def ensure_yolo6d_preview(
    yolo_root: Path,
    *,
    fps: float | None = None,
    force: bool = False,
) -> Path | None:
    """
    Write ``preview_6d.mp4`` under ``yolo_root`` from rgb/ + labels/.
    Skips if file already exists unless force=True. Returns path or None.
    """
    yolo_root = Path(yolo_root)
    out_path = yolo_root / "preview_6d.mp4"
    if out_path.exists() and not force:
        return out_path

    lab_dir = yolo_root / "labels"
    if not lab_dir.is_dir():
        return None

    imgs = _iter_rgb_images(yolo_root)
    if not imgs:
        # fallback via scene for keyframe exports with empty rgb/
        pairs = _resolve_preview_rgb_pairs(yolo_root)
        imgs = [p[0] for p in pairs]
    if not imgs:
        return None

    import cv2

    sample = cv2.imread(str(imgs[0]))
    if sample is None:
        return None
    h, w = sample.shape[:2]
    if fps is None:
        # keyframe exports ~74 frames → 5fps; dense full video → 30fps
        fps = 30.0 if len(imgs) > 200 else 5.0

    tmp_path = yolo_root / "preview_6d_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed: {tmp_path}")

    color_edge = (0, 255, 80)
    color_front = (0, 200, 255)
    color_center = (0, 80, 255)
    color_text = (255, 255, 255)
    front = {1, 3, 5, 7}

    for i, img_path in enumerate(imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        lab_path = lab_dir / f"{img_path.stem}.txt"
        if lab_path.exists():
            vals = [float(x) for x in lab_path.read_text().strip().split()]
            if len(vals) >= 19:
                pts = []
                for k in range(9):
                    u = vals[1 + 2 * k] * w
                    v = vals[2 + 2 * k] * h
                    pts.append((int(round(u)), int(round(v))))
                for a, b in _BOX_EDGES:
                    pa, pb = pts[1 + a], pts[1 + b]
                    c = color_front if (a in front and b in front) else color_edge
                    cv2.line(img, pa, pb, c, 2, cv2.LINE_AA)
                cv2.circle(img, pts[0], 5, color_center, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{img_path.stem}  {i + 1}/{len(imgs)}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color_text,
            2,
            cv2.LINE_AA,
        )
        writer.write(img)
    writer.release()
    return _finalize_preview_mp4(tmp_path, out_path)


def ensure_yolo6d_mask_preview(
    yolo_root: Path,
    *,
    fps: float | None = None,
    force: bool = False,
    alpha: float = 0.45,
) -> Path | None:
    """
    Write ``preview_mask.mp4``: RGB with semi-transparent green mask overlay.

    Skips if file already exists unless force=True.
    """
    yolo_root = Path(yolo_root)
    out_path = yolo_root / "preview_mask.mp4"
    if out_path.exists() and not force:
        return out_path

    pairs = _resolve_preview_rgb_pairs(yolo_root)
    if not pairs:
        return None

    import cv2

    sample = cv2.imread(str(pairs[0][0]))
    if sample is None:
        return None
    h, w = sample.shape[:2]
    if fps is None:
        fps = 30.0 if len(pairs) > 200 else 5.0

    tmp_path = yolo_root / "preview_mask_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed: {tmp_path}")

    tint = np.array([0, 220, 80], dtype=np.float32)  # BGR green
    color_text = (255, 255, 255)
    a = float(np.clip(alpha, 0.05, 0.9))

    for i, (img_path, mask_path) in enumerate(pairs):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            fg = mask > 127
            if fg.any():
                overlay = img.astype(np.float32)
                overlay[fg] = (1.0 - a) * overlay[fg] + a * tint
                img = np.clip(overlay, 0, 255).astype(np.uint8)
                # thin contour for readability
                cnts, _ = cv2.findContours(
                    fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(img, cnts, -1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{img_path.stem}  {i + 1}/{len(pairs)}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color_text,
            2,
            cv2.LINE_AA,
        )
        writer.write(img)
    writer.release()
    return _finalize_preview_mp4(tmp_path, out_path)

# Files that must stay for the live annotator session after a snapshot.
_ANNOTATOR_CARRY = (
    "scene.json",
    "points.bin",
    "plane.json",
    "scale.json",
    "last_mask.png",
)
_ANNOTATOR_CARRY_DIRS = ("masks",)


def snapshot_annotator(run_dir: Path, frame: dict, *, snapshot: bool = True) -> dict:
    """
    Snapshot existing ``annotator/`` then recreate a fresh working copy.

    Old dir → ``annotator_snap_YYYYMMDD_HHMMSS/``. New ``annotator/`` keeps
    scene/points/plane/scale (copied from snap) and writes the new object_frame.

    Returns dict with annotator_dir, snap_path, object_frame_path.
    """
    run_dir = Path(run_dir)
    ann = run_dir / "annotator"
    snap_path = None
    if snapshot and ann.exists() and any(ann.iterdir()):
        snap_path = _unique_snap_dir(run_dir, prefix="annotator_snap")
        ann.rename(snap_path)
        ann.mkdir(parents=True, exist_ok=True)
        for name in _ANNOTATOR_CARRY:
            src = snap_path / name
            if src.exists():
                shutil.copy2(src, ann / name)
        for name in _ANNOTATOR_CARRY_DIRS:
            src = snap_path / name
            if src.is_dir():
                shutil.copytree(src, ann / name, dirs_exist_ok=True)
    else:
        ann.mkdir(parents=True, exist_ok=True)

    out_frame = ann / "object_frame.json"
    out_frame.write_text(json.dumps(frame, indent=2), encoding="utf-8")
    return {
        "annotator_dir": ann,
        "snap_path": snap_path,
        "object_frame_path": out_frame,
    }


def export_yolo6d_from_frame(
    run_dir: Path,
    frame: dict,
    *,
    snapshot: bool = True,
) -> dict:
    """
    Export keyframe YOLO6D labels.

    By default does not overwrite: existing ``yolo6d/`` is renamed to a timestamped
    snapshot first. Returns dict with n_ok, yolo_root, snap_path.
    """
    run_dir = Path(run_dir)
    scene = json.loads((run_dir / "annotator" / "scene.json").read_text())
    K = np.asarray(scene["K"], dtype=np.float64)
    W, H = int(scene["width"]), int(scene["height"])
    _, _, pts9_w = obj_to_world(frame)
    corners_w = pts9_w[1:]  # 8 box corners (skip center)
    class_id = int(frame.get("class_id", 0))

    yolo_root, snap_path = prepare_yolo6d_outdir(run_dir, snapshot=snapshot)
    label_dir = yolo_root / "labels"
    rgb_dir = yolo_root / "rgb"
    mask_dir = yolo_root / "mask"
    label_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    interactive_masks = run_dir / "annotator" / "masks"

    train_list = []
    n_ok = 0
    mask_sources: dict[str, int] = {}
    for fr in scene["frames"]:
        stem = fr["stem"]
        w2c = np.asarray(fr["w2c"], dtype=np.float64)
        uv, z = project(K, w2c, pts9_w)
        if z[0] <= 0:
            continue
        uvn = uv.copy()
        uvn[:, 0] /= W
        uvn[:, 1] /= H
        xs, ys = uvn[:, 0], uvn[:, 1]
        x_range = float(xs.max() - xs.min())
        y_range = float(ys.max() - ys.min())
        vals = [float(class_id)]
        for i in range(9):
            vals.extend([float(xs[i]), float(ys[i])])
        vals.extend([x_range, y_range])
        line = " ".join(f"{v:.6f}" for v in vals) + "\n"
        (label_dir / f"{stem}.txt").write_text(line, encoding="utf-8")
        img_name = Path(fr["image"]).name
        src = Path(fr["image"])
        dst = rgb_dir / img_name
        if not dst.exists() and src.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        img_for_mask = dst if dst.exists() else src
        mask_u8, src_tag = resolve_frame_mask(
            stem=stem,
            image_path=img_for_mask,
            K=K,
            w2c=w2c,
            corners_w=corners_w,
            width=W,
            height=H,
            interactive_mask_dir=interactive_masks,
        )
        write_mask_png(mask_dir / f"{stem}.png", mask_u8)
        mask_sources[src_tag.split(":")[0]] = mask_sources.get(src_tag.split(":")[0], 0) + 1
        train_list.append(f"rgb/{img_name}")
        n_ok += 1

    (yolo_root / "train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8")
    (yolo_root / "test.txt").write_text("\n".join(train_list[::5]) + "\n", encoding="utf-8")
    (yolo_root / "object_frame.json").write_text(json.dumps(frame, indent=2), encoding="utf-8")
    write_box_ply(yolo_root / "object.ply", frame["size"])
    meta = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "n_labels": n_ok,
        "snapshot_of": str(snap_path.name) if snap_path else None,
        "mask_sources": mask_sources,
    }
    (yolo_root / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    preview_path = None
    preview_mask_path = None
    try:
        preview_path = ensure_yolo6d_preview(yolo_root, force=True)
    except Exception as e:
        print(f"[export] preview_6d.mp4 failed: {e}")
    try:
        preview_mask_path = ensure_yolo6d_mask_preview(yolo_root, force=True)
    except Exception as e:
        print(f"[export] preview_mask.mp4 failed: {e}")
    return {
        "n_ok": n_ok,
        "yolo_root": yolo_root,
        "snap_path": snap_path,
        "preview_path": preview_path,
        "preview_mask_path": preview_mask_path,
    }


def write_box_ply(path: Path, size):
    corners = object_corners_local(size)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(corners)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z in corners:
            f.write(f"{x} {y} {z}\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite yolo6d/ in place (default: snapshot old then write new)",
    )
    args = p.parse_args()
    frame = json.loads((args.run / "annotator" / "object_frame.json").read_text())
    info = export_yolo6d_from_frame(args.run, frame, snapshot=not args.overwrite)
    snap = info["snap_path"]
    extra = f" (old → {snap.name})" if snap else ""
    print(f"Exported {info['n_ok']} labels → {info['yolo_root']}{extra}")
