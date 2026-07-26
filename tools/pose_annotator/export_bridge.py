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

    rgb_dir = yolo_root / "rgb"
    lab_dir = yolo_root / "labels"
    if not rgb_dir.is_dir() or not lab_dir.is_dir():
        return None

    imgs = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    # Prefer full_* then frame_* ordering by stem
    imgs = sorted({p.resolve(): p for p in imgs}.values(), key=lambda p: p.name)
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

    # Prefer H.264 for playback
    import shutil
    import subprocess

    if shutil.which("ffmpeg"):
        h264 = yolo_root / "preview_6d_h264.mp4"
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
            tmp_path.rename(out_path)
            h264.unlink(missing_ok=True)
    else:
        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)

    return out_path if out_path.exists() else None


# Files that must stay for the live annotator session after a snapshot.
_ANNOTATOR_CARRY = (
    "scene.json",
    "points.bin",
    "plane.json",
    "scale.json",
    "last_mask.png",
)


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
    class_id = int(frame.get("class_id", 0))

    yolo_root, snap_path = prepare_yolo6d_outdir(run_dir, snapshot=snapshot)
    label_dir = yolo_root / "labels"
    rgb_dir = yolo_root / "rgb"
    label_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    train_list = []
    n_ok = 0
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
        train_list.append(f"rgb/{img_name}")
        src = Path(fr["image"])
        dst = rgb_dir / img_name
        if not dst.exists() and src.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        n_ok += 1

    (yolo_root / "train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8")
    (yolo_root / "test.txt").write_text("\n".join(train_list[::5]) + "\n", encoding="utf-8")
    (yolo_root / "object_frame.json").write_text(json.dumps(frame, indent=2), encoding="utf-8")
    write_box_ply(yolo_root / "object.ply", frame["size"])
    meta = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "n_labels": n_ok,
        "snapshot_of": str(snap_path.name) if snap_path else None,
    }
    (yolo_root / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    preview_path = None
    try:
        preview_path = ensure_yolo6d_preview(yolo_root, force=True)
    except Exception as e:
        print(f"[export] preview_6d.mp4 failed: {e}")
    return {
        "n_ok": n_ok,
        "yolo_root": yolo_root,
        "snap_path": snap_path,
        "preview_path": preview_path,
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
