"""Apply metric scale to annotator scene (points, poses, object frame)."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def read_points_bin(path: Path):
    raw = path.read_bytes()
    n = struct.unpack_from("<I", raw, 0)[0]
    xyz = np.frombuffer(raw, dtype="<f4", count=n * 3, offset=4).reshape(n, 3).copy()
    rgb = np.frombuffer(raw, dtype="u1", count=n * 3, offset=4 + n * 12).reshape(n, 3).copy()
    return xyz, rgb


def write_points_bin(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    n = len(xyz)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(np.asarray(xyz, dtype="<f4").tobytes())
        f.write(np.asarray(rgb, dtype="u1").tobytes())


def scale_w2c(w2c: np.ndarray, scale: float) -> np.ndarray:
    """Scale world: X'=sX → w2c translation t' = s*t (R unchanged)."""
    out = np.asarray(w2c, dtype=np.float64).copy()
    out[:3, 3] *= scale
    return out


def apply_scale_to_run(
    run_dir: Path,
    scale: float,
    object_frame: dict | None = None,
    meta: dict | None = None,
    *,
    locked: bool | None = None,
    mode: str = "auto",
) -> dict:
    """
    Multiply SfM/annotator geometry by `scale`.

    mode:
      - "convert": establishing / changing units (typically SfM → meters). Sets metric_locked=True after.
      - "refine": already metric; corrective rescale. Unit stays "m", lock stays True.
      - "auto": refine if scene.metric_locked else convert.

    locked: if provided, written to scene after apply (overrides default lock policy).
    """
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"Invalid scale: {scale}")

    run_dir = Path(run_dir)
    ann = run_dir / "annotator"
    scene_path = ann / "scene.json"
    pts_path = ann / "points.bin"
    if not scene_path.exists() or not pts_path.exists():
        raise FileNotFoundError("annotator scene.json / points.bin missing")

    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    prev = float(scene.get("metric_scale", 1.0) or 1.0)
    was_locked = bool(scene.get("metric_locked", False))
    if mode == "auto":
        mode = "refine" if was_locked else "convert"
    if mode not in ("convert", "refine"):
        raise ValueError(f"Invalid mode: {mode}")

    # No-op refine when already consistent
    skipped = False
    if mode == "refine" and abs(scale - 1.0) < 1e-8:
        skipped = True
    else:
        xyz, rgb = read_points_bin(pts_path)
        xyz *= scale
        write_points_bin(pts_path, xyz, rgb)

        center = xyz.mean(axis=0)
        extent = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
        scene["center"] = center.tolist()
        scene["extent"] = extent
        scene["metric_scale"] = prev * scale

        for fr in scene["frames"]:
            w2c = scale_w2c(np.asarray(fr["w2c"], dtype=np.float64), scale)
            fr["w2c"] = w2c.tolist()
            export_pose = run_dir / "export" / "poses_w2c" / f"{fr['stem']}.txt"
            if export_pose.exists():
                np.savetxt(export_pose, w2c)

        frame_path = ann / "object_frame.json"
        if object_frame is None and frame_path.exists():
            object_frame = json.loads(frame_path.read_text(encoding="utf-8"))
        if object_frame is None:
            object_frame = scene.get("object_frame_default") or {}

        if object_frame:
            of = dict(object_frame)
            of["center"] = [float(v) * scale for v in of["center"]]
            of["size"] = [float(v) * scale for v in of["size"]]
            of["metric_scale"] = scene["metric_scale"]
            frame_path.write_text(json.dumps(of, indent=2), encoding="utf-8")
            scene["object_frame_default"] = {
                **scene.get("object_frame_default", {}),
                "center": of["center"],
                "size": of["size"],
                "euler_deg": of.get("euler_deg", [0, 0, 0]),
            }
            object_frame = of

    # Unit / lock policy
    scene["metric_unit"] = "m"
    if locked is None:
        # convert or refine → stay/become locked (unit fixed as meters)
        scene["metric_locked"] = True
    else:
        scene["metric_locked"] = bool(locked)

    if skipped:
        # still refresh object_frame file if provided
        frame_path = ann / "object_frame.json"
        if object_frame is not None:
            of = dict(object_frame)
            of["metric_scale"] = scene.get("metric_scale", prev)
            frame_path.write_text(json.dumps(of, indent=2), encoding="utf-8")
        else:
            object_frame = (
                json.loads(frame_path.read_text(encoding="utf-8"))
                if frame_path.exists()
                else scene.get("object_frame_default")
            )
        xyz, _ = read_points_bin(pts_path)
        scene["center"] = xyz.mean(axis=0).tolist()
        scene["extent"] = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))

    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")

    # Persist plane in *current* world units for align-to-plane UI
    if meta:
        meta = dict(meta)
        s_plane = 1.0 if skipped else float(scale)
        if s_plane != 1.0:
            if meta.get("plane_centroid"):
                meta["plane_centroid"] = [float(v) * s_plane for v in meta["plane_centroid"]]
            if meta.get("target"):
                meta["target"] = [float(v) * s_plane for v in meta["target"]]
            if "plane_offset" in meta and meta["plane_offset"] is not None:
                meta["plane_offset"] = float(meta["plane_offset"]) * s_plane
        if meta.get("plane_normal") and meta.get("plane_centroid"):
            plane_payload = {
                "centroid": meta["plane_centroid"],
                "normal": meta["plane_normal"],
                "offset": float(meta.get("plane_offset") or 0),
                "samples": [],
            }
            (ann / "plane.json").write_text(
                json.dumps(plane_payload, indent=2), encoding="utf-8"
            )

    info = {
        "ok": True,
        "mode": mode,
        "scale_applied": 1.0 if skipped else scale,
        "skipped_noop": skipped,
        "metric_scale_cumulative": float(scene.get("metric_scale", prev)),
        "metric_locked": bool(scene.get("metric_locked", True)),
        "metric_unit": scene.get("metric_unit", "m"),
        "center": scene["center"],
        "extent": scene["extent"],
        "n_points": int(len(read_points_bin(pts_path)[0])),
        "object_frame": object_frame,
        "meta": meta or {},
    }
    (ann / "scale.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    export = run_dir / "export"
    if export.exists():
        (export / "scale.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    return info


def set_metric_lock(run_dir: Path, locked: bool) -> dict:
    """Toggle metric_locked flag without changing geometry."""
    run_dir = Path(run_dir)
    scene_path = run_dir / "annotator" / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["metric_locked"] = bool(locked)
    if locked:
        scene["metric_unit"] = scene.get("metric_unit") or "m"
    scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "metric_locked": bool(scene["metric_locked"]),
        "metric_scale": float(scene.get("metric_scale", 1.0) or 1.0),
        "metric_unit": scene.get("metric_unit", "sfm"),
    }
