#!/usr/bin/env python3
"""CLI: export YOLO6D labels from saved object_frame.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "pose_annotator"))
from export_bridge import export_yolo6d_from_frame  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("outputs/run1"))
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite yolo6d/ in place (default: rename old to yolo6d_snap_*)",
    )
    args = p.parse_args()
    frame_path = args.run / "annotator" / "object_frame.json"
    if not frame_path.exists():
        raise SystemExit(f"Missing {frame_path} — save from the annotator UI first")
    frame = json.loads(frame_path.read_text())
    info = export_yolo6d_from_frame(args.run, frame, snapshot=not args.overwrite)
    snap = info["snap_path"]
    extra = f" (old → {snap.name})" if snap else ""
    print(f"Exported {info['n_ok']} labels → {info['yolo_root']}{extra}")


if __name__ == "__main__":
    main()
