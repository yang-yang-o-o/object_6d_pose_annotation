#!/usr/bin/env python3
"""Extract frames from a phone orbit video for SfM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=2.0,
                   help="Target sampling rate. 2 fps is a good start for ~30s orbits.")
    p.add_argument("--every_n", type=int, default=0,
                   help="If >0, take every N-th frame instead of --fps.")
    p.add_argument("--max_side", type=int, default=1600,
                   help="Downscale so max(H,W)=max_side. 0 = native resolution. "
                        "Default 1600 balances quality vs speed (run1 config).")
    p.add_argument("--max_frames", type=int, default=0, help="0 = no limit.")
    p.add_argument("--ext", choices=["jpg", "png"], default="jpg")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.every_n > 0:
        step = args.every_n
    else:
        step = max(1, int(round(src_fps / args.fps)))

    scale = 1.0
    if args.max_side > 0:
        scale = min(1.0, args.max_side / float(max(w0, h0)))
    w1 = int(round(w0 * scale))
    h1 = int(round(h0 * scale))

    meta = {
        "video": str(args.video.resolve()),
        "src_fps": src_fps,
        "src_size": [w0, h0],
        "out_size": [w1, h1],
        "scale": scale,
        "step": step,
        "approx_out_fps": src_fps / step,
    }
    print(json.dumps(meta, indent=2))

    idx_out = 0
    frame_i = 0
    pbar = tqdm(total=n_total, desc="extract")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step == 0:
            if scale < 1.0:
                frame = cv2.resize(frame, (w1, h1), interpolation=cv2.INTER_AREA)
            name = f"frame_{idx_out:06d}.{args.ext}"
            path = args.out_dir / name
            if args.ext == "jpg":
                cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(str(path), frame)
            idx_out += 1
            if args.max_frames > 0 and idx_out >= args.max_frames:
                break
        frame_i += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    meta["n_frames"] = idx_out
    (args.out_dir.parent / "extract_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote {idx_out} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
