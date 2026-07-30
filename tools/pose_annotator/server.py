#!/usr/bin/env python3
"""
Local pose annotator for Cursor Simple Browser / any browser.

  python tools/pose_annotator/server.py --run outputs/run1
  → open http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=ROOT / "outputs/run1")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


class Handler(BaseHTTPRequestHandler):
    run_dir: Path = ROOT / "outputs/run1"
    images_dir: Path = ROOT / "data/frames"

    def log_message(self, fmt, *args):
        print(f"[annotator] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            data = (STATIC / "index.html").read_bytes()
            return self._send(200, data, "text/html; charset=utf-8")
        if path == "/app.js":
            return self._send(200, (STATIC / "app.js").read_bytes(), "application/javascript")
        if path == "/scene.json":
            p = self.run_dir / "annotator" / "scene.json"
            if not p.exists():
                return self._send(404, b"scene.json missing - run prepare_annotator_scene.py", "text/plain")
            return self._send(200, p.read_bytes(), "application/json")
        if path == "/points.bin":
            p = self.run_dir / "annotator" / "points.bin"
            return self._send(200, p.read_bytes(), "application/octet-stream")
        if path == "/object_frame.json":
            p = self.run_dir / "annotator" / "object_frame.json"
            if not p.exists():
                return self._send(404, b"{}", "application/json")
            return self._send(200, p.read_bytes(), "application/json")
        if path == "/plane.json":
            p = self.run_dir / "annotator" / "plane.json"
            if not p.exists():
                return self._send(404, b"{}", "application/json")
            return self._send(200, p.read_bytes(), "application/json")
        if path == "/scale.json":
            p = self.run_dir / "annotator" / "scale.json"
            if not p.exists():
                return self._send(404, b"{}", "application/json")
            return self._send(200, p.read_bytes(), "application/json")
        if path == "/api/mask":
            # GET ?frame_index=N → load dumped interactive mask for that keyframe
            try:
                from segment import mask_to_png_b64
                import cv2

                qs = parse_qs(urlparse(self.path).query)
                frame_i = int((qs.get("frame_index") or ["0"])[0])
                sc = json.loads((self.run_dir / "annotator" / "scene.json").read_text())
                fr = sc["frames"][frame_i]
                stem = str(fr.get("stem") or Path(fr["image"]).stem)
                mask_path = self.run_dir / "annotator" / "masks" / f"{stem}.png"
                if not mask_path.exists():
                    body = json.dumps({"ok": False, "stem": stem, "error": "no_mask"}).encode()
                    return self._send(200, body, "application/json")
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise RuntimeError(f"Cannot read mask: {mask_path}")
                fg = (m > 127).astype("uint8")
                msg = {
                    "ok": True,
                    "stem": stem,
                    "mask_path": f"masks/{stem}.png",
                    "mask_png_b64": mask_to_png_b64(fg),
                    "fg_pixels": int(fg.sum()),
                    "width": int(fg.shape[1]),
                    "height": int(fg.shape[0]),
                }
            except Exception as e:
                msg = {"ok": False, "error": str(e)}
            body = json.dumps(msg).encode()
            return self._send(200, body, "application/json")
        if path.startswith("/frames/"):
            name = Path(path).name
            fp = self.images_dir / name
            if not fp.exists():
                return self._send(404, b"image not found", "text/plain")
            ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
            return self._send(200, fp.read_bytes(), ctype)
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/save_frame":
            data = self._read_json()
            try:
                from export_bridge import export_yolo6d_from_frame, snapshot_annotator

                # Snapshot old annotator/, then write pose into a fresh annotator/
                ann_info = snapshot_annotator(self.run_dir, data, snapshot=True)
                info = export_yolo6d_from_frame(self.run_dir, data, snapshot=True)
                ann_snap = ann_info.get("snap_path")
                msg = {
                    "ok": True,
                    "path": str(ann_info["object_frame_path"]),
                    "labels": info["n_ok"],
                    "yolo_root": str(info["yolo_root"]),
                    "snap_path": str(info["snap_path"]) if info["snap_path"] else None,
                    "annotator_snap": str(ann_snap) if ann_snap else None,
                }
            except Exception as e:
                msg = {"ok": False, "export_error": str(e)}
            body = json.dumps(msg).encode()
            return self._send(200, body, "application/json")
        if path == "/api/segment":
            data = self._read_json()
            try:
                from segment import mask_to_png_b64, save_mask_png, segment_sam2

                frame_i = int(data.get("frame_index", 0))
                sc = json.loads((self.run_dir / "annotator" / "scene.json").read_text())
                fr = sc["frames"][frame_i]
                stem = str(fr.get("stem") or Path(fr["image"]).stem)
                img_path = Path(fr["image"])
                if not img_path.exists():
                    name = Path(fr.get("image_rel", "")).name
                    img_path = self.images_dir / name
                fg = data.get("fg") or []
                bg = data.get("bg") or []
                mask, info = segment_sam2(img_path, fg, bg)
                ann = self.run_dir / "annotator"
                # Per-frame dump (all interactive SAM2 views) + legacy last_mask
                per_frame = save_mask_png(ann / "masks" / f"{stem}.png", mask)
                save_mask_png(ann / "last_mask.png", mask)
                msg = {
                    "ok": True,
                    "mask_png_b64": mask_to_png_b64(mask),
                    "stem": stem,
                    "mask_path": str(per_frame.relative_to(ann)),
                    **info,
                }
            except Exception as e:
                msg = {"ok": False, "error": str(e)}
            body = json.dumps(msg).encode()
            return self._send(200, body, "application/json")
        if path == "/api/apply_scale":
            data = self._read_json()
            try:
                from export_bridge import export_yolo6d_from_frame, snapshot_annotator
                from scale_bridge import apply_scale_to_run

                scale = float(data["scale"])
                frame = data.get("object_frame")
                if not isinstance(frame, dict) or "center" not in frame:
                    fp = self.run_dir / "annotator" / "object_frame.json"
                    frame = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
                meta = data.get("meta") or {}
                do_export = bool(data.get("export", True))
                mode = data.get("mode") or "auto"
                locked = data.get("locked", None)

                # Snapshot pre-scale annotator/, then apply scale on a fresh copy
                ann_info = snapshot_annotator(self.run_dir, frame, snapshot=True)
                info = apply_scale_to_run(
                    self.run_dir,
                    scale,
                    object_frame=frame,
                    meta=meta,
                    mode=mode,
                    locked=locked,
                )
                info["annotator_snap"] = (
                    str(ann_info["snap_path"]) if ann_info.get("snap_path") else None
                )
                out_frame = info.get("object_frame") or frame
                if do_export and out_frame:
                    exp = export_yolo6d_from_frame(self.run_dir, out_frame, snapshot=True)
                    info["labels"] = exp["n_ok"]
                    info["exported"] = True
                    info["yolo_root"] = str(exp["yolo_root"])
                    info["snap_path"] = str(exp["snap_path"]) if exp["snap_path"] else None
                else:
                    info["exported"] = False
                msg = info
            except Exception as e:
                msg = {"ok": False, "error": str(e)}
            body = json.dumps(msg).encode()
            return self._send(200, body, "application/json")
        if path == "/api/set_scale_lock":
            data = self._read_json()
            try:
                from scale_bridge import set_metric_lock

                msg = set_metric_lock(self.run_dir, bool(data.get("locked", True)))
            except Exception as e:
                msg = {"ok": False, "error": str(e)}
            body = json.dumps(msg).encode()
            return self._send(200, body, "application/json")
        if path == "/api/save_plane":
            data = self._read_json()
            out = self.run_dir / "annotator" / "plane.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            if data.get("clear"):
                if out.exists():
                    out.unlink()
                msg = {"ok": True, "cleared": True}
            else:
                payload = {
                    "centroid": data.get("centroid"),
                    "normal": data.get("normal"),
                    "offset": float(data.get("offset") or 0),
                    "samples": data.get("samples") or [],
                }
                out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                msg = {"ok": True, "path": str(out)}
            body = json.dumps(msg).encode()
            return self._send(200, body, "application/json")
        return self._send(404, b"not found", "text/plain")


def main():
    args = parse_args()
    run = args.run.resolve()
    scene = run / "annotator" / "scene.json"
    if not scene.exists():
        raise SystemExit(
            f"Missing {scene}\nRun first:\n  python scripts/prepare_annotator_scene.py --run {run}"
        )
    # images from scene
    sc = json.loads(scene.read_text())
    img0 = Path(sc["frames"][0]["image"])
    Handler.run_dir = run
    Handler.images_dir = img0.parent

    # allow import of export helper next to server
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # Preload SAM2 so first click is not cold-start
    try:
        from segment import warmup

        print("[sam2] warming up…")
        warmup(image_path=img0 if img0.exists() else None)
        print("[sam2] ready")
    except Exception as e:
        print(f"[sam2] warmup skipped: {e}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print("=" * 60)
    print("Pose annotator ready (SAM2 interactive segment)")
    print(f"  Open in Cursor Simple Browser / browser: {url}")
    print(f"  Run: {run}")
    print("  Segment: Ultralytics SAM2.1-t · click FG/BG on 2D views")
    print("=" * 60)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
