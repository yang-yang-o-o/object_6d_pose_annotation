"""Interactive segmentation via Ultralytics SAM2 (point prompts)."""
from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "sam2.1_t.pt"

_lock = threading.Lock()
_model = None
_model_path: Path | None = None
_last_image: str | None = None


def mask_to_png_b64(mask01: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (mask01.astype(np.uint8) * 255))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def get_sam(weights: Path | str | None = None):
    """Lazy-load / reuse a single SAM2 model (GPU if available)."""
    global _model, _model_path
    path = Path(weights) if weights else DEFAULT_WEIGHTS
    if not path.exists():
        # Fall back to Ultralytics auto-download name
        path = Path("sam2.1_t.pt")
    with _lock:
        if _model is None or _model_path != path.resolve():
            from ultralytics import SAM

            t0 = time.time()
            _model = SAM(str(path))
            _model_path = path.resolve() if path.exists() else path
            print(f"[sam2] loaded {path} in {time.time() - t0:.2f}s")
        return _model


def warmup(weights: Path | str | None = None, image_path: Path | None = None):
    """Load weights and optionally run one dummy encode so first click is fast."""
    model = get_sam(weights)
    if image_path and Path(image_path).exists():
        h, w = 64, 64
        # tiny synthetic prompt on real image is better for real encode cache
        import cv2 as _cv

        im = _cv.imread(str(image_path))
        if im is not None:
            hh, ww = im.shape[:2]
            model.predict(
                source=str(image_path),
                points=[[ww // 2, hh // 2]],
                labels=[1],
                verbose=False,
            )
            global _last_image
            _last_image = str(Path(image_path).resolve())
    return model


def segment_sam2(
    image_path: Path,
    fg_seeds: list[list[float]],
    bg_seeds: list[list[float]] | None = None,
    weights: Path | str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Point-prompt SAM2 segmentation.
    fg_seeds / bg_seeds: [[u, v], ...] in original image pixels.
    Returns uint8 mask {0,1} HxW and info dict.
    """
    global _last_image
    if not fg_seeds:
        raise ValueError("Need at least one foreground seed")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    bg_seeds = bg_seeds or []
    points_flat: list[list[float]] = []
    labels_flat: list[int] = []
    for s in fg_seeds:
        u, v = float(s[0]), float(s[1])
        u = min(max(u, 0), w - 1)
        v = min(max(v, 0), h - 1)
        points_flat.append([u, v])
        labels_flat.append(1)
    for s in bg_seeds:
        u, v = float(s[0]), float(s[1])
        u = min(max(u, 0), w - 1)
        v = min(max(v, 0), h - 1)
        points_flat.append([u, v])
        labels_flat.append(0)

    # CRITICAL: Ultralytics treats 2D points (N,2) as N separate objects.
    # For one object with multi FG/BG clicks use shape (1, P, 2) / (1, P).
    points = [points_flat]
    labels = [labels_flat]

    model = get_sam(weights)
    t0 = time.time()
    # Same-image re-prompts reuse encoder features inside Ultralytics predictor (~0.1s)
    results = model.predict(
        source=str(image_path),
        points=points,
        labels=labels,
        verbose=False,
    )
    dt = time.time() - t0
    _last_image = str(image_path.resolve())

    if not results or results[0].masks is None:
        raise RuntimeError("SAM2 returned no masks")

    masks = results[0].masks.data  # (N, H, W)
    # Single-object multi-point → N=1
    m = masks[0].detach().float().cpu().numpy()
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    fg = (m > 0.5).astype(np.uint8)

    info = {
        "width": w,
        "height": h,
        "fg_pixels": int(fg.sum()),
        "backend": "sam2",
        "weights": str(_model_path),
        "infer_s": round(dt, 3),
        "n_points": len(points_flat),
        "n_fg": sum(1 for x in labels_flat if x == 1),
        "n_bg": sum(1 for x in labels_flat if x == 0),
    }
    return fg, info


# Back-compat alias used by older callers
def segment_grabcut(*args, **kwargs):
    return segment_sam2(*args, **kwargs)
