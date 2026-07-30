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
_temporal_predictor = None
_temporal_model_path: Path | None = None


def mask_to_png_b64(mask01: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (mask01.astype(np.uint8) * 255))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def clean_binary_mask(
    mask: np.ndarray,
    *,
    min_area: int = 64,
    keep_largest_only: bool = True,
    fill_holes: bool = True,
    close_ksize: int = 5,
) -> np.ndarray:
    """
    Denoise a SAM2 mask: drop small speckles, keep main object, fill holes, light close.

    Accepts {0,1} or {0,255}; returns uint8 {0,255}.
    """
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    fg = (m > 127).astype(np.uint8) if m.max() > 1 else (m > 0).astype(np.uint8)
    if int(fg.sum()) == 0:
        return fg * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if keep_largest_only:
            keep_ids = {1 + int(np.argmax(areas))}
        else:
            largest = int(areas.max())
            thr = max(min_area, int(0.05 * largest))
            keep_ids = {1 + i for i, a in enumerate(areas) if int(a) >= thr}
        cleaned = np.zeros_like(fg)
        for i in keep_ids:
            cleaned[labels == i] = 1
        fg = cleaned

    if fill_holes and int(fg.sum()) > 0:
        inv = (1 - fg).astype(np.uint8)
        n2, lab2, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        h, w = fg.shape
        for j in range(1, n2):
            ys, xs = np.where(lab2 == j)
            if ys.size == 0:
                continue
            # Background component not touching the image border → hole
            if ys.min() > 0 and xs.min() > 0 and ys.max() < h - 1 and xs.max() < w - 1:
                fg[lab2 == j] = 1

    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

    return (fg.astype(np.uint8) * 255)


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


def _mask_from_results(results, h: int, w: int) -> np.ndarray:
    if not results or results[0].masks is None:
        raise RuntimeError("SAM2 returned no masks")
    masks = results[0].masks.data  # (N, H, W)
    m = masks[0].detach().float().cpu().numpy()
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return (m > 0.5).astype(np.uint8)


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
    fg = _mask_from_results(results, h, w)
    fg = (clean_binary_mask(fg) > 0).astype(np.uint8)

    info = {
        "width": w,
        "height": h,
        "fg_pixels": int(fg.sum()),
        "backend": "sam2",
        "prompt": "points",
        "weights": str(_model_path),
        "infer_s": round(dt, 3),
        "n_points": len(points_flat),
        "n_fg": sum(1 for x in labels_flat if x == 1),
        "n_bg": sum(1 for x in labels_flat if x == 0),
        "cleaned": True,
    }
    return fg, info


def segment_sam2_bbox(
    image_path: Path,
    bbox_xyxy: list[float] | np.ndarray,
    weights: Path | str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Box-prompt SAM2 segmentation.
    bbox_xyxy: [x1, y1, x2, y2] in original image pixels.
    Returns uint8 mask {0,1} HxW and info dict.
    """
    global _last_image
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    xa, xb = min(x1, x2), max(x1, x2)
    ya, yb = min(y1, y2), max(y1, y2)
    x1 = float(np.clip(xa, 0, w - 1))
    x2 = float(np.clip(xb, 0, w - 1))
    y1 = float(np.clip(ya, 0, h - 1))
    y2 = float(np.clip(yb, 0, h - 1))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError(f"Degenerate bbox for SAM2: {[x1, y1, x2, y2]}")

    model = get_sam(weights)
    t0 = time.time()
    results = model.predict(
        source=str(image_path),
        bboxes=[[x1, y1, x2, y2]],
        verbose=False,
    )
    dt = time.time() - t0
    _last_image = str(image_path.resolve())
    fg = _mask_from_results(results, h, w)
    fg = (clean_binary_mask(fg) > 0).astype(np.uint8)

    info = {
        "width": w,
        "height": h,
        "fg_pixels": int(fg.sum()),
        "backend": "sam2",
        "prompt": "bbox",
        "bbox_xyxy": [x1, y1, x2, y2],
        "weights": str(_model_path),
        "infer_s": round(dt, 3),
        "cleaned": True,
    }
    return fg, info


def get_temporal_predictor(weights: Path | str | None = None):
    """Lazy SAM2DynamicInteractivePredictor for temporal mask propagation."""
    global _temporal_predictor, _temporal_model_path
    path = Path(weights) if weights else DEFAULT_WEIGHTS
    if not path.exists():
        path = Path("sam2.1_t.pt")
    key = path.resolve() if path.exists() else path
    with _lock:
        if _temporal_predictor is None or _temporal_model_path != key:
            from ultralytics.models.sam import SAM2DynamicInteractivePredictor

            overrides = dict(
                # SAM2 temporal scores are often below detector-style 0.25;
                # 0.01 avoids dropping the tracked object after a few frames.
                conf=0.01,
                task="segment",
                mode="predict",
                imgsz=1024,
                model=str(path),
                save=False,
                verbose=False,
            )
            t0 = time.time()
            _temporal_predictor = SAM2DynamicInteractivePredictor(
                overrides=overrides, max_obj_num=1
            )
            _temporal_model_path = key
            print(f"[sam2-temporal] ready {path} in {time.time() - t0:.2f}s")
        return _temporal_predictor


def _mask_u8_from_results(results, h: int, w: int) -> np.ndarray | None:
    if not results or results[0].masks is None:
        return None
    data = results[0].masks.data
    if data is None or data.numel() == 0 or data.shape[0] == 0:
        return None
    m = data[0].detach().float().cpu().numpy()
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return clean_binary_mask((m > 0.5).astype(np.uint8) * 255)


def propagate_masks_temporal(
    frames: list[dict],
    *,
    weights: Path | str | None = None,
    obj_id: int = 0,
) -> dict[str, np.ndarray]:
    """
    Temporal SAM2 mask propagation along an ordered frame list.

    Each item in ``frames``::
        {"stem": str, "image": Path, "seed_mask": np.ndarray|None}

    When ``seed_mask`` is set, that frame updates memory (interactive correction).
    Otherwise the previous memory is propagated. Returns stem → cleaned uint8 {0,255}.
    """
    global _temporal_predictor, _temporal_model_path
    if not frames:
        return {}
    # Fresh predictor per sequence so memory does not leak across exports
    with _lock:
        _temporal_predictor = None
        _temporal_model_path = None
    pred = get_temporal_predictor(weights)
    out: dict[str, np.ndarray] = {}
    memory_ready = False

    for fr in frames:
        stem = str(fr["stem"])
        image = Path(fr["image"])
        if not image.exists():
            print(f"[sam2-temporal] missing image for {stem}: {image}")
            continue
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[sam2-temporal] cannot read {image}")
            continue
        h, w = img.shape[:2]
        seed = fr.get("seed_mask")
        try:
            if seed is not None:
                seed_u8 = clean_binary_mask(seed)
                if seed_u8.shape[0] != h or seed_u8.shape[1] != w:
                    seed_u8 = cv2.resize(seed_u8, (w, h), interpolation=cv2.INTER_NEAREST)
                pred(
                    source=str(image),
                    masks=[seed_u8],
                    obj_ids=[obj_id],
                    update_memory=True,
                )
                memory_ready = True
                # Keep cleaned interactive dump as GT for seed frames
                out[stem] = seed_u8
            elif memory_ready:
                results = pred(source=str(image))
                m = _mask_u8_from_results(results, h, w)
                if m is None:
                    raise RuntimeError("SAM2 returned an empty temporal mask")
                out[stem] = m
            else:
                print(f"[sam2-temporal] skip {stem}: no memory yet (need a seed first)")
        except Exception as e:
            print(f"[sam2-temporal] failed @ {stem}: {e}")
            if out:
                # Re-seed current frame from the previous cleaned mask instead
                # of silently copying it. This lets SAM2 recover its memory.
                prev = next(reversed(out.values())).copy()
                try:
                    retry = pred(
                        source=str(image),
                        masks=[prev],
                        obj_ids=[obj_id],
                        update_memory=True,
                    )
                    recovered = _mask_u8_from_results(retry, h, w)
                    out[stem] = prev if recovered is None else recovered
                    print(f"[sam2-temporal] recovered @ {stem} with previous-mask prompt")
                except Exception as retry_e:
                    print(f"[sam2-temporal] recovery failed @ {stem}: {retry_e}; hold previous")
                    out[stem] = prev
    return out


def save_mask_png(path: Path, mask01_or_u8: np.ndarray) -> Path:
    """Write grayscale PNG; accepts {0,1} or {0,255}. Auto-cleans speckles."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = clean_binary_mask(mask01_or_u8)
    if not cv2.imwrite(str(path), m):
        raise RuntimeError(f"Failed to write mask: {path}")
    return path


# Back-compat alias used by older callers
def segment_grabcut(*args, **kwargs):
    return segment_sam2(*args, **kwargs)
