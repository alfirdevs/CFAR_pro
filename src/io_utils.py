
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None


def load_image_grayscale(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"} and tifffile is not None:
        arr = tifffile.imread(str(path))
    else:
        with Image.open(path) as im:
            arr = np.array(im)

    if arr.ndim == 3:
        # RGB/RGBA to gray
        arr = arr[..., :3].astype(np.float32)
        arr = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def log_normalize_for_processing(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.maximum(arr.astype(np.float32), 0.0)
    arr = np.log1p(arr + eps)
    p1 = float(np.percentile(arr, 1.0))
    p99 = float(np.percentile(arr, 99.5))
    if p99 <= p1:
        p99 = p1 + 1.0
    arr = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
    return arr.astype(np.float32)


def downscale_for_display(arr: np.ndarray, max_dim: int = 1800) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale == 1.0:
        return arr
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im = im.resize(new_size, Image.Resampling.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def save_grayscale_png(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    im.save(path)


def save_chip(chip: np.ndarray, path: str | Path) -> None:
    save_grayscale_png(chip, path)
