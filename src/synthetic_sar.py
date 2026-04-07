
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np

from .io_utils import save_grayscale_png


def generate_synthetic_sar(height: int = 1024, width: int = 1024, num_targets: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    texture = rng.gamma(shape=2.0, scale=1.0, size=(height, width)).astype(np.float32)
    speckle = rng.gamma(shape=1.5, scale=1.0, size=(height, width)).astype(np.float32)
    img = texture * speckle
    img /= float(np.percentile(img, 99.9) + 1e-6)

    boxes = []
    for _ in range(num_targets):
        cy = int(rng.integers(40, height - 40))
        cx = int(rng.integers(40, width - 40))
        hh = int(rng.integers(4, 10))
        ww = int(rng.integers(8, 20))
        amp = float(rng.uniform(2.5, 6.0))
        y0, y1 = max(0, cy - hh), min(height, cy + hh)
        x0, x1 = max(0, cx - ww), min(width, cx + ww)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        blob = amp * np.exp(-(((yy - cy) / max(1, hh)) ** 2 + ((xx - cx) / max(1, ww)) ** 2))
        img[y0:y1, x0:x1] += blob.astype(np.float32)
        boxes.append((x0, y0, x1 - 1, y1 - 1))

    img = np.clip(np.log1p(np.maximum(img, 0.0)), 0, None)
    img /= float(img.max() + 1e-6)
    return img.astype(np.float32), boxes


def create_synthetic_example(path: str | Path, seed: int = 0) -> None:
    img, _ = generate_synthetic_sar(seed=seed)
    save_grayscale_png(img, path)
