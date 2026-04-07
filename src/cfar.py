
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import time
import numpy as np

from .backend import ArrayBackend


@dataclass
class CFARConfig:
    guard_cells: int = 4
    training_cells: int = 12
    pfa: float = 1e-4
    threshold_scale: float = 1.0
    censor: bool = False
    censor_percentile: float = 99.5
    min_training_cells: int = 32


def ca_cfar_alpha(num_training_cells: int, pfa: float) -> float:
    # CA-CFAR alpha for exponential clutter
    n = max(1, int(num_training_cells))
    return n * ((pfa ** (-1.0 / n)) - 1.0)


def _integral_image(xp, arr):
    return xp.cumsum(xp.cumsum(arr, axis=0), axis=1)


def _rect_sum(xp, sat, top, left, bottom, right):
    # all coords inclusive, arrays of same shape
    out = sat[bottom, right].copy()
    mask_t = top > 0
    out[mask_t] -= sat[top[mask_t] - 1, right[mask_t]]
    mask_l = left > 0
    out[mask_l] -= sat[bottom[mask_l], left[mask_l] - 1]
    mask_both = mask_t & mask_l
    out[mask_both] += sat[top[mask_both] - 1, left[mask_both] - 1]
    return out


def run_cfar_tile(tile: np.ndarray, cfg: CFARConfig, backend: ArrayBackend) -> Tuple[np.ndarray, Dict[str, float]]:
    xp = backend.xp
    tile_x = backend.asarray(tile.astype(np.float32))

    event_start = backend.new_event()
    event_end = backend.new_event()
    if event_start is not None:
        event_start.record()
    wall_start = time.perf_counter()

    proc = tile_x
    if cfg.censor:
        clip_val = xp.percentile(proc, cfg.censor_percentile)
        proc = xp.minimum(proc, clip_val)

    h, w = proc.shape
    yy, xx = xp.meshgrid(xp.arange(h), xp.arange(w), indexing="ij")

    tr = cfg.training_cells
    gd = cfg.guard_cells
    r_outer = tr + gd
    r_inner = gd

    top_o = xp.maximum(yy - r_outer, 0)
    left_o = xp.maximum(xx - r_outer, 0)
    bottom_o = xp.minimum(yy + r_outer, h - 1)
    right_o = xp.minimum(xx + r_outer, w - 1)

    top_i = xp.maximum(yy - r_inner, 0)
    left_i = xp.maximum(xx - r_inner, 0)
    bottom_i = xp.minimum(yy + r_inner, h - 1)
    right_i = xp.minimum(xx + r_inner, w - 1)

    sat = _integral_image(xp, proc)
    ones = xp.ones_like(proc, dtype=xp.float32)
    sat_count = _integral_image(xp, ones)

    outer_sum = _rect_sum(xp, sat, top_o, left_o, bottom_o, right_o)
    outer_count = _rect_sum(xp, sat_count, top_o, left_o, bottom_o, right_o)
    inner_sum = _rect_sum(xp, sat, top_i, left_i, bottom_i, right_i)
    inner_count = _rect_sum(xp, sat_count, top_i, left_i, bottom_i, right_i)

    train_sum = outer_sum - inner_sum
    train_count = outer_count - inner_count
    train_count = xp.maximum(train_count, 1.0)

    mu = train_sum / train_count
    nominal_n = int((2 * (tr + gd) + 1) ** 2 - (2 * gd + 1) ** 2)
    alpha = cfg.threshold_scale * ca_cfar_alpha(max(nominal_n, cfg.min_training_cells), cfg.pfa)

    det = (tile_x > (alpha * mu)) & (train_count >= cfg.min_training_cells)

    # suppress border where window is incomplete
    valid = (yy >= r_outer) & (yy < h - r_outer) & (xx >= r_outer) & (xx < w - r_outer)
    det &= valid

    wall_end = time.perf_counter()
    gpu_seconds = 0.0
    if event_end is not None:
        event_end.record()
        gpu_seconds = backend.elapsed_event_seconds(event_start, event_end)
    backend.synchronize()

    return backend.to_numpy(det).astype(np.uint8), {
        "wall_seconds": float(wall_end - wall_start),
        "gpu_event_seconds": float(gpu_seconds),
    }
