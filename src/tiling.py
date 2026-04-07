
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple
import numpy as np


@dataclass
class TileSpec:
    y0: int
    y1: int
    x0: int
    x1: int
    core_y0: int
    core_y1: int
    core_x0: int
    core_x1: int


def generate_tiles(h: int, w: int, tile_size: int, overlap: int) -> Iterator[TileSpec]:
    step = max(1, tile_size - 2 * overlap)
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    for y in ys:
        for x in xs:
            y0 = max(0, y - overlap)
            x0 = max(0, x - overlap)
            y1 = min(h, y + step + overlap)
            x1 = min(w, x + step + overlap)

            core_y0 = y
            core_x0 = x
            core_y1 = min(h, y + step)
            core_x1 = min(w, x + step)

            yield TileSpec(y0=y0, y1=y1, x0=x0, x1=x1,
                           core_y0=core_y0, core_y1=core_y1,
                           core_x0=core_x0, core_x1=core_x1)


def paste_core(global_mask: np.ndarray, tile_mask: np.ndarray, spec: TileSpec) -> None:
    local_core_y0 = spec.core_y0 - spec.y0
    local_core_y1 = spec.core_y1 - spec.y0
    local_core_x0 = spec.core_x0 - spec.x0
    local_core_x1 = spec.core_x1 - spec.x0
    global_mask[spec.core_y0:spec.core_y1, spec.core_x0:spec.core_x1] = tile_mask[
        local_core_y0:local_core_y1, local_core_x0:local_core_x1
    ]
