
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class ArrayBackend:
    use_gpu: bool = False

    def __post_init__(self) -> None:
        self.xp = cp if (self.use_gpu and CUPY_AVAILABLE) else np
        self.on_gpu = self.xp is not np

    def asarray(self, arr: Any):
        return self.xp.asarray(arr)

    def to_numpy(self, arr: Any) -> np.ndarray:
        if self.on_gpu:
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def synchronize(self) -> None:
        if self.on_gpu:
            cp.cuda.Stream.null.synchronize()

    def new_event(self):
        if self.on_gpu:
            return cp.cuda.Event()
        return None

    def elapsed_event_seconds(self, start_event, end_event) -> float:
        if self.on_gpu:
            end_event.synchronize()
            return float(cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0)
        return 0.0
