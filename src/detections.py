
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple
import csv
import numpy as np
from scipy import ndimage


@dataclass
class Detection:
    det_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    x_center: float
    y_center: float
    width: int
    height: int
    area_pixels: int
    max_score: float
    mean_score: float


def extract_detections(mask: np.ndarray, score_image: np.ndarray, min_area: int = 6, dilation: int = 1) -> List[Detection]:
    work = mask.astype(bool)
    if dilation > 0:
        work = ndimage.binary_dilation(work, iterations=dilation)

    labeled, num = ndimage.label(work)
    objects = ndimage.find_objects(labeled)

    detections: List[Detection] = []
    det_id = 1
    for label_idx, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        ys, xs = slc
        y0, y1 = int(ys.start), int(ys.stop)
        x0, x1 = int(xs.start), int(xs.stop)
        region_mask = labeled[slc] == label_idx
        area = int(region_mask.sum())
        if area < min_area:
            continue
        region_scores = score_image[slc][region_mask]
        detections.append(Detection(
            det_id=det_id,
            x_min=x0,
            y_min=y0,
            x_max=x1 - 1,
            y_max=y1 - 1,
            x_center=(x0 + x1 - 1) / 2.0,
            y_center=(y0 + y1 - 1) / 2.0,
            width=x1 - x0,
            height=y1 - y0,
            area_pixels=area,
            max_score=float(np.max(region_scores)) if region_scores.size else 0.0,
            mean_score=float(np.mean(region_scores)) if region_scores.size else 0.0,
        ))
        det_id += 1
    return detections


def save_detections_csv(detections: List[Detection], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(detections[0]).keys()) if detections else [
        "det_id", "x_min", "y_min", "x_max", "y_max", "x_center", "y_center",
        "width", "height", "area_pixels", "max_score", "mean_score"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for det in detections:
            writer.writerow(asdict(det))
