
from __future__ import annotations

from pathlib import Path
from typing import List
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .detections import Detection
from .io_utils import downscale_for_display, save_chip


def save_overlay(image: np.ndarray, detections: List[Detection], path: str | Path, max_dim: int = 1800) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scale_img = downscale_for_display(image, max_dim=max_dim)
    scale_y = scale_img.shape[0] / image.shape[0]
    scale_x = scale_img.shape[1] / image.shape[1]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(scale_img, cmap="gray", vmin=0, vmax=1)
    for det in detections:
        rect = Rectangle(
            (det.x_min * scale_x, det.y_min * scale_y),
            det.width * scale_x,
            det.height * scale_y,
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(det.x_min * scale_x, max(0, det.y_min * scale_y - 2), str(det.det_id), fontsize=7)
    ax.set_title(f"Detections: {len(detections)}")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_chips(image: np.ndarray, detections: List[Detection], out_dir: str | Path, pad: int = 16) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = image.shape
    for det in detections:
        y0 = max(0, det.y_min - pad)
        x0 = max(0, det.x_min - pad)
        y1 = min(h, det.y_max + 1 + pad)
        x1 = min(w, det.x_max + 1 + pad)
        chip = image[y0:y1, x0:x1]
        save_chip(chip, out_dir / f"chip_{det.det_id:04d}.png")


def save_html_viewer(image_name: str, detections: List[Detection], overlay_path: str | Path, html_path: str | Path) -> None:
    overlay_path = Path(overlay_path)
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><td>{d.det_id}</td><td>{d.x_center:.1f}</td><td>{d.y_center:.1f}</td>"
        f"<td>{d.width}</td><td>{d.height}</td><td>{d.area_pixels}</td></tr>"
        for d in detections
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>SAR CFAR Viewer</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
img {{ max-width: 100%; border: 1px solid #ccc; }}
table {{ border-collapse: collapse; margin-top: 16px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
th {{ background: #f2f2f2; }}
</style>
</head>
<body>
<h1>SAR CFAR Detection Viewer</h1>
<p><strong>Image:</strong> {image_name}</p>
<p><strong>Detections:</strong> {len(detections)}</p>
<img src="{overlay_path.name}" alt="Overlay" />
<table>
<thead><tr><th>ID</th><th>X center</th><th>Y center</th><th>Width</th><th>Height</th><th>Area</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")
