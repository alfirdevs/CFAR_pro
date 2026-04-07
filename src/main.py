
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np

from .backend import ArrayBackend, CUPY_AVAILABLE
from .cfar import CFARConfig, run_cfar_tile
from .detections import extract_detections, save_detections_csv
from .io_utils import load_image_grayscale, log_normalize_for_processing, save_grayscale_png
from .synthetic_sar import create_synthetic_example, generate_synthetic_sar
from .tiling import generate_tiles, paste_core
from .visualization import save_overlay, save_chips, save_html_viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU/CPU tiled SAR ship detection with CFAR and optional censoring.")
    parser.add_argument("--input", type=str, default="", help="Path to input image (.png/.jpg/.tif/.tiff).")
    parser.add_argument("--mode", type=str, default="real", choices=["real", "synthetic"])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--tile_size", type=int, default=1536)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--guard_cells", type=int, default=4)
    parser.add_argument("--training_cells", type=int, default=12)
    parser.add_argument("--pfa", type=float, default=1e-4)
    parser.add_argument("--threshold_scale", type=float, default=1.0)
    parser.add_argument("--censor", type=int, default=1)
    parser.add_argument("--censor_percentile", type=float, default=99.5)
    parser.add_argument("--min_area", type=int, default=6)
    parser.add_argument("--chip_pad", type=int, default=16)
    parser.add_argument("--synthetic_size", type=int, default=1024)
    parser.add_argument("--synthetic_targets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_input(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.mode == "synthetic":
        img, _ = generate_synthetic_sar(height=args.synthetic_size, width=args.synthetic_size,
                                        num_targets=args.synthetic_targets, seed=args.seed)
        return img, f"synthetic_{args.synthetic_size}"
    if not args.input:
        raise ValueError("For mode=real, please provide --input.")
    raw = load_image_grayscale(args.input)
    proc = log_normalize_for_processing(raw)
    return proc, Path(args.input).name


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.perf_counter()
    image, image_name = build_input(args)

    save_grayscale_png(image, output_dir / "normalized_input.png")

    backend = ArrayBackend(use_gpu=bool(args.use_gpu))
    cfg = CFARConfig(
        guard_cells=args.guard_cells,
        training_cells=args.training_cells,
        pfa=args.pfa,
        threshold_scale=args.threshold_scale,
        censor=bool(args.censor),
        censor_percentile=args.censor_percentile,
    )

    h, w = image.shape
    global_mask = np.zeros((h, w), dtype=np.uint8)

    tile_records = []
    for idx, spec in enumerate(generate_tiles(h, w, args.tile_size, args.overlap), start=1):
        tile = image[spec.y0:spec.y1, spec.x0:spec.x1]
        tile_mask, metrics = run_cfar_tile(tile, cfg, backend)
        paste_core(global_mask, tile_mask, spec)
        tile_records.append({
            "tile_index": idx,
            "y0": spec.y0, "y1": spec.y1, "x0": spec.x0, "x1": spec.x1,
            "core_y0": spec.core_y0, "core_y1": spec.core_y1,
            "core_x0": spec.core_x0, "core_x1": spec.core_x1,
            **metrics
        })

    detections = extract_detections(global_mask, image, min_area=args.min_area, dilation=1)

    save_detections_csv(detections, output_dir / "detections.csv")
    save_grayscale_png(global_mask.astype(np.float32), output_dir / "detection_mask.png")
    save_overlay(image, detections, output_dir / "overlay.png")
    save_chips(image, detections, output_dir / "chips", pad=args.chip_pad)
    save_html_viewer(image_name, detections, output_dir / "overlay.png", output_dir / "viewer.html")

    overall_end = time.perf_counter()
    run_metrics = {
        "image_name": image_name,
        "image_height": int(h),
        "image_width": int(w),
        "num_detections": int(len(detections)),
        "use_gpu_requested": bool(args.use_gpu),
        "gpu_available": bool(CUPY_AVAILABLE),
        "gpu_used": bool(backend.on_gpu),
        "overall_wall_seconds": float(overall_end - overall_start),
        "tile_wall_seconds_sum": float(sum(t["wall_seconds"] for t in tile_records)),
        "tile_gpu_event_seconds_sum": float(sum(t["gpu_event_seconds"] for t in tile_records)),
        "config": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "guard_cells": args.guard_cells,
            "training_cells": args.training_cells,
            "pfa": args.pfa,
            "threshold_scale": args.threshold_scale,
            "censor": bool(args.censor),
            "censor_percentile": args.censor_percentile,
            "min_area": args.min_area,
        }
    }

    (output_dir / "run_metrics.json").write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    (output_dir / "tile_timing.json").write_text(json.dumps(tile_records, indent=2), encoding="utf-8")

    print(json.dumps(run_metrics, indent=2))
    print(f"Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
