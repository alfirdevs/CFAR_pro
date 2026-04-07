# GPU-Accelerated SAR Ship Detection with Tiled CFAR

This project implements a **large-image SAR ship detection pipeline** using **tiled CA-CFAR** with **optional censoring** and optional **GPU acceleration with CuPy**.

## What is actually implemented

This code implements:

- **Tiled mean-based CA-CFAR**
- **Optional censoring** by clipping tile intensities above a chosen percentile before local mean estimation
- **Overlapping divide-and-conquer processing** for large images
- **Global mask stitching** using the tile core region so overlap duplicates are handled cleanly
- **Detection extraction** from the final global mask
- **Bounding-box overlay**
- **Per-detection chips**
- **CSV output of all detections**
- **Wall-clock timing** with `time.perf_counter()`
- **GPU event timing** with CuPy CUDA events when GPU is active

It does **not** implement full K-distribution parameter estimation. The statistical motivation is that SAR sea clutter is often modeled as a **Gamma scale mixture**, which leads to **K-distributed intensity**. The implemented detector is a **practical, robust approximation** based on local mean estimation plus optional censoring.

## Statistical model behind the project

A common SAR clutter model is:

\[
I = S \cdot T
\]

where:

- \(S\) is **speckle**
- \(T\) is **texture**

If both are Gamma-distributed, the intensity follows a **K-distribution**. In practice, explicit local estimation of such compound models is costly, so this project uses a robust CFAR detector.

### Implemented detection rule

For each cell under test (CUT), the detector computes a local training-cell mean:

\[
\hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} x_i
\]

and declares a detection when:

\[
x_{CUT} > \alpha \hat{\mu}
\]

with:

\[
\alpha = N \left(P_{FA}^{-1/N} - 1\right)
\]

for a CA-CFAR-style threshold multiplier.

### Optional censoring

When censoring is enabled, the tile intensity image is clipped above a chosen percentile before the local mean is computed. This reduces contamination from bright outliers and nearby targets.

## Repository structure

```text
gpu_sar_ship_detection_full/
├── run.py
├── run.sh
├── requirements.txt
├── README.md
├── data/
│   └── input_images/
├── results/
└── src/
    ├── __init__.py
    ├── backend.py
    ├── io_utils.py
    ├── cfar.py
    ├── tiling.py
    ├── detections.py
    ├── visualization.py
    ├── synthetic_sar.py
    └── main.py
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, install the CuPy package that matches your CUDA version. Example:

```bash
pip install cupy-cuda12x
```

## Quick start

### Synthetic demo

```bash
python run.py --mode synthetic --use_gpu 1 --output_dir results
```

### Real image

```bash
python run.py \
  --mode real \
  --input "data/input_images/sample_panama_canal_cfar_ready.png" \
  --use_gpu 1 \
  --tile_size 1536 \
  --overlap 96 \
  --guard_cells 4 \
  --training_cells 12 \
  --pfa 1e-4 \
  --threshold_scale 1.0 \
  --censor 1 \
  --censor_percentile 99.5 \
  --output_dir results
```

### Your Gibraltar Sentinel-1 TIFF

```bash
python run.py \
  --mode real \
  --input "C:/Users/Operator/Downloads/S1A_IW_GRDH_1SDV_20260329T181812_20260329T181837_063846_08077F_80E6.SAFE/measurement/s1a-iw-grd-vv-20260329t181812-20260329t181837-063846-08077f-001.tiff" \
  --use_gpu 1 \
  --tile_size 1536 \
  --overlap 96 \
  --guard_cells 4 \
  --training_cells 12 \
  --pfa 1e-4 \
  --threshold_scale 1.0 \
  --censor 1 \
  --censor_percentile 99.5 \
  --output_dir results_gibraltar
```

## Outputs

The code writes:

- `results/normalized_input.png`
- `results/detection_mask.png`
- `results/detections.csv`
- `results/overlay.png`
- `results/viewer.html`
- `results/chips/chip_XXXX.png`
- `results/run_metrics.json`
- `results/tile_timing.json`

## Timing

This project reports both:

- **Wall-clock time** using `time.perf_counter()`
- **GPU event time** using CuPy CUDA events

Wall-clock time reflects the user-visible runtime. GPU event time isolates device-side processing more precisely.

## Large-image strategy

The image is split into overlapping tiles. Each tile is processed independently. Only the **core** of each tile is written back to the global mask. This prevents double-counting and avoids box-merging problems from overlap regions.

## Notes on geospatial overlays

This code creates **image-space overlays** and chips. It does not claim true Google Maps georeferencing for raw Sentinel-1 measurement TIFFs, because those files are not necessarily terrain-corrected map products. For a true geospatial overlay, a geocoded GeoTIFF or terrain-corrected product would be needed.

## Suggested Coursera submission description

This project presents a GPU-accelerated SAR ship detection pipeline based on tiled CFAR with optional censoring. It supports both synthetic SAR-like imagery and real Sentinel-1 images, handles large files through overlapping tile-based processing, provides bounding-box overlays and detection chips, and reports both wall-clock and GPU event timing.
