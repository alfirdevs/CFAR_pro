#!/usr/bin/env bash
set -euo pipefail

python run.py   --mode synthetic   --use_gpu 1   --tile_size 1024   --overlap 64   --guard_cells 4   --training_cells 12   --pfa 1e-4   --threshold_scale 1.0   --censor 1   --censor_percentile 99.5   --output_dir results
