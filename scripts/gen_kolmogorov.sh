#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup environment (uncomment if needed)
# source activate
# conda activate da

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

# Avoid filling $HOME with JAX/XLA temp + cache.
# If $HOME is full, JAX GPU compilation (ptxas) can fail with "Internal error: writing file".
export TMPDIR="/data/da_outputs/tmp"
export XDG_CACHE_HOME="/data/da_outputs/jax_cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

DATA_DIR="/data/da_outputs/datasets"

echo "=========================================="
echo "Generating Kolmogorov Flow Datasets"
echo "=========================================="

# --- Fully observed (default, for sanity checking / baselines) ---
# obs_grid_size=150 → all 150×150=22,500 locations × 2 components = 45,000 obs
echo ""
echo "[1/2] Fully observed (obs_grid_size=150) with obs noise std 0.0 and 0.1"
python generate.py \
    --system kolmogorov \
    --num-trajectories 200 \
    --kol-num-steps 200 \
    --kol-warmup-steps 100 \
    --kol-dt 0.04 \
    --kol-parallel-gpus 8 \
    --obs-frequency 5 \
    --obs-grid-size 150 \
    --obs-noise-variations "0.0,0.1" \
    --re-min 500 \
    --re-max 1500 \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --seed 42 \
    --output-dir "$DATA_DIR" \
    --force

# --- Sparse observed (paper setting) ---
# obs_grid_size=10 → 10×10=100 spatial locations × 2 components = 200 obs
# ≈ 0.44% of the domain, matching the paper's regular observation case
echo ""
echo "[2/2] Sparse observed — paper setting (obs_grid_size=10) with obs noise std 0.0 and 0.1"
python generate.py \
    --system kolmogorov \
    --num-trajectories 200 \
    --kol-num-steps 200 \
    --kol-warmup-steps 100 \
    --kol-dt 0.04 \
    --kol-parallel-gpus 8 \
    --obs-frequency 5 \
    --obs-grid-size 10 \
    --obs-noise-variations "0.0,0.1" \
    --re-min 500 \
    --re-max 1500 \
    --train-ratio 0.6 \
    --val-ratio 0.2 \
    --test-ratio 0.2 \
    --seed 42 \
    --output-dir "$DATA_DIR" \
    --force

echo ""
echo "=========================================="
echo "Done."
echo "=========================================="
