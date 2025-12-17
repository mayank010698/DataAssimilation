#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# GPU Selection
export CUDA_VISIBLE_DEVICES=${GPU_DEVICE:-0}
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

echo "=========================================="
echo "Training RF Proposals (With Observations)"
echo "=========================================="

echo ""
echo "Training: 1024 trajs, len=100 (Scaled), WITH observations"
echo "----------------------------------------"
python proposals/train_rf.py \
    --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p000_freq1_comp0_arctan" \
    --output_dir "rf_runs/lorenz63_n1024_len100_obs0p000_freq1_comp0_arctan" \
    --max_epochs 500 \
    --batch_size 64 \
    --gpus 1 \
    --use_observations \
    --obs_dim 1 \
    --evaluate

echo ""
echo "=========================================="
echo "Training with observations completed!"
echo "=========================================="

