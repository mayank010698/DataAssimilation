#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Lorenz96 Deterministic Model Training"
echo "=========================================="
echo "This is a sanity check to verify the ResNet1D architecture"
echo "before using it for flow matching."
echo "Dataset: 10-dimensional Lorenz96 with 10 observed components"
echo "=========================================="

# Lorenz96 with ResNet1D (Deterministic) + AdaLN conditioning
# Using same hyperparameters as the flow matching training for fair comparison
# Note: ResNet1DDeterministic always uses outer residual (x_next = x_prev + delta)
# This is baked into the architecture, no flag needed.
# Training stability fixes applied:
#   - Lower LR (3e-4 default, can override)
#   - L2 regularization on output layer (--output_l2_reg 1e-4)
#   - More aggressive scheduler (patience=10)
echo "Starting: Lorenz96 Deterministic ResNet1D ..."
CUDA_VISIBLE_DEVICES=7 nohup python proposals/train_deterministic.py \
    --data_dir "datasets/lorenz96_n4096_len100_dt0p0100_obs0p100_freq1_comp10of10_arctan" \
    --output_dir "deterministic_runs/lorenz96_10dim_resnet1d_no_resid" \
    --state_dim 10 \
    --obs_dim 10 \
    --channels 64 \
    --num_blocks 10 \
    --kernel_size 5 \
    --conditioning_method concat \
    --cond_embed_dim 128 \
    --use_observations \
    --batch_size 256 \
    --learning_rate 3e-4 \
    --weight_decay 1e-4 \
    --scheduler_patience 10 \
    --output_l2_reg 1e-4 \
    --max_epochs 500 \
    --gpus 1 \
    --evaluate \
    --no_predict_residual \
    > logs/train_deterministic_lorenz96_10dim_no_resid.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

echo "=========================================="
echo "Training started in background."
echo "Log: logs/train_deterministic_lorenz96_10dim.log"
echo "Monitor with: tail -f logs/train_deterministic_lorenz96_10dim.log"
echo "=========================================="

