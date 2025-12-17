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
echo "Lorenz96 ResNet1D Training"
echo "=========================================="

# Lorenz96 with ResNet1D + AdaLN conditioning
echo "Starting: Lorenz96 ResNet1D with AdaLN conditioning..."
CUDA_VISIBLE_DEVICES=7 nohup python proposals/train_rf.py \
    --data_dir "datasets/lorenz96_n4096_len100_dt0p0100_obs0p100_freq1_comp50of50_arctan" \
    --output_dir "rf_runs/lorenz96_resnet1d_adaln" \
    --state_dim 50 \
    --obs_dim 50 \
    --architecture resnet1d \
    --channels 64 \
    --num_blocks 10 \
    --kernel_size 5 \
    --conditioning_method adaln \
    --cond_embed_dim 128 \
    --use_observations \
    --batch_size 1024 \
    --learning_rate 4e-3 \
    --max_epochs 500 \
    --gpus 1 \
    --evaluate \
    > logs/train_lorenz96_resnet1d_adaln.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

echo "=========================================="
echo "Training started in background."
echo "Log: logs/train_lorenz96_resnet1d_adaln.log"
echo "Monitor with: tail -f logs/train_lorenz96_resnet1d_adaln.log"
echo "=========================================="

