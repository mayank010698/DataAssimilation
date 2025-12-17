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
echo "Starting Debug Evaluations (Lorenz 96)"
echo "=========================================="

DATA_DIR="datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan"

# 1. Low process noise
echo "Starting Exp 1: Debug Low Process Noise on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python eval.py \
    --data-dir "$DATA_DIR" \
    --proposal-type transition \
    --n-particles 10000 \
    --batch-size 128 \
    --process-noise-std 0.05 \
    --experiment-label debug_low_pnoise \
    --num-eval-trajectories 10 \
    --device cuda \
    > logs/eval_96_debug_low_pnoise.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

# 2. Inflated observation noise
echo "Starting Exp 2: Debug Inflated Obs Noise on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python eval.py \
    --data-dir "$DATA_DIR" \
    --proposal-type transition \
    --n-particles 10000 \
    --batch-size 128 \
    --obs-noise-std 1.0 \
    --experiment-label debug_inflated \
    --num-eval-trajectories 10 \
    --device cuda \
    > logs/eval_96_debug_inflated.log 2>&1 &
PID2=$!
echo "Started PID $PID2"

echo "=========================================="
echo "All experiments started in background."
echo "Logs are being written to logs/"
echo "  - logs/eval_96_debug_low_pnoise.log"
echo "  - logs/eval_96_debug_inflated.log"
echo "You can monitor them with: tail -f logs/eval_96_debug_*.log"
echo "=========================================="
