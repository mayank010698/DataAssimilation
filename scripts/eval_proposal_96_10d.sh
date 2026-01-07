#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

# Create logs directory if it doesn't exist
mkdir -p logs

# Dataset paths (10D)
DATA_DIR_NO_NOISE="/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp10of10_arctan_init3p000"
DATA_DIR_WITH_NOISE="/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp10of10_arctan_pnoise0p100_init3p000"

echo "=========================================="
echo "Starting Parallel Evaluation for Lorenz 96 10D (RF Proposals)"
echo "=========================================="

# 1. l96_10d_nonoise_half (GPU 0)
echo "Starting Eval 1: l96_10d_nonoise_half on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python proposals/eval_proposal.py \
    --checkpoint "/data/da_outputs/rf_runs/l96_10d_nonoise_half/final_model.ckpt" \
    --data-dir "$DATA_DIR_NO_NOISE" \
    --run-name "eval_l96_10d_nonoise_half" \
    > logs/eval_l96_10d_nonoise_half.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

# 2. l96_10d_nonoise_full (GPU 1)
echo "Starting Eval 2: l96_10d_nonoise_full on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python proposals/eval_proposal.py \
    --checkpoint "/data/da_outputs/rf_runs/l96_10d_nonoise_full/final_model.ckpt" \
    --data-dir "$DATA_DIR_NO_NOISE" \
    --run-name "eval_l96_10d_nonoise_full" \
    > logs/eval_l96_10d_nonoise_full.log 2>&1 &
PID2=$!
echo "Started PID $PID2"

# 3. l96_10d_noise_half (GPU 2)
echo "Starting Eval 3: l96_10d_noise_half on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python proposals/eval_proposal.py \
    --checkpoint "/data/da_outputs/rf_runs/l96_10d_noise_half/final_model.ckpt" \
    --data-dir "$DATA_DIR_WITH_NOISE" \
    --run-name "eval_l96_10d_noise_half" \
    > logs/eval_l96_10d_noise_half.log 2>&1 &
PID3=$!
echo "Started PID $PID3"

# 4. l96_10d_noise_full (GPU 3)
echo "Starting Eval 4: l96_10d_noise_full on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup python proposals/eval_proposal.py \
    --checkpoint "/data/da_outputs/rf_runs/l96_10d_noise_full/final_model.ckpt" \
    --data-dir "$DATA_DIR_WITH_NOISE" \
    --run-name "eval_l96_10d_noise_full" \
    > logs/eval_l96_10d_noise_full.log 2>&1 &
PID4=$!
echo "Started PID $PID4"

echo "=========================================="
echo "All 4 evaluation experiments started in background."
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/eval_l96_10d_*.log"
echo "=========================================="

