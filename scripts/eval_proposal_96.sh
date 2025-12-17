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
echo "Starting Parallel Evaluation for Lorenz 96 (NO Guidance)"
echo "=========================================="

# 1. l96_nonoise_no_obs (GPU 0)
echo "Starting Eval 1: l96_nonoise_no_obs on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/l96_nonoise_no_obs/final_model.ckpt" \
    --data-dir "datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan" \
    --run-name "eval_l96_nonoise_no_obs" \
    > logs/eval_l96_nonoise_no_obs.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

# 2. l96_nonoise_tenth (GPU 1)
echo "Starting Eval 2: l96_nonoise_tenth on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/l96_nonoise_tenth/final_model.ckpt" \
    --data-dir "datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan" \
    --run-name "eval_l96_nonoise_tenth" \
    > logs/eval_l96_nonoise_tenth.log 2>&1 &
PID2=$!
echo "Started PID $PID2"

# 3. l96_nonoise_half (GPU 2)
echo "Starting Eval 3: l96_nonoise_half on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/l96_nonoise_half/final_model.ckpt" \
    --data-dir "datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan" \
    --run-name "eval_l96_nonoise_half" \
    > logs/eval_l96_nonoise_half.log 2>&1 &
PID3=$!
echo "Started PID $PID3"

# 4. l96_nonoise_full (GPU 3)
echo "Starting Eval 4: l96_nonoise_full on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/l96_nonoise_full/final_model.ckpt" \
    --data-dir "datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan" \
    --run-name "eval_l96_nonoise_full" \
    > logs/eval_l96_nonoise_full.log 2>&1 &
PID4=$!
echo "Started PID $PID4"

# 5. l96_noise0p1_no_obs (GPU 4)
echo "Starting Eval 5: l96_noise0p1_no_obs on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/l96_noise0p1_no_obs/final_model.ckpt" \
    --data-dir "datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_pnoise0p100" \
    --run-name "eval_l96_noise0p1_no_obs" \
    > logs/eval_l96_noise0p1_no_obs.log 2>&1 &
PID5=$!
echo "Started PID $PID5"

# 6. l96_noise0p1_tenth (GPU 5)
echo "Starting Eval 6: l96_noise0p1_tenth on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/l96_noise0p1_tenth/final_model.ckpt" \
    --data-dir "datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_pnoise0p100" \
    --run-name "eval_l96_noise0p1_tenth" \
    > logs/eval_l96_noise0p1_tenth.log 2>&1 &
PID6=$!
echo "Started PID $PID6"

# 7. l96_noise0p1_half (GPU 6)
echo "Starting Eval 7: l96_noise0p1_half on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/l96_noise0p1_half/final_model.ckpt" \
    --data-dir "datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_pnoise0p100" \
    --run-name "eval_l96_noise0p1_half" \
    > logs/eval_l96_noise0p1_half.log 2>&1 &
PID7=$!
echo "Started PID $PID7"

# 8. l96_noise0p1_full (GPU 7)
echo "Starting Eval 8: l96_noise0p1_full on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/l96_noise0p1_full/final_model.ckpt" \
    --data-dir "datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_pnoise0p100" \
    --run-name "eval_l96_noise0p1_full" \
    > logs/eval_l96_noise0p1_full.log 2>&1 &
PID8=$!
echo "Started PID $PID8"

echo "=========================================="
echo "All 8 evaluation experiments started in background."
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/eval_l96_*.log"
echo "=========================================="
