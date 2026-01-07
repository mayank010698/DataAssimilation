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
echo "Starting Parallel Evaluation for Lorenz 96 50D (FlowDAS Guidance)"
echo "=========================================="
echo "Configuration:"
echo "- Using 'No Obs' trained models (unconditional score)"
echo "- Applying guidance with FULL observations (50/50)"
echo "=========================================="

# Dataset paths (50D, Full Obs)
DATA_DIR_NO_NOISE="/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p200_freq1_comp50of50_arctan_init3p000"
DATA_DIR_WITH_NOISE="/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p200_freq1_comp50of50_arctan_pnoise0p100_init3p000"

# Model paths (No Obs models)
MODEL_NO_NOISE="/data/da_outputs/runs_flowdas/l96_nonoise_no_obs_1230/checkpoint_best.pth"
MODEL_WITH_NOISE="/data/da_outputs/runs_flowdas/l96_noise0p1_no_obs_1230/checkpoint_best.pth"

# 1. Eval 1: No Noise (GPU 0)
echo "Starting Eval 1: No Noise, Full Guidance on GPU 0..."
CUDA_VISIBLE_DEVICES=3 nohup python scripts/eval_flowdas.py \
    --checkpoint "$MODEL_NO_NOISE" \
    --data_dir "$DATA_DIR_NO_NOISE" \
    --results_dir "/data/da_outputs/results/flowdas_eval/l96_nonoise_guidance_full" \
    --run_name "eval_flowdas_l96_nonoise_guidance_full" \
    --num_trajs 10 \
    --use_wandb \
    > logs/eval_flowdas_l96_nonoise_guidance_full.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

# 2. Eval 2: With Noise (GPU 1)
echo "Starting Eval 2: With Noise, Full Guidance on GPU 1..."
CUDA_VISIBLE_DEVICES=3 nohup python scripts/eval_flowdas.py \
    --checkpoint "$MODEL_WITH_NOISE" \
    --data_dir "$DATA_DIR_WITH_NOISE" \
    --results_dir "/data/da_outputs/results/flowdas_eval/l96_noise0p1_guidance_full" \
    --run_name "eval_flowdas_l96_noise0p1_guidance_full" \
    --num_trajs 10 \
    --use_wandb \
    > logs/eval_flowdas_l96_noise0p1_guidance_full.log 2>&1 &
PID2=$!
echo "Started PID $PID2"

echo "=========================================="
echo "All evaluation experiments started in background."
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/eval_flowdas_l96_*.log"
echo "=========================================="

