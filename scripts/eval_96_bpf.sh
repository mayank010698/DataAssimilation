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
echo "Starting Evaluations (Lorenz 96) - BPF (Transition Proposal)"
echo "=========================================="

# DATA_DIR from scripts/eval_96_steps.sh
DATA_DIR="datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_init3p000"

# Common args
COMMON_ARGS="--data-dir $DATA_DIR --proposal-type transition --process-noise-std 0.1 --obs-noise-std 0.1 --num-eval-trajectories 50 --device cuda"

# 1. 1000 Particles
echo "Starting Run 1: 1000 Particles (Transition) on GPU 1..."
CUDA_VISIBLE_DEVICES=0 nohup python eval.py \
    $COMMON_ARGS \
    --n-particles 1000 \
    --batch-size 128 \
    --resampling-threshold 0.1 \
    --experiment-label eval_96_bpf_1k \
    > logs/eval_96_bpf_1k_threshold0p1.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

# # 2. 5000 Particles
# echo "Starting Run 2: 5000 Particles (Transition) on GPU 2..."
# CUDA_VISIBLE_DEVICES=1 nohup python eval.py \
#     $COMMON_ARGS \
#     --n-particles 5000 \
#     --batch-size 32 \
#     --experiment-label eval_96_bpf_5k \
#     > logs/eval_96_bpf_5k.log 2>&1 &
# PID2=$!
# echo "Started PID $PID2"

# # 3. 10000 Particles
# echo "Starting Run 3: 10000 Particles (Transition) on GPU 3..."
# CUDA_VISIBLE_DEVICES=2 nohup python eval.py \
#     $COMMON_ARGS \
#     --n-particles 10000 \
#     --batch-size 32 \
#     --experiment-label eval_96_bpf_10k \
#     > logs/eval_96_bpf_10k.log 2>&1 &
# PID3=$!
# echo "Started PID $PID3"

echo "=========================================="
echo "All experiments started in background."
echo "Logs are being written to logs/"
echo "  - logs/eval_96_bpf_1k.log"
echo "  - logs/eval_96_bpf_5k.log"
echo "  - logs/eval_96_bpf_10k.log"
echo "You can monitor them with: tail -f logs/eval_96_bpf_*.log"
echo "=========================================="

