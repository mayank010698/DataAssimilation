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
echo "Starting Evaluations (Lorenz 96) - RF Proposal"
echo "=========================================="

DATA_DIR="datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_pnoise0p100"
RF_CKPT="rf_runs/l96_noise0p1_full/final_model.ckpt"

# Common args
# Limiting to 10 trajectories by default due to high computational cost of RF + high particles
# Remove --num-eval-trajectories 10 to evaluate on the full dataset
COMMON_ARGS="--data-dir $DATA_DIR --proposal-type rf --rf-checkpoint $RF_CKPT --rf-sampling-steps 100 --rf-likelihood-steps 100 --process-noise-std 0.1 --obs-noise-std 0.1 --num-eval-trajectories 10 --device cuda"

# # 1. 1000 Particles
# echo "Starting Run 1: 1000 Particles on GPU 0..."
# CUDA_VISIBLE_DEVICES=0 nohup python eval.py \
#     $COMMON_ARGS \
#     --n-particles 1000 \
#     --batch-size 128 \
#     --experiment-label eval_96_rf_1k \
#     > logs/eval_96_rf_1k.log 2>&1 &
# PID1=$!
# echo "Started PID $PID1"

# 1a. 1000 Particles, Rademacher, 1 Probe
echo "Starting Run 1a: 1000 Particles (Rademacher, 1 probe) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python eval.py \
    $COMMON_ARGS \
    --n-particles 1000 \
    --batch-size 128 \
    --rf-trace-estimator rademacher \
    --rf-num-probes 1 \
    --experiment-label eval_96_rf_1k_rad1 \
    > logs/eval_96_rf_1k_rad1.log 2>&1 &
PID1A=$!
echo "Started PID $PID1A"

# 1b. 1000 Particles, Rademacher, 5 Probes
echo "Starting Run 1b: 1000 Particles (Rademacher, 5 probes) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python eval.py \
    $COMMON_ARGS \
    --n-particles 1000 \
    --batch-size 128 \
    --rf-trace-estimator rademacher \
    --rf-num-probes 5 \
    --experiment-label eval_96_rf_1k_rad5 \
    > logs/eval_96_rf_1k_rad5.log 2>&1 &
PID1B=$!
echo "Started PID $PID1B"

# 2. 1000 Particles, standard transition proposal
# echo "Starting Run 2: 1000 Particles (Transition) on GPU 1..."
# CUDA_VISIBLE_DEVICES=1 nohup python eval.py \
#     --data-dir $DATA_DIR \
#     --proposal-type transition \
#     --num-eval-trajectories 10 \
#     --device cuda \
#     --n-particles 1000 \
#     --batch-size 128 \
#     --experiment-label eval_96_transition_1k \
#     > logs/eval_96_transition_1k.log 2>&1 &
# PID2=$!
# echo "Started PID $PID2"

# # 3. 5000 Particles
# echo "Starting Run 3: 5000 Particles on GPU 2..."
# CUDA_VISIBLE_DEVICES=2 nohup python eval.py \
#     $COMMON_ARGS \
#     --n-particles 5000 \
#     --batch-size 128 \
#     --experiment-label eval_96_rf_5k \
#     > logs/eval_96_rf_5k.log 2>&1 &
# PID3=$!
# echo "Started PID $PID3"

# # 3. 10000 Particles
# echo "Starting Run 3: 10000 Particles on GPU 2..."
# CUDA_VISIBLE_DEVICES=2 nohup python eval.py \
#     $COMMON_ARGS \
#     --n-particles 10000 \
#     --batch-size 128 \
#     --experiment-label eval_96_rf_10k \
#     > logs/eval_96_rf_10k.log 2>&1 &
# PID3=$!
# echo "Started PID $PID3"

echo "=========================================="
echo "All experiments started in background."
echo "Logs are being written to logs/"
echo "  - logs/eval_96_rf_1k.log"
echo "  - logs/eval_96_rf_1k_rad1.log"
echo "  - logs/eval_96_rf_1k_rad5.log"
echo "  - logs/eval_96_transition_1k.log"
echo "  - logs/eval_96_rf_5k.log"
echo "  - logs/eval_96_rf_10k.log"
echo "You can monitor them with: tail -f logs/eval_96_rf_*.log"
echo "=========================================="
