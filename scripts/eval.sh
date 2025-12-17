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
echo "Starting Parallel Evaluations"
echo "=========================================="

DATA_DIR="datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan"
NO_NOISE_DATA_DIR="datasets/lorenz63_n1024_len100_dt0p0100_obs0p000_freq1_comp0_arctan"

# 1. RF with observations
# Checkpoint: ...arctan/checkpoints/rf-epoch=280-val_loss=0.000219.ckpt
echo "Starting Exp 1: RF with observations on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python eval.py \
    --data-dir "$DATA_DIR" \
    --n-particles 100 \
    --proposal-type rf \
    --rf-checkpoint "/home/cnagda/DataAssimilation/rf_runs/lorenz63_n1024_len100_obs0p250_freq1_comp0_arctan/checkpoints/rf-epoch=280-val_loss=0.000219.ckpt" \
    --rf-sampling-steps 100 \
    --rf-likelihood-steps 100 \
    --batch-size 64 \
    --device cuda \
    --experiment-label "eval_rf_obs" \
    > logs/eval_rf_obs.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

# 2. RF without observations
# Checkpoint: ...arctan_obsFalse/checkpoints/rf-epoch=329-val_loss=0.000226.ckpt
echo "Starting Exp 2: RF with no observations on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python eval.py \
    --data-dir "$DATA_DIR" \
    --n-particles 100 \
    --proposal-type rf \
    --rf-checkpoint "/home/cnagda/DataAssimilation/rf_runs/lorenz63_n1024_len100_obs0p250_freq1_comp0_arctan_obsFalse/checkpoints/rf-epoch=329-val_loss=0.000226.ckpt" \
    --rf-sampling-steps 100 \
    --rf-likelihood-steps 100 \
    --batch-size 64 \
    --device cuda \
    --experiment-label "eval_rf_no_obs" \
    > logs/eval_rf_obs.log 2>&1 &
PID2=$!
echo "Started PID $PID2"

# 3. RF with no noise observations
echo "Starting Exp 3: RF with no noise observations on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python eval.py \
    --data-dir "$NO_NOISE_DATA_DIR" \
    --n-particles 100 \
    --proposal-type rf \
    --rf-checkpoint "/home/cnagda/DataAssimilation/rf_runs/lorenz63_n1024_len100_obs0p000_freq1_comp0_arctan/checkpoints/rf-epoch=232-val_loss=0.000332.ckpt" \
    --rf-sampling-steps 100 \
    --rf-likelihood-steps 100 \
    --batch-size 64 \
    --device cuda \
    --experiment-label "eval_rf_no_noise_obs" \
    > logs/eval_rf_no_noise_obs.log 2>&1 &
PID3=$!
echo "Started PID $PID3"

# 4. Standard BPF
echo "Starting Exp 4: Standard BPF on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup python eval.py \
    --data-dir "$DATA_DIR" \
    --n-particles 100 \
    --proposal-type transition \
    --batch-size 64 \
    --device cuda \
    --experiment-label "eval_bpf_standard" \
    > logs/eval_bpf.log 2>&1 &
PID4=$!
echo "Started PID $PID4"

echo "=========================================="
echo "All experiments started in background."
echo "Logs are being written to logs/"
echo "  - logs/eval_rf_no_obs.log"
echo "  - logs/eval_rf_obs.log"
echo "  - logs/eval_bpf.log"
echo "You can monitor them with: tail -f logs/*.log"
echo "=========================================="
