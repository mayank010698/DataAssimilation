#!/bin/bash

# Define paths
DATA_STD="datasets/double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p200"
DATA_LARGE="datasets/double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p300"
PYTHON="/home/cnagda/miniconda3/envs/da/bin/python"

# Ensure we are in the project root
cd "$(dirname "$0")/.." || exit

echo "Starting Double Well Training Pipeline (Parallel)"
echo "==============================================="

# # 1. RF Proposal - Standard Variance (GPU 2)
# echo "[1/4] Starting RF Proposal - Standard Variance on GPU 2..."
# CUDA_VISIBLE_DEVICES=2 $PYTHON proposals/train_rf.py \
#     --data_dir "$DATA_STD" \
#     --output_dir "rf_runs/dw_std_mlp" \
#     --state_dim 1 \
#     --obs_dim 1 \
#     --architecture mlp \
#     --hidden_dim 128 \
#     --depth 3 \
#     --use_observations \
#     --train-cond-method concat \
#     --predict_delta \
#     --wandb_project dw-train \
#     --max_epochs 200 > logs/train_rf_dw_std.log 2>&1 &
# PID1=$!

# # 2. RF Proposal - Large Variance (GPU 4)
# echo "[2/4] Starting RF Proposal - Large Variance on GPU 4..."
# CUDA_VISIBLE_DEVICES=4 $PYTHON proposals/train_rf.py \
#     --data_dir "$DATA_LARGE" \
#     --output_dir "rf_runs/dw_large_mlp" \
#     --state_dim 1 \
#     --obs_dim 1 \
#     --architecture mlp \
#     --hidden_dim 128 \
#     --depth 3 \
#     --use_observations \
#     --train-cond-method concat \
#     --predict_delta \
#     --wandb_project dw-train \
#     --max_epochs 200 > logs/train_rf_dw_large.log 2>&1 &
# PID2=$!

# 3. FlowDAS - Standard Variance (GPU 5)
echo "[3/4] Starting FlowDAS - Standard Variance on GPU 5..."
CUDA_VISIBLE_DEVICES=5 $PYTHON scripts/train_flowdas.py \
    --config "$DATA_STD/config.yaml" \
    --data_dir "$DATA_STD" \
    --run_dir "runs/flowdas_dw_std" \
    --architecture mlp \
    --width 128 \
    --depth 3 \
    --wandb_project dw-train \
    --epochs 200 > logs/train_flowdas_dw_std.log 2>&1 &
PID3=$!

# 4. FlowDAS - Large Variance (GPU 6)
echo "[4/4] Starting FlowDAS - Large Variance on GPU 6..."
CUDA_VISIBLE_DEVICES=6 $PYTHON scripts/train_flowdas.py \
    --config "$DATA_LARGE/config.yaml" \
    --data_dir "$DATA_LARGE" \
    --run_dir "runs/flowdas_dw_large" \
    --architecture mlp \
    --width 128 \
    --depth 3 \
    --wandb_project dw-train \
    --epochs 200 > logs/train_flowdas_dw_large.log 2>&1 &
PID4=$!

echo "Jobs started with PIDs: $PID1, $PID2, $PID3, $PID4"
echo "Waiting for completion..."
wait
echo "==============================================="
echo "All parallel training runs completed."
