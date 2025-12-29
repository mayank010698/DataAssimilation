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
echo "Starting Parallel Training for FlowDAS (Lorenz 63)"
echo "=========================================="

# 1. Lorenz 63, 0 noise, 1 dim observed (GPU 0)
echo "Starting Exp 1: L63, 0 noise, 1 dim obs on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_flowdas.py \
    --config "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan/config.yaml" \
    --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan" \
    --run_dir "runs_flowdas/lorenz63_comp0_arctan" \
    --architecture mlp \
    --width 128 \
    --depth 2 \
    --batch_size 64 \
    --epochs 500 \
    --lr 1e-3 \
    --device cuda \
    --use_wandb \
    > logs/train_flowdas_l63_comp0.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

# 2. Lorenz 63, 0 noise, 2 dim observed (GPU 1)
echo "Starting Exp 2: L63, 0 noise, 2 dim obs on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_flowdas.py \
    --config "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan/config.yaml" \
    --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan" \
    --run_dir "runs_flowdas/lorenz63_comp01_arctan" \
    --architecture mlp \
    --width 128 \
    --depth 2 \
    --batch_size 64 \
    --epochs 500 \
    --lr 1e-3 \
    --device cuda \
    --use_wandb \
    > logs/train_flowdas_l63_comp01.log 2>&1 &
PID2=$!
echo "Started PID $PID2"

# 3. Lorenz 63, 0 noise, 3 dim observed (GPU 2)
echo "Starting Exp 3: L63, 0 noise, 3 dim obs on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python scripts/train_flowdas.py \
    --config "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan/config.yaml" \
    --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan" \
    --run_dir "runs_flowdas/lorenz63_comp012_arctan" \
    --architecture mlp \
    --width 128 \
    --depth 2 \
    --batch_size 64 \
    --epochs 500 \
    --lr 1e-3 \
    --device cuda \
    --use_wandb \
    > logs/train_flowdas_l63_comp012.log 2>&1 &
PID3=$!
echo "Started PID $PID3"

# 4. Lorenz 63, 0.25 pnoise, 1 dim observed (GPU 3)
echo "Starting Exp 4: L63, 0.25 pnoise, 1 dim obs on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup python scripts/train_flowdas.py \
    --config "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan_pnoise0p250/config.yaml" \
    --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan_pnoise0p250" \
    --run_dir "runs_flowdas/lorenz63_comp0_arctan_pnoise0p250" \
    --architecture mlp \
    --width 128 \
    --depth 2 \
    --batch_size 64 \
    --epochs 500 \
    --lr 1e-3 \
    --device cuda \
    --use_wandb \
    > logs/train_flowdas_l63_comp0_pnoise.log 2>&1 &
PID4=$!
echo "Started PID $PID4"

# 5. Lorenz 63, 0.25 pnoise, 2 dim observed (GPU 4)
echo "Starting Exp 5: L63, 0.25 pnoise, 2 dim obs on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup python scripts/train_flowdas.py \
    --config "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan_pnoise0p250/config.yaml" \
    --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan_pnoise0p250" \
    --run_dir "runs_flowdas/lorenz63_comp01_arctan_pnoise0p250" \
    --architecture mlp \
    --width 128 \
    --depth 2 \
    --batch_size 64 \
    --epochs 500 \
    --lr 1e-3 \
    --device cuda \
    --use_wandb \
    > logs/train_flowdas_l63_comp01_pnoise.log 2>&1 &
PID5=$!
echo "Started PID $PID5"

# 6. Lorenz 63, 0.25 pnoise, 3 dim observed (GPU 5)
echo "Starting Exp 6: L63, 0.25 pnoise, 3 dim obs on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup python scripts/train_flowdas.py \
    --config "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250/config.yaml" \
    --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250" \
    --run_dir "runs_flowdas/lorenz63_comp012_arctan_pnoise0p250" \
    --architecture mlp \
    --width 128 \
    --depth 2 \
    --batch_size 64 \
    --epochs 500 \
    --lr 1e-3 \
    --device cuda \
    --use_wandb \
    > logs/train_flowdas_l63_comp012_pnoise.log 2>&1 &
PID6=$!
echo "Started PID $PID6"

echo "=========================================="
echo "All FlowDAS training experiments started in background."
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/train_flowdas_*.log"
echo "=========================================="

