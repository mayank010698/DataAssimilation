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
echo "Lorenz96 (10D) Training Experiments"
echo "=========================================="
echo "Configuration:"
echo "- 8 Experiments total (RF and FlowDAS)"
echo "- Variations: No Noise vs Noise, Full (10) vs Half (5) Obs"
echo "- GPUs: 0, 1, 2 (cycled)"
echo "=========================================="

# Common parameters
STATE_DIM=10
BATCH_SIZE=512
LR=3e-4
MAX_EPOCHS=500
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
COND_EMBED_DIM=128

# Dataset paths (10D)
DATA_DIR_NO_NOISE="/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp10of10_arctan_init3p000"
DATA_DIR_WITH_NOISE="/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp10of10_arctan_pnoise0p100_init3p000"

# Indices strings
COMPONENTS_FULL="0,1,2,3,4,5,6,7,8,9"
COMPONENTS_HALF="0,2,4,6,8"

# --- EXPERIMENTS ---

# 1. RF - No Noise - Full Obs (10) - GPU 0
echo "Starting Exp 1: RF, No Noise, Full Obs (10) (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 nohup python proposals/train_rf.py \
    --data_dir "$DATA_DIR_NO_NOISE" \
    --output_dir "/data/da_outputs/rf_runs/l96_10d_nonoise_full" \
    --state_dim $STATE_DIM \
    --obs_components "$COMPONENTS_FULL" \
    --architecture resnet1d \
    --channels $CHANNELS \
    --num_blocks $NUM_BLOCKS \
    --kernel_size $KERNEL_SIZE \
    --train-cond-method adaln \
    --cond_embed_dim $COND_EMBED_DIM \
    --use_observations \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --max_epochs $MAX_EPOCHS \
    --gpus 1 \
    --evaluate \
    > logs/l96_10d_rf_nonoise_full.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

# 2. FlowDAS - No Noise - Full Obs (10) - GPU 1
echo "Starting Exp 2: FlowDAS, No Noise, Full Obs (10) (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_flowdas.py \
    --config "$DATA_DIR_NO_NOISE/config.yaml" \
    --data_dir "$DATA_DIR_NO_NOISE" \
    --run_dir "/data/da_outputs/runs_flowdas/l96_10d_nonoise_full" \
    --obs_components "$COMPONENTS_FULL" \
    --architecture resnet1d \
    --channels $CHANNELS \
    --num_blocks $NUM_BLOCKS \
    --kernel_size $KERNEL_SIZE \
    --use_observations \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $MAX_EPOCHS \
    --use_wandb \
    --evaluate \
    > logs/l96_10d_flowdas_nonoise_full.log 2>&1 &
PID2=$!
echo "Started PID $PID2"

# 3. RF - No Noise - Half Obs (5) - GPU 2
echo "Starting Exp 3: RF, No Noise, Half Obs (5) (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 nohup python proposals/train_rf.py \
    --data_dir "$DATA_DIR_NO_NOISE" \
    --output_dir "/data/da_outputs/rf_runs/l96_10d_nonoise_half" \
    --state_dim $STATE_DIM \
    --obs_components "$COMPONENTS_HALF" \
    --architecture resnet1d \
    --channels $CHANNELS \
    --num_blocks $NUM_BLOCKS \
    --kernel_size $KERNEL_SIZE \
    --train-cond-method adaln \
    --cond_embed_dim $COND_EMBED_DIM \
    --use_observations \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --max_epochs $MAX_EPOCHS \
    --gpus 1 \
    --evaluate \
    > logs/l96_10d_rf_nonoise_half.log 2>&1 &
PID3=$!
echo "Started PID $PID3"

# 4. FlowDAS - No Noise - Half Obs (5) - GPU 0
echo "Starting Exp 4: FlowDAS, No Noise, Half Obs (5) (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_flowdas.py \
    --config "$DATA_DIR_NO_NOISE/config.yaml" \
    --data_dir "$DATA_DIR_NO_NOISE" \
    --run_dir "/data/da_outputs/runs_flowdas/l96_10d_nonoise_half" \
    --obs_components "$COMPONENTS_HALF" \
    --architecture resnet1d \
    --channels $CHANNELS \
    --num_blocks $NUM_BLOCKS \
    --kernel_size $KERNEL_SIZE \
    --use_observations \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $MAX_EPOCHS \
    --use_wandb \
    --evaluate \
    > logs/l96_10d_flowdas_nonoise_half.log 2>&1 &
PID4=$!
echo "Started PID $PID4"

# 5. RF - With Noise - Full Obs (10) - GPU 1
echo "Starting Exp 5: RF, With Noise, Full Obs (10) (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 nohup python proposals/train_rf.py \
    --data_dir "$DATA_DIR_WITH_NOISE" \
    --output_dir "/data/da_outputs/rf_runs/l96_10d_noise_full" \
    --state_dim $STATE_DIM \
    --obs_components "$COMPONENTS_FULL" \
    --architecture resnet1d \
    --channels $CHANNELS \
    --num_blocks $NUM_BLOCKS \
    --kernel_size $KERNEL_SIZE \
    --train-cond-method adaln \
    --cond_embed_dim $COND_EMBED_DIM \
    --use_observations \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --max_epochs $MAX_EPOCHS \
    --gpus 1 \
    --evaluate \
    > logs/l96_10d_rf_noise_full.log 2>&1 &
PID5=$!
echo "Started PID $PID5"

# 6. FlowDAS - With Noise - Full Obs (10) - GPU 2
echo "Starting Exp 6: FlowDAS, With Noise, Full Obs (10) (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 nohup python scripts/train_flowdas.py \
    --config "$DATA_DIR_WITH_NOISE/config.yaml" \
    --data_dir "$DATA_DIR_WITH_NOISE" \
    --run_dir "/data/da_outputs/runs_flowdas/l96_10d_noise_full" \
    --obs_components "$COMPONENTS_FULL" \
    --architecture resnet1d \
    --channels $CHANNELS \
    --num_blocks $NUM_BLOCKS \
    --kernel_size $KERNEL_SIZE \
    --use_observations \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $MAX_EPOCHS \
    --use_wandb \
    --evaluate \
    > logs/l96_10d_flowdas_noise_full.log 2>&1 &
PID6=$!
echo "Started PID $PID6"

# 7. RF - With Noise - Half Obs (5) - GPU 0
echo "Starting Exp 7: RF, With Noise, Half Obs (5) (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 nohup python proposals/train_rf.py \
    --data_dir "$DATA_DIR_WITH_NOISE" \
    --output_dir "/data/da_outputs/rf_runs/l96_10d_noise_half" \
    --state_dim $STATE_DIM \
    --obs_components "$COMPONENTS_HALF" \
    --architecture resnet1d \
    --channels $CHANNELS \
    --num_blocks $NUM_BLOCKS \
    --kernel_size $KERNEL_SIZE \
    --train-cond-method adaln \
    --cond_embed_dim $COND_EMBED_DIM \
    --use_observations \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --max_epochs $MAX_EPOCHS \
    --gpus 1 \
    --evaluate \
    > logs/l96_10d_rf_noise_half.log 2>&1 &
PID7=$!
echo "Started PID $PID7"

# 8. FlowDAS - With Noise - Half Obs (5) - GPU 1
echo "Starting Exp 8: FlowDAS, With Noise, Half Obs (5) (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_flowdas.py \
    --config "$DATA_DIR_WITH_NOISE/config.yaml" \
    --data_dir "$DATA_DIR_WITH_NOISE" \
    --run_dir "/data/da_outputs/runs_flowdas/l96_10d_noise_half" \
    --obs_components "$COMPONENTS_HALF" \
    --architecture resnet1d \
    --channels $CHANNELS \
    --num_blocks $NUM_BLOCKS \
    --kernel_size $KERNEL_SIZE \
    --use_observations \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $MAX_EPOCHS \
    --use_wandb \
    --evaluate \
    > logs/l96_10d_flowdas_noise_half.log 2>&1 &
PID8=$!
echo "Started PID $PID8"

echo "=========================================="
echo "All 8 training experiments started in background."
echo "Logs are being written to logs/"
echo "Monitor with: tail -f logs/l96_10d_*.log"
echo "=========================================="
