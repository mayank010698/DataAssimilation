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
echo "Lorenz96 (20D) DBF Training Experiments"
echo "=========================================="
echo "Training RF proposals for 2 scenarios (pnoise 0.1 and 0.2)"
echo "=========================================="

# Common parameters
STATE_DIM=20
BATCH_SIZE=1024
LR=3e-4
MAX_EPOCHS=500
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
COND_EMBED_DIM=128
WANDB_PROJECT="rf-train-96-dbf"
SEED=0

# Generate 0..19 string
OBS_COMPONENTS=$(seq -s, 0 19)

# Base path
DATA_BASE="/data/da_outputs/datasets"

# 1. pnoise 0.1, quad operator, 20/20 observed
JOB1_GPU=6
JOB1_NAME="l96_dbf_20d_quad_pnoise0p1"
JOB1_DATA="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp20of20_quad_capped_10_pnoise0p100_initUnim10_10"

# 2. pnoise 0.2, quad operator, 20/20 observed
JOB2_GPU=7
JOB2_NAME="l96_dbf_20d_quad_pnoise0p2"
JOB2_DATA="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp20of20_quad_capped_10_pnoise0p200_initUnim10_10"


# Function to launch job
launch_job() {
    GPU=$1
    NAME=$2
    DATA=$3

    echo "Starting $NAME on GPU $GPU..."
    echo "  Data: $DATA"

    CUDA_VISIBLE_DEVICES=$GPU nohup python proposals/train_rf.py \
        --data_dir "$DATA" \
        --output_dir "rf_runs/$NAME" \
        --state_dim $STATE_DIM \
        --obs_components "$OBS_COMPONENTS" \
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
        --wandb_project "$WANDB_PROJECT" \
        --seed $SEED \
        --evaluate \
        > "logs/${NAME}.log" 2>&1 &

    PID=$!
    echo "  PID: $PID"
}

# Launch both jobs
launch_job $JOB1_GPU $JOB1_NAME "$JOB1_DATA"
launch_job $JOB2_GPU $JOB2_NAME "$JOB2_DATA"

echo "=========================================="
echo "Both 20D training experiments started."
echo "Logs in logs/"
echo "=========================================="
