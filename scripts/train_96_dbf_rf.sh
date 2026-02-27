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
echo "Lorenz96 (40D) DBF Training Experiments"
echo "=========================================="
echo "Training RF proposals for 6 scenarios (3 Obs Noise x 2 Operators)"
echo "Training on process noise = 0.1 datasets"
echo "=========================================="

# Common parameters
STATE_DIM=40
BATCH_SIZE=1024
LR=3e-4
MAX_EPOCHS=500
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
COND_EMBED_DIM=128
WANDB_PROJECT="rf-train-96-dbf"
SEED=0

# Generate 0..39 string
OBS_COMPONENTS=$(seq -s, 0 39)

# Base path
DATA_BASE="/data/da_outputs/datasets"

# Define the 6 jobs
# Format: GPU OBS_NOISE OPERATOR NOISY_DIR_SUFFIX
# Note: We construct the full path dynamically

# 1. Obs 1.0, Identity
# Noisy Path: lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_identity_pnoise0p100_initUnim10_10
JOB1_GPU=0
JOB1_NAME="l96_dbf_obs1_identity"
JOB1_DATA="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_identity_pnoise0p100_initUnim10_10"

# 2. Obs 1.0, Quad
# Noisy Path: lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10
JOB2_GPU=1
JOB2_NAME="l96_dbf_obs1_quad"
JOB2_DATA="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10"

# 3. Obs 3.0, Identity
# Noisy Path: lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_identity_pnoise0p100_initUnim10_10
JOB3_GPU=2
JOB3_NAME="l96_dbf_obs3_identity"
JOB3_DATA="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_identity_pnoise0p100_initUnim10_10"

# 4. Obs 3.0, Quad
# Noisy Path: lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10
JOB4_GPU=3
JOB4_NAME="l96_dbf_obs3_quad"
JOB4_DATA="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10"

# 5. Obs 5.0, Identity
# Noisy Path: lorenz96_n2048_len80_dt0p0300_obs5p000_freq1_comp40of40_identity_pnoise0p100_initUnim10_10
JOB5_GPU=4
JOB5_NAME="l96_dbf_obs5_identity"
JOB5_DATA="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs5p000_freq1_comp40of40_identity_pnoise0p100_initUnim10_10"

# 6. Obs 5.0, Quad
# Noisy Path: lorenz96_n2048_len80_dt0p0300_obs5p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10
JOB6_GPU=5
JOB6_NAME="l96_dbf_obs5_quad"
JOB6_DATA="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs5p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10"


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

# Launch all 6
launch_job $JOB1_GPU $JOB1_NAME "$JOB1_DATA"
launch_job $JOB2_GPU $JOB2_NAME "$JOB2_DATA"
launch_job $JOB3_GPU $JOB3_NAME "$JOB3_DATA"
launch_job $JOB4_GPU $JOB4_NAME "$JOB4_DATA"
launch_job $JOB5_GPU $JOB5_NAME "$JOB5_DATA"
launch_job $JOB6_GPU $JOB6_NAME "$JOB6_DATA"

echo "=========================================="
echo "All 6 training experiments started."
echo "Logs in logs/"
echo "=========================================="
