#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Configuration: Available GPUs
# ============================================================================
# List of available GPU IDs to use for training
# The script will rotate through these GPUs for each experiment
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)

# ============================================================================
# Data directories
# ============================================================================
# DATA_DIR_NOISE="/data/da_outputs/datasets/lorenz63_n1024_len1000_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250"
# DATA_DIR_NO_NOISE="/data/da_outputs/datasets/lorenz63_n1024_len1000_dt0p0100_obs0p250_freq1_comp0,1,2_arctan"
DATA_DIR_NOISE="/data/da_outputs/datasets/lorenz63_n512_len500_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250"
DATA_DIR_NO_NOISE="/data/da_outputs/datasets/lorenz63_n512_len500_dt0p0100_obs0p250_freq1_comp0,1,2_arctan"

# ============================================================================
# Experiment configurations
# ============================================================================
# Define experiments as: "exp_name:data_dir:output_dir:obs_components:log_file"
declare -a EXPERIMENTS=(
    # "L63, 0 noise, all 3 dims obs:$DATA_DIR_NO_NOISE:rf_runs/lorenz63_len1000_obs_all3_no_noise:0,1,2:train_l63_obs_all3_no_noise.log"
    # "L63, 0 noise, only 1st dim obs:$DATA_DIR_NO_NOISE:rf_runs/lorenz63_len1000_obs_dim0_no_noise:0:train_l63_obs_dim0_no_noise.log"
    "L63, 0.25 pnoise, all 3 dims obs:$DATA_DIR_NOISE:rf_runs/lorenz63_len500_obs_all3_pnoise0p250:0,1,2:train_l63_obs_all3_pnoise.log"
    "L63, 0.25 pnoise, only 1st dim obs:$DATA_DIR_NOISE:rf_runs/lorenz63_len500_obs_dim0_pnoise0p250:0:train_l63_obs_dim0_pnoise.log"
)

echo "=========================================="
echo "Starting Parallel Training for Lorenz 63 (MLP with Observations)"
echo "=========================================="
echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
echo "Number of experiments: ${#EXPERIMENTS[@]}"
echo "=========================================="

# Array to store PIDs
declare -a PIDS=()

# Launch each experiment
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name data_dir output_dir obs_components log_file <<< "${EXPERIMENTS[$i]}"
    
    # Get GPU for this experiment (rotate through available GPUs)
    gpu_idx=$((i % ${#AVAILABLE_GPUS[@]}))
    gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
    
    exp_num=$((i + 1))
    echo "Starting Exp $exp_num: $exp_name on GPU $gpu_id..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python proposals/train_rf.py \
        --data_dir "$data_dir" \
        --output_dir "$output_dir" \
        --state_dim 3 \
        --architecture mlp \
        --batch_size 256 \
        --max_epochs 500 \
        --gpus 1 \
        --use_observations \
        --obs_components "$obs_components" \
        --predict_delta \
        --wandb_project "rf-train-63" \
        --evaluate \
        > logs/"$log_file" 2>&1 &
    
    pid=$!
    PIDS+=($pid)
    echo "Started PID $pid"
done

echo "=========================================="
echo "All training experiments started in background."
echo "Started PIDs: ${PIDS[*]}"
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/*.log"
echo "=========================================="
