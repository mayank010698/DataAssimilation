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
AVAILABLE_GPUS=(0 1 2 3)

# ============================================================================
# Common parameters
# ============================================================================
BATCH_SIZE=64
LR=1e-3
EPOCHS=500
WIDTH=128  # Embedding/Hidden dimension (equivalent to RF's hidden_dim)
DEPTH=4   # Network depth (matching RF's depth)

# ============================================================================
# Data directories
# ============================================================================
DATA_DIR_NOISE="/data/da_outputs/datasets/lorenz63_n1024_len1000_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250"
DATA_DIR_NO_NOISE="/data/da_outputs/datasets/lorenz63_n1024_len1000_dt0p0100_obs0p250_freq1_comp0,1,2_arctan"

# ============================================================================
# Experiment configurations
# ============================================================================
# Define experiments as: "exp_name:data_dir:run_dir:obs_components:log_file"
declare -a EXPERIMENTS=(
    "L63, 0 noise, all 3 dims obs:$DATA_DIR_NO_NOISE:/data/da_outputs/runs_flowdas/l63_nonoise_all3:0,1,2:flowdas_l63_nonoise_all3.log"
    "L63, 0 noise, only 1st dim obs:$DATA_DIR_NO_NOISE:/data/da_outputs/runs_flowdas/l63_nonoise_dim0:0:flowdas_l63_nonoise_dim0.log"
    "L63, 0.25 pnoise, all 3 dims obs:$DATA_DIR_NOISE:/data/da_outputs/runs_flowdas/l63_pnoise0p250_all3:0,1,2:flowdas_l63_pnoise0p250_all3.log"
    "L63, 0.25 pnoise, only 1st dim obs:$DATA_DIR_NOISE:/data/da_outputs/runs_flowdas/l63_pnoise0p250_dim0:0:flowdas_l63_pnoise0p250_dim0.log"
)

echo "=========================================="
echo "Lorenz 63 FlowDAS Training Experiments"
echo "=========================================="
echo "Configuration:"
echo "- 4 Experiments total (2 No Noise, 2 With Noise)"
echo "- Training: NO observations (unconditional)"
echo "- Inference: Will use guidance with observations (all 3 dims or 1st dim)"
echo "- Architecture: MLP (matching RF training)"
echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
echo "Number of experiments: ${#EXPERIMENTS[@]}"
echo "=========================================="

# Array to store PIDs
declare -a PIDS=()

# Launch each experiment
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name data_dir run_dir obs_components log_file <<< "${EXPERIMENTS[$i]}"
    
    # Get GPU for this experiment (rotate through available GPUs)
    gpu_idx=$((i % ${#AVAILABLE_GPUS[@]}))
    gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
    
    exp_num=$((i + 1))
    echo "Starting Exp $exp_num: $exp_name on GPU $gpu_id..."
    
    # Build command arguments
    # NOTE: We do NOT use --use_observations during training
    # Observations will be used only at inference time via guidance
    cmd_args=(
        --config "$data_dir/config.yaml"
        --data_dir "$data_dir"
        --run_dir "$run_dir"
        --architecture mlp
        --width $WIDTH
        --depth $DEPTH
        --batch_size $BATCH_SIZE
        --lr $LR
        --epochs $EPOCHS
        --use_wandb
        --wandb_project "flowdas-train-63"
        --evaluate
    )
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python scripts/train_flowdas.py "${cmd_args[@]}" \
        > logs/"$log_file" 2>&1 &
    
    pid=$!
    PIDS+=($pid)
    echo "Started PID $pid"
done

echo "=========================================="
echo "All FlowDAS training experiments started in background."
echo "Started PIDs: ${PIDS[*]}"
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/flowdas_l63_*.log"
echo "=========================================="
