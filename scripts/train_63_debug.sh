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
# List of available GPU IDs to use for training (leave 0 open)
# The script will rotate through these GPUs for each experiment
AVAILABLE_GPUS=(1 2 3 4 5 6 7)

# ============================================================================
# Data directories
# ============================================================================
# Use the noisy dataset for all debug experiments
DATA_DIR_NOISE="/data/da_outputs/datasets/lorenz63_n512_len500_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250"

declare -a EXPERIMENTS=()

# Helper function to build experiment name
build_exp_name() {
    local predict_delta=$1
    local use_obs=$2
    local debug_obs=$3
    local debug_prev=$4
    local obs_comp=$5
    local use_time_step=$6
    
    local name="debug_l63_len500"
    
    if [ "$predict_delta" = "true" ]; then
        name="${name}_delta"
    else
        name="${name}_nodelta"
    fi
    
    if [ "$use_obs" = "true" ]; then
        name="${name}_obs"
        if [ "$obs_comp" = "0,1,2" ]; then
            name="${name}_all3"
        else
            name="${name}_dim0"
        fi
    else
        name="${name}_noobs"
    fi
    
    if [ "$debug_obs" = "true" ]; then
        name="${name}_randobs"
    fi
    
    if [ "$debug_prev" = "true" ]; then
        name="${name}_randprev"
    fi
    
    if [ "$use_time_step" = "true" ]; then
        name="${name}_timestep"
    fi
    
    echo "$name"
}

# Generate all combinations
for predict_delta in "true" "false"; do
    for use_obs in "true" "false"; do
        for debug_obs in "true" "false"; do
            for debug_prev in "true" "false"; do
                for use_time_step in "true" "INVALID"; do
                    # Skip invalid combinations:
                    # - If debug_prev=true AND use_time_step=true, skip (random state makes time step useless)
                    if [ "$debug_prev" = "true" ] && [ "$use_time_step" = "true" ]; then
                        continue
                    fi

                    if [ "$use_time_step" = "INVALID" ]; then
                        continue
                    fi
                    
                    for obs_comp in "0,1,2" "0"; do
                        # Skip invalid combinations: if use_obs=false, obs_comp doesn't matter
                        # But we'll still generate them for completeness (obs_comp will be ignored)
                        
                        exp_name=$(build_exp_name "$predict_delta" "$use_obs" "$debug_obs" "$debug_prev" "$obs_comp" "$use_time_step")
                        output_dir="/data/da_outputs/rf_runs/${exp_name}"
                        log_file="debug_${exp_name}.log"
                        
                        # Build flags
                        predict_delta_flag=""
                        if [ "$predict_delta" = "true" ]; then
                            predict_delta_flag="--predict_delta"
                        fi
                        
                        use_obs_flag=""
                        if [ "$use_obs" = "true" ]; then
                            use_obs_flag="--use_observations"
                        fi
                        
                        debug_obs_flag=""
                        if [ "$debug_obs" = "true" ]; then
                            debug_obs_flag="--debug_random_obs"
                        fi
                        
                        debug_prev_flag=""
                        if [ "$debug_prev" = "true" ]; then
                            debug_prev_flag="--debug_random_prev_state"
                        fi
                        
                        use_time_step_flag=""
                        if [ "$use_time_step" = "true" ]; then
                            use_time_step_flag="--use_time_step"
                        fi
                        
                        # Format: "exp_name:data_dir:output_dir:obs_components:log_file:predict_delta_flag:use_obs_flag:debug_obs_flag:debug_prev_flag:use_time_step_flag"
                        EXPERIMENTS+=("${exp_name}:${DATA_DIR_NOISE}:${output_dir}:${obs_comp}:${log_file}:${predict_delta_flag}:${use_obs_flag}:${debug_obs_flag}:${debug_prev_flag}:${use_time_step_flag}")
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "Starting Debug Training for Lorenz 63"
echo "=========================================="
echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
echo "Number of experiments: ${#EXPERIMENTS[@]}"
echo "=========================================="

# Array to store PIDs
declare -a PIDS=()

# Launch each experiment
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name data_dir output_dir obs_components log_file predict_delta_flag use_obs_flag debug_obs_flag debug_prev_flag use_time_step_flag <<< "${EXPERIMENTS[$i]}"
    
    # Get GPU for this experiment (rotate through available GPUs)
    gpu_idx=$((i % ${#AVAILABLE_GPUS[@]}))
    gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
    
    exp_num=$((i + 1))
    echo "Starting Exp $exp_num/$(( ${#EXPERIMENTS[@]} )): $exp_name on GPU $gpu_id..."
    
    # Build command with conditional flags
    PYTHON_CMD="/home/cnagda/miniconda3/envs/da/bin/python"
    
    # Start building the command
    CMD_ARGS=(
        "$PYTHON_CMD" "proposals/train_rf.py"
        "--data_dir" "$data_dir"
        "--output_dir" "$output_dir"
        "--state_dim" "3"
        "--architecture" "mlp"
        "--batch_size" "256"
        "--max_epochs" "500"
        "--gpus" "1"
    )
    
    # Add conditional flags
    if [ -n "$predict_delta_flag" ]; then
        CMD_ARGS+=("$predict_delta_flag")
    fi
    if [ -n "$use_obs_flag" ]; then
        CMD_ARGS+=("$use_obs_flag" "--obs_components" "$obs_components")
    fi
    if [ -n "$debug_obs_flag" ]; then
        CMD_ARGS+=("$debug_obs_flag")
    fi
    if [ -n "$debug_prev_flag" ]; then
        CMD_ARGS+=("$debug_prev_flag")
    fi
    if [ -n "$use_time_step_flag" ]; then
        CMD_ARGS+=("$use_time_step_flag")
    fi
    
    CMD_ARGS+=(
        "--wandb_project" "rf-train-63-debug"
        "--evaluate"
    )
    
    # Execute command
    CUDA_VISIBLE_DEVICES=$gpu_id nohup "${CMD_ARGS[@]}" > logs/"$log_file" 2>&1 &
    
    pid=$!
    PIDS+=($pid)
    echo "Started PID $pid"
    
    # Small delay to avoid overwhelming the system
    sleep 2
done

echo "=========================================="
echo "All debug training experiments started in background."
echo "Started PIDs: ${PIDS[*]}"
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/debug_*.log"
echo "=========================================="

