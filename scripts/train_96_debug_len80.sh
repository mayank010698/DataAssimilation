#!/bin/bash

# Same experiment grid as train_63_debug.sh but:
# - Datasets from gen_96_dbf.sh (listed in datasets_len80.txt) - Lorenz 96 len80
# - predict_delta fixed to true (no delta vs nodelta comparison)
# - state_dim=10, obs_components always all 10

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.." || exit

export PYTHONPATH=$(pwd)

mkdir -p logs

# ============================================================================
# Configuration: Available GPUs
# ============================================================================
AVAILABLE_GPUS=(4 5 6 7)

# ============================================================================
# Data directories: Lorenz 96 len80 datasets from datasets_len80.txt
# ============================================================================
DATASETS_FILE="datasets_len80.txt"
if [ ! -f "$DATASETS_FILE" ]; then
    echo "Error: $DATASETS_FILE not found."
    exit 1
fi

# Read dataset paths (trimmed, skip empty)
mapfile -t DATA_DIRS < <(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' "$DATASETS_FILE" | grep -v '^$')

declare -a EXPERIMENTS=()

# All state components for obs_comp (state_dim=10)
ALL_OBS=$(echo {0..9} | tr ' ' ',')

# Helper: build experiment name (predict_delta always true, so name always includes _delta)
build_exp_name() {
    local use_obs=$1
    local debug_obs=$2
    local debug_prev=$3
    local obs_comp=$4
    local use_time_step=$5

    local name="resnet_debug_len80_delta"

    if [ "$use_obs" = "true" ]; then
        name="${name}_obs"
        if [ "$obs_comp" = "$ALL_OBS" ]; then
            name="${name}_all10"
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

# Dataset slug from path for output dirs and logs (basename)
dataset_slug() {
    local path=$1
    basename "$path"
}

# Generate all combinations: one grid per dataset, predict_delta always true
# debug_obs always false: real obs when use_obs true, no obs when use_obs false
for data_dir in "${DATA_DIRS[@]}"; do
    [ -d "$data_dir" ] || continue
    slug=$(dataset_slug "$data_dir")

    for use_obs in "true" "false"; do
        debug_obs="false"
        for debug_prev in "true" "false"; do
            for use_time_step in "true"; do
                # Skip invalid: random prev state makes time step meaningless
                if [ "$debug_prev" = "true" ] && [ "$use_time_step" = "true" ]; then
                    continue
                fi

                # obs_components always all 10 (no single-component runs)
                obs_comp="$ALL_OBS"
                exp_name=$(build_exp_name "$use_obs" "$debug_obs" "$debug_prev" "$obs_comp" "$use_time_step")
                output_dir="/data/da_outputs/rf_runs/len80_debug/${slug}/${exp_name}"
                log_file="len80_debug_${slug}_${exp_name}.log"
                # Sanitize log filename: replace / with _
                log_file="${log_file//\//_}"

                use_obs_flag=""
                [ "$use_obs" = "true" ] && use_obs_flag="--use_observations"

                debug_obs_flag=""
                # debug_obs always false: never pass --debug_random_obs

                debug_prev_flag=""
                [ "$debug_prev" = "true" ] && debug_prev_flag="--debug_random_prev_state"

                use_time_step_flag=""
                [ "$use_time_step" = "true" ] && use_time_step_flag="--use_time_step"

                # predict_delta always true (not in tuple)
                EXPERIMENTS+=("${exp_name}:${data_dir}:${output_dir}:${obs_comp}:${log_file}:${use_obs_flag}:${debug_obs_flag}:${debug_prev_flag}:${use_time_step_flag}")
            done
        done
    done
done

echo "=========================================="
echo "Starting Debug Training for Lorenz 96 len80"
echo "=========================================="
echo "Datasets: ${#DATA_DIRS[@]}"
echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
echo "Number of experiments: ${#EXPERIMENTS[@]}"
echo "=========================================="

declare -a PIDS=()

for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name data_dir output_dir obs_components log_file use_obs_flag debug_obs_flag debug_prev_flag use_time_step_flag <<< "${EXPERIMENTS[$i]}"

    gpu_idx=$((i % ${#AVAILABLE_GPUS[@]}))
    gpu_id=${AVAILABLE_GPUS[$gpu_idx]}

    exp_num=$((i + 1))
    echo "Starting Exp $exp_num/${#EXPERIMENTS[@]}: $exp_name on GPU $gpu_id..."

    PYTHON_CMD="/home/cnagda/miniconda3/envs/da/bin/python"

    CMD_ARGS=(
        "$PYTHON_CMD" "proposals/train_rf.py"
        "--data_dir" "$data_dir"
        "--output_dir" "$output_dir"
        "--state_dim" "10"
        "--architecture" "resnet1d"
        "--channels" "64"
        "--num_blocks" "10"
        "--kernel_size" "5"
        "--cond_embed_dim" "128"
        "--batch_size" "256"
        "--max_epochs" "500"
        "--gpus" "1"
        "--predict_delta"
    )

    [ -n "$use_obs_flag" ] && CMD_ARGS+=("$use_obs_flag" "--obs_components" "$obs_components")
    [ -n "$debug_obs_flag" ] && CMD_ARGS+=("$debug_obs_flag")
    [ -n "$debug_prev_flag" ] && CMD_ARGS+=("$debug_prev_flag")
    [ -n "$use_time_step_flag" ] && CMD_ARGS+=("$use_time_step_flag")

    CMD_ARGS+=(
        "--wandb_project" "rf-train-96-len80-debug"
        "--evaluate"
    )

    CUDA_VISIBLE_DEVICES=$gpu_id nohup "${CMD_ARGS[@]}" > logs/"$log_file" 2>&1 &

    pid=$!
    PIDS+=($pid)
    echo "Started PID $pid"

    sleep 2
done

echo "=========================================="
echo "All len80 debug training experiments started in background."
echo "Started PIDs: ${PIDS[*]}"
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/len80_debug_*.log"
echo "=========================================="
