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
echo "Starting Parallel Evaluation for Lorenz 96 (FlowDAS)"
echo "Runs from: 20260109_193733"
echo "Evaluation on No Noise datasets only"
echo "=========================================="

# Available GPUs: 0, 2, 3, 4, 7
GPUS=(2 3 4 5)
NUM_GPUS=${#GPUS[@]}
GPU_IDX=0

# Timestamp for run finding
TIMESTAMP="20260109_193733"

# Dimensions to evaluate
DIMS=(5)

# Model types
TYPES=("noise" "nonoise")

for DIM in "${DIMS[@]}"; do
    # Dataset path (No Noise version) - using obs0p100 as seen in dataset directory
    DATASET_NAME="lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp${DIM}of${DIM}_arctan_init3p000"
    DATA_DIR="/data/da_outputs/datasets/${DATASET_NAME}"
    
    # Check if dataset exists
    if [ ! -d "$DATA_DIR" ]; then
        echo "Warning: Dataset $DATA_DIR does not exist. Skipping dimension $DIM."
        continue
    fi

    for TYPE in "${TYPES[@]}"; do
        RUN_NAME="flowdas_l96_${TYPE}_dim${DIM}_${TIMESTAMP}"
        MODEL_DIR="/data/da_outputs/flowdas_runs_96/${RUN_NAME}"
        CHECKPOINT="${MODEL_DIR}/checkpoint_best.pth"
        
        if [ ! -f "$CHECKPOINT" ]; then
             echo "Warning: Checkpoint $CHECKPOINT does not exist. Skipping."
             continue
        fi

        RESULTS_DIR="/data/da_outputs/results/flowdas_eval/${RUN_NAME}"
        LOG_FILE="logs/eval_${RUN_NAME}.log"
        
        # Select GPU
        CURRENT_GPU=${GPUS[$GPU_IDX]}
        GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
        
        echo "Starting Eval: ${RUN_NAME} on GPU ${CURRENT_GPU}..."
        
        CUDA_VISIBLE_DEVICES=${CURRENT_GPU} nohup python scripts/eval_flowdas.py \
            --checkpoint "$CHECKPOINT" \
            --data_dir "$DATA_DIR" \
            --results_dir "$RESULTS_DIR" \
            --run_name "eval_${RUN_NAME}" \
            --num_trajs 100 \
            --use_wandb \
            > "$LOG_FILE" 2>&1 &
            
        PID=$!
        echo "Started PID $PID"
        
        # Sleep briefly to stagger starts and avoid potential race conditions or spikes
        sleep 2
    done
done

echo "=========================================="
echo "All evaluation experiments started in background."
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/eval_flowdas_l96_*.log"
echo "=========================================="

