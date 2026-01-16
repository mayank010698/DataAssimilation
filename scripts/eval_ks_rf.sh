#!/bin/bash

BASE_DIR="/data/da_outputs/rf_runs_ks"
SUFFIX="003851"
WANDB_PROJECT="pf-rf-eval-ks"

# Filter configuration
INCLUDE_J128=false # Set to true to include J128 datasets

# Find runs
RUN_DIRS=$(find "$BASE_DIR" -maxdepth 1 -type d -name "*$SUFFIX")

GPU_ID=0
N_PARTICLES=5000

for RUN_DIR in $RUN_DIRS; do
    RUN_NAME=$(basename "$RUN_DIR")

    # Filter logic: Default to J64, optionally include J128
    if [[ "$RUN_NAME" == *"J64"* ]]; then
        echo "Processing J64 Run: $RUN_NAME"
    elif [[ "$INCLUDE_J128" == "true" ]] && [[ "$RUN_NAME" == *"J128"* ]]; then
        echo "Processing J128 Run: $RUN_NAME"
    else
        echo "Skipping Run: $RUN_NAME (Not J64, and J128 include is $INCLUDE_J128)"
        continue
    fi

    # Find config.yaml (it's inside wandb directory)
    CONFIG_FILE=$(find "$RUN_DIR/wandb" -name "config.yaml" | head -n 1)
    
    if [ -z "$CONFIG_FILE" ]; then
        echo "  Error: config.yaml not found in $RUN_DIR"
        continue
    fi

    # Extract data_dir using grep/awk as fallback or python if available
    # Trying grep first as it is lightweight and the format is known
    DATA_DIR=$(grep -A 1 "data_dir:" "$CONFIG_FILE" | tail -n 1 | awk '{print $2}')
    
    if [ -z "$DATA_DIR" ]; then
        echo "  Error: Could not extract data_dir from $CONFIG_FILE"
        continue
    fi
    
    echo "  Data Dir: $DATA_DIR"

    # Find checkpoint
    CKPT_FILE="$RUN_DIR/final_model.ckpt"
    if [ ! -f "$CKPT_FILE" ]; then
        echo "  Error: $CKPT_FILE not found"
        continue
    fi
    echo "  Checkpoint: $CKPT_FILE"

    # Launch evaluations
    for NUM_TRAJ in 25; do
        echo "  Launching eval with $N_PARTICLES particles, $NUM_TRAJ trajectories on GPU $GPU_ID..."
        
        LOG_FILE="eval_${RUN_NAME}_N${N_PARTICLES}_T${NUM_TRAJ}.log"
        
        CUDA_VISIBLE_DEVICES=$GPU_ID /home/cnagda/miniconda3/envs/da/bin/python eval.py \
            --data-dir "$DATA_DIR" \
            --n-particles "$N_PARTICLES" \
            --proposal-type rf \
            --rf-checkpoint "$CKPT_FILE" \
            --rf-likelihood-steps 100 \
            --rf-sampling-steps 100 \
            --obs-frequency 10 \
            --num-eval-trajectories "$NUM_TRAJ" \
            --batch-size 5 \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "eval_${RUN_NAME}_N${N_PARTICLES}_T${NUM_TRAJ}" \
            --device cuda \
            > "$LOG_FILE" 2>&1 &
            
        GPU_ID=$((GPU_ID + 1))
        
        # Wrap around if we run out of GPUs
        if [ $GPU_ID -ge 8 ]; then
            GPU_ID=0
        fi
    done
done

echo "All evaluations launched in background."
