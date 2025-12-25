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
echo "Starting Step Grid Search Evaluations (Lorenz 96)"
echo "=========================================="

DATA_DIR="datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_init3p000"
RF_CKPT="rf_runs/l96_noise0p1_full/final_model.ckpt"

# Common args
# Using 1000 particles, 16 batch size, rademacher trace estimator, 1 probe
# Limiting to 50 trajectories
COMMON_ARGS="--data-dir $DATA_DIR --proposal-type rf --rf-checkpoint $RF_CKPT --process-noise-std 0.1 --obs-noise-std 0.1 --num-eval-trajectories 50 --device cuda --n-particles 1000 --batch-size 16 --rf-trace-estimator rademacher --rf-num-probes 1 --use-opt-weight-update"

# Grid search parameters
likelihood_steps=(50 100)
sampling_steps=(20 100)

# Generate experiments list
experiments=()
idx=0

# Available GPUs
available_gpus=(3 6)
num_gpus=${#available_gpus[@]}

for samp_steps in "${sampling_steps[@]}"; do
    for like_steps in "${likelihood_steps[@]}"; do
        # Assign GPU (cycling through 4, 5, 6, 7)
        gpu_index=$((idx % num_gpus))
        gpu_id=${available_gpus[$gpu_index]}
        
        label="eval_96_steps_S${samp_steps}_L${like_steps}_opt"
        
        experiments+=("$samp_steps|$like_steps|$gpu_id|$label")
        ((idx++))
    done
done

for experiment in "${experiments[@]}"; do
    IFS='|' read -r samp_steps like_steps gpu_id label <<< "$experiment"
    
    echo "Starting Exp: Sampling=$samp_steps, Likelihood=$like_steps on GPU $gpu_id..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python eval.py \
        $COMMON_ARGS \
        --rf-sampling-steps "$samp_steps" \
        --rf-likelihood-steps "$like_steps" \
        --experiment-label "$label" \
        > "logs/$label.log" 2>&1 &
        
    pid=$!
    echo "Started PID $pid"
done

echo "=========================================="
echo "All 16 step grid search experiments started in background."
echo "Logs are being written to logs/"
echo "Monitor with: tail -f logs/eval_96_steps_*.log"
echo "=========================================="
