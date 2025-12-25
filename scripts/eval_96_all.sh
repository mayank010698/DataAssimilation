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
echo "Starting Evaluations (Lorenz 96) - All Models"
echo "=========================================="

DATA_DIR="datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_init3p000"

# Common args
# Limiting to 10 trajectories by default due to high computational cost of RF + high particles
COMMON_ARGS="--data-dir $DATA_DIR --proposal-type rf --rf-sampling-steps 100 --rf-likelihood-steps 100 --process-noise-std 0.1 --obs-noise-std 0.1 --num-eval-trajectories 50 --device cuda --n-particles 1000 --batch-size 128 --rf-trace-estimator rademacher --rf-num-probes 1"

# Define experiments
# Format: MODEL_DIR|GPU_ID|EXPERIMENT_LABEL

experiments=(
    "l96_nonoise_full|0|eval_96_nonoise_full_1k_rad1"
    "l96_nonoise_half|1|eval_96_nonoise_half_1k_rad1"
    "l96_nonoise_tenth|2|eval_96_nonoise_tenth_1k_rad1"
    "l96_nonoise_no_obs|3|eval_96_nonoise_no_obs_1k_rad1"
    "l96_noise0p1_full|4|eval_96_noise0p1_full_1k_rad1"
    "l96_noise0p1_half|5|eval_96_noise0p1_half_1k_rad1"
    "l96_noise0p1_tenth|6|eval_96_noise0p1_tenth_1k_rad1"
    "l96_noise0p1_no_obs|7|eval_96_noise0p1_no_obs_1k_rad1"
)

for experiment in "${experiments[@]}"; do
    IFS='|' read -r model_dir gpu_id label <<< "$experiment"
    
    echo "Starting Exp: $label on GPU $gpu_id..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python eval.py \
        $COMMON_ARGS \
        --rf-checkpoint "rf_runs/$model_dir/final_model.ckpt" \
        --experiment-label "$label" \
        > "logs/$label.log" 2>&1 &
        
    pid=$!
    echo "Started PID $pid"
done

echo "=========================================="
echo "All experiments started in background."
echo "Logs are being written to logs/"
echo "Monitor with: tail -f logs/eval_96_*.log"
echo "=========================================="
