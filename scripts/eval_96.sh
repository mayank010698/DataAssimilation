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
echo "Starting Evaluations (Lorenz 96) - 8 Proposals"
echo "=========================================="

# Dataset from eval_96_all.sh (using the init3p000 one as it seems to be the standard for these models)
DATA_DIR="/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p200_freq1_comp50of50_arctan_init3p000"

# Observation Component Definitions
# Full (50): 0..49
COMPS_FULL=$(seq -s, 0 49)
# Half (25): 0..24
COMPS_HALF=$(seq -s, 0 24)
# Tenth (10): 0..9 (Based on training script 1/5 obs)
COMPS_TENTH=$(seq -s, 0 9)
# No Obs: Empty string
COMPS_NONE=""

# Common args
# --num-eval-trajectories 10 (User requested)
# --n-particles 1000 (User requested)
# --rf-trace-estimator rademacher (User requested/default)
# --rf-num-probes 1 (User requested)
# --rf-sampling-steps 100 (User requested)
# --rf-likelihood-steps 100 (User requested)
# --process-noise-std 0.1 (Matched to typical eval settings)
COMMON_ARGS="--data-dir $DATA_DIR --proposal-type rf --rf-sampling-steps 100 --rf-likelihood-steps 100 --process-noise-std 0.1 --num-eval-trajectories 10 --device cuda --n-particles 1000 --batch-size 128 --rf-trace-estimator rademacher --rf-num-probes 1"

# Define experiments
# Format: MODEL_DIR|GPU_ID|OBS_COMPONENTS|LABEL_SUFFIX
experiments=(
    "l96_nonoise_full|0|$COMPS_FULL|nonoise_full"
    "l96_nonoise_half|1|$COMPS_HALF|nonoise_half"
    "l96_nonoise_tenth|2|$COMPS_TENTH|nonoise_tenth"
    "l96_nonoise_no_obs|3|$COMPS_NONE|nonoise_no_obs"
    "l96_noise0p1_full|4|$COMPS_FULL|noise0p1_full"
    "l96_noise0p1_half|5|$COMPS_HALF|noise0p1_half"
    "l96_noise0p1_tenth|6|$COMPS_TENTH|noise0p1_tenth"
    "l96_noise0p1_no_obs|7|$COMPS_NONE|noise0p1_no_obs"
)

for experiment in "${experiments[@]}"; do
    IFS='|' read -r model_dir gpu_id obs_comps label_suffix <<< "$experiment"
    
    label="eval_96_rf_1k_$label_suffix"
    log_file="logs/$label.log"
    
    echo "Starting Exp: $label on GPU $gpu_id..."
    
    # Construct command
    # Conditionally add --obs-components if not empty
    OBS_ARG=""
    if [ -n "$obs_comps" ]; then
        OBS_ARG="--obs-components $obs_comps"
    else
        # For No Obs, we pass empty string to override dataset config
        OBS_ARG="--obs-components \"\""
    fi

    # Note: We must use eval or carefully handle the empty string argument for obs-components
    # To avoid shell parsing issues with empty string, we'll use a slightly different approach for the command execution
    
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id nohup python eval.py \
        $COMMON_ARGS \
        --rf-checkpoint rf_runs/$model_dir/final_model.ckpt \
        --experiment-label $label \
        $OBS_ARG \
        > $log_file 2>&1 &"

    # Execute
    eval "$cmd"
        
    pid=$!
    echo "Started PID $pid (Log: $log_file)"
done

echo "=========================================="
echo "All experiments started in background."
echo "Logs are being written to logs/"
echo "Monitor with: tail -f logs/eval_96_rf_*.log"
echo "=========================================="
