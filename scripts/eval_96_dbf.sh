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
echo "Lorenz96 (40D) DBF Evaluation Experiments"
echo "=========================================="
echo "Evaluating EnKF, EnSF, and EnSF+RF on 6 datasets"
echo "Testing Process Noise: 0.1, 0.2"
echo "=========================================="

# Common parameters
BATCH_SIZE=25
N_PARTICLES_ENKF=50
N_PARTICLES_ENSF=100
INFLATION=1.0
ENSF_STEPS=50
ENSF_EPS_A=0.5
ENSF_EPS_B=0.025
RF_LIK_STEPS=50
RF_SAMP_STEPS=50
WANDB_PROJECT="eval-96-dbf"
DATE_PREFIX=$(date +%Y%m%d)

# Datasets (Noiseless versions)
# Note: These must match the order of RF checkpoints below
DATASETS=(
    "/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_identity_initUnim10_10"
    "/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_initUnim10_10"
    "/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_identity_initUnim10_10"
    "/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_quad_capped_10_initUnim10_10"
    "/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs5p000_freq1_comp40of40_identity_initUnim10_10"
    "/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs5p000_freq1_comp40of40_quad_capped_10_initUnim10_10"
)

# RF Checkpoints (corresponding to datasets)
RF_CKPTS=(
    "rf_runs/l96_dbf_obs1_identity/final_model.ckpt"
    "rf_runs/l96_dbf_obs1_quad/final_model.ckpt"
    "rf_runs/l96_dbf_obs3_identity/final_model.ckpt"
    "rf_runs/l96_dbf_obs3_quad/final_model.ckpt"
    "rf_runs/l96_dbf_obs5_identity/final_model.ckpt"
    "rf_runs/l96_dbf_obs5_quad/final_model.ckpt"
)

# Dataset Labels for logging
LABELS=(
    "obs1_identity"
    "obs1_quad"
    "obs3_identity"
    "obs3_quad"
    "obs5_identity"
    "obs5_quad"
)

# Process Noise Levels
PROC_NOISES=(0.1 0.2)

# Models to evaluate
MODELS=("enkf" "ensf" "ensf_rf")

# Counter for GPU assignment
GPU_COUNTER=0
MAX_GPUS=8

# Function to launch evaluation job
launch_eval() {
    DATA_IDX=$1
    P_NOISE=$2
    MODEL=$3
    GPU=$4
    
    DATA_DIR="${DATASETS[$DATA_IDX]}"
    RF_CKPT="${RF_CKPTS[$DATA_IDX]}"
    LABEL="${LABELS[$DATA_IDX]}"
    
    # Construct Run Name
    RUN_NAME="${DATE_PREFIX}_${MODEL}_${LABEL}_pnoise${P_NOISE}"
    LOG_FILE="logs/${RUN_NAME}.log"
    
    echo "Launching $RUN_NAME on GPU $GPU..."
    
    # Base command
    CMD="/home/cnagda/miniconda3/envs/da/bin/python eval.py \
        --data-dir \"$DATA_DIR\" \
        --process-noise-std $P_NOISE \
        --batch-size $BATCH_SIZE \
        --wandb-project \"$WANDB_PROJECT\" \
        --run-name \"$RUN_NAME\" \
        --device cuda"
        
    # Model specific args
    if [ "$MODEL" == "enkf" ]; then
        CMD="$CMD --method enkf --n-particles $N_PARTICLES_ENKF --inflation $INFLATION"
    elif [ "$MODEL" == "ensf" ]; then
        CMD="$CMD --method ensf --n-particles $N_PARTICLES_ENSF --ensf-steps $ENSF_STEPS --ensf-eps-a $ENSF_EPS_A --ensf-eps-b $ENSF_EPS_B"
    elif [ "$MODEL" == "ensf_rf" ]; then
        CMD="$CMD --method ensf --proposal-type rf --rf-checkpoint \"$RF_CKPT\" \
             --n-particles $N_PARTICLES_ENSF --ensf-steps $ENSF_STEPS --ensf-eps-a $ENSF_EPS_A --ensf-eps-b $ENSF_EPS_B \
             --rf-likelihood-steps $RF_LIK_STEPS --rf-sampling-steps $RF_SAMP_STEPS --ensf-fallback-physical"
    fi
    
    # Execute in background
    CUDA_VISIBLE_DEVICES=$GPU nohup $CMD > "$LOG_FILE" 2>&1 &
}

# Main Loop
# Iterate through all combinations and schedule jobs
for ((i=0; i<${#DATASETS[@]}; i++)); do
    for P_NOISE in "${PROC_NOISES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            
            # Determine GPU
            GPU=$((GPU_COUNTER % MAX_GPUS))
            
            # Launch job
            launch_eval $i $P_NOISE $MODEL $GPU
            
            # Increment counter
            GPU_COUNTER=$((GPU_COUNTER + 1))
            
            # Simple flow control: if we've filled all GPUs, wait a bit to stagger starts
            # Note: This doesn't wait for completion, just distributes round-robin.
            # Since we have 36 jobs and 8 GPUs, this will queue multiple jobs per GPU in background.
            # To avoid overloading, we can use a simple wait if we want strictly sequential per GPU,
            # but bash backgrounding with same GPU ID will run them in parallel on the same GPU which is bad.
            # BETTER APPROACH: We need to chain commands for the same GPU.
            
        done
    done
done

echo "=========================================="
echo "NOTE: The above loop launched all jobs simultaneously in background."
echo "This might overload GPUs if not managed."
echo "Switching to a per-GPU queue approach..."
echo "=========================================="

# Kill any jobs just started by the naive loop above (if any ran)
# Actually, let's rewrite the loop to build command strings per GPU and execute them sequentially.

# Clear arrays
GPU_CMDS=()
for ((g=0; g<MAX_GPUS; g++)); do
    GPU_CMDS[$g]=""
done

GPU_COUNTER=0

# Re-iterate to build queues
for ((i=0; i<${#DATASETS[@]}; i++)); do
    for P_NOISE in "${PROC_NOISES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            
            GPU=$((GPU_COUNTER % MAX_GPUS))
            GPU_COUNTER=$((GPU_COUNTER + 1))
            
            DATA_DIR="${DATASETS[$i]}"
            RF_CKPT="${RF_CKPTS[$i]}"
            LABEL="${LABELS[$i]}"
            RUN_NAME="${DATE_PREFIX}_${MODEL}_${LABEL}_pnoise${P_NOISE}"
            LOG_FILE="logs/${RUN_NAME}.log"
            
            # Construct command
            CMD="/home/cnagda/miniconda3/envs/da/bin/python eval.py \
                --data-dir \"$DATA_DIR\" \
                --process-noise-std $P_NOISE \
                --batch-size $BATCH_SIZE \
                --wandb-project \"$WANDB_PROJECT\" \
                --run-name \"$RUN_NAME\" \
                --device cuda"
                
            if [ "$MODEL" == "enkf" ]; then
                CMD="$CMD --method enkf --n-particles $N_PARTICLES_ENKF --inflation $INFLATION"
            elif [ "$MODEL" == "ensf" ]; then
                CMD="$CMD --method ensf --n-particles $N_PARTICLES_ENSF --ensf-steps $ENSF_STEPS --ensf-eps-a $ENSF_EPS_A --ensf-eps-b $ENSF_EPS_B"
            elif [ "$MODEL" == "ensf_rf" ]; then
                CMD="$CMD --method ensf --proposal-type rf --rf-checkpoint \"$RF_CKPT\" \
                     --n-particles $N_PARTICLES_ENSF --ensf-steps $ENSF_STEPS --ensf-eps-a $ENSF_EPS_A --ensf-eps-b $ENSF_EPS_B \
                     --rf-likelihood-steps $RF_LIK_STEPS --rf-sampling-steps $RF_SAMP_STEPS --ensf-fallback-physical"
            fi
            
            # Append to GPU queue
            # Use '&&' to ensure sequential execution
            if [ -z "${GPU_CMDS[$GPU]}" ]; then
                GPU_CMDS[$GPU]="$CMD > \"$LOG_FILE\" 2>&1"
            else
                GPU_CMDS[$GPU]="${GPU_CMDS[$GPU]} && $CMD > \"$LOG_FILE\" 2>&1"
            fi
            
        done
    done
done

# Now launch the queues
echo "Launching GPU Queues..."
for ((g=0; g<MAX_GPUS; g++)); do
    if [ ! -z "${GPU_CMDS[$g]}" ]; then
        echo "Starting queue on GPU $g..."
        # echo "${GPU_CMDS[$g]}"
        CUDA_VISIBLE_DEVICES=$g eval "${GPU_CMDS[$g]}" &
    fi
done

echo "=========================================="
echo "All evaluation queues started."
echo "Monitor logs in logs/"
echo "=========================================="
