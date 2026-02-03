#!/bin/bash

# Configuration Arrays
# Each index corresponds to one run configuration

# 1. 16pi, pnoise 0.1, obs 0.2
DATA_DIRS[0]="/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_pnoise0p100_J64_L50p27"
CKPT_DIRS[0]="/data/da_outputs/rf_runs_ks/ks_pnoise0p1_J64_16pi_cd0p1_20260117_192257"
P_NOISES[0]="0.1"
OBS_NOISES[0]="0.2"
RUN_LABELS[0]="J64_16pi_p0.1_o0.2"

# 2. 16pi, pnoise 0.1, obs 0.1
DATA_DIRS[1]="/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_pnoise0p100_J64_L50p27"
CKPT_DIRS[1]="/data/da_outputs/rf_runs_ks/ks_pnoise0p1_J64_16pi_cd0p1_20260117_192257"
P_NOISES[1]="0.1"
OBS_NOISES[1]="0.1"
RUN_LABELS[1]="J64_16pi_p0.1_o0.1"

# 3. 16pi, pnoise 0.05, obs 0.1
DATA_DIRS[2]="/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_pnoise0p050_J64_L50p27"
CKPT_DIRS[2]="/data/da_outputs/rf_runs_ks/ks_pnoise0p05_J64_16pi_cd0p1_20260117_192257"
P_NOISES[2]="0.05"
OBS_NOISES[2]="0.1"
RUN_LABELS[2]="J64_16pi_p0.05_o0.1"

# 4. 32pi, pnoise 0.1, obs 0.1
DATA_DIRS[3]="/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_pnoise0p100_J64_L100p53"
CKPT_DIRS[3]="/data/da_outputs/rf_runs_ks/ks_pnoise0p1_J64_32pi_cd0p1_20260117_192257"
P_NOISES[3]="0.1"
OBS_NOISES[3]="0.1"
RUN_LABELS[3]="J64_32pi_p0.1_o0.1"

# 5. 32pi, pnoise 0.05, obs 0.1
DATA_DIRS[4]="/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_pnoise0p050_J64_L100p53"
CKPT_DIRS[4]="/data/da_outputs/rf_runs_ks/ks_pnoise0p05_J64_32pi_cd0p1_20260117_192257"
P_NOISES[4]="0.05"
OBS_NOISES[4]="0.1"
RUN_LABELS[4]="J64_32pi_p0.05_o0.1"

# 6. 32pi, pnoise 0.1, obs 0.2
DATA_DIRS[5]="/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_pnoise0p100_J64_L100p53"
CKPT_DIRS[5]="/data/da_outputs/rf_runs_ks/ks_pnoise0p1_J64_32pi_cd0p1_20260117_192257"
P_NOISES[5]="0.1"
OBS_NOISES[5]="0.2"
RUN_LABELS[5]="J64_32pi_p0.1_o0.2"

WANDB_PROJECT="pf-rf-eval-ks"
GPU_ID=0
N_PARTICLES=5000
NUM_TRAJ=25
PYTHON_BIN="/home/cnagda/miniconda3/envs/da/bin/python"

# Check if python exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python binary not found at $PYTHON_BIN"
    exit 1
fi

echo "Launching 6 experiments..."

for i in {0..5}; do
    DATA_DIR="${DATA_DIRS[$i]}"
    RUN_DIR="${CKPT_DIRS[$i]}"
    P_NOISE="${P_NOISES[$i]}"
    OBS_NOISE="${OBS_NOISES[$i]}"
    LABEL="${RUN_LABELS[$i]}"
    
    # Checkpoint file
    CKPT_FILE="$RUN_DIR/final_model.ckpt"
    if [ ! -f "$CKPT_FILE" ]; then
        echo "Error: Checkpoint not found at $CKPT_FILE (Run $i: $LABEL)"
        # continue # Uncomment to skip missing checkpoints, but for now we want to know
    fi
    
    echo "------------------------------------------------"
    echo "Launching Run $i: $LABEL"
    echo "  Data: $DATA_DIR"
    echo "  Ckpt: $CKPT_FILE"
    echo "  PNoise: $P_NOISE, ObsNoise: $OBS_NOISE"
    echo "  GPU: $GPU_ID"
    
    LOG_FILE="eval_${LABEL}_N${N_PARTICLES}.log"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN eval.py \
        --data-dir "$DATA_DIR" \
        --n-particles "$N_PARTICLES" \
        --proposal-type rf \
        --rf-checkpoint "$CKPT_FILE" \
        --rf-likelihood-steps 100 \
        --rf-sampling-steps 100 \
        --obs-frequency 10 \
        --process-noise-std "$P_NOISE" \
        --obs-noise-std "$OBS_NOISE" \
        --num-eval-trajectories "$NUM_TRAJ" \
        --batch-size 5 \
        --wandb-project "$WANDB_PROJECT" \
        --run-name "eval_${LABEL}_N${N_PARTICLES}" \
        --device cuda \
        > "$LOG_FILE" 2>&1 &
        
    GPU_ID=$((GPU_ID + 1))
    
    # Wrap around GPUs (assuming 8 GPUs available 0-7)
    if [ $GPU_ID -ge 8 ]; then
        GPU_ID=0
    fi
done

echo "All evaluations launched."
