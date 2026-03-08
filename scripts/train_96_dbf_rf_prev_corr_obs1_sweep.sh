#!/bin/bash

# Sweep for previous-state corruption (no gating).
# Dataset: L96 DBF with pnoise0p100 (process noise std = 0.1), obs1p000 (observation noise 1.0).
# Fixed: sigma_corr=0.05 (0.5*proc), prev_state_corr_p0=0.3. Mask ratio 0.4 only where used.
#
# 4 runs total. All use: resnet1d, adaln, use_observations.
# Summary:
#   1  No corruption (EnSF+RF baseline)
#   2  sigma=0.05 mask=0     (Gaussian-only)
#   3  sigma=0   mask=0.4   (mask-only)
#   4  sigma=0.05 mask=0.4  (Gaussian + mask)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.." || exit 1

export PYTHONPATH=$(pwd)

STATE_DIM=40
BATCH_SIZE=1024
LR=3e-4
MAX_EPOCHS=500
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
COND_EMBED_DIM=128
WANDB_PROJECT="rf-train-96-prev-corr-obs1"
SEED=0
P_MIN=0.05
P0=0.3

OBS_COMPONENTS=$(seq -s, 0 39)

DATA_BASE="/data/da_outputs/datasets"
DATA_DIR="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_pnoise0p200_initUnim10_10"

# sigma_proc = 0.2 from dataset name pnoise0p200; only sigma=0.1 for this sweep
SIGMA=0.1  # 0.5 * sigma_proc

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

PYTHON_BIN="/home/cnagda/miniconda3/envs/da/bin/python"

mkdir -p logs

JOB_IDX=0

run_job() {
  local NAME=$1
  local GPU=${GPUS[$((JOB_IDX % NUM_GPUS))]}
  shift
  echo "=========================================="
  echo "Job $JOB_IDX on GPU $GPU: $NAME"
  echo "  $*"
  echo "=========================================="
  CMD=(
    "$PYTHON_BIN" proposals/train_rf.py
    --data_dir "$DATA_DIR"
    --output_dir "rf_runs/$NAME"
    --state_dim "$STATE_DIM"
    --obs_components "$OBS_COMPONENTS"
    --architecture resnet1d
    --channels "$CHANNELS"
    --num_blocks "$NUM_BLOCKS"
    --kernel_size "$KERNEL_SIZE"
    --train-cond-method adaln
    --cond_embed_dim "$COND_EMBED_DIM"
    --use_observations
    --batch_size "$BATCH_SIZE"
    --learning_rate "$LR"
    --max_epochs "$MAX_EPOCHS"
    --gpus 1
    --wandb_project "$WANDB_PROJECT"
    --seed "$SEED"
    --prev_state_corr_p0 "$P0"
    --prev_state_corr_p_min "$P_MIN"
    "$@"
  )
  CUDA_VISIBLE_DEVICES="$GPU" nohup "${CMD[@]}" > "logs/${NAME}.log" 2>&1 &
  JOB_IDX=$((JOB_IDX + 1))
}

# 1. Baseline: EnSF+RF, no previous-state corruption
run_job "l96_dbf_baseline_ensf_rf" \
  --prev_state_corr_p0 0

# 2. Gaussian-only (sigma=0.05)
run_job "l96_dbf_prevcorr_p030_sig050" \
  --prev_state_corr_sigma "$SIGMA"

# 3. Mask-only (mask=0.4, no Gaussian)
run_job "l96_dbf_prevcorr_p030_mask040_only" \
  --prev_state_corr_mask_ratio 0.4

# 4. Gaussian + mask 0.4
run_job "l96_dbf_prevcorr_p030_sig050_mask040" \
  --prev_state_corr_sigma "$SIGMA" --prev_state_corr_mask_ratio 0.4

echo "=========================================="
echo "Submitted $JOB_IDX prev-state corruption training jobs."
echo "Logs in logs/ and runs in rf_runs/."
echo "=========================================="
