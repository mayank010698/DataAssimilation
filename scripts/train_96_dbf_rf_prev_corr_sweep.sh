#!/bin/bash

# Sweep for previous-state corruption (no gating).
# Dataset: L96 DBF with pnoise0p100 (process noise std = 0.1).
#   sigma_corr: 0.025=0.25*proc, 0.05=0.5*proc, 0.075=0.75*proc, 0.10=1.0*proc
#
# 16 runs total (8 GPUs × 2 jobs each). All use: p_min=0.05, resnet1d, adaln, use_observations.
# Summary:
#   1   p0=0.3  sigma=0.05   mask=0     (Gaussian-only)
#   2   p0=0.2  sigma=0.05   mask=0     (Gaussian-only)
#   3   p0=0.3  sigma=0.025 mask=0     (Gaussian-only)
#   4   p0=0.3  sigma=0     mask=0.2    (mask-only)
#   5   p0=0.3  sigma=0     mask=0.4    (mask-only)
#   6   p0=0.3  sigma=0.05  mask=0.2   (Gaussian + mask)
#   7   p0=0.3  sigma=0.05  mask=0.4   (Gaussian + mask)
#   8   p0=0.3  sigma=0.10  mask=0     (Gaussian-only, stronger)
#   9   p0=0.3  sigma=0.075 mask=0     (Gaussian-only)
#   10  p0=0.3  sigma=0     mask=0.3    (mask-only)
#   11  p0=0.3  sigma=0.05  mask=0.3   (Gaussian + mask)
#   12  p0=0.2  sigma=0.025 mask=0     (Gaussian-only, lighter)
#   13  p0=0.3  sigma=0.05  mask=0     p_min=0 (full anneal)
#   14  p0=0.4  sigma=0.05  mask=0     (Gaussian-only, higher p)
#   15  p0=0.3  sigma=0.05  mask=0.5   (Gaussian + mask)
#   16  p0=0.2  sigma=0.05  mask=0.2   (Gaussian + mask)

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
WANDB_PROJECT="rf-train-96-prev-corr"
SEED=0
P_MIN=0.05

OBS_COMPONENTS=$(seq -s, 0 39)

DATA_BASE="/data/da_outputs/datasets"
DATA_DIR="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10"

# sigma_proc = 0.1 from dataset name pnoise0p100
SIGMA_05=0.05    # 0.5 * sigma_proc
SIGMA_025=0.025  # 0.25 * sigma_proc
SIGMA_075=0.075  # 0.75 * sigma_proc
SIGMA_10=0.10    # 1.0 * sigma_proc

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
    --prev_state_corr_p_min "$P_MIN"
    "$@"
  )
  CUDA_VISIBLE_DEVICES="$GPU" nohup "${CMD[@]}" > "logs/${NAME}.log" 2>&1 &
  JOB_IDX=$((JOB_IDX + 1))
}

# 1. p0=0.3, sigma=0.5*proc
run_job "l96_dbf_prevcorr_p030_sig050" \
  --prev_state_corr_p0 0.3 --prev_state_corr_sigma "$SIGMA_05"

# 2. p0=0.2, sigma=0.5*proc
run_job "l96_dbf_prevcorr_p020_sig050" \
  --prev_state_corr_p0 0.2 --prev_state_corr_sigma "$SIGMA_05"

# 3. p0=0.3, sigma=0.25*proc
run_job "l96_dbf_prevcorr_p030_sig025" \
  --prev_state_corr_p0 0.3 --prev_state_corr_sigma "$SIGMA_025"

# 4. Mask-only: p0=0.3, mask=0.2 (no Gaussian)
run_job "l96_dbf_prevcorr_p030_mask020_only" \
  --prev_state_corr_p0 0.3 --prev_state_corr_mask_ratio 0.2

# 5. Mask-only: p0=0.3, mask=0.4 (no Gaussian)
run_job "l96_dbf_prevcorr_p030_mask040_only" \
  --prev_state_corr_p0 0.3 --prev_state_corr_mask_ratio 0.4

# 6. Gaussian + mask 0.2
run_job "l96_dbf_prevcorr_p030_sig050_mask020" \
  --prev_state_corr_p0 0.3 --prev_state_corr_sigma "$SIGMA_05" --prev_state_corr_mask_ratio 0.2

# 7. Gaussian + mask 0.4
run_job "l96_dbf_prevcorr_p030_sig050_mask040" \
  --prev_state_corr_p0 0.3 --prev_state_corr_sigma "$SIGMA_05" --prev_state_corr_mask_ratio 0.4

# 8. p0=0.3, sigma=1.0*proc (stronger Gaussian)
run_job "l96_dbf_prevcorr_p030_sig100" \
  --prev_state_corr_p0 0.3 --prev_state_corr_sigma "$SIGMA_10"

# 9. p0=0.3, sigma=0.75*proc
run_job "l96_dbf_prevcorr_p030_sig075" \
  --prev_state_corr_p0 0.3 --prev_state_corr_sigma "$SIGMA_075"

# 10. Mask-only: p0=0.3, mask=0.3 (no Gaussian)
run_job "l96_dbf_prevcorr_p030_mask030_only" \
  --prev_state_corr_p0 0.3 --prev_state_corr_mask_ratio 0.3

# 11. Gaussian + mask 0.3
run_job "l96_dbf_prevcorr_p030_sig050_mask030" \
  --prev_state_corr_p0 0.3 --prev_state_corr_sigma "$SIGMA_05" --prev_state_corr_mask_ratio 0.3

# 12. p0=0.2, sigma=0.025 (lighter)
run_job "l96_dbf_prevcorr_p020_sig025" \
  --prev_state_corr_p0 0.2 --prev_state_corr_sigma "$SIGMA_025"

# 13. p_min=0 (full anneal), p0=0.3, sigma=0.05
run_job "l96_dbf_prevcorr_p030_sig050_pmin0" \
  --prev_state_corr_p0 0.3 --prev_state_corr_p_min 0.0 --prev_state_corr_sigma "$SIGMA_05"

# 14. p0=0.4, sigma=0.05 (higher corruption prob)
run_job "l96_dbf_prevcorr_p040_sig050" \
  --prev_state_corr_p0 0.4 --prev_state_corr_sigma "$SIGMA_05"

# 15. Gaussian + mask 0.5
run_job "l96_dbf_prevcorr_p030_sig050_mask050" \
  --prev_state_corr_p0 0.3 --prev_state_corr_sigma "$SIGMA_05" --prev_state_corr_mask_ratio 0.5

# 16. p0=0.2, sigma=0.05, mask=0.2
run_job "l96_dbf_prevcorr_p020_sig050_mask020" \
  --prev_state_corr_p0 0.2 --prev_state_corr_sigma "$SIGMA_05" --prev_state_corr_mask_ratio 0.2

echo "=========================================="
echo "Submitted $JOB_IDX prev-state corruption training jobs."
echo "Logs in logs/ and runs in rf_runs/."
echo "=========================================="
