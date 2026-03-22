#!/bin/bash

# Sweep over observation-consistency weights for RF proposal on Lorenz-96 DBF dataset.
# - 8 total runs:
#   * 1 baseline RF (obs_consistency_weight = 0.0)
#   * 7 runs with increasing obs_consistency_weight
# - Previous-state corruption and gating are DISABLED in all runs.
#
# This is analogous in structure to:
#   - scripts/train_96_dbf_rf_prev_corr_obs1_sweep.sh
#   - scripts/train_96_dbf_rf_gated_grid.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.." || exit 1

export PYTHONPATH=$(pwd)

# Common RF training parameters
STATE_DIM=40
BATCH_SIZE=1024
LR=3e-4
MAX_EPOCHS=500
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
COND_EMBED_DIM=128
WANDB_PROJECT="rf-train-96-obscons-obs1"
SEED=0

# Observe all 40 components (0..39)
OBS_COMPONENTS=$(seq -s, 0 39)

# Dataset: same DBF L96 setup used for gated RF training (obs noise 1.0, pnoise 0.1)
DATA_BASE="/data/da_outputs/datasets"
DATA_DIR="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10"

# GPUs to use
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
  echo "  Extra args: $*"
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
    # Explicitly disable previous-state corruption
    --prev_state_corr_p0 0.0
    --prev_state_corr_sigma 0.0
    --prev_state_corr_mask_ratio 0.0
    # No gating flags passed: use_gated_obs_correction remains False
    "$@"
  )

  LOGFILE="logs/${NAME}.log"
  echo "Running command (background):"
  echo "  CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]} > $LOGFILE 2>&1 &"
  CUDA_VISIBLE_DEVICES="$GPU" nohup "${CMD[@]}" > "$LOGFILE" 2>&1 &

  JOB_IDX=$((JOB_IDX + 1))
}

# ---------------------------------------------------------------------------
# Baseline: EnSF+RF-style RF trained with no observation-consistency penalty
# ---------------------------------------------------------------------------
run_job "l96_dbf_rf_obscons_w000" \
  --obs-consistency-weight 0.0

# ---------------------------------------------------------------------------
# Observation-consistency weights to sweep
# (rough log-spaced from 1e-4 up to 1e-1)
# ---------------------------------------------------------------------------
run_job "l96_dbf_rf_obscons_w1e4"  --obs-consistency-weight 0.0001
run_job "l96_dbf_rf_obscons_w3e4"  --obs-consistency-weight 0.0003
run_job "l96_dbf_rf_obscons_w1e3"  --obs-consistency-weight 0.001
run_job "l96_dbf_rf_obscons_w3e3"  --obs-consistency-weight 0.003
run_job "l96_dbf_rf_obscons_w1e2"  --obs-consistency-weight 0.01
run_job "l96_dbf_rf_obscons_w3e2"  --obs-consistency-weight 0.03
run_job "l96_dbf_rf_obscons_w1e1"  --obs-consistency-weight 0.1

echo "=========================================="
echo "Submitted $JOB_IDX RF training jobs with obs-consistency sweep."
echo "Runs in rf_runs/l96_dbf_rf_obscons_* and logs in logs/."
echo "=========================================="

