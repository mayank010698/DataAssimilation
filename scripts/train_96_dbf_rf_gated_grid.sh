#!/bin/bash

# Grid search over gated RF proposal hyperparameters for Lorenz-96 DBF dataset.
# This script runs 17 jobs:
#   1 baseline (no gating, no prior zero init)
#   16 gated configs:
#     use_gated_obs_correction: true
#     gate_type: scalar/spatial
#     gate_init_bias: 0.0/0.1/0.2/0.3
#     prior_zero_init: true/false

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit 1

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

# Common parameters
STATE_DIM=40
BATCH_SIZE=1024
LR=3e-4
MAX_EPOCHS=500
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
COND_EMBED_DIM=128
WANDB_PROJECT="rf-train-96-gated-grid-obs1"
SEED=0

# Generate 0..39 string for observed components
OBS_COMPONENTS=$(seq -s, 0 39)

# Dataset base path and target dataset (obs noise 1.0, pnoise 0.1)
DATA_BASE="/data/da_outputs/datasets"
DATA_DIR="$DATA_BASE/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10"

# GPUs available (you can edit this if needed)
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

PYTHON_BIN="/home/cnagda/miniconda3/envs/da/bin/python"

mkdir -p logs

JOB_IDX=0

# ---------------------------------------------------------------------------
# 1) Baseline RF run: no gating, no prior zero init
# ---------------------------------------------------------------------------
GPU=${GPUS[$((JOB_IDX % NUM_GPUS))]}
BASELINE_NAME="l96_dbf_rf_baseline_nogate_nopzi"

echo "=========================================="
echo "Job $JOB_IDX on GPU $GPU: $BASELINE_NAME"
echo "  use_gated_obs_correction = false"
echo "  gate_type               = scalar (unused)"
echo "  gate_init_bias          = 0.0 (unused)"
echo "  prior_zero_init         = false"
echo "=========================================="

CMD_BASELINE=(
  "$PYTHON_BIN" proposals/train_rf.py
  --data_dir "$DATA_DIR"
  --output_dir "rf_runs/$BASELINE_NAME"
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
  --gate-type scalar
  --gate-init-bias 0.0
  --no-obs-zero-init
  --no-prior-zero-init
)

LOGFILE_BASELINE="logs/${BASELINE_NAME}.log"
echo "Running command (background):"
echo "  CUDA_VISIBLE_DEVICES=$GPU ${CMD_BASELINE[*]} > $LOGFILE_BASELINE 2>&1 &"
CUDA_VISIBLE_DEVICES="$GPU" nohup "${CMD_BASELINE[@]}" > "$LOGFILE_BASELINE" 2>&1 &

JOB_IDX=$((JOB_IDX + 1))

# ---------------------------------------------------------------------------
# 2) Gated RF grid (16 configs): gate_type × gate_bias × prior_zero_init
# ---------------------------------------------------------------------------
for GATE_TYPE in scalar spatial; do
  for GATE_BIAS in 0.0 0.1 0.2 0.3; do
    for PRIOR_INIT in true false; do
      GPU=${GPUS[$((JOB_IDX % NUM_GPUS))]}

      NAME="l96_dbf_gated_useTrue_gt${GATE_TYPE}_gb${GATE_BIAS}_pzi${PRIOR_INIT}"

      echo "=========================================="
      echo "Job $JOB_IDX on GPU $GPU: $NAME"
      echo "  use_gated_obs_correction = true"
      echo "  gate_type               = $GATE_TYPE"
      echo "  gate_init_bias          = $GATE_BIAS"
      echo "  prior_zero_init         = $PRIOR_INIT"
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
        --gate-type "$GATE_TYPE"
        --gate-init-bias "$GATE_BIAS"
        --no-obs-zero-init
        --use-gated-obs-correction
      )

      # prior_zero_init (BooleanOptionalAction)
      if [ "$PRIOR_INIT" = "true" ]; then
        CMD+=(--prior-zero-init)
      else
        CMD+=(--no-prior-zero-init)
      fi

      LOGFILE="logs/${NAME}.log"
      echo "Running command (background):"
      echo "  CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]} > $LOGFILE 2>&1 &"

      CUDA_VISIBLE_DEVICES="$GPU" nohup "${CMD[@]}" > "$LOGFILE" 2>&1 &

      JOB_IDX=$((JOB_IDX + 1))
    done
  done
done

echo "=========================================="
echo "Submitted $JOB_IDX RF training jobs in background."
echo "Baseline: rf_runs/$BASELINE_NAME"
echo "Gated runs: rf_runs/l96_dbf_gated_useTrue_*"
echo "=========================================="

