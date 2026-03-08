#!/bin/bash

# Evaluation script for Lorenz-96 DBF with previous-state corruption RF proposal sweep.
# - Single noiseless dataset (matching the prev-corr training dataset but without injected process noise).
# - One EnKF baseline run.
# - One EnSF baseline run.
# - One EnSF+RF run per trained prev-corr proposal (16 total).

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit 1

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

mkdir -p logs

echo "============================================="
echo "Lorenz96 (40D) DBF Prev-Corr RF Evaluation"
echo "============================================="

# Common parameters (mirroring eval_96_dbf.sh where sensible)
BATCH_SIZE=50
NUM_EVAL_TRAJECTORIES=50
N_PARTICLES_ENKF=50
N_PARTICLES_ENSF=100
INFLATION=1.0
ENSF_STEPS=50
ENSF_EPS_A=0.5
ENSF_EPS_B=0.025
RF_LIK_STEPS=100
RF_SAMP_STEPS=100
PROC_NOISE=0.2

WANDB_PROJECT="eval-96-prev-corr"
DATE_PREFIX=$(date +%Y%m%d)

PYTHON_BIN="/home/cnagda/miniconda3/envs/da/bin/python"

# Single noiseless dataset corresponding to the prev-corr training dataset
# Training used:
#   /data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10
# We use the matching noiseless dataset (no pnoise suffix):
DATASET="/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_quad_capped_10_initUnim10_10"

# ---------------------------------------------------------------------------
# Prev-corr RF checkpoints
# Names mirror those in train_96_dbf_rf_prev_corr_sweep.sh:
#   l96_dbf_prevcorr_p030_sig050
#   l96_dbf_prevcorr_p020_sig050
#   ...
# ---------------------------------------------------------------------------

PREVCORR_NAMES=(
  "l96_dbf_prevcorr_p030_sig050"
  "l96_dbf_prevcorr_p020_sig050"
  "l96_dbf_prevcorr_p030_sig025"
  "l96_dbf_prevcorr_p030_mask020_only"
  "l96_dbf_prevcorr_p030_mask040_only"
  "l96_dbf_prevcorr_p030_sig050_mask020"
  "l96_dbf_prevcorr_p030_sig050_mask040"
  "l96_dbf_prevcorr_p030_sig100"
  "l96_dbf_prevcorr_p030_sig075"
  "l96_dbf_prevcorr_p030_mask030_only"
  "l96_dbf_prevcorr_p030_sig050_mask030"
  "l96_dbf_prevcorr_p020_sig025"
  "l96_dbf_prevcorr_p030_sig050_pmin0"
  "l96_dbf_prevcorr_p040_sig050"
  "l96_dbf_prevcorr_p030_sig050_mask050"
  "l96_dbf_prevcorr_p020_sig050_mask020"
)

NUM_PROPOSALS=${#PREVCORR_NAMES[@]}

RF_CKPTS=()
LABELS=()
for NAME in "${PREVCORR_NAMES[@]}"; do
  RF_CKPTS+=("rf_runs/${NAME}/final_model.ckpt")
  LABELS+=("${NAME}")
done

# Models to evaluate
MODELS=("enkf" "ensf" "ensf_rf")

# GPU scheduling
MAX_GPUS=8
JOBS_PER_GPU=2
MAX_CONCURRENT=$((MAX_GPUS * JOBS_PER_GPU))

JOB_GPUS=()
JOB_CMDS=()
JOB_LOGS=()

GPU_COUNTER=0

add_job() {
  local MODEL=$1
  local LABEL=$2
  local RF_CKPT=$3

  local GPU=$((GPU_COUNTER % MAX_GPUS))
  GPU_COUNTER=$((GPU_COUNTER + 1))

  local RUN_NAME="${DATE_PREFIX}_${MODEL}_${LABEL}_pnoise0p2"
  local LOG_FILE="logs/${RUN_NAME}.log"

  echo "Queueing job: MODEL=${MODEL}, LABEL=${LABEL}, GPU=${GPU}"

  local CMD="$PYTHON_BIN eval.py \
    --data-dir \"$DATASET\" \
    --process-noise-std $PROC_NOISE \
    --batch-size $BATCH_SIZE \
    --num-eval-trajectories $NUM_EVAL_TRAJECTORIES \
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
  else
    echo "Unknown MODEL=$MODEL"
    return 1
  fi

  JOB_GPUS+=("$GPU")
  JOB_CMDS+=("$CMD")
  JOB_LOGS+=("$LOG_FILE")
}

echo "Building job queues..."

# 1) Baseline EnKF (no RF)
add_job "enkf" "baseline" ""

# 2) Baseline EnSF (no RF)
add_job "ensf" "baseline" ""

# 3) EnSF + RF: one per prev-corr proposal
for ((i=0; i<NUM_PROPOSALS; i++)); do
  LABEL="${LABELS[$i]}"
  RF_CKPT="${RF_CKPTS[$i]}"
  add_job "ensf_rf" "$LABEL" "$RF_CKPT"
done

echo "Launching jobs..."
echo "  MAX_GPUS=$MAX_GPUS"
echo "  JOBS_PER_GPU=$JOBS_PER_GPU"
echo "  MAX_CONCURRENT=$MAX_CONCURRENT"

RUNNING=0
TOTAL_JOBS=${#JOB_CMDS[@]}

for ((i=0; i<TOTAL_JOBS; i++)); do
  GPU="${JOB_GPUS[$i]}"
  CMD="${JOB_CMDS[$i]}"
  LOG_FILE="${JOB_LOGS[$i]}"

  echo "Starting job $((i + 1))/$TOTAL_JOBS on GPU $GPU (log: $LOG_FILE)"

  CUDA_VISIBLE_DEVICES=$GPU bash -lc "$CMD" > "$LOG_FILE" 2>&1 &
  RUNNING=$((RUNNING + 1))

  if [ "$RUNNING" -ge "$MAX_CONCURRENT" ]; then
    wait -n
    RUNNING=$((RUNNING - 1))
  fi
done

wait

echo "============================================="
echo "All prev-corr RF evaluation queues started."
echo "Monitor logs in logs/."
echo "============================================="

