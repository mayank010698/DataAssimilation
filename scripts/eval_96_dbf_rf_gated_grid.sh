#!/bin/bash

# Evaluation script for Lorenz-96 DBF with gated RF proposal sweep.
# - Single noiseless dataset (matching the gated training dataset but without injected process noise).
# - One EnKF baseline run.
# - One EnSF baseline run.
# - One EnSF+RF run per trained gated proposal (32 total).

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit 1

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

mkdir -p logs

echo "============================================="
echo "Lorenz96 (40D) DBF Gated RF Evaluation"
echo "============================================="

# Common parameters (mirroring eval_96_dbf.sh where sensible)
BATCH_SIZE=25
N_PARTICLES_ENKF=50
N_PARTICLES_ENSF=100
INFLATION=1.0
ENSF_STEPS=50
ENSF_EPS_A=0.5
ENSF_EPS_B=0.025
RF_LIK_STEPS=100
RF_SAMP_STEPS=100
PROC_NOISE=0.2

WANDB_PROJECT="eval-96-gated-grid"
DATE_PREFIX=$(date +%Y%m%d)

PYTHON_BIN="/home/cnagda/miniconda3/envs/da/bin/python"

# Single noiseless dataset corresponding to the gated training dataset
# Training used:
#   /data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10
# We use the matching noiseless dataset (no pnoise suffix):
DATASET="/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp40of40_quad_capped_10_initUnim10_10"

# ---------------------------------------------------------------------------
# Gated RF checkpoints
# Names mirror those in train_96_dbf_rf_gated_grid.sh:
#   NAME=\"l96_dbf_gated_use${USE_GATED}_gt${GATE_TYPE}_gb${GATE_BIAS}_pzi${PRIOR_INIT}\"
#   output_dir: rf_runs/$NAME
# ---------------------------------------------------------------------------

GATED_NAMES=()

for USE_GATED in true false; do
  for GATE_TYPE in scalar spatial; do
    for GATE_BIAS in 0.0 0.1 0.2 0.3; do
      for PRIOR_INIT in true false; do
        NAME="l96_dbf_gated_use${USE_GATED}_gt${GATE_TYPE}_gb${GATE_BIAS}_pzi${PRIOR_INIT}"
        GATED_NAMES+=("$NAME")
      done
    done
  done
done

NUM_PROPOSALS=${#GATED_NAMES[@]}

RF_CKPTS=()
LABELS=()
for NAME in "${GATED_NAMES[@]}"; do
  RF_CKPTS+=("rf_runs/${NAME}/final_model.ckpt")
  LABELS+=("${NAME}")
done

# Models to evaluate
MODELS=("enkf" "ensf" "ensf_rf")

# GPU scheduling
MAX_GPUS=8
GPU_CMDS=()
for ((g=0; g<MAX_GPUS; g++)); do
  GPU_CMDS[$g]=""
done

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

  if [ -z "${GPU_CMDS[$GPU]}" ]; then
    GPU_CMDS[$GPU]="$CMD > \"$LOG_FILE\" 2>&1"
  else
    GPU_CMDS[$GPU]="${GPU_CMDS[$GPU]} && $CMD > \"$LOG_FILE\" 2>&1"
  fi
}

echo "Building job queues..."

# 1) Baseline EnKF (no RF)
add_job "enkf" "baseline" ""

# 2) Baseline EnSF (no RF)
add_job "ensf" "baseline" ""

# 3) EnSF + RF: one per gated proposal
for ((i=0; i<NUM_PROPOSALS; i++)); do
  LABEL="${LABELS[$i]}"
  RF_CKPT="${RF_CKPTS[$i]}"
  add_job "ensf_rf" "$LABEL" "$RF_CKPT"
done

echo "Launching GPU queues..."
for ((g=0; g<MAX_GPUS; g++)); do
  if [ -n "${GPU_CMDS[$g]}" ]; then
    echo "Starting queue on GPU $g..."
    CUDA_VISIBLE_DEVICES=$g eval "${GPU_CMDS[$g]}" &
  fi
done

echo "============================================="
echo "All gated RF evaluation queues started."
echo "Monitor logs in logs/."
echo "============================================="

