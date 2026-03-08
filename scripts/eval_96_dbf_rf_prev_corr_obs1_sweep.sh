#!/bin/bash

# Evaluation script for Lorenz-96 DBF with previous-state corruption RF proposal (obs1 sweep).
# Uses the 4 trained models from train_96_dbf_rf_prev_corr_obs1_sweep.sh.
# - Uses lowest val_loss checkpoint (not final_model)
# - 3 seeds per experiment, seed in wandb run name
# - New wandb project: eval-96-prev-corr-obs1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.." || exit 1

export PYTHONPATH=$(pwd)

mkdir -p logs

echo "============================================="
echo "Lorenz96 (40D) DBF Prev-Corr RF Evaluation (obs1)"
echo "============================================="

# Common parameters
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

WANDB_PROJECT="eval-96-prev-corr-obs1"
DATE_PREFIX=$(date +%Y%m%d)
SEEDS=(0 1 2)

PYTHON_BIN="/home/cnagda/miniconda3/envs/da/bin/python"

# Matching noiseless dataset (training used obs1p000, pnoise0p200)
DATASET="/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_initUnim10_10"

# ---------------------------------------------------------------------------
# Find best (lowest val_loss) checkpoint in rf_runs/NAME/checkpoints/
# PyTorch Lightning saves: rf-epoch=007-val_loss=0.034167.ckpt (or rf-042-0.123456.ckpt)
# ---------------------------------------------------------------------------
get_best_ckpt() {
  local run_dir=$1
  local ckpt_dir="$run_dir/checkpoints"
  if [ ! -d "$ckpt_dir" ]; then
    echo ""
    return 1
  fi
  # List val_loss and path, sort numerically, take lowest
  for f in "$ckpt_dir"/rf-*.ckpt; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .ckpt)
    # Extract val_loss: Lightning format rf-epoch=007-val_loss=0.034167 or legacy rf-042-0.123456
    val_loss=$(echo "$base" | sed -n 's/.*val_loss=\([0-9.]*\).*/\1/p')
    [ -z "$val_loss" ] && val_loss=$(echo "$base" | sed -n 's/.*-\([0-9]\+\.[0-9]*\)$/\1/p')
    [ -n "$val_loss" ] && printf "%s\t%s\n" "$val_loss" "$f"
  done | sort -n | head -1 | cut -f2-
}

# ---------------------------------------------------------------------------
# 4 prev-corr RF runs from train_96_dbf_rf_prev_corr_obs1_sweep.sh
# ---------------------------------------------------------------------------
PREVCORR_NAMES=(
  "l96_dbf_baseline_ensf_rf"
  "l96_dbf_prevcorr_p030_sig050"
  "l96_dbf_prevcorr_p030_mask040_only"
  "l96_dbf_prevcorr_p030_sig050_mask040"
)

NUM_PROPOSALS=${#PREVCORR_NAMES[@]}

# Resolve best checkpoints
RF_CKPTS=()
LABELS=()
for NAME in "${PREVCORR_NAMES[@]}"; do
  RUN_DIR="rf_runs/${NAME}"
  CKPT=$(get_best_ckpt "$RUN_DIR")
  if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "WARNING: No best checkpoint found for $NAME in $RUN_DIR/checkpoints/, skipping"
    continue
  fi
  RF_CKPTS+=("$CKPT")
  LABELS+=("${NAME}")
done

NUM_PROPOSALS=${#RF_CKPTS[@]}
echo "Found $NUM_PROPOSALS RF checkpoints (best val_loss):"
for ((i=0; i<NUM_PROPOSALS; i++)); do
  echo "  ${LABELS[$i]}: ${RF_CKPTS[$i]}"
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
  local SEED=$4

  local GPU=$((GPU_COUNTER % MAX_GPUS))
  GPU_COUNTER=$((GPU_COUNTER + 1))

  local RUN_NAME="${DATE_PREFIX}_${MODEL}_${LABEL}_seed${SEED}_pnoise0p2"
  local LOG_FILE="logs/${RUN_NAME}.log"

  echo "Queueing job: MODEL=${MODEL}, LABEL=${LABEL}, SEED=${SEED}, GPU=${GPU}"

  local CMD="$PYTHON_BIN eval.py \
    --data-dir \"$DATASET\" \
    --process-noise-std $PROC_NOISE \
    --batch-size $BATCH_SIZE \
    --num-eval-trajectories $NUM_EVAL_TRAJECTORIES \
    --wandb-project \"$WANDB_PROJECT\" \
    --run-name \"$RUN_NAME\" \
    --seed $SEED \
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

# 1) Baseline EnKF (no RF) - 3 seeds
for SEED in "${SEEDS[@]}"; do
  add_job "enkf" "baseline" "" "$SEED"
done

# 2) Baseline EnSF (no RF) - 3 seeds
for SEED in "${SEEDS[@]}"; do
  add_job "ensf" "baseline" "" "$SEED"
done

# 3) EnSF + RF: one per prev-corr proposal, 3 seeds each
for ((i=0; i<NUM_PROPOSALS; i++)); do
  LABEL="${LABELS[$i]}"
  RF_CKPT="${RF_CKPTS[$i]}"
  for SEED in "${SEEDS[@]}"; do
    add_job "ensf_rf" "$LABEL" "$RF_CKPT" "$SEED"
  done
done

echo "Launching jobs..."
echo "  MAX_GPUS=$MAX_GPUS"
echo "  JOBS_PER_GPU=$JOBS_PER_GPU"
echo "  MAX_CONCURRENT=$MAX_CONCURRENT"
echo "  Total jobs: ${#JOB_CMDS[@]}"

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
echo "All prev-corr obs1 RF evaluation jobs completed."
echo "Monitor logs in logs/."
echo "============================================="
