#!/bin/bash

# Evaluation script for Lorenz-96 DBF with gated RF proposal sweep.
# - Single noiseless dataset (matching the obs1 gated training dataset but without injected process noise).
# - One EnKF baseline run.
# - One EnSF baseline run.
# - One EnSF+RF baseline run (no gating, no prior zero init).
# - One EnSF+RF run per trained gated proposal (16 total).

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

WANDB_PROJECT="eval-96-gated-grid-obs1"
DATE_PREFIX=$(date +%Y%m%d)

PYTHON_BIN="/home/cnagda/miniconda3/envs/da/bin/python"

# Single noiseless dataset corresponding to the gated training dataset (obs noise 1.0)
# Training used:
#   /data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_pnoise0p100_initUnim10_10
# We use the matching noiseless dataset (no pnoise suffix):
DATASET="/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_initUnim10_10"

# ---------------------------------------------------------------------------
# RF checkpoints
# Names mirror those in train_96_dbf_rf_gated_grid.sh (wandb run name = output_dir.name).
# We use the checkpoint with lowest val_loss in each run's checkpoints/ dir.
#   Baseline: rf_runs/l96_dbf_rf_baseline_nogate_nopzi
#   Gated:    rf_runs/l96_dbf_gated_useTrue_gt{scalar|spatial}_gb{0.0..0.3}_pzi{true|false}
# ---------------------------------------------------------------------------

WANDB_TRAIN_PROJECT="rf-train-96-gated-grid-obs1"
WANDB_ENTITY="ml-climate"

BASELINE_NAME="l96_dbf_rf_baseline_nogate_nopzi"

GATED_NAMES=()
for GATE_TYPE in scalar spatial; do
  for GATE_BIAS in 0.0 0.1 0.2 0.3; do
    for PRIOR_INIT in true false; do
      NAME="l96_dbf_gated_useTrue_gt${GATE_TYPE}_gb${GATE_BIAS}_pzi${PRIOR_INIT}"
      GATED_NAMES+=("$NAME")
    done
  done
done

# Build list of expected run names (must match wandb run names from training)
EXPECTED_RUN_NAMES=("$BASELINE_NAME" "${GATED_NAMES[@]}")

# ---------------------------------------------------------------------------
# Sanity check: expected run names exist in wandb project rf-train-96-gated-grid-obs1
# ---------------------------------------------------------------------------
echo "Sanity check: verifying run names exist in wandb project ${WANDB_ENTITY}/${WANDB_TRAIN_PROJECT}..."
CHECK_SCRIPT=$(cat << 'PYEOF'
import sys
try:
  import wandb
  api = wandb.Api()
  runs = list(api.runs(f"{sys.argv[1]}/{sys.argv[2]}"))
  wandb_names = {r.name for r in runs}
  expected = sys.argv[3].split("\n")
  missing = [n for n in expected if n and n not in wandb_names]
  extra = wandb_names - set(n for n in expected if n)
  if missing:
    print("ERROR: The following expected run names are NOT in the wandb project:", file=sys.stderr)
    for n in missing:
      print("  -", n, file=sys.stderr)
    sys.exit(1)
  if extra:
    print("Note: wandb project has extra runs not in this eval list:", list(extra)[:5], "..." if len(extra) > 5 else "")
  print("OK: All", len(expected), "expected run names found in wandb.")
except Exception as e:
  print("WARNING: Could not verify wandb runs:", e, file=sys.stderr)
  sys.exit(0)
PYEOF
)
EXPECTED_NEWLINES=$(printf '%s\n' "${EXPECTED_RUN_NAMES[@]}")
if ! $PYTHON_BIN -c "$CHECK_SCRIPT" "$WANDB_ENTITY" "$WANDB_TRAIN_PROJECT" "$EXPECTED_NEWLINES"; then
  echo "Sanity check failed. Fix wandb run names or project and re-run."
  exit 1
fi

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
  for f in "$ckpt_dir"/rf-*.ckpt; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .ckpt)
    val_loss=$(echo "$base" | sed -n 's/.*val_loss=\([0-9.]*\).*/\1/p')
    [ -z "$val_loss" ] && val_loss=$(echo "$base" | sed -n 's/.*-\([0-9]\+\.[0-9]*\)$/\1/p')
    [ -n "$val_loss" ] && printf "%s\t%s\n" "$val_loss" "$f"
  done | sort -n | head -1 | cut -f2-
}

# Resolve best checkpoints (skip missing with warning)
RF_CKPTS=()
LABELS=()
for NAME in "${EXPECTED_RUN_NAMES[@]}"; do
  RUN_DIR="rf_runs/${NAME}"
  CKPT=$(get_best_ckpt "$RUN_DIR")
  if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "WARNING: No best checkpoint found for $NAME in $RUN_DIR/checkpoints/, skipping"
    continue
  fi
  RF_CKPTS+=("$CKPT")
  LABELS+=("${NAME}")
done

TOTAL_RF_MODELS=${#RF_CKPTS[@]}
echo "Using $TOTAL_RF_MODELS RF checkpoints (lowest val_loss):"
for ((i=0; i<TOTAL_RF_MODELS; i++)); do
  echo "  ${LABELS[$i]}: ${RF_CKPTS[$i]}"
done

# Models to evaluate
MODELS=("enkf" "ensf" "ensf_rf")

# GPU scheduling: multiple jobs per GPU, cap total concurrent jobs
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

# 3) EnSF + RF: one for baseline RF, then one per gated proposal
for ((i=0; i<TOTAL_RF_MODELS; i++)); do
  LABEL="${LABELS[$i]}"
  RF_CKPT="${RF_CKPTS[$i]}"
  add_job "ensf_rf" "$LABEL" "$RF_CKPT"
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
echo "All RF evaluation jobs completed."
echo "Monitor logs in logs/."
echo "============================================="

