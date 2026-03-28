#!/bin/bash

# Evaluation sweep for Lorenz-96 DBF using RF proposal checkpoint obscons_w000.
# Focus: EnKF baseline, standard BPF baseline (transition proposal),
# and RF-proposal BPF/ABPF with ABPF first-stage correction toggle + resampling threshold.
# N particles is fixed to 5000 for all runs.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.." || exit 1

export PYTHONPATH=$(pwd)
mkdir -p logs

echo "============================================="
echo "Lorenz96 (40D) DBF ABPF RF Evaluation Sweep"
echo "============================================="

# Common parameters
BATCH_SIZE=20
NUM_EVAL_TRAJECTORIES=40
N_PARTICLES=5000
N_PARTICLES_ENKF=50
PROC_NOISE=0.2
RF_LIK_STEPS=100
RF_SAMP_STEPS=100

WANDB_PROJECT="eval-96-abpf"
DATE_PREFIX=$(date +%Y%m%d)

PYTHON_BIN="/home/cnagda/miniconda3/envs/da/bin/python"

DATASET="/data/da_outputs/datasets/lorenz96_n2048_len80_dt0p0300_obs1p000_freq1_comp40of40_quad_capped_10_initUnim10_10"
RF_CKPT="rf_runs/l96_dbf_rf_obscons_w000/final_model.ckpt"

# Sweep axes
METHODS=("bpf" "bpf_abpf")
ABPF_CORRECTIONS=("off" "on")
RESAMPLING_THRESHOLDS=(0.33 0.5)

# GPU scheduling
MAX_GPUS=8
JOBS_PER_GPU=1
MAX_CONCURRENT=$((MAX_GPUS * JOBS_PER_GPU))

JOB_GPUS=()
JOB_CMDS=()
JOB_LOGS=()
GPU_COUNTER=0

add_job() {
  local METHOD=$1
  local PROPOSAL=$2
  local ABPF_CORR=$3
  local RESAMP=$4

  local GPU=$((GPU_COUNTER % MAX_GPUS))
  GPU_COUNTER=$((GPU_COUNTER + 1))

  if [ "$METHOD" == "enkf" ]; then
    local RUN_NAME="${DATE_PREFIX}_enkf_baseline_N${N_PARTICLES_ENKF}"
    local LOG_FILE="logs/${RUN_NAME}.log"
    echo "Queueing job: METHOD=enkf (baseline), GPU=${GPU}"
    local CMD="$PYTHON_BIN eval.py \
      --data-dir \"$DATASET\" \
      --method enkf \
      --n-particles $N_PARTICLES_ENKF \
      --process-noise-std $PROC_NOISE \
      --batch-size $BATCH_SIZE \
      --num-eval-trajectories $NUM_EVAL_TRAJECTORIES \
      --wandb-project \"$WANDB_PROJECT\" \
      --run-name \"$RUN_NAME\" \
      --device cuda"
    JOB_GPUS+=("$GPU")
    JOB_CMDS+=("$CMD")
    JOB_LOGS+=("$LOG_FILE")
    return
  fi

  local CORR_TAG="none"
  if [ "$METHOD" == "bpf_abpf" ]; then
    CORR_TAG="$ABPF_CORR"
  fi

  local PROPOSAL_TAG="$PROPOSAL"
  local RESAMP_TAG=${RESAMP/./p}
  local RUN_NAME="${DATE_PREFIX}_${METHOD}_${PROPOSAL_TAG}_corr${CORR_TAG}_N${N_PARTICLES}_rs${RESAMP_TAG}"
  if [ "$PROPOSAL" == "rf" ]; then
    RUN_NAME="${RUN_NAME}_rfw000"
  fi
  local LOG_FILE="logs/${RUN_NAME}.log"

  echo "Queueing job: METHOD=${METHOD}, PROPOSAL=${PROPOSAL}, CORR=${CORR_TAG}, RS=${RESAMP}, GPU=${GPU}"

  local CMD="$PYTHON_BIN eval.py \
    --data-dir \"$DATASET\" \
    --method $METHOD \
    --proposal-type $PROPOSAL \
    --n-particles $N_PARTICLES \
    --process-noise-std $PROC_NOISE \
    --resampling-threshold $RESAMP \
    --batch-size $BATCH_SIZE \
    --num-eval-trajectories $NUM_EVAL_TRAJECTORIES \
    --wandb-project \"$WANDB_PROJECT\" \
    --run-name \"$RUN_NAME\" \
    --device cuda"

  if [ "$PROPOSAL" == "rf" ]; then
    CMD="$CMD --rf-checkpoint \"$RF_CKPT\" \
      --rf-likelihood-steps $RF_LIK_STEPS \
      --rf-sampling-steps $RF_SAMP_STEPS"
  fi

  if [ "$METHOD" == "bpf_abpf" ] && [ "$ABPF_CORR" == "on" ]; then
    CMD="$CMD --abpf-first-stage-correction"
  fi

  JOB_GPUS+=("$GPU")
  JOB_CMDS+=("$CMD")
  JOB_LOGS+=("$LOG_FILE")
}

echo "Building job queue..."

# Baseline EnKF
# add_job "enkf" "transition" "off" "0.33"

# Standard BPF baseline with transition proposal
# add_job "bpf" "transition" "off" "0.33"

for RESAMP in "${RESAMPLING_THRESHOLDS[@]}"; do
  # RF-proposal BPF baseline per threshold
  add_job "bpf" "rf" "off" "$RESAMP"

  # RF-proposal ABPF with and without first-stage correction
  for CORR in "${ABPF_CORRECTIONS[@]}"; do
    add_job "bpf_abpf" "rf" "$CORR" "$RESAMP"
  done
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
echo "All ABPF evaluation jobs completed."
echo "Monitor logs in logs/."
echo "============================================="
