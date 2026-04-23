#!/bin/bash
set -euo pipefail

# Sequentially rerun only the failed tasks from array job 8193368 on a specific
# allocated compute node via SSH (default: ccc0284).
#
# Usage:
#   bash scripts/rerun_enkf_failed_via_ssh.sh
#   SSH_NODE=ccc0284 BATCH_SIZE=50 CONCURRENCY_TAG=rerun_d50_bs50 \
#     bash scripts/rerun_enkf_failed_via_ssh.sh
#   FAILED_TASKS="51 52 53 54 55 56 57 58 59 110 111 112 113 114 115 116 117 118 119" \
#     bash scripts/rerun_enkf_failed_via_ssh.sh

SSH_NODE="${SSH_NODE:-ccc0284}"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/DataAssimilation}"
WANDB_PROJECT="${WANDB_PROJECT:-eval-enkf-l96-neurips}"
CONCURRENCY_TAG="${CONCURRENCY_TAG:-rerun8193368}"

# Default to the failed task ids from the last sweep.
FAILED_TASKS="${FAILED_TASKS:-51 52 53 54 55 56 57 58 59 110 111 112 113 114 115 116 117 118 119}"
FAILED_TASKS_CSV="${FAILED_TASKS_CSV:-${FAILED_TASKS// /,}}"

# Match your new plan.
BATCH_SIZE="${BATCH_SIZE:-50}"

echo "Node          : ${SSH_NODE}"
echo "Project root  : ${PROJECT_ROOT}"
echo "W&B project   : ${WANDB_PROJECT}"
echo "Batch size    : ${BATCH_SIZE}"
echo "Tasks         : ${FAILED_TASKS_CSV}"
echo

ssh -o StrictHostKeyChecking=no "${SSH_NODE}" 'bash -l -s' -- \
  "${SSH_NODE}" \
  "${PROJECT_ROOT}" \
  "${WANDB_PROJECT}" \
  "${CONCURRENCY_TAG}" \
  "${FAILED_TASKS_CSV}" \
  "${BATCH_SIZE}" <<'REMOTE'
set -euo pipefail

SSH_NODE_ENV="${1}"
PROJECT_ROOT="${2}"
WANDB_PROJECT="${3}"
CONCURRENCY_TAG="${4}"
FAILED_TASKS_CSV="${5}"
BATCH_SIZE="${6}"
shift 6

module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate da
module load cuda/12.8

cd "${PROJECT_ROOT}"
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0

DIMS=(5 10 15 20 25 50)
NOISES=(0.2 0.5 1 3 5)
OPS_DIR=(arctan quad_capped_10)
OPS_TAG=(arctan quad_capped)
ENSEMBLE_SIZES=(20 50)

NUM_DIMS=${#DIMS[@]}
NUM_NOISE=${#NOISES[@]}
NUM_OPS=${#OPS_DIR[@]}
NUM_ENSEMBLES=${#ENSEMBLE_SIZES[@]}
BASE_GRID_TOTAL=$((NUM_DIMS * NUM_NOISE * NUM_OPS))
TOTAL=$((BASE_GRID_TOTAL * NUM_ENSEMBLES))

noise_to_obs_token() {
  case "$1" in
    0.2) echo "obs0p200" ;;
    0.5) echo "obs0p500" ;;
    1)   echo "obs1p000" ;;
    3)   echo "obs3p000" ;;
    5)   echo "obs5p000" ;;
    *)   echo "Unsupported obs noise: $1" >&2; exit 1 ;;
  esac
}

DATA_ROOT="${DATA_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims}"
INFLATION="${INFLATION:-1.0}"
PROCESS_NOISE_STD="${PROCESS_NOISE_STD:-0.2}"
NUM_EVAL_TRAJECTORIES="${NUM_EVAL_TRAJECTORIES:-100}"
SEED="${SEED:-42}"

IFS=',' read -r -a TASK_ID_LIST <<< "${FAILED_TASKS_CSV}"

run_counter=0
for TASK_ID in "${TASK_ID_LIST[@]}"; do
  if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
    echo "Skipping invalid TASK_ID=${TASK_ID} (valid range: [0, ${TOTAL}))"
    continue
  fi

  base_task_id=$((TASK_ID % BASE_GRID_TOTAL))
  ens_idx=$((TASK_ID / BASE_GRID_TOTAL))
  op_idx=$((base_task_id % NUM_OPS))
  tmp=$((base_task_id / NUM_OPS))
  noise_idx=$((tmp % NUM_NOISE))
  dim_idx=$((tmp / NUM_NOISE))

  STATE_DIM="${DIMS[$dim_idx]}"
  OBS_NOISE="${NOISES[$noise_idx]}"
  OBS_OP_DIR="${OPS_DIR[$op_idx]}"
  OBS_OP_TAG="${OPS_TAG[$op_idx]}"
  N_PARTICLES="${ENSEMBLE_SIZES[$ens_idx]}"
  OBS_TOKEN="$(noise_to_obs_token "${OBS_NOISE}")"

  DATASET_NAME="lorenz96_n2048_len200_dt0p0100_${OBS_TOKEN}_freq1_comp${STATE_DIM}of${STATE_DIM}_${OBS_OP_DIR}_init3p000"
  DATA_DIR="${DATA_ROOT}/${DATASET_NAME}"
  if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: DATA_DIR not found for TASK_ID=${TASK_ID}: ${DATA_DIR}"
    exit 1
  fi

  RUN_NAME="eval_enkf_d${STATE_DIM}_${OBS_TOKEN}_${OBS_OP_TAG}_${CONCURRENCY_TAG}_t${TASK_ID}_bs${BATCH_SIZE}"

  echo "==============================================================="
  echo "[$((++run_counter))] task=${TASK_ID} node=${SSH_NODE_ENV} dim=${STATE_DIM} obs=${OBS_TOKEN} op=${OBS_OP_TAG} N=${N_PARTICLES} bs=${BATCH_SIZE}"
  echo "run_name=${RUN_NAME}"
  echo "==============================================================="

  python eval.py \
    --data-dir "${DATA_DIR}" \
    --method enkf \
    --n-particles "${N_PARTICLES}" \
    --inflation "${INFLATION}" \
    --process-noise-std "${PROCESS_NOISE_STD}" \
    --batch-size "${BATCH_SIZE}" \
    --num-eval-trajectories "${NUM_EVAL_TRAJECTORIES}" \
    --seed "${SEED}" \
    --device cuda \
    --experiment-label "${RUN_NAME}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-tags "eval_enkf_l96_neurips,rerun_ssh,d${STATE_DIM},${OBS_TOKEN},${OBS_OP_TAG},N${N_PARTICLES},infl${INFLATION},bs${BATCH_SIZE},node_${SSH_NODE_ENV}" \
    --run-name "${RUN_NAME}"
done

echo "All requested tasks completed."
REMOTE

