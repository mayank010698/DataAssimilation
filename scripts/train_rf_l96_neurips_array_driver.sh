#!/usr/bin/env bash
# Slurm job-array driver: one RF training run per array task.
# Expects SLURM_ARRAY_TASK_ID in 0..59 and a configured project root (cwd).
#
# Environment (set by the .sbatch wrapper):
#   MAX_EPOCHS          — training epochs (500 prod, 1 smoke)
#   WANDB_PROJECT       — W&B project name
#   DATA_ROOT           — parent of per-run dataset dirs (l96_dims)
#   OUTPUT_ROOT         — parent directory for checkpoints/logs
#   SEED                — optional; if unset, train_rf uses its default

set -euo pipefail

MAX_EPOCHS="${MAX_EPOCHS:?Set MAX_EPOCHS}"
WANDB_PROJECT="${WANDB_PROJECT:?Set WANDB_PROJECT}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

DIMS=(5 10 15 20 25 50)
NOISES=(0.2 0.5 1 3 5)
# Directory tokens under DATA_ROOT (see generate.py / data.py naming)
OPS_DIR=(arctan quad_capped_10)
# Short tags for paths / W&B run names (quad_capped_10 -> quad_capped)
OPS_TAG=(arctan quad_capped)

NUM_DIMS=${#DIMS[@]}
NUM_NOISE=${#NOISES[@]}
NUM_OPS=${#OPS_DIR[@]}
TOTAL=$((NUM_DIMS * NUM_NOISE * NUM_OPS))

tid="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
if ((tid < 0 || tid >= TOTAL)); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${tid} (expected 0..$((TOTAL - 1)))"
    exit 1
fi

op_idx=$((tid % NUM_OPS))
tmp=$((tid / NUM_OPS))
noise_idx=$((tmp % NUM_NOISE))
dim_idx=$((tmp / NUM_NOISE))

STATE_DIM="${DIMS[$dim_idx]}"
OBS_NOISE="${NOISES[$noise_idx]}"
OBS_OP_DIR="${OPS_DIR[$op_idx]}"
OBS_OP_TAG="${OPS_TAG[$op_idx]}"

noise_to_obs_token() {
    case "$1" in
        0.2) echo "obs0p200" ;;
        0.5) echo "obs0p500" ;;
        1) echo "obs1p000" ;;
        3) echo "obs3p000" ;;
        5) echo "obs5p000" ;;
        *)
            echo "Unsupported obs noise: $1" >&2
            exit 1
            ;;
    esac
}

OBS_TOKEN="$(noise_to_obs_token "${OBS_NOISE}")"
DATASET_NAME="lorenz96_n2048_len200_dt0p0100_${OBS_TOKEN}_freq1_comp${STATE_DIM}of${STATE_DIM}_${OBS_OP_DIR}_pnoise0p100_init3p000"
DATA_DIR="${DATA_ROOT%/}/${DATASET_NAME}"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Missing dataset directory:"
    echo "  ${DATA_DIR}"
    exit 1
fi

OBS_COMPONENTS="$(seq -s, 0 "$((STATE_DIM - 1))")"

JOB_TAG="${SLURM_JOB_ID:-manual}"
ARRAY_TAG="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="l96_d${STATE_DIM}_${OBS_TOKEN}_${OBS_OP_TAG}_a${ARRAY_TAG}"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${RUN_NAME}_job${JOB_TAG}"

mkdir -p "${OUTPUT_DIR}" logs

echo "=========================================="
echo "RF L96 Neurips grid task"
echo "  array_task_id=${tid} / ${TOTAL}"
echo "  state_dim=${STATE_DIM}"
echo "  obs_noise_std=${OBS_NOISE}  (${OBS_TOKEN})"
echo "  obs_operator_dir=${OBS_OP_DIR}  (tag=${OBS_OP_TAG})"
echo "  data_dir=${DATA_DIR}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  max_epochs=${MAX_EPOCHS}"
echo "  wandb_project=${WANDB_PROJECT}"
echo "=========================================="

TRAIN_LOG="logs/${RUN_NAME}_job${JOB_TAG}_a${ARRAY_TAG}.log"

SEED_ARGS=()
if [[ -n "${SEED:-}" ]]; then
    SEED_ARGS=(--seed "${SEED}")
fi

# NOTE: train_rf.py exposes classifier-free style conditioning dropout as --cond_dropout
# (there is no --obs_dropout on this entrypoint; localized RF uses --obs_dropout).
srun python proposals/train_rf.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --state_dim "${STATE_DIM}" \
    --obs_components "${OBS_COMPONENTS}" \
    --architecture resnet1d \
    --channels 64 \
    --num_blocks 10 \
    --kernel_size 5 \
    --train-cond-method adaln \
    --cond_embed_dim 128 \
    --use_observations \
    --predict_delta \
    --batch_size 1024 \
    --learning_rate 1e-3 \
    --max_epochs "${MAX_EPOCHS}" \
    --num_workers 4 \
    --gpus 1 \
    --cond_dropout 0.1 \
    --prev_state_corr_p0 0.3 \
    --prev_state_corr_p_min 0.05 \
    --prev_state_corr_mask_ratio 0.4 \
    --prev_state_corr_sigma 0.0 \
    --wandb_project "${WANDB_PROJECT}" \
    --evaluate \
    "${SEED_ARGS[@]}" \
    2>&1 | tee "${TRAIN_LOG}"

exit "${PIPESTATUS[0]}"
