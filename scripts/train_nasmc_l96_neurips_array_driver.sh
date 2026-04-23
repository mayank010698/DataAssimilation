#!/usr/bin/env bash
# Slurm job-array driver: one NASMC training run per array task.
# Expects SLURM_ARRAY_TASK_ID in 0..23 and a configured project root (cwd).
#
# Layout (24 tasks = 6 dims * 2 operators * 2 presets, single obs noise):
#   preset_idx = tid % NUM_PRESETS
#   tmp        = tid / NUM_PRESETS
#   op_idx     = tmp % NUM_OPS
#   dim_idx    = tmp / NUM_OPS
#
# Environment (set by the .sbatch wrapper):
#   OBS_NOISE                  -- fixed obs noise level (e.g. 0.2)
#   MAX_EPOCHS_PRETRAIN        -- phase-1 (MLE) epochs
#   MAX_EPOCHS_REFINE          -- phase-2 (NASMC) epochs
#   BOOTSTRAP_WARMUP_EPOCHS    -- epochs at start of phase 2 using bootstrap proposal
#   WANDB_PROJECT              -- W&B project name
#   WANDB_ENTITY               -- W&B entity (team) name
#   DATA_ROOT                  -- parent of per-run dataset dirs (l96_dims)
#   OUTPUT_ROOT                -- parent directory for checkpoints/logs
#   SEED                       -- optional; forwarded to train_nasmc if set

set -euo pipefail

OBS_NOISE="${OBS_NOISE:?Set OBS_NOISE}"
MAX_EPOCHS_PRETRAIN="${MAX_EPOCHS_PRETRAIN:?Set MAX_EPOCHS_PRETRAIN}"
MAX_EPOCHS_REFINE="${MAX_EPOCHS_REFINE:?Set MAX_EPOCHS_REFINE}"
BOOTSTRAP_WARMUP_EPOCHS="${BOOTSTRAP_WARMUP_EPOCHS:?Set BOOTSTRAP_WARMUP_EPOCHS}"
WANDB_PROJECT="${WANDB_PROJECT:?Set WANDB_PROJECT}"
WANDB_ENTITY="${WANDB_ENTITY:?Set WANDB_ENTITY}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

DIMS=(5 10 15 20 25 50)
# Directory tokens under DATA_ROOT (see generate.py / data.py naming)
OPS_DIR=(arctan quad_capped_10)
# Short tags for paths / W&B run names (quad_capped_10 -> quad_capped)
OPS_TAG=(arctan quad_capped)
# NASMC presets (see proposals/train_nasmc.py PRESETS dict).
#   pure_nasmc_K1 -- single-Gaussian (Gu et al. 2015 base variant)
#   pure_nasmc    -- K=3 mixture of diagonal Gaussians (-MD- variant)
PRESETS=(pure_nasmc_K1 pure_nasmc)
# Short tags used in RUN_NAME / wandb_run_name (must be filesystem-safe).
PRESETS_TAG=(nasmc_k1 nasmc_mdn)

NUM_DIMS=${#DIMS[@]}
NUM_OPS=${#OPS_DIR[@]}
NUM_PRESETS=${#PRESETS[@]}
TOTAL=$((NUM_DIMS * NUM_OPS * NUM_PRESETS))

tid="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"
if ((tid < 0 || tid >= TOTAL)); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${tid} (expected 0..$((TOTAL - 1)))"
    exit 1
fi

preset_idx=$((tid % NUM_PRESETS))
tmp=$((tid / NUM_PRESETS))
op_idx=$((tmp % NUM_OPS))
dim_idx=$((tmp / NUM_OPS))

STATE_DIM="${DIMS[$dim_idx]}"
OBS_OP_DIR="${OPS_DIR[$op_idx]}"
OBS_OP_TAG="${OPS_TAG[$op_idx]}"
PRESET="${PRESETS[$preset_idx]}"
PRESET_TAG="${PRESETS_TAG[$preset_idx]}"

noise_to_obs_token() {
    case "$1" in
        0.2) echo "obs0p200" ;;
        0.5) echo "obs0p500" ;;
        1)   echo "obs1p000" ;;
        3)   echo "obs3p000" ;;
        5)   echo "obs5p000" ;;
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

# Architecture / optimisation hyperparameters match the dim=5 NASMC pilot
# (train_nasmc_l96_dim5.sbatch) so capacity and training dynamics are
# comparable across the grid.
ARCH=resnet1d
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
TIME_EMBED_DIM=128
BATCH_SIZE=64
# Phase-2 unrolls an SMC sweep of SEGMENT_LENGTH * NUM_PARTICLES extra
# activations per sample; keep the refine-phase batch smaller to avoid OOM.
BATCH_SIZE_REFINE=64
LR=3e-4
NUM_PARTICLES=32
SEGMENT_LENGTH=64
RESAMPLE_THRESHOLD=0.5

JOB_TAG="${SLURM_JOB_ID:-manual}"
ARRAY_TAG="${SLURM_ARRAY_TASK_ID:-0}"
RUN_NAME="l96_d${STATE_DIM}_${OBS_TOKEN}_${OBS_OP_TAG}_${PRESET_TAG}_a${ARRAY_TAG}"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${RUN_NAME}_job${JOB_TAG}"

mkdir -p "${OUTPUT_DIR}" logs

echo "=========================================="
echo "NASMC L96 Neurips grid task"
echo "  array_task_id=${tid} / ${TOTAL}"
echo "  state_dim=${STATE_DIM}"
echo "  obs_noise_std=${OBS_NOISE}  (${OBS_TOKEN})"
echo "  obs_operator_dir=${OBS_OP_DIR}  (tag=${OBS_OP_TAG})"
echo "  preset=${PRESET}  (tag=${PRESET_TAG})"
echo "  data_dir=${DATA_DIR}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  max_epochs_pretrain=${MAX_EPOCHS_PRETRAIN}"
echo "  max_epochs_refine=${MAX_EPOCHS_REFINE}"
echo "  bootstrap_warmup_epochs=${BOOTSTRAP_WARMUP_EPOCHS}"
echo "  wandb_project=${WANDB_PROJECT}"
echo "=========================================="

TRAIN_LOG="logs/${RUN_NAME}_job${JOB_TAG}_a${ARRAY_TAG}.log"

SEED_ARGS=()
if [[ -n "${SEED:-}" ]]; then
    SEED_ARGS=(--seed "${SEED}")
fi

srun python -m proposals.train_nasmc \
    --preset "${PRESET}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --state_dim "${STATE_DIM}" \
    --obs_components "${OBS_COMPONENTS}" \
    --use_observations \
    --architecture "${ARCH}" \
    --channels "${CHANNELS}" \
    --num_blocks "${NUM_BLOCKS}" \
    --kernel_size "${KERNEL_SIZE}" \
    --time_embed_dim "${TIME_EMBED_DIM}" \
    --predict_delta \
    --batch_size "${BATCH_SIZE}" \
    --batch_size_refine "${BATCH_SIZE_REFINE}" \
    --learning_rate "${LR}" \
    --max_epochs_pretrain "${MAX_EPOCHS_PRETRAIN}" \
    --max_epochs_refine "${MAX_EPOCHS_REFINE}" \
    --bootstrap_warmup_epochs "${BOOTSTRAP_WARMUP_EPOCHS}" \
    --num_particles "${NUM_PARTICLES}" \
    --segment_length "${SEGMENT_LENGTH}" \
    --resample_threshold "${RESAMPLE_THRESHOLD}" \
    --num_workers 4 \
    --gpus 1 \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_run_name "${RUN_NAME}" \
    "${SEED_ARGS[@]}" \
    2>&1 | tee "${TRAIN_LOG}"

exit "${PIPESTATUS[0]}"
