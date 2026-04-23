#!/bin/bash
#
# 60-way evaluation: for every cell of the NeurIPS training grid
# (dims {5,10,15,20,25,50} x obs_noise {0.2,0.5,1,3,5} x ops {arctan, quad_capped_10})
# run a Bootstrap Particle Filter whose proposal is the best (lowest val_loss)
# RectifiedFlowProposal checkpoint from the matching training run.
#
# Each array task evaluates ONE cell on the *no-noise* version of the training
# dataset (i.e. no `pnoise0p100` suffix in the dataset dirname).
#
# === PF / proposal config (fixed across the sweep) ===
#   method            = bpf
#   proposal_type     = rf
#   n_particles       = 1000  for dims {5, 10}
#                       5000  for dims {15, 20, 25, 50}
#   batch_size        = max batch size from the probe
#                       ({5,10}->100, 15->50, 20->25, 25->25, 50->10)
#   process_noise_std = 0.2
#   sampling grid     = uniform    (32 steps)
#   likelihood grid   = power_back (32 steps, gamma=2.0)
#   resampling_thr.   = 0.33
#   num_eval_trajs    = 100
#   max_timesteps     = unset (full 200)
#   seed              = 42
#
# === Task decoding (matches train_rf_l96_neurips_array_driver.sh) ===
#   op_idx    = tid % 2
#   noise_idx = (tid / 2) % 5
#   dim_idx   = tid / 10
#
# === Outputs (shared across array tasks) ===
#   logs/eval_rf_l96_neurips_<array_job_id>/<run_name>.log   per-task stdout
#   logs/eval_rf_l96_neurips_<array_job_id>/summary.csv      one row per cell
#   W&B project:   eval-rf-l96-neurips (overridable)
#
#SBATCH --job-name="eval-rf-l96-neurips"
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-59
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%A_%a.log

set -euo pipefail

echo "=== Slurm context ==="
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-N/A}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "====================="

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/eval.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [ ! -f "${PROJECT_ROOT}/eval.py" ]; then
    echo "ERROR: Cannot find project root (expected eval.py at PROJECT_ROOT)."
    echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    echo "  tried PROJECT_ROOT=$PROJECT_ROOT"
    exit 1
fi
cd "$PROJECT_ROOT" || exit 1

module load anaconda3/2024.10
source activate
conda activate da
module load cuda/12.8

export PYTHONPATH="$(pwd)"

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script expects SLURM_ARRAY_TASK_ID (submit via --array).}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_DIR="logs/eval_rf_l96_neurips_${JOB_TAG}"
mkdir -p "${RUN_DIR}"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
CSV_LOCK="${RUN_DIR}/summary.csv.lock"

# --- Grid definition (mirrors train_rf_l96_neurips_array_driver.sh) ---
DIMS=(5 10 15 20 25 50)
NOISES=(0.2 0.5 1 3 5)
OPS_DIR=(arctan quad_capped_10)
OPS_TAG=(arctan quad_capped)

NUM_DIMS=${#DIMS[@]}
NUM_NOISE=${#NOISES[@]}
NUM_OPS=${#OPS_DIR[@]}
TOTAL=$((NUM_DIMS * NUM_NOISE * NUM_OPS))

if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} is out of range [0, ${TOTAL})."
    exit 1
fi

op_idx=$((TASK_ID % NUM_OPS))
tmp=$((TASK_ID / NUM_OPS))
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
        1)   echo "obs1p000" ;;
        3)   echo "obs3p000" ;;
        5)   echo "obs5p000" ;;
        *)   echo "Unsupported obs noise: $1" >&2; exit 1 ;;
    esac
}
OBS_TOKEN="$(noise_to_obs_token "${OBS_NOISE}")"

# --- Per-dim PF config (from probe_rf_batch_size_l96.sbatch, job 8180334) ---
#    d=5,10 -> N=1000, bs=100
#    d=15   -> N=5000, bs=50
#    d=20   -> N=5000, bs=25
#    d=25   -> N=5000, bs=25
#    d=50   -> N=5000, bs=10
case "$STATE_DIM" in
    5|10) N_PARTICLES=1000; BATCH_SIZE=100 ;;
    15)   N_PARTICLES=5000; BATCH_SIZE=50  ;;
    20)   N_PARTICLES=5000; BATCH_SIZE=25  ;;
    25)   N_PARTICLES=5000; BATCH_SIZE=25  ;;
    50)   N_PARTICLES=5000; BATCH_SIZE=10  ;;
    *)    echo "ERROR: no per-dim N/batch config for dim=${STATE_DIM}" >&2; exit 1 ;;
esac

# --- Resolve training run dir & best checkpoint for this cell ---
# Training output layout (from train_rf_l96_neurips_array_driver.sh):
#   ${TRAIN_RUN_ROOT}/l96_d${D}_${OBS_TOKEN}_${OP_TAG}_a${ARRAY_TAG}_job${JOB_TAG}
# We don't know ${ARRAY_TAG}/${JOB_TAG} here, so glob on the stable prefix.
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_l96_neurips}"
RUN_GLOB="${TRAIN_RUN_ROOT}/l96_d${STATE_DIM}_${OBS_TOKEN}_${OBS_OP_TAG}_a*"

shopt -s nullglob
RUN_MATCHES=( $RUN_GLOB )
shopt -u nullglob

if (( ${#RUN_MATCHES[@]} == 0 )); then
    echo "ERROR: no training run dir found matching: ${RUN_GLOB}"
    exit 1
fi
if (( ${#RUN_MATCHES[@]} > 1 )); then
    echo "WARN: multiple training run dirs for cell d=${STATE_DIM} ${OBS_TOKEN} ${OBS_OP_TAG}:"
    printf '  %s\n' "${RUN_MATCHES[@]}"
    echo "Using the first match; override TRAIN_RUN_ROOT if needed."
fi
RUN_ROOT="${RUN_MATCHES[0]}"

CKPT_DIR="${RUN_ROOT}/checkpoints"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint dir missing: ${CKPT_DIR}"; exit 1
fi

# Pick the RF ckpt with the lowest val_loss (filename format is
# `rf-epoch=XXX-val_loss=YYY.ckpt`). Sort numerically on the val_loss field.
CHECKPOINT="$(
    ls -1 "${CKPT_DIR}"/rf-epoch=*-val_loss=*.ckpt 2>/dev/null \
    | awk -F'val_loss=' '{print $2, $0}' \
    | sort -g \
    | head -n1 \
    | awk '{print $2}'
)"
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: no val_loss checkpoint resolved under ${CKPT_DIR}"; exit 1
fi

# --- Resolve no-noise evaluation dataset ---
DATA_ROOT="${DATA_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims}"
DATASET_NAME="lorenz96_n2048_len200_dt0p0100_${OBS_TOKEN}_freq1_comp${STATE_DIM}of${STATE_DIM}_${OBS_OP_DIR}_init3p000"
DATA_DIR="${DATA_ROOT}/${DATASET_NAME}"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: eval DATA_DIR not found: ${DATA_DIR}"
    echo "  (expected the *no-noise* dataset, i.e. without pnoise0p100)"
    exit 1
fi

# --- Fixed PF / proposal settings (match the probe) ---
PROCESS_NOISE_STD=0.2
SAMPLING_GRID="uniform"
LIKELIHOOD_GRID="power_back"
SAMPLING_STEPS=32
LIKELIHOOD_STEPS=32
RESAMPLING_THRESHOLD=0.33
NUM_EVAL_TRAJECTORIES=100
SEED=42

WANDB_PROJECT="${WANDB_PROJECT:-eval-rf-l96-neurips}"

RUN_NAME="eval_rf_d${STATE_DIM}_${OBS_TOKEN}_${OBS_OP_TAG}_${JOB_TAG}_t${TASK_ID}"
LOG_FILE="${RUN_DIR}/${RUN_NAME}.log"

echo "=========================================="
echo "eval_rf_l96_neurips_grid.sbatch (task ${TASK_ID}/${TOTAL})"
echo "  state_dim        = ${STATE_DIM}"
echo "  obs_noise        = ${OBS_NOISE} (${OBS_TOKEN})"
echo "  obs_operator     = ${OBS_OP_DIR} (tag=${OBS_OP_TAG})"
echo "  n_particles      = ${N_PARTICLES}"
echo "  batch_size       = ${BATCH_SIZE}"
echo "  train_run_root   = ${RUN_ROOT}"
echo "  checkpoint       = ${CHECKPOINT}"
echo "  data_dir         = ${DATA_DIR}"
echo "  sampling grid    = ${SAMPLING_GRID} / ${SAMPLING_STEPS}"
echo "  likelihood grid  = ${LIKELIHOOD_GRID} / ${LIKELIHOOD_STEPS} (gamma=2.0)"
echo "  num_eval_trajs   = ${NUM_EVAL_TRAJECTORIES}"
echo "  max_timesteps    = unset (full 200)"
echo "  process_noise    = ${PROCESS_NOISE_STD}"
echo "  wandb_project    = ${WANDB_PROJECT}"
echo "  log_file         = ${LOG_FILE}"
echo "=========================================="

# --- Ensure summary CSV header is written once ---
(
    flock -x 200
    if [ ! -s "${SUMMARY_CSV}" ]; then
        echo "task_id,dim,obs_noise,obs_op,n_particles,batch_size,wall_time_sec,mean_rmse,std_rmse,mean_crps,std_crps,mean_ess,run_name,checkpoint,data_dir,log_file" > "${SUMMARY_CSV}"
    fi
) 200>"${CSV_LOCK}"

# Extract a "value" line ("  Some label: 1.2345 ± 0.0123") from a log file.
extract_metric() {
    local log_file="$1"
    local label="$2"
    grep -F "${label}" "${log_file}" | tail -n1 \
        | awk -F ':' '{print $2}' \
        | awk '{print $1}'
}
extract_metric_std() {
    local log_file="$1"
    local label="$2"
    grep -F "${label}" "${log_file}" | tail -n1 \
        | awk -F '±' '{print $2}' \
        | awk '{print $1}'
}

eval_cmd=(
    python eval.py
    --data-dir "$DATA_DIR"
    --method bpf --proposal-type rf
    --rf-checkpoint "$CHECKPOINT"
    --rf-sampling-steps "$SAMPLING_STEPS"
    --rf-likelihood-steps "$LIKELIHOOD_STEPS"
    --sampling-grid-type "$SAMPLING_GRID"
    --likelihood-grid-type "$LIKELIHOOD_GRID"
    --likelihood-grid-param '{"gamma": 2.0}'
    --n-particles "$N_PARTICLES"
    --batch-size "$BATCH_SIZE"
    --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES"
    --resampling-threshold "$RESAMPLING_THRESHOLD"
    --process-noise-std "$PROCESS_NOISE_STD"
    --seed "$SEED"
    --device cuda
    --experiment-label "$RUN_NAME"
    --wandb-project "$WANDB_PROJECT"
    --wandb-tags "eval_rf_l96_neurips,d${STATE_DIM},${OBS_TOKEN},${OBS_OP_TAG},N${N_PARTICLES},bs${BATCH_SIZE}"
    --run-name "$RUN_NAME"
)

echo "--> running: ${eval_cmd[*]}"
t0=$(date +%s)
set +e
CUDA_VISIBLE_DEVICES=0 "${eval_cmd[@]}" > "$LOG_FILE" 2>&1
rc=$?
set -e
t1=$(date +%s)
dt=$(( t1 - t0 ))

mean_rmse="" std_rmse="" mean_crps="" std_crps="" mean_ess=""
if [ "$rc" -eq 0 ]; then
    mean_rmse="$(extract_metric     "$LOG_FILE" "Mean RMSE across trajectories")"
    std_rmse="$( extract_metric_std "$LOG_FILE" "Mean RMSE across trajectories")"
    mean_crps="$(extract_metric     "$LOG_FILE" "Mean CRPS across trajectories")"
    std_crps="$( extract_metric_std "$LOG_FILE" "Mean CRPS across trajectories")"
    mean_ess="$( extract_metric     "$LOG_FILE" "Mean ESS")"
    echo "    OK   rc=${rc} wall=${dt}s  rmse=${mean_rmse:-?}  crps=${mean_crps:-?}  ess=${mean_ess:-?}"
else
    echo "    FAIL rc=${rc} wall=${dt}s (see ${LOG_FILE})"
fi

(
    flock -x 200
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$TASK_ID" "$STATE_DIM" "$OBS_NOISE" "$OBS_OP_TAG" \
        "$N_PARTICLES" "$BATCH_SIZE" "$dt" \
        "${mean_rmse:-}" "${std_rmse:-}" "${mean_crps:-}" "${std_crps:-}" \
        "${mean_ess:-}" "$RUN_NAME" "$CHECKPOINT" "$DATA_DIR" "$LOG_FILE" \
        >> "${SUMMARY_CSV}"
) 200>"${CSV_LOCK}"

exit "$rc"
