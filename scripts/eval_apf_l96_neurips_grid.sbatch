#!/bin/bash
#
# 60-way APF evaluation over the NeurIPS L96 grid:
# (dims {5,10,15,20,25,50} x obs_noise {0.2,0.5,1,3,5} x ops {arctan, quad_capped_10})
#
# Each array task evaluates ONE cell on the *no-noise* dataset variant.
#
# === APF config (fixed unless noted) ===
#   method              = apf
#   proposal_type       = transition (q = f)
#   apf_adjustment      = point
#   n_particles         = 1000  for dims {5,10}
#                         5000  for dims {15,20,25,50}
#   batch_size          = 100 for dims {5,10}
#                         50 for dim {15}
#                         25 for dims {20,25}
#                         10 for dim {50}
#   process_noise_std   = 0.2
#   resampling_threshold = 0.33
#   num_eval_trajectories = 100
#   max_timesteps       = unset (full 200)
#   seed                = 42
#
#SBATCH --job-name="eval-apf-l96-neurips"
#SBATCH --partition=scavenger
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-59%8
#SBATCH --requeue
#SBATCH --exclude=ccc0236
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
RUN_DIR="logs/eval_apf_l96_neurips_${JOB_TAG}"
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

# Per-dimension policy (kept aligned with FPPF and EnKF grid scripts).
case "$STATE_DIM" in
    5|10) N_PARTICLES=1000; BATCH_SIZE=100 ;;
    15)   N_PARTICLES=5000; BATCH_SIZE=50  ;;
    20)   N_PARTICLES=5000; BATCH_SIZE=25  ;;
    25)   N_PARTICLES=5000; BATCH_SIZE=25  ;;
    50)   N_PARTICLES=5000; BATCH_SIZE=10  ;;
    *)    echo "ERROR: no per-dim N/batch config for dim=${STATE_DIM}" >&2; exit 1 ;;
esac

DATA_ROOT="${DATA_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims}"
DATASET_NAME="lorenz96_n2048_len200_dt0p0100_${OBS_TOKEN}_freq1_comp${STATE_DIM}of${STATE_DIM}_${OBS_OP_DIR}_init3p000"
DATA_DIR="${DATA_ROOT}/${DATASET_NAME}"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: eval DATA_DIR not found: ${DATA_DIR}"
    echo "  (expected the *no-noise* dataset, i.e. without pnoise0p100)"
    exit 1
fi

APF_ADJUSTMENT="point"
PROCESS_NOISE_STD=0.2
RESAMPLING_THRESHOLD=0.33
NUM_EVAL_TRAJECTORIES=100
SEED=42

WANDB_PROJECT="${WANDB_PROJECT:-eval-apf-l96-neurips}"

RUN_NAME="eval_apf_d${STATE_DIM}_${OBS_TOKEN}_${OBS_OP_TAG}_${JOB_TAG}_t${TASK_ID}"
LOG_FILE="${RUN_DIR}/${RUN_NAME}.log"

echo "=========================================="
echo "eval_apf_l96_neurips_grid.sbatch (task ${TASK_ID}/${TOTAL})"
echo "  state_dim          = ${STATE_DIM}"
echo "  obs_noise          = ${OBS_NOISE} (${OBS_TOKEN})"
echo "  obs_operator       = ${OBS_OP_DIR} (tag=${OBS_OP_TAG})"
echo "  apf_adjustment     = ${APF_ADJUSTMENT}"
echo "  proposal_type      = transition"
echo "  n_particles        = ${N_PARTICLES}"
echo "  batch_size         = ${BATCH_SIZE}"
echo "  process_noise_std  = ${PROCESS_NOISE_STD}"
echo "  resamp_threshold   = ${RESAMPLING_THRESHOLD}"
echo "  data_dir           = ${DATA_DIR}"
echo "  num_eval_trajs     = ${NUM_EVAL_TRAJECTORIES}"
echo "  max_timesteps      = unset (full 200)"
echo "  wandb_project      = ${WANDB_PROJECT}"
echo "  log_file           = ${LOG_FILE}"
echo "=========================================="

(
    flock -x 200
    if [ ! -s "${SUMMARY_CSV}" ]; then
        echo "task_id,dim,obs_noise,obs_op,apf_adjustment,n_particles,batch_size,process_noise_std,resampling_threshold,wall_time_sec,mean_rmse,std_rmse,mean_crps,std_crps,mean_ess,run_name,data_dir,log_file" > "${SUMMARY_CSV}"
    fi
) 200>"${CSV_LOCK}"

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
    --method apf
    --proposal-type transition
    --apf-adjustment "$APF_ADJUSTMENT"
    --n-particles "$N_PARTICLES"
    --batch-size "$BATCH_SIZE"
    --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES"
    --resampling-threshold "$RESAMPLING_THRESHOLD"
    --process-noise-std "$PROCESS_NOISE_STD"
    --seed "$SEED"
    --device cuda
    --experiment-label "$RUN_NAME"
    --wandb-project "$WANDB_PROJECT"
    --wandb-tags "eval_apf_l96_neurips,d${STATE_DIM},${OBS_TOKEN},${OBS_OP_TAG},N${N_PARTICLES},bs${BATCH_SIZE},adj${APF_ADJUSTMENT}"
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
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$TASK_ID" "$STATE_DIM" "$OBS_NOISE" "$OBS_OP_TAG" "$APF_ADJUSTMENT" \
        "$N_PARTICLES" "$BATCH_SIZE" "$PROCESS_NOISE_STD" "$RESAMPLING_THRESHOLD" "$dt" \
        "${mean_rmse:-}" "${std_rmse:-}" "${mean_crps:-}" "${std_crps:-}" \
        "${mean_ess:-}" "$RUN_NAME" "$DATA_DIR" "$LOG_FILE" \
        >> "${SUMMARY_CSV}"
) 200>"${CSV_LOCK}"

exit "$rc"
