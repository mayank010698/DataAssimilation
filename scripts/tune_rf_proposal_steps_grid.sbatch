#!/bin/bash
#
# Hyperparameter tuning pilot: sweep RF proposal integration grid type AND
# the number of integration steps used for (i) sampling and (ii) likelihood
# evaluation, inside a BPF driven by a trained global RectifiedFlowProposal.
#
# Goal: figure out how aggressively we can shrink num_sampling_steps and
# num_likelihood_steps (default has been 100 for both) before filter quality
# (RMSE / CRPS) degrades. Also check if a non-uniform integration grid
# (cosine or power_back) buys us accuracy at low step counts.
#
# The "pilot" variant: one trained checkpoint, few trajectories, small N.
# If results look stable across N_particles, we can reuse the winning
# (sampling_grid, likelihood_grid, sampling_steps, likelihood_steps) tuple
# for the heavier test-time eval runs without re-tuning each cell of the
# 60-way grid.
#
# === Execution model ===
# Each SLURM array task runs EXACTLY ONE (sampling_grid, likelihood_grid,
# sampling_steps, likelihood_steps) configuration. SLURM_ARRAY_TASK_ID is
# mapped to a tuple via a lexicographic index over
#   (pair_idx, sampling_steps_idx, likelihood_steps_idx).
# All tasks share one output directory (keyed by SLURM_ARRAY_JOB_ID) and
# append one CSV row via flock.
#
# Default sweep = 5 pairs * 4 * 4 = 80 tasks  (array=0-79).
# If you change GRID_PAIRS or STEP_GRID, either resize --array on the
# `sbatch` command line or override ARRAY_SIZE env var and re-submit with
# a matching --array=0-<ARRAY_SIZE-1>.
#
# Pairs (default):
#   uniform:uniform          (baseline)
#   cosine:cosine            (non-uniform applied to both)
#   power_back:power_back    (non-uniform applied to both)
#   uniform:cosine           (non-uniform applied to likelihood only)
#   uniform:power_back       (non-uniform applied to likelihood only)
# Step grid (default): {32, 64, 80, 100}^2
#
# Fixed for this pilot:
#   method=bpf, proposal-type=rf, N=500, 10 trajectories, 100 timesteps,
#   process_noise_std=0.1 (matches training dataset pnoise0p100),
#   resampling_threshold=0.33, seed=42.
#
# Output (shared across array tasks):
#   logs/tune_rf_steps_<array_job_id>/<run_name>.log   per-task stdout
#   logs/tune_rf_steps_<array_job_id>/summary.csv      one row per config
#   W&B project: tune-rf-proposal-steps-grid (overridable)
#
# Pilot target (override via env vars):
#   CHECKPOINT = rf-epoch=137-val_loss=0.020530.ckpt  (best val on d5/obs0.2/arctan)
#   DATA_DIR   = lorenz96 ... comp5of5_arctan_pnoise0p100_init3p000
#
#SBATCH --job-name="tune-rf-steps"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=01:00:00
#SBATCH --array=0-79
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/tune_rf_steps_%A_%a.log

set -euo pipefail

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
# Share one output dir across all array tasks.
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_DIR="logs/tune_rf_steps_${JOB_TAG}"
mkdir -p "${RUN_DIR}"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
CSV_LOCK="${RUN_DIR}/summary.csv.lock"

# --- Pilot target (override via env) ---
RUN_ROOT_DEFAULT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_l96_neurips/l96_d5_obs0p200_arctan_a0_job8153697"
CHECKPOINT_DEFAULT="${RUN_ROOT_DEFAULT}/checkpoints/rf-epoch=137-val_loss=0.020530.ckpt"
DATA_DIR_DEFAULT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims/lorenz96_n2048_len200_dt0p0100_obs0p200_freq1_comp5of5_arctan_pnoise0p100_init3p000"

CHECKPOINT="${CHECKPOINT:-$CHECKPOINT_DEFAULT}"
DATA_DIR="${DATA_DIR:-$DATA_DIR_DEFAULT}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT not found: $CHECKPOINT"; exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1
fi

# --- PF eval configuration (fixed across the sweep) ---
N_PARTICLES="${N_PARTICLES:-500}"
NUM_EVAL_TRAJECTORIES="${NUM_EVAL_TRAJECTORIES:-10}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-10}"
RESAMPLING_THRESHOLD="${RESAMPLING_THRESHOLD:-0.33}"
PROCESS_NOISE_STD="${PROCESS_NOISE_STD:-0.1}"
SEED="${SEED:-42}"

# --- Sweep grid (override via env) ---
# GRID_PAIRS: comma-separated list of "<sampling_grid>:<likelihood_grid>"
#             tuples. Each pair is swept over STEP_GRID x STEP_GRID.
# STEP_GRID : comma-separated list of integer step counts used for BOTH
#             sampling_steps and likelihood_steps (full cross product).
GRID_PAIRS_CSV="${GRID_PAIRS:-uniform:uniform,cosine:cosine,power_back:power_back,uniform:cosine,uniform:power_back}"
STEP_GRID_CSV="${STEP_GRID:-32,64,80,100}"

IFS=',' read -r -a GRID_PAIRS_ARR <<< "${GRID_PAIRS_CSV}"
IFS=',' read -r -a STEP_GRID_ARR  <<< "${STEP_GRID_CSV}"

N_PAIRS=${#GRID_PAIRS_ARR[@]}
N_STEPS=${#STEP_GRID_ARR[@]}
TOTAL_RUNS=$(( N_PAIRS * N_STEPS * N_STEPS ))

if (( TASK_ID < 0 || TASK_ID >= TOTAL_RUNS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} is out of range [0, ${TOTAL_RUNS})."
    echo "  Adjust #SBATCH --array (or pass --array=0-$((TOTAL_RUNS - 1))) to match"
    echo "  your GRID_PAIRS (${N_PAIRS}) * STEP_GRID^2 (${N_STEPS}^2) = ${TOTAL_RUNS} runs."
    exit 1
fi

# Lexicographic decoding: pair_idx outer, sampling_steps middle, likelihood_steps inner.
LIK_IDX=$(( TASK_ID % N_STEPS ))
SAMP_IDX=$(( (TASK_ID / N_STEPS) % N_STEPS ))
PAIR_IDX=$(( TASK_ID / (N_STEPS * N_STEPS) ))

PAIR="${GRID_PAIRS_ARR[$PAIR_IDX]}"
SAMP_GRID="${PAIR%%:*}"
LIK_GRID="${PAIR##*:}"
S_SAMP="${STEP_GRID_ARR[$SAMP_IDX]}"
S_LIK="${STEP_GRID_ARR[$LIK_IDX]}"

if [ -z "$SAMP_GRID" ] || [ -z "$LIK_GRID" ] || [ "$SAMP_GRID" = "$PAIR" ]; then
    echo "ERROR: malformed GRID_PAIRS entry '${PAIR}' (expected '<samp>:<lik>')"; exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-tune-rf-proposal-steps-grid}"
DATASET_BASENAME="$(basename "$DATA_DIR")"
RUN_ROOT_TAG="$(basename "$(dirname "$(dirname "$CHECKPOINT")")")"

# W&B caps individual tags at 64 chars, so we can't use the full dataset
# dirname (often ~80 chars). Build a short, human-readable tag from the
# salient tokens (dim, obs noise, observation operator). Fall back to a
# 60-char truncation if none of the patterns match.
_dim_tok="$( echo "$DATASET_BASENAME" | grep -oE 'comp[0-9]+of[0-9]+' | head -1 || true)"
_obs_tok="$( echo "$DATASET_BASENAME" | grep -oE 'obs[0-9]+p[0-9]+'   | head -1 || true)"
_op_tok="$(  echo "$DATASET_BASENAME" | grep -oE '(arctan|quad_capped_10|identity|linear)' | head -1 || true)"
if [ -n "${_dim_tok}${_obs_tok}${_op_tok}" ]; then
    DATASET_TAG="$(printf '%s_%s_%s' "${_dim_tok}" "${_obs_tok}" "${_op_tok}" | sed -E 's/^_+//; s/_+$//; s/__+/_/g')"
else
    DATASET_TAG="${DATASET_BASENAME:0:60}"
fi

RUN_NAME="${DATASET_TAG}_sg${SAMP_GRID}_lg${LIK_GRID}_ss${S_SAMP}_ls${S_LIK}_${JOB_TAG}_t${TASK_ID}"
LOG_FILE="${RUN_DIR}/${RUN_NAME}.log"

echo "=========================================="
echo "tune_rf_proposal_steps_grid.sbatch (array task ${TASK_ID}/${TOTAL_RUNS})"
echo "  pair_idx    = ${PAIR_IDX}   (${SAMP_GRID}:${LIK_GRID})"
echo "  samp_idx    = ${SAMP_IDX}   sampling_steps   = ${S_SAMP}"
echo "  lik_idx     = ${LIK_IDX}    likelihood_steps = ${S_LIK}"
echo "  checkpoint  = ${CHECKPOINT}"
echo "  data_dir    = ${DATA_DIR}"
echo "  W&B dataset = ${DATASET_TAG}"
echo "  W&B runroot = ${RUN_ROOT_TAG}"
echo "  log_file    = ${LOG_FILE}"
echo "=========================================="

# --- Ensure the summary CSV has a header (write-once under flock) ---
(
    flock -x 200
    if [ ! -s "${SUMMARY_CSV}" ]; then
        echo "task_id,sampling_grid,likelihood_grid,sampling_steps,likelihood_steps,wall_time_sec,mean_rmse,std_rmse,mean_crps,std_crps,mean_ess,mean_step_time,n_trajectories,run_name,log_file" > "${SUMMARY_CSV}"
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
    --rf-sampling-steps "$S_SAMP"
    --rf-likelihood-steps "$S_LIK"
    --sampling-grid-type "$SAMP_GRID"
    --likelihood-grid-type "$LIK_GRID"
    --n-particles "$N_PARTICLES"
    --batch-size "$BATCH_SIZE"
    --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES"
    --max-timesteps "$MAX_TIMESTEPS"
    --resampling-threshold "$RESAMPLING_THRESHOLD"
    --process-noise-std "$PROCESS_NOISE_STD"
    --seed "$SEED"
    --device cuda
    --experiment-label "$RUN_NAME"
    --wandb-project "$WANDB_PROJECT"
    --wandb-tags "tune_rf_steps,${RUN_ROOT_TAG},${DATASET_TAG},sg_${SAMP_GRID},lg_${LIK_GRID},ss_${S_SAMP},ls_${S_LIK}"
    --run-name "$RUN_NAME"
)

# Non-uniform grid params (match scripts/eval_lg_grid.sbatch conventions).
if [ "$SAMP_GRID" = "power_front" ] || [ "$SAMP_GRID" = "power_back" ]; then
    eval_cmd+=( --sampling-grid-param '{"gamma": 2.0}' )
fi
if [ "$LIK_GRID" = "power_front" ] || [ "$LIK_GRID" = "power_back" ]; then
    eval_cmd+=( --likelihood-grid-param '{"gamma": 2.0}' )
fi

echo "--> sg=${SAMP_GRID}(${S_SAMP})  lg=${LIK_GRID}(${S_LIK})  -> ${LOG_FILE}"
t0=$(date +%s)
set +e
CUDA_VISIBLE_DEVICES=0 "${eval_cmd[@]}" > "$LOG_FILE" 2>&1
rc=$?
set -e
t1=$(date +%s)
dt=$(( t1 - t0 ))

mean_rmse="" std_rmse="" mean_crps="" std_crps="" mean_ess="" mean_step_time=""
if [ "$rc" -eq 0 ]; then
    mean_rmse="$(extract_metric     "$LOG_FILE" "Mean RMSE across trajectories")"
    std_rmse="$( extract_metric_std "$LOG_FILE" "Mean RMSE across trajectories")"
    mean_crps="$(extract_metric     "$LOG_FILE" "Mean CRPS across trajectories")"
    std_crps="$( extract_metric_std "$LOG_FILE" "Mean CRPS across trajectories")"
    mean_ess="$( extract_metric     "$LOG_FILE" "Mean ESS")"
    # eval.py doesn't print mean_step_time to stdout (it only logs it to wandb),
    # so this column is left blank. Add a print to eval.py if you want it here.
    echo "    OK  rc=${rc} wall=${dt}s  rmse=${mean_rmse:-?}  crps=${mean_crps:-?}  ess=${mean_ess:-?}"
else
    echo "    FAIL rc=${rc} wall=${dt}s (see ${LOG_FILE})"
fi

(
    flock -x 200
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$TASK_ID" "$SAMP_GRID" "$LIK_GRID" "$S_SAMP" "$S_LIK" "$dt" \
        "${mean_rmse:-}" "${std_rmse:-}" "${mean_crps:-}" "${std_crps:-}" \
        "${mean_ess:-}" "${mean_step_time:-}" "$NUM_EVAL_TRAJECTORIES" \
        "$RUN_NAME" "$LOG_FILE" \
        >> "${SUMMARY_CSV}"
) 200>"${CSV_LOCK}"

exit "$rc"
