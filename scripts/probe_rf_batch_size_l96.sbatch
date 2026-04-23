#!/bin/bash
#
# GPU batch-size probe for the NeurIPS-grid RF proposals before running the
# full eval sweep. One array task per L96 dimensionality. Each task descends
# through a list of candidate batch sizes and stops at the first one whose
# eval.py probe finishes without `CUDA out of memory`. The winning batch
# size (or "OOM" if every candidate failed) is appended to a shared CSV.
#
# === PF settings (must match the planned full-eval config) ===
#   method             = bpf
#   proposal_type      = rf
#   n_particles        = 1000  for dims {5, 10}
#                        5000  for dims {15, 20, 25, 50}
#   process_noise_std  = 0.2
#   sampling grid      = uniform,    32 steps
#   likelihood grid    = power_back, 32 steps (gamma=2.0)
#   resampling_thr.    = 0.33
#
# === Probe settings (shortened — peak memory is allocated on the first
# few PF steps, so we don't need full-length trajectories to trigger OOM) ===
#   num_eval_trajectories = <current batch_size>   (exactly one batch)
#   max_timesteps         = 20
#
# === Datasets ===
# No-noise version of the per-dim training datasets:
#   lorenz96_n2048_len200_dt0p0100_obs0p200_freq1_comp{D}of{D}_arctan_init3p000
#
# === Checkpoints ===
# Best (lowest val_loss) obs0p200 / arctan RF checkpoint per dim, from the
# neurips training grid.
#
# === Output ===
#   logs/probe_rf_bs_<array_job_id>/max_batch_size.csv
#     header: dim,n_particles,max_batch_size,attempts,winning_wall_sec,run_name
#   logs/probe_rf_bs_<array_job_id>/probe_d<D>_bs<BS>.log  per attempt
#
#SBATCH --job-name="probe-rf-bs"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%A_%a.log

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
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_DIR="logs/probe_rf_bs_${JOB_TAG}"
mkdir -p "${RUN_DIR}"
SUMMARY_CSV="${RUN_DIR}/max_batch_size.csv"
CSV_LOCK="${RUN_DIR}/max_batch_size.csv.lock"

# --- Per-dim configuration ---
DIMS=(5 10 15 20 25 50)
# Matches the "no noise" dataset naming; obs_token is fixed at obs0p200 and
# the observation operator is `arctan` (same cell the tuning pilot used).
DATASET_ROOT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims"
RUN_ROOTS=(
    "/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_l96_neurips/l96_d5_obs0p200_arctan_a0_job8153697"
    "/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_l96_neurips/l96_d10_obs0p200_arctan_a10_job8156580"
    "/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_l96_neurips/l96_d15_obs0p200_arctan_a20_job8157442"
    "/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_l96_neurips/l96_d20_obs0p200_arctan_a30_job8158757"
    "/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_l96_neurips/l96_d25_obs0p200_arctan_a40_job8163011"
    "/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_l96_neurips/l96_d50_obs0p200_arctan_a50_job8164358"
)
# Best val_loss ckpt per dim (from `rf-epoch=XXX-val_loss=YYY.ckpt`).
CKPT_NAMES=(
    "rf-epoch=137-val_loss=0.020530.ckpt"
    "rf-epoch=075-val_loss=0.019843.ckpt"
    "rf-epoch=096-val_loss=0.019723.ckpt"
    "rf-epoch=123-val_loss=0.019890.ckpt"
    "rf-epoch=108-val_loss=0.020386.ckpt"
    "rf-epoch=008-val_loss=0.020305.ckpt"
)

NUM_DIMS=${#DIMS[@]}
if (( TASK_ID < 0 || TASK_ID >= NUM_DIMS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} is out of range [0, ${NUM_DIMS})."
    exit 1
fi

DIM="${DIMS[$TASK_ID]}"
RUN_ROOT="${RUN_ROOTS[$TASK_ID]}"
CKPT_NAME="${CKPT_NAMES[$TASK_ID]}"
CHECKPOINT="${RUN_ROOT}/checkpoints/${CKPT_NAME}"
DATA_DIR="${DATASET_ROOT}/lorenz96_n2048_len200_dt0p0100_obs0p200_freq1_comp${DIM}of${DIM}_arctan_init3p000"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: CHECKPOINT not found: $CHECKPOINT"; exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR not found: $DATA_DIR"; exit 1
fi

# n_particles policy from the user's full-eval spec.
if (( DIM <= 10 )); then
    N_PARTICLES=1000
else
    N_PARTICLES=5000
fi

# Candidate batch sizes, descending. First non-OOM wins.
BATCH_SIZES=(100 50 25 20 10 5)

# --- Fixed PF / proposal settings ---
PROCESS_NOISE_STD=0.2
SAMPLING_GRID="uniform"
LIKELIHOOD_GRID="power_back"
SAMPLING_STEPS=32
LIKELIHOOD_STEPS=32
RESAMPLING_THRESHOLD=0.33
PROBE_MAX_TIMESTEPS=20
SEED=42

# --- Ensure shared summary CSV has a header (write-once under flock) ---
(
    flock -x 200
    if [ ! -s "${SUMMARY_CSV}" ]; then
        echo "dim,n_particles,max_batch_size,attempts,winning_wall_sec,run_name" > "${SUMMARY_CSV}"
    fi
) 200>"${CSV_LOCK}"

echo "=========================================="
echo "probe_rf_batch_size_l96.sbatch (task ${TASK_ID}, dim ${DIM})"
echo "  n_particles      = ${N_PARTICLES}"
echo "  checkpoint       = ${CHECKPOINT}"
echo "  data_dir         = ${DATA_DIR}"
echo "  process_noise    = ${PROCESS_NOISE_STD}"
echo "  sampling grid    = ${SAMPLING_GRID} / ${SAMPLING_STEPS}"
echo "  likelihood grid  = ${LIKELIHOOD_GRID} / ${LIKELIHOOD_STEPS} (gamma=2.0)"
echo "  probe timesteps  = ${PROBE_MAX_TIMESTEPS}"
echo "  batch sizes      = ${BATCH_SIZES[*]}"
echo "  summary_csv      = ${SUMMARY_CSV}"
echo "=========================================="

# OOM detection: treat a run as "OOM" if it exited nonzero AND the log
# mentions one of the known CUDA OOM signatures. Non-OOM failures are
# surfaced as errors (we don't want to silently count a model bug as OOM).
is_oom_log() {
    local log_file="$1"
    grep -qiE "CUDA out of memory|CUDAOutOfMemoryError|torch\.OutOfMemoryError|OutOfMemoryError|cuBLAS.*CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation" "$log_file"
}

WINNING_BS=""
WINNING_WALL=""
WINNING_RUN_NAME=""
ATTEMPTS=0

for BS in "${BATCH_SIZES[@]}"; do
    ATTEMPTS=$((ATTEMPTS + 1))
    RUN_NAME="probe_d${DIM}_N${N_PARTICLES}_bs${BS}_${JOB_TAG}"
    LOG_FILE="${RUN_DIR}/${RUN_NAME}.log"

    echo "--> [dim=${DIM}] attempting batch_size=${BS} (N=${N_PARTICLES})  -> ${LOG_FILE}"

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
        --batch-size "$BS"
        --num-eval-trajectories "$BS"
        --max-timesteps "$PROBE_MAX_TIMESTEPS"
        --resampling-threshold "$RESAMPLING_THRESHOLD"
        --process-noise-std "$PROCESS_NOISE_STD"
        --seed "$SEED"
        --device cuda
        --disable-wandb
        --experiment-label "$RUN_NAME"
        --run-name "$RUN_NAME"
    )

    t0=$(date +%s)
    set +e
    CUDA_VISIBLE_DEVICES=0 "${eval_cmd[@]}" > "$LOG_FILE" 2>&1
    rc=$?
    set -e
    t1=$(date +%s)
    dt=$(( t1 - t0 ))

    if [ "$rc" -eq 0 ]; then
        echo "    OK   rc=${rc} wall=${dt}s  bs=${BS} fits."
        WINNING_BS="$BS"
        WINNING_WALL="$dt"
        WINNING_RUN_NAME="$RUN_NAME"
        break
    fi

    if is_oom_log "$LOG_FILE"; then
        echo "    OOM  rc=${rc} wall=${dt}s  bs=${BS} (see ${LOG_FILE}); trying next smaller."
        # Flush CUDA context before the next attempt. Python exit frees it,
        # but we sleep briefly to let the driver settle.
        sleep 2
        continue
    fi

    # Some other failure — don't silently pretend this is an OOM.
    echo "    FAIL rc=${rc} wall=${dt}s  bs=${BS} (NOT an OOM; see ${LOG_FILE})"
    echo "    Aborting probe for dim=${DIM}. Fix the underlying error and re-run."
    (
        flock -x 200
        printf "%s,%s,%s,%s,%s,%s\n" \
            "$DIM" "$N_PARTICLES" "ERROR_bs${BS}" "$ATTEMPTS" "$dt" "$RUN_NAME" \
            >> "${SUMMARY_CSV}"
    ) 200>"${CSV_LOCK}"
    exit "$rc"
done

(
    flock -x 200
    if [ -n "$WINNING_BS" ]; then
        printf "%s,%s,%s,%s,%s,%s\n" \
            "$DIM" "$N_PARTICLES" "$WINNING_BS" "$ATTEMPTS" \
            "$WINNING_WALL" "$WINNING_RUN_NAME" \
            >> "${SUMMARY_CSV}"
    else
        printf "%s,%s,%s,%s,%s,%s\n" \
            "$DIM" "$N_PARTICLES" "OOM" "$ATTEMPTS" "" "" \
            >> "${SUMMARY_CSV}"
    fi
) 200>"${CSV_LOCK}"

if [ -z "$WINNING_BS" ]; then
    echo "[dim=${DIM}] ALL batch sizes OOM'd. Full-eval will need bs<5 or fewer particles."
    exit 2
fi

echo "[dim=${DIM}] max batch size = ${WINNING_BS} (attempts=${ATTEMPTS}, wall=${WINNING_WALL}s)"
exit 0
