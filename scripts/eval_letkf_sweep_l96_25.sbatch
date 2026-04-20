#!/bin/bash
#
# Hyperparameter sweep for LETKF on L96-25.
#
# Sweeps the standard LETKF tuning knobs (multiplicative inflation x
# localization radius, optionally also ensemble size) on the L96-25 obs0p100
# arctan dataset and logs each configuration as its own wandb run. Modeled on
# scripts/eval_lfppf_integration_l96_25.sbatch (sequential dispatch with a
# status summary so a single bad config doesn't abort the queue).
#
# LETKF only needs --inflation, --localization-radius, --n-particles, and
# --process-noise-std (see eval.py L820-842 and models/enkf.py
# LocalEnsembleTransformKalmanFilter). There is no learned proposal, so no
# checkpoints are required.
#
# Sweep grid is the cartesian product
#       INFLATION_VALUES x LOCALIZATION_VALUES x N_PARTICLES_VALUES
# Defaults match the grids used in scripts/tune_enkf.py / run_enkf_tune_96.py
# so results line up with the existing EnKF/LETKF tuning history. Override any
# of the three arrays via env vars (comma-separated), e.g.:
#
#   INFLATION_VALUES="1.0,1.1,1.2" \
#   LOCALIZATION_VALUES="3,4,5" \
#   N_PARTICLES_VALUES="50" \
#   sbatch scripts/eval_letkf_sweep_l96_25.sbatch
#
# Usage:
#   sbatch scripts/eval_letkf_sweep_l96_25.sbatch
#
#SBATCH --job-name="letkf-sweep-l96-25"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/eval.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1

module load anaconda3/2024.10
source activate
conda activate da
export PYTHONPATH="$(pwd)"
mkdir -p logs

DATA_DIR="${DATA_DIR:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_init3p000}"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR does not exist: $DATA_DIR"; exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-letkf-sweep-l96-25}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

# --- Sweep grids ---
# Comma-separated; cartesian product is taken below.
INFLATION_VALUES="${INFLATION_VALUES:-1.0,1.05,1.1,1.15,1.2,1.3,1.4,1.5}"
LOCALIZATION_VALUES="${LOCALIZATION_VALUES:-2,3,4,5}"
N_PARTICLES_VALUES="${N_PARTICLES_VALUES:-50}"

# --- Fixed eval hyperparameters ---
# Match the L-FPPF integration script so RMSE/CRPS numbers are comparable to
# the particle-filter arms run on the same dataset.
BATCH_SIZE="${BATCH_SIZE:-10}"
NUM_EVAL_TRAJECTORIES="${NUM_EVAL_TRAJECTORIES:-10}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-100}"
PROCESS_NOISE_STD="${PROCESS_NOISE_STD:-0.2}"
SEED="${SEED:-42}"
INIT_MODE="${INIT_MODE:-truth}"  # "truth" or "climatology"

EXTRA_TAGS="${EXTRA_TAGS:-}"  # optional comma-separated extra wandb tags

# Max concurrent eval.py processes on the single allocated GPU. LETKF with
# n_particles=50 and batch_size=10 is light enough that ~4 fit comfortably;
# bump down if you see OOMs or contention.
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# --- Build sweep list ---
IFS=',' read -ra INFL_ARR <<< "$INFLATION_VALUES"
IFS=',' read -ra LOC_ARR  <<< "$LOCALIZATION_VALUES"
IFS=',' read -ra NP_ARR   <<< "$N_PARTICLES_VALUES"

total=$(( ${#INFL_ARR[@]} * ${#LOC_ARR[@]} * ${#NP_ARR[@]} ))
echo "LETKF sweep: ${#INFL_ARR[@]} infl x ${#LOC_ARR[@]} loc x ${#NP_ARR[@]} npart = ${total} configs"
echo "  data_dir            = $DATA_DIR"
echo "  inflation_values    = $INFLATION_VALUES"
echo "  localization_values = $LOCALIZATION_VALUES"
echo "  n_particles_values  = $N_PARTICLES_VALUES"
echo "  process_noise_std   = $PROCESS_NOISE_STD"
echo "  batch_size          = $BATCH_SIZE"
echo "  num_eval_trajs      = $NUM_EVAL_TRAJECTORIES"
echo "  max_timesteps       = $MAX_TIMESTEPS"
echo "  init_mode           = $INIT_MODE"
echo "  max_parallel        = $MAX_PARALLEL"
echo "  wandb_project       = $WANDB_PROJECT"

run_labels=()
run_status=()

# Slot bookkeeping for bounded-parallel dispatch (FIFO).
slot_pids=()
slot_labels=()

# Sanitize a float for use in a label/run name (replace "." with "p").
fmt() { printf "%s" "$1" | tr '.' 'p'; }

# Block until the *oldest* in-flight job finishes; record OK/FAIL and free
# its slot. FIFO is fine here because all configs have similar runtimes;
# we lose only a tiny bit of slot occupancy vs. a true wait-any scheme but
# avoid bash version / wait -n portability concerns.
reap_oldest() {
    [ "${#slot_pids[@]}" -eq 0 ] && return 0
    local pid="${slot_pids[0]}"
    local label="${slot_labels[0]}"
    local rc=0
    set +e
    wait "$pid"
    rc=$?
    set -e
    run_labels+=("$label")
    if [ "$rc" -eq 0 ]; then
        echo "  OK   ${label}"
        run_status+=("OK")
    else
        echo "  FAIL ${label} (rc=${rc})"
        run_status+=("FAIL")
    fi
    if [ "${#slot_pids[@]}" -gt 1 ]; then
        slot_pids=("${slot_pids[@]:1}")
        slot_labels=("${slot_labels[@]:1}")
    else
        slot_pids=()
        slot_labels=()
    fi
}

launch_one() {
    local infl="$1"
    local loc="$2"
    local np="$3"
    local idx="$4"

    local infl_tag loc_tag pn_tag
    infl_tag=$(fmt "$infl")
    loc_tag=$(fmt "$loc")
    pn_tag=$(fmt "$PROCESS_NOISE_STD")

    local label="letkf_L96_25_N${np}_infl${infl_tag}_loc${loc_tag}_pn${pn_tag}_${INIT_MODE}_${JOB_TAG}"
    local log="logs/${label}.log"

    local tags="sweep,letkf,L96_25,init=${INIT_MODE}"
    if [ -n "$EXTRA_TAGS" ]; then tags="${tags},${EXTRA_TAGS}"; fi

    # Background subshell. Output goes to per-config log only (interleaved
    # tee from N parallel jobs would be unreadable on the SLURM stdout).
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method letkf \
            --n-particles "$np" \
            --inflation "$infl" \
            --localization-radius "$loc" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --batch-size "$BATCH_SIZE" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --max-timesteps "$MAX_TIMESTEPS" \
            --init-mode "$INIT_MODE" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-tags "$tags" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda \
            > "$log" 2>&1
    ) &
    local pid=$!
    slot_pids+=("$pid")
    slot_labels+=("$label")
    echo "[${idx}/${total}] launched N=${np} infl=${infl} loc=${loc}  pid=${pid}  log=${log}"
}

# --- Dispatch ---
# Bounded-parallel: keep up to MAX_PARALLEL eval.py processes on the GPU
# at any time. A single bad config (e.g. NaNs / OOM) does not abort the
# rest of the sweep; we collect statuses and exit non-zero at the end if
# any failed.
i=0
for np in "${NP_ARR[@]}"; do
    for loc in "${LOC_ARR[@]}"; do
        for infl in "${INFL_ARR[@]}"; do
            i=$((i+1))
            while [ "${#slot_pids[@]}" -ge "$MAX_PARALLEL" ]; do
                reap_oldest
            done
            launch_one "$infl" "$loc" "$np" "$i"
        done
    done
done

# Drain remaining in-flight jobs.
while [ "${#slot_pids[@]}" -gt 0 ]; do
    reap_oldest
done

echo "Summary of ${#run_labels[@]} runs:"
failures=0
for idx in "${!run_labels[@]}"; do
    label="${run_labels[$idx]}"
    status="${run_status[$idx]}"
    echo "  ${status}   ${label}"
    if [ "$status" = "FAIL" ]; then failures=$((failures+1)); fi
done
if [ "$failures" -gt 0 ]; then echo "Finished with ${failures} failure(s)."; exit 1; fi
echo "LETKF sweep completed."
