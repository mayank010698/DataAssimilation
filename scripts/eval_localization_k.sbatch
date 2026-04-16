#!/bin/bash
#
# BPF vs L-BPF particle-count sweep on Lorenz96 (dim=25, fully obs).
#
# Varies N across {1000, 2000, 3000, 4000, 5000} for both methods.
# Set RUN_BPF / RUN_LBPF to 0 to disable either method.
#
# Usage:  cd /path/to/DataAssimilation && sbatch scripts/eval_localization_k.sbatch
#
#SBATCH --job-name="eval-loc-k"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

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

export PYTHONPATH="$(pwd)"

mkdir -p logs

# ======================================================================
# Job enable/disable flags  —  set to 0 to skip
# ======================================================================

RUN_BPF=1      # Standard bootstrap PF (transition proposal)
RUN_LBPF=1     # Localized bootstrap PF (transition proposal)

# ======================================================================
# Configuration — shared
# ======================================================================

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_init3p000"

WANDB_PROJECT="${WANDB_PROJECT:-eval-lpf-l96}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

BATCH_SIZE=10
NUM_EVAL_TRAJECTORIES=10
RESAMPLING_THRESHOLD=0.33
PROCESS_NOISE_STD=0.1
SEED=42

PARTICLE_COUNTS=(100 500)

# --- Localized PF hyperparameters -------------------------------------
LPF_BLOCK_SIZE=1
LPF_LOC_RADIUS=3.0
LPF_TAPER="gaussian"
LPF_POST_REG="none"
LPF_WEIGHT_SMOOTHING=""   # empty = off (flag not passed)

# ======================================================================
# Sanity checks
# ======================================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data dir not found: $DATA_DIR"
    exit 1
fi

TOTAL_JOBS=0
(( RUN_BPF  )) && TOTAL_JOBS=$((TOTAL_JOBS + ${#PARTICLE_COUNTS[@]}))
(( RUN_LBPF )) && TOTAL_JOBS=$((TOTAL_JOBS + ${#PARTICLE_COUNTS[@]}))

echo "=========================================="
echo "BPF vs L-BPF particle-count sweep — Lorenz96 d=25"
echo "Project root : $PROJECT_ROOT"
echo "Data         : $DATA_DIR"
echo "Slurm Job ID : ${SLURM_JOB_ID:-N/A}"
echo ""
echo "Particle counts : ${PARTICLE_COUNTS[*]}"
echo "Batch size      : $BATCH_SIZE"
echo "Eval trajs      : $NUM_EVAL_TRAJECTORIES"
echo "Resamp. thresh  : $RESAMPLING_THRESHOLD"
echo "Process noise   : $PROCESS_NOISE_STD"
echo ""
echo "LPF block_size           : $LPF_BLOCK_SIZE"
echo "LPF localization_radius  : $LPF_LOC_RADIUS"
echo "LPF taper                : $LPF_TAPER"
echo "LPF post_regularization  : $LPF_POST_REG"
echo "LPF weight_smoothing     : ${LPF_WEIGHT_SMOOTHING:-off}"
echo ""
echo "Enabled jobs : BPF=$RUN_BPF  L-BPF=$RUN_LBPF"
echo "Launching $TOTAL_JOBS job(s) in parallel …"
echo "=========================================="

# ======================================================================
# Launch jobs
# ======================================================================

pids=()
run_names=()

lpf_extra_flags=()
if [ -n "$LPF_WEIGHT_SMOOTHING" ]; then
    lpf_extra_flags+=(--weight-smoothing)
fi

# ------------------------------------------------------------------
# BPF (standard bootstrap PF)
# ------------------------------------------------------------------
if (( RUN_BPF )); then
for N in "${PARTICLE_COUNTS[@]}"; do
    label="bpf_d25_${N}_${JOB_TAG}_0.1"
    log="logs/eval_bpf_d25_${N}_${JOB_TAG}.log"

    echo "  -> BPF  N=${N}  (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf \
            --proposal-type transition \
            --n-particles "$N" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "$log" &
    pids+=("$!")
    run_names+=("bpf_N${N}")
done
fi

# ------------------------------------------------------------------
# L-BPF (localized bootstrap PF)
# ------------------------------------------------------------------
if (( RUN_LBPF )); then
for N in "${PARTICLE_COUNTS[@]}"; do
    label="lbpf_d25_${N}_bs${LPF_BLOCK_SIZE}_r${LPF_LOC_RADIUS}_${JOB_TAG}"
    log="logs/eval_lbpf_d25_${N}_${JOB_TAG}.log"

    echo "  -> L-BPF  N=${N}  (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method lpf \
            --proposal-type transition \
            --n-particles "$N" \
            --batch-size "$BATCH_SIZE" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --block-size "$LPF_BLOCK_SIZE" \
            --localization-radius "$LPF_LOC_RADIUS" \
            --lpf-taper "$LPF_TAPER" \
            --post-regularization "$LPF_POST_REG" \
            "${lpf_extra_flags[@]}" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "$log" &
    pids+=("$!")
    run_names+=("lbpf_N${N}")
done
fi

# ======================================================================
# Wait for all jobs
# ======================================================================

echo ""
echo "All ${#pids[@]} job(s) launched. Waiting for completion …"
echo ""

failures=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    run="${run_names[$idx]}"
    if wait "$pid"; then
        echo "  OK   ${run}  (pid=${pid})"
    else
        echo "  FAIL ${run}  (pid=${pid})"
        failures=$((failures + 1))
    fi
done

echo ""
if [ "$failures" -gt 0 ]; then
    echo "Finished with ${failures} failure(s)."
    exit 1
fi

echo "All ${#run_names[@]} experiments completed successfully."
