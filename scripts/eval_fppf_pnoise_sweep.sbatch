#!/bin/bash
#
# FPPF process-noise sweep on Lorenz96 (dim=25, fully obs).
#
# Grid: N ∈ {500, 1000} × process_noise_std ∈ {0.05, 0.075, 0.1, 0.15, 0.2, 0.25}
# RF config: uniform grid, 100 sampling steps, 50 likelihood steps.
# Trajectories truncated to 100 timesteps (--max-timesteps 100).
# Total: 12 jobs in parallel on 1 GPU.
#
# Usage:  cd /path/to/DataAssimilation && sbatch scripts/eval_fppf_pnoise_sweep.sbatch
#
#SBATCH --job-name="eval-fppf-pnoise"
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
# Configuration
# ======================================================================

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_init3p000"

WANDB_PROJECT="${WANDB_PROJECT:-eval-lpf-l96}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

BATCH_SIZE=10
NUM_EVAL_TRAJECTORIES=10
RESAMPLING_THRESHOLD=0.33
MAX_TIMESTEPS=100
SEED=42

PARTICLE_COUNTS=(500 1000)
PROCESS_NOISE_STDS=(0.05 0.075 0.1 0.15 0.2 0.25)

RF_CHECKPOINT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96/l96_noise_dim25_20260109_193733/checkpoints/rf-epoch=067-val_loss=0.020633.ckpt"
RF_SAMPLING_STEPS=100
RF_LIKELIHOOD_STEPS=50

# ======================================================================
# Sanity checks
# ======================================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data dir not found: $DATA_DIR"
    exit 1
fi
if [ ! -f "$RF_CHECKPOINT" ]; then
    echo "ERROR: RF checkpoint not found: $RF_CHECKPOINT"
    exit 1
fi

TOTAL_JOBS=$(( ${#PARTICLE_COUNTS[@]} * ${#PROCESS_NOISE_STDS[@]} ))

echo "=========================================="
echo "FPPF process-noise sweep — Lorenz96 d=25"
echo "Project root : $PROJECT_ROOT"
echo "Data         : $DATA_DIR"
echo "Slurm Job ID : ${SLURM_JOB_ID:-N/A}"
echo ""
echo "Particle counts    : ${PARTICLE_COUNTS[*]}"
echo "Process noise stds : ${PROCESS_NOISE_STDS[*]}"
echo "Max timesteps      : $MAX_TIMESTEPS"
echo "Batch size         : $BATCH_SIZE"
echo "Eval trajs         : $NUM_EVAL_TRAJECTORIES"
echo "Resamp. thresh     : $RESAMPLING_THRESHOLD"
echo "RF checkpoint      : $RF_CHECKPOINT"
echo "RF sampling steps  : $RF_SAMPLING_STEPS"
echo "RF likelihood steps: $RF_LIKELIHOOD_STEPS"
echo "Grid type          : uniform"
echo ""
echo "Launching $TOTAL_JOBS job(s) in parallel …"
echo "=========================================="

# ======================================================================
# Launch jobs
# ======================================================================

pids=()
run_names=()

for N in "${PARTICLE_COUNTS[@]}"; do
for pnoise in "${PROCESS_NOISE_STDS[@]}"; do
    label="fppf_d25_N${N}_pn${pnoise}_s${RF_SAMPLING_STEPS}_l${RF_LIKELIHOOD_STEPS}_${JOB_TAG}"
    log="logs/eval_fppf_N${N}_pn${pnoise}_${JOB_TAG}.log"

    echo "  -> FPPF  N=${N}  pnoise=${pnoise}  (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf \
            --proposal-type rf \
            --rf-checkpoint "$RF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --sampling-grid-type uniform \
            --sampling-grid-param '{}' \
            --likelihood-grid-type uniform \
            --likelihood-grid-param '{}' \
            --n-particles "$N" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$pnoise" \
            --max-timesteps "$MAX_TIMESTEPS" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "$log" &
    pids+=("$!")
    run_names+=("fppf_N${N}_pn${pnoise}")
done
done

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
