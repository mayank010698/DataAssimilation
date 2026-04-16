#!/bin/bash
#
# Evaluate BPF / LPF / FPPF / Localized-FPPF on Lorenz96 (dim=25, fully obs).
#
# Runs up to 8 independent jobs in parallel on 1 GPU (4 methods × 2 particle counts).
#   Job 1: BPF    (bootstrap PF, transition proposal)
#   Job 2: LPF    (localized PF, transition proposal)
#   Job 3: FPPF   (standard PF with rectified-flow proposal, non-localized)
#   Job 4: L-FPPF (localized PF with rectified-flow proposal)
#
# To disable a job, set its RUN_* flag to 0 below.
#
# Usage:  cd /path/to/DataAssimilation && sbatch scripts/eval_lpf.sbatch
#
#SBATCH --job-name="eval-lpf"
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
# Job enable/disable flags  —  set to 0 to skip a job
# ======================================================================

RUN_BPF=0        # Job 1 — Bootstrap PF (transition proposal)
RUN_LPF=0        # Job 2 — Localized PF (transition proposal)
RUN_FPPF=1       # Job 3 — FPPF (rectified-flow proposal, non-localized)
RUN_LFPPF=1      # Job 4 — Localized FPPF (rectified-flow proposal + localization)

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

# --- Localized PF hyperparameters (shared by LPF and L-FPPF) ---------
LPF_BLOCK_SIZE=1
LPF_LOC_RADIUS=3.0
LPF_TAPER="gaussian"
LPF_POST_REG="none"
LPF_WEIGHT_SMOOTHING=""   # empty = off (flag not passed)

# --- FPPF / L-FPPF hyperparameters -----------------------------------
RF_CHECKPOINT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96/l96_noise_dim25_20260109_193733/checkpoints/rf-epoch=067-val_loss=0.020633.ckpt"
RF_SAMPLING_STEPS=64
RF_LIKELIHOOD_STEPS=16
SAMPLING_GRID_TYPE="power_back"
SAMPLING_GRID_PARAM='{"gamma": 2.0}'
LIKELIHOOD_GRID_TYPE="power_back"
LIKELIHOOD_GRID_PARAM='{"gamma": 2.0}'

# ======================================================================
# Sanity checks
# ======================================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data dir not found: $DATA_DIR"
    exit 1
fi

TOTAL_JOBS=0
(( RUN_BPF  )) && TOTAL_JOBS=$((TOTAL_JOBS + ${#PARTICLE_COUNTS[@]}))
(( RUN_LPF  )) && TOTAL_JOBS=$((TOTAL_JOBS + ${#PARTICLE_COUNTS[@]}))
(( RUN_FPPF )) && TOTAL_JOBS=$((TOTAL_JOBS + ${#PARTICLE_COUNTS[@]}))
(( RUN_LFPPF)) && TOTAL_JOBS=$((TOTAL_JOBS + ${#PARTICLE_COUNTS[@]}))

echo "=========================================="
echo "BPF / LPF / FPPF / L-FPPF — Lorenz96 d=25"
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
echo "RF checkpoint            : $RF_CHECKPOINT"
echo "RF sampling steps        : $RF_SAMPLING_STEPS"
echo "RF likelihood steps      : $RF_LIKELIHOOD_STEPS"
echo "Sampling grid            : $SAMPLING_GRID_TYPE ($SAMPLING_GRID_PARAM)"
echo "Likelihood grid          : $LIKELIHOOD_GRID_TYPE ($LIKELIHOOD_GRID_PARAM)"
echo ""
echo "Enabled jobs : BPF=$RUN_BPF  LPF=$RUN_LPF  FPPF=$RUN_FPPF  L-FPPF=$RUN_LFPPF"
echo "Launching $TOTAL_JOBS job(s) in parallel …"
echo "=========================================="

# ======================================================================
# Launch jobs
# ======================================================================

pids=()
run_names=()

# Helper to build localization flags
lpf_extra_flags=()
if [ -n "$LPF_WEIGHT_SMOOTHING" ]; then
    lpf_extra_flags+=(--weight-smoothing)
fi

# ------------------------------------------------------------------
# Job 1 — BPF (standard bootstrap PF)
# ------------------------------------------------------------------
if (( RUN_BPF )); then
for N in "${PARTICLE_COUNTS[@]}"; do
    bpf_name="bpf_d25_${N}k_${JOB_TAG}"
    bpf_log="logs/eval_bpf_d25_${N}_${JOB_TAG}.log"

    echo "  -> BPF  N=${N}  (${bpf_name})"
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
            --experiment-label "$bpf_name" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$bpf_name" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "$bpf_log" &
    pids+=("$!")
    run_names+=("bpf_N${N}")
done
fi

# ------------------------------------------------------------------
# Job 2 — LPF (localized PF, transition proposal)
# ------------------------------------------------------------------
if (( RUN_LPF )); then
for N in "${PARTICLE_COUNTS[@]}"; do
    lpf_name="lpf_d25_${N}k_bs${LPF_BLOCK_SIZE}_r${LPF_LOC_RADIUS}_${JOB_TAG}"
    lpf_log="logs/eval_lpf_d25_${N}_${JOB_TAG}.log"

    echo "  -> LPF  N=${N}  (${lpf_name})"
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
            --experiment-label "$lpf_name" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$lpf_name" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "$lpf_log" &
    pids+=("$!")
    run_names+=("lpf_N${N}")
done
fi

# ------------------------------------------------------------------
# Job 3 — FPPF (rectified-flow proposal, non-localized)
# ------------------------------------------------------------------
if (( RUN_FPPF )); then
for N in "${PARTICLE_COUNTS[@]}"; do
    fppf_name="fppf_d25_${N}k_s${RF_SAMPLING_STEPS}_l${RF_LIKELIHOOD_STEPS}_${JOB_TAG}"
    fppf_log="logs/eval_fppf_d25_${N}_${JOB_TAG}.log"

    echo "  -> FPPF  N=${N}  (${fppf_name})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf \
            --proposal-type rf \
            --rf-checkpoint "$RF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --sampling-grid-type "$SAMPLING_GRID_TYPE" \
            --sampling-grid-param "$SAMPLING_GRID_PARAM" \
            --likelihood-grid-type "$LIKELIHOOD_GRID_TYPE" \
            --likelihood-grid-param "$LIKELIHOOD_GRID_PARAM" \
            --n-particles "$N" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$fppf_name" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$fppf_name" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "$fppf_log" &
    pids+=("$!")
    run_names+=("fppf_N${N}")
done
fi

# ------------------------------------------------------------------
# Job 4 — L-FPPF (localized PF + rectified-flow proposal)
# ------------------------------------------------------------------
if (( RUN_LFPPF )); then
for N in "${PARTICLE_COUNTS[@]}"; do
    lfppf_name="lfppf_d25_${N}k_bs${LPF_BLOCK_SIZE}_r${LPF_LOC_RADIUS}_s${RF_SAMPLING_STEPS}_l${RF_LIKELIHOOD_STEPS}_${JOB_TAG}"
    lfppf_log="logs/eval_lfppf_d25_${N}_${JOB_TAG}.log"

    echo "  -> L-FPPF  N=${N}  (${lfppf_name})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method lpf \
            --proposal-type rf \
            --rf-checkpoint "$RF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --sampling-grid-type "$SAMPLING_GRID_TYPE" \
            --sampling-grid-param "$SAMPLING_GRID_PARAM" \
            --likelihood-grid-type "$LIKELIHOOD_GRID_TYPE" \
            --likelihood-grid-param "$LIKELIHOOD_GRID_PARAM" \
            --n-particles "$N" \
            --batch-size "$BATCH_SIZE" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --block-size "$LPF_BLOCK_SIZE" \
            --localization-radius "$LPF_LOC_RADIUS" \
            --lpf-taper "$LPF_TAPER" \
            --post-regularization "$LPF_POST_REG" \
            "${lpf_extra_flags[@]}" \
            --experiment-label "$lfppf_name" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$lfppf_name" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "$lfppf_log" &
    pids+=("$!")
    run_names+=("lfppf_N${N}")
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
