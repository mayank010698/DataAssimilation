#!/bin/bash
#
# FPPF ablation on Lorenz96 (dim=25, fully obs).
#
# Job enable/disable flags below make it easy to comment out runs.
#
# Round 1 (commented out): step-count & schedule ablation, N=500
#   1a. FPPF  uniform,    s=100, l=100
#   1b. FPPF  uniform,    s=100, l=50
#   1c. FPPF  power_back, s=100, l=50
#   1d. FPPF  power_back, s=64,  l=16  (original baseline)
#
# Round 2: BPF vs FPPF head-to-head, N=5000, uniform s=100 l=100
#   2a. BPF   (transition proposal)
#   2b. FPPF  (rectified-flow proposal)
#
# Usage:  cd /path/to/DataAssimilation && sbatch scripts/eval_fppf_96_steps.sbatch
#
#SBATCH --job-name="eval-fppf-steps"
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
# Configuration — shared
# ======================================================================

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_init3p000"

WANDB_PROJECT="${WANDB_PROJECT:-eval-lpf-l96}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

BATCH_SIZE=10
NUM_EVAL_TRAJECTORIES=10
RESAMPLING_THRESHOLD=0.33
PROCESS_NOISE_STD=0.25
SEED=42

RF_CHECKPOINT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96/l96_noise_dim25_20260109_193733/checkpoints/rf-epoch=067-val_loss=0.020633.ckpt"

# ======================================================================
# Job enable/disable flags  —  set to 0 to skip a job
# ======================================================================

# --- Round 1: FPPF step-count & schedule ablation (N=500) -------------
RUN_1A=0   # FPPF  uniform,    s=100, l=100, N=500
RUN_1B=0   # FPPF  uniform,    s=100, l=50,  N=500
RUN_1C=0   # FPPF  power_back, s=100, l=50,  N=500
RUN_1D=0   # FPPF  power_back, s=64,  l=16,  N=500

# --- Round 2: BPF vs FPPF head-to-head (N=5000) ----------------------
RUN_2A=1   # BPF   (transition proposal), N=5000
RUN_2B=1   # FPPF  (uniform, s=100, l=100), N=5000

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

TOTAL_JOBS=$(( RUN_1A + RUN_1B + RUN_1C + RUN_1D + RUN_2A + RUN_2B ))

echo "=========================================="
echo "FPPF ablation — Lorenz96 d=25"
echo "Project root : $PROJECT_ROOT"
echo "Data         : $DATA_DIR"
echo "Slurm Job ID : ${SLURM_JOB_ID:-N/A}"
echo ""
echo "Batch size      : $BATCH_SIZE"
echo "Eval trajs      : $NUM_EVAL_TRAJECTORIES"
echo "Resamp. thresh  : $RESAMPLING_THRESHOLD"
echo "Process noise   : $PROCESS_NOISE_STD"
echo "RF checkpoint   : $RF_CHECKPOINT"
echo ""
echo "Enabled jobs : 1a=$RUN_1A 1b=$RUN_1B 1c=$RUN_1C 1d=$RUN_1D 2a=$RUN_2A 2b=$RUN_2B"
echo "Launching $TOTAL_JOBS job(s) in parallel …"
echo "=========================================="

# ======================================================================
# Launch jobs
# ======================================================================

pids=()
run_names=()

# ------------------------------------------------------------------
# Round 1: FPPF step-count & schedule ablation (N=500)
# ------------------------------------------------------------------
R1_PARTICLES=500

# 1a — FPPF uniform s=100 l=100
if (( RUN_1A )); then
    label="fppf_d25_${R1_PARTICLES}k_s100_l100_uniform_${JOB_TAG}"
    echo "  -> 1a: FPPF uniform s=100 l=100 N=${R1_PARTICLES}  (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf \
            --proposal-type rf \
            --rf-checkpoint "$RF_CHECKPOINT" \
            --rf-sampling-steps 100 \
            --rf-likelihood-steps 100 \
            --sampling-grid-type uniform \
            --sampling-grid-param '{}' \
            --likelihood-grid-type uniform \
            --likelihood-grid-param '{}' \
            --n-particles "$R1_PARTICLES" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "logs/eval_1a_${JOB_TAG}.log" &
    pids+=("$!"); run_names+=("1a_fppf_uni_s100_l100")
fi

# 1b — FPPF uniform s=100 l=50
if (( RUN_1B )); then
    label="fppf_d25_${R1_PARTICLES}k_s100_l50_uniform_${JOB_TAG}"
    echo "  -> 1b: FPPF uniform s=100 l=50 N=${R1_PARTICLES}  (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf \
            --proposal-type rf \
            --rf-checkpoint "$RF_CHECKPOINT" \
            --rf-sampling-steps 100 \
            --rf-likelihood-steps 50 \
            --sampling-grid-type uniform \
            --sampling-grid-param '{}' \
            --likelihood-grid-type uniform \
            --likelihood-grid-param '{}' \
            --n-particles "$R1_PARTICLES" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "logs/eval_1b_${JOB_TAG}.log" &
    pids+=("$!"); run_names+=("1b_fppf_uni_s100_l50")
fi

# 1c — FPPF power_back s=100 l=50
if (( RUN_1C )); then
    label="fppf_d25_${R1_PARTICLES}k_s100_l50_pb_${JOB_TAG}"
    echo "  -> 1c: FPPF power_back s=100 l=50 N=${R1_PARTICLES}  (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf \
            --proposal-type rf \
            --rf-checkpoint "$RF_CHECKPOINT" \
            --rf-sampling-steps 100 \
            --rf-likelihood-steps 50 \
            --sampling-grid-type power_back \
            --sampling-grid-param '{"gamma":2.0}' \
            --likelihood-grid-type power_back \
            --likelihood-grid-param '{"gamma":2.0}' \
            --n-particles "$R1_PARTICLES" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "logs/eval_1c_${JOB_TAG}.log" &
    pids+=("$!"); run_names+=("1c_fppf_pb_s100_l50")
fi

# 1d — FPPF power_back s=64 l=16 (original baseline)
if (( RUN_1D )); then
    label="fppf_d25_${R1_PARTICLES}k_s64_l16_pb_${JOB_TAG}"
    echo "  -> 1d: FPPF power_back s=64 l=16 N=${R1_PARTICLES}  (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf \
            --proposal-type rf \
            --rf-checkpoint "$RF_CHECKPOINT" \
            --rf-sampling-steps 64 \
            --rf-likelihood-steps 16 \
            --sampling-grid-type power_back \
            --sampling-grid-param '{"gamma":2.0}' \
            --likelihood-grid-type power_back \
            --likelihood-grid-param '{"gamma":2.0}' \
            --n-particles "$R1_PARTICLES" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "logs/eval_1d_${JOB_TAG}.log" &
    pids+=("$!"); run_names+=("1d_fppf_pb_s64_l16")
fi

# ------------------------------------------------------------------
# Round 2: BPF vs FPPF head-to-head (N=5000, uniform s=100 l=100)
# ------------------------------------------------------------------
R2_PARTICLES=5000

# 2a — BPF (transition proposal)
if (( RUN_2A )); then
    label="bpf_d25_${R2_PARTICLES}_${JOB_TAG}"
    echo "  -> 2a: BPF N=${R2_PARTICLES}  (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf \
            --proposal-type transition \
            --n-particles "$R2_PARTICLES" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "logs/eval_2a_bpf_${JOB_TAG}.log" &
    pids+=("$!"); run_names+=("2a_bpf_N${R2_PARTICLES}")
fi

# 2b — FPPF (uniform, s=100, l=100)
if (( RUN_2B )); then
    label="fppf_d25_${R2_PARTICLES}_s100_l100_uniform_${JOB_TAG}"
    echo "  -> 2b: FPPF uniform s=100 l=100 N=${R2_PARTICLES}  (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf \
            --proposal-type rf \
            --rf-checkpoint "$RF_CHECKPOINT" \
            --rf-sampling-steps 100 \
            --rf-likelihood-steps 100 \
            --sampling-grid-type uniform \
            --sampling-grid-param '{}' \
            --likelihood-grid-type uniform \
            --likelihood-grid-param '{}' \
            --n-particles "$R2_PARTICLES" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "logs/eval_2b_fppf_${JOB_TAG}.log" &
    pids+=("$!"); run_names+=("2b_fppf_N${R2_PARTICLES}")
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
