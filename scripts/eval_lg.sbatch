#!/bin/bash
#
# Oracle / Kalman evaluation for Linear Gaussian (proposals/eval_lg.py).
#
# Run from repo root so Slurm sees the project:
#   cd /path/to/DataAssimilation && sbatch scripts/eval_lg.sbatch
#
#SBATCH --job-name="eval-lg-grid-bg"
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

# module load cuda/12.8

export PYTHONPATH="$(pwd)"

mkdir -p logs

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/linear_gaussian_d8_n1024_len200_freq1"

# Lightning "best" checkpoint from training, or point to checkpoints/ema_weights.ckpt if you prefer EMA.
CHECKPOINT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_lg/fm_lg_d8_nomask_7954968/checkpoints/fm-epoch=096-val_loss=0.419213.ckpt"

# Set to empty to skip W&B (only stdout / Slurm log).
WANDB_PROJECT="${WANDB_PROJECT:-eval-lg-d8}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

N_SAMPLES=64
N_PAIRS=1000
BATCH_SIZE=512
SEED=42

SAMPLING_STEPS_LIST=(64 64 64 64 64 64)
LIKELIHOOD_STEPS_LIST=(128 64 32 16 8 4)

N_EXPERIMENTS="${#LIKELIHOOD_STEPS_LIST[@]}"
if [ "${#SAMPLING_STEPS_LIST[@]}" -ne "$N_EXPERIMENTS" ]; then
    echo "ERROR: sampling/likelihood step lists must have equal length."
    exit 1
fi

echo "=========================================="
echo "eval_lg.py — Linear Gaussian oracle eval"
echo "Project root: $PROJECT_ROOT"
echo "Checkpoint: $CHECKPOINT"
echo "Data: $DATA_DIR"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Launching ${N_EXPERIMENTS} experiments in parallel background processes"
echo "=========================================="

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data dir not found: $DATA_DIR"
    exit 1
fi

pids=()
run_names=()
for i in "${!LIKELIHOOD_STEPS_LIST[@]}"; do
    sampling_steps="${SAMPLING_STEPS_LIST[$i]}"
    likelihood_steps="${LIKELIHOOD_STEPS_LIST[$i]}"
    run_suffix="s${sampling_steps}_l${likelihood_steps}"
    run_name="fm_lg_d8_${JOB_TAG}_${run_suffix}"
    log_file="logs/eval_lg_${JOB_TAG}_${run_suffix}.log"

    eval_cmd=(
        python proposals/eval_lg.py
        --checkpoint "$CHECKPOINT"
        --data_dir "$DATA_DIR"
        --n_samples "$N_SAMPLES"
        --n_pairs "$N_PAIRS"
        --batch_size "$BATCH_SIZE"
        --device cuda
        --seed "$SEED"
        --num_sampling_steps "$sampling_steps"
        --num_likelihood_steps "$likelihood_steps"
    )
    if [ -n "$WANDB_PROJECT" ]; then
        eval_cmd+=( --wandb_project "$WANDB_PROJECT" --wandb_run_name "$run_name" )
    fi

    echo "Launching ${run_suffix} -> ${log_file}"
    (
        CUDA_VISIBLE_DEVICES=0 "${eval_cmd[@]}"
    ) 2>&1 | tee "$log_file" &
    pids+=("$!")
    run_names+=("$run_suffix")
done

failures=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    run_suffix="${run_names[$idx]}"
    if wait "$pid"; then
        echo "Completed ${run_suffix} (pid=${pid})"
    else
        echo "FAILED ${run_suffix} (pid=${pid})"
        failures=$((failures + 1))
    fi
done

if [ "$failures" -gt 0 ]; then
    echo "eval_lg finished with ${failures} failing experiment(s)."
    exit 1
fi

echo "eval_lg finished successfully for all ${N_EXPERIMENTS} experiments."
