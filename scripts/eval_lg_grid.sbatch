#!/bin/bash
#
# RFProposal likelihood-grid ablation for Linear Gaussian eval.
#
# Sampling integration is fixed: uniform grid, num_sampling_steps below.
# Only the likelihood / backward integration grid varies (per array task and K).
#
# Usage:
#   cd /projects/illinois/eng/cs/arindamb/cnagda2/da/DataAssimilation
#   sbatch scripts/eval_lg_grid.sbatch
#
# Strategy:
# - Slurm array index selects ONE likelihood grid type per GPU.
# - Each array task launches all budgets K in background on that GPU.
#
#SBATCH --job-name="eval-lg-grid-ablate"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=04:00:00
#SBATCH --array=0-4
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

export PYTHONPATH="$(pwd)"
mkdir -p logs

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/linear_gaussian_d8_n1024_len200_freq1"
CHECKPOINT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_lg/fm_lg_d8_nomask_7954968/checkpoints/fm-epoch=096-val_loss=0.419213.ckpt"

# Set empty to disable W&B. Default project tags uniform-sampling sweeps vs older coupled runs.
WANDB_PROJECT="${WANDB_PROJECT:-eval-lg-d8-grid-ablation-uniform-samp}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-0}"

N_SAMPLES=100
N_PAIRS=1000
BATCH_SIZE=100
SEED=42

# Fixed budget list for this sweep.
BUDGETS=(128 64 32 16 8 4)

# Array index -> one grid type per GPU.
GRID_TYPES=(uniform cosine power_front power_back piecewise)
GRID_LABELS=(uniform cosine power_front_g2 power_back_g2 piecewise)

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: This script expects SLURM_ARRAY_TASK_ID via --array."
    exit 1
fi

if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${#GRID_TYPES[@]}" ]; then
    echo "ERROR: Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

grid_type="${GRID_TYPES[$SLURM_ARRAY_TASK_ID]}"
grid_label="${GRID_LABELS[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "eval_lg.py — RFProposal grid ablation"
echo "Project root: $PROJECT_ROOT"
echo "Checkpoint: $CHECKPOINT"
echo "Data: $DATA_DIR"
echo "Array task: ${SLURM_ARRAY_TASK_ID} -> likelihood_grid_type=${grid_type}"
echo "Sampling: uniform, 64 steps (fixed across this sweep)"
echo "Budgets (likelihood K): ${BUDGETS[*]}"
echo "W&B project (default): ${WANDB_PROJECT:-<unset>}"
echo "Launching ${#BUDGETS[@]} runs in parallel on one GPU"
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

for K in "${BUDGETS[@]}"; do
    run_name="${grid_label}_k${K}"
    wandb_run_name="fm_lg_d8_uniform_samp_kl_${JOB_TAG}_${run_name}"
    log_file="logs/eval_lg_grid_${JOB_TAG}_${run_name}.log"

    # Fixed sampling budget; likelihood budget K sweeps per task.
    sampling_steps="64"
    likelihood_steps="${K}"

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
        --sampling_grid_type uniform
        --likelihood_grid_type "$grid_type"
    )

    # Likelihood-only grid params (sampling stays uniform; no --sampling_grid_param).
    # - uniform/cosine: omit likelihood_grid_param
    # - power_*: gamma fixed at 2.0
    # - piecewise: step_allocations sum to K; boundaries [0,0.2,0.8,1], 1:2:1 split
    if [ "$grid_type" = "power_front" ] || [ "$grid_type" = "power_back" ]; then
        eval_cmd+=( --likelihood_grid_param '{"gamma": 2.0}' )
    elif [ "$grid_type" = "piecewise" ]; then
        la=$((K / 4))
        lb=$((K / 2))
        lc=$((K / 4))
        if [ $((la + lb + lc)) -ne "$K" ]; then
            echo "ERROR: piecewise likelihood allocation does not sum to K=${K}: ${la}+${lb}+${lc}"
            exit 1
        fi
        likelihood_piecewise_json="{\"boundaries\": [0.0, 0.2, 0.8, 1.0], \"step_allocations\": [${la}, ${lb}, ${lc}]}"
        eval_cmd+=( --likelihood_grid_param "$likelihood_piecewise_json" )
    fi

    if [ -n "$WANDB_PROJECT" ]; then
        eval_cmd+=( --wandb_project "$WANDB_PROJECT" --wandb_run_name "$wandb_run_name" )
    fi

    echo "Launching ${run_name} -> ${log_file}"
    (
        CUDA_VISIBLE_DEVICES=0 "${eval_cmd[@]}"
    ) 2>&1 | tee "$log_file" &
    pids+=("$!")
    run_names+=("$run_name")
done

failures=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    run_name="${run_names[$idx]}"
    if wait "$pid"; then
        echo "Completed ${run_name} (pid=${pid})"
    else
        echo "FAILED ${run_name} (pid=${pid})"
        failures=$((failures + 1))
    fi
done

if [ "$failures" -gt 0 ]; then
    echo "eval_lg_grid finished with ${failures} failing experiment(s) for grid_type=${grid_type}."
    exit 1
fi

echo "eval_lg_grid finished successfully for grid_type=${grid_type}."
