#!/bin/bash
#
# RFProposal likelihood-grid ablation for Linear Gaussian eval — uniform grid only.
# Interactive version of eval_lg_grid.sbatch (array task 0).
#
# Usage (from an interactive GPU session):
#   cd /projects/illinois/eng/cs/arindamb/cnagda2/da/DataAssimilation
#   bash scripts/eval_lg_grid_uniform.sh
#

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ ! -f "${PROJECT_ROOT}/eval.py" ]; then
    echo "ERROR: Cannot find project root (expected eval.py at PROJECT_ROOT)."
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

WANDB_PROJECT="${WANDB_PROJECT:-eval-lg-d8-grid-ablation-uniform-samp}"
JOB_TAG="interactive_$$"

N_SAMPLES=100
N_PAIRS=1000
BATCH_SIZE=100
SEED=42

BUDGETS=(4 8 16 32 64 128)

grid_type="uniform"
grid_label="uniform"

echo "=========================================="
echo "eval_lg.py — RFProposal grid ablation"
echo "Project root: $PROJECT_ROOT"
echo "Checkpoint: $CHECKPOINT"
echo "Data: $DATA_DIR"
echo "Likelihood grid type: ${grid_type}"
echo "Sampling: uniform, 64 steps (fixed)"
echo "Budgets (likelihood K): ${BUDGETS[*]}"
echo "W&B project: ${WANDB_PROJECT:-<unset>}"
echo "Running ${#BUDGETS[@]} budgets sequentially on one GPU"
echo "=========================================="

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data dir not found: $DATA_DIR"
    exit 1
fi

failures=0

for K in "${BUDGETS[@]}"; do
    run_name="${grid_label}_k${K}"
    wandb_run_name="fm_lg_d8_uniform_samp_kl_${JOB_TAG}_${run_name}"
    log_file="logs/eval_lg_grid_${JOB_TAG}_${run_name}.log"

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

    if [ -n "$WANDB_PROJECT" ]; then
        eval_cmd+=( --wandb_project "$WANDB_PROJECT" --wandb_run_name "$wandb_run_name" )
    fi

    echo "Running ${run_name} -> ${log_file}"
    if CUDA_VISIBLE_DEVICES=0 "${eval_cmd[@]}" 2>&1 | tee "$log_file"; then
        echo "Completed ${run_name}"
    else
        echo "FAILED ${run_name}"
        failures=$((failures + 1))
    fi
done

if [ "$failures" -gt 0 ]; then
    echo "eval_lg_grid_uniform finished with ${failures} failing experiment(s)."
    exit 1
fi

echo "eval_lg_grid_uniform finished successfully."
