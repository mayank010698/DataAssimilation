#!/bin/bash
#
# Autoregressive proposal-quality comparison on L96-25:
#   arm A: LocalizedRFProposal (patched, radius r)
#   arm B: global RFProposal   (same dataset, same training recipe)
#
# Both arms call ``proposals.eval_proposal`` which rolls the proposal out
# autoregressively (x_t ~ q(x_t | x_{t-1}, y_t)) over full test trajectories
# and logs RMSE / CRPS / observation error (global + over time) to wandb
# under the ``rf-proposal-eval`` project.
#
# Required env vars (or edit the defaults below):
#   LRF_CHECKPOINT         — LocalizedRFProposal checkpoint
#   GLOBAL_RF_CHECKPOINT   — global RFProposal checkpoint
#   DATA_DIR               — dataset dir (must match training dataset scaling)
#
# Optional:
#   N_TRAJECTORIES=32
#   N_SAMPLES_PER_TRAJ=20
#   NUM_SAMPLING_STEPS=25
#   NUM_LIKELIHOOD_STEPS=25
#
# Usage:
#   LRF_CHECKPOINT=/path/to/lrf.ckpt \
#   GLOBAL_RF_CHECKPOINT=/path/to/rf.ckpt \
#   sbatch scripts/eval_proposal_lrf_vs_rf_l96_25.sbatch
#
#SBATCH --job-name="eval-lrf-vs-rf-96"
#SBATCH --partition=dali
#SBATCH --account=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/eval.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1

module load anaconda3/2024.10
source activate
conda activate da
module load cuda/12.8

export PYTHONPATH="$(pwd)"
mkdir -p logs

DATA_DIR="${DATA_DIR:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_pnoise0p100_init3p000}"
LRF_CHECKPOINT="${LRF_CHECKPOINT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/lrf_runs/run_8092322/checkpoints/lrf-epoch=062-val_loss=0.022068.ckpt}"
GLOBAL_RF_CHECKPOINT="${GLOBAL_RF_CHECKPOINT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96/l96_noise_dim25_20260109_193733/checkpoints/rf-epoch=067-val_loss=0.020633.ckpt}"

N_TRAJECTORIES="${N_TRAJECTORIES:-32}"
N_SAMPLES_PER_TRAJ="${N_SAMPLES_PER_TRAJ:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-25}"
NUM_LIKELIHOOD_STEPS="${NUM_LIKELIHOOD_STEPS:-25}"

# Writable wandb run dir (default next to lrf_runs/rf_runs_96).
WANDB_DIR="${WANDB_DIR:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/wandb}"
mkdir -p "$WANDB_DIR"

for path_var in LRF_CHECKPOINT GLOBAL_RF_CHECKPOINT; do
    path_val="${!path_var}"
    if [ -z "$path_val" ] || [ ! -f "$path_val" ]; then
        echo "ERROR: $path_var='$path_val' is missing or not a file." >&2
        exit 1
    fi
done
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR='$DATA_DIR' is not a directory." >&2
    exit 1
fi

STAMP="${SLURM_JOB_ID:-manual}"

echo "=========================================="
echo "Proposal-quality eval: LRF vs global RF on L96-25"
echo "Data dir           : $DATA_DIR"
echo "LRF ckpt           : $LRF_CHECKPOINT"
echo "Global RF ckpt     : $GLOBAL_RF_CHECKPOINT"
echo "n_trajectories     : $N_TRAJECTORIES"
echo "n_samples_per_traj : $N_SAMPLES_PER_TRAJ"
echo "num_sampling_steps : $NUM_SAMPLING_STEPS"
echo "=========================================="

run_eval() {
    local label="$1"
    local ckpt="$2"
    local log_file="logs/eval_${label}_${STAMP}.log"

    echo "--- Arm: $label ---"
    echo "Log: $log_file"
    srun python -m proposals.eval_proposal \
        --checkpoint "$ckpt" \
        --data-dir "$DATA_DIR" \
        --n-trajectories "$N_TRAJECTORIES" \
        --batch-size "$BATCH_SIZE" \
        --n-samples-per-traj "$N_SAMPLES_PER_TRAJ" \
        --num-sampling-steps "$NUM_SAMPLING_STEPS" \
        --num-likelihood-steps "$NUM_LIKELIHOOD_STEPS" \
        --run-name "eval-${label}-l96-25-${STAMP}" \
        --wandb-dir "$WANDB_DIR" \
        2>&1 | tee "$log_file"
}

run_eval "lrf" "$LRF_CHECKPOINT"
run_eval "rf"  "$GLOBAL_RF_CHECKPOINT"

echo "=========================================="
echo "Done. Compare the two runs in wandb project 'rf-proposal-eval'."
echo "Key metrics: eval/global_mean_rmse, eval/global_mean_crps,"
echo "             eval/global_mean_obs_error, and *_over_time curves."
echo "=========================================="
