#!/bin/bash
#
#SBATCH --job-name="train-lrf-l96"
#SBATCH --partition=dali
#SBATCH --account=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
#SBATCH --time=12:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
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
TRAIN_LOG="logs/train_lrf_l96_${SLURM_JOB_ID:-manual}.log"

echo "=========================================="
echo "Training Localized Rectified Flow on L96"
echo "Project root: $(pwd)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

# L96-25 pnoise dataset (override as needed)
DATA_DIR="${DATA_DIR:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_pnoise0p100_init3p000}"
STATE_DIM="${STATE_DIM:-25}"
RADIUS="${RADIUS:-4}"
ARCH="${ARCH:-local_mlp}"
WANDB_PROJECT="${WANDB_PROJECT:-lrf-train-96}"

srun python -m proposals.train_lrf \
    --data_dir "$DATA_DIR" \
    --output_dir /projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/lrf_runs/run_${SLURM_JOB_ID:-manual} \
    --radius "$RADIUS" \
    --state_dim "$STATE_DIM" \
    --architecture "$ARCH" \
    --hidden_dim 128 \
    --depth 4 \
    --channels 32 \
    --num_blocks 2 \
    --kernel_size 3 \
    --time_embed_dim 64 \
    --batch_size 512 \
    --learning_rate 3e-4 \
    --max_epochs 100 \
    --num_workers 4 \
    --use_observations \
    --num_sampling_steps 25 \
    --num_likelihood_steps 25 \
    --gpus 1 \
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee "$TRAIN_LOG"

echo "=========================================="
echo "Training complete."
echo "=========================================="
