#!/bin/bash
#
#SBATCH --job-name="fm-lg-d8-nomask"
#SBATCH --partition=dali
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

set -euo pipefail

# In Slurm, use submit directory; fallback to script-relative project root.
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

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/linear_gaussian_d8_n1024_len200_freq1"
OBS_COMPONENTS="0,1,2,3,4,5,6,7"

WANDB_PROJECT="rf-train-lg-d8"

STATE_DIM=8
BATCH_SIZE=512
LR=3e-4
MAX_EPOCHS=100
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
COND_EMBED_DIM=128

JOB_TAG="${SLURM_JOB_ID:-manual}"
RUN_BASE="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_lg"
OUTPUT_DIR="${RUN_BASE}/fm_lg_d8_nomask_${JOB_TAG}"

echo "=========================================="
echo "FM teacher (no prev-state mask), Linear Gaussian d=8"
echo "Project root: $(pwd)"
echo "Data dir: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "WandB project: $WANDB_PROJECT"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python proposals/train_fm.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --state_dim "$STATE_DIM" \
    --obs_components "$OBS_COMPONENTS" \
    --architecture resnet1d \
    --channels "$CHANNELS" \
    --num_blocks "$NUM_BLOCKS" \
    --kernel_size "$KERNEL_SIZE" \
    --train_cond_method adaln \
    --cond_embed_dim "$COND_EMBED_DIM" \
    --use_observations \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --gpus 1 \
    --wandb_project "$WANDB_PROJECT" \
    --evaluate \
    2>&1 | tee "logs/fm_lg_d8_nomask_${JOB_TAG}.log"

echo "Training finished."
