#!/bin/bash
#
#SBATCH --job-name="train-rf96-1ep"
#SBATCH --partition=dali
#SBATCH --account=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
#SBATCH --time=06:00:00
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

# Activate conda env on DALI
module load anaconda3/2024.10
source activate
conda activate da

# Load CUDA module used across project scripts
module load cuda/12.8

export PYTHONPATH="$(pwd)"

mkdir -p logs
GPU_LOG="logs/gpu_mem_train_rf_96_1epoch_${SLURM_JOB_ID:-manual}.csv"
TRAIN_LOG="logs/train_rf_96_1epoch_${SLURM_JOB_ID:-manual}.log"

echo "=========================================="
echo "Training OG RF model (1 epoch)"
echo "Project root: $(pwd)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "GPU monitor log: $GPU_LOG"
echo "Train log: $TRAIN_LOG"
echo "=========================================="

# Monitor GPU memory/utilization every 2 seconds in the background.
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 2 > "$GPU_LOG" 2>&1 &
SMI_PID=$!

cleanup() {
    if [ -n "${SMI_PID:-}" ]; then
        kill "$SMI_PID" 2>/dev/null || true
        wait "$SMI_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_pnoise0p100_init3p000"

srun python /projects/illinois/eng/cs/arindamb/cnagda2/da/DataAssimilation/proposals/train_rf.py \
    --data_dir "$DATA_DIR" \
    --output_dir /projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims_1epoch \
    --state_dim 5 \
    --obs_components 0,1,2,3,4 \
    --architecture resnet1d \
    --channels 64 \
    --num_blocks 10 \
    --kernel_size 5 \
    --train-cond-method adaln \
    --cond_embed_dim 128 \
    --use_observations \
    --batch_size 512 \
    --learning_rate 0.0003 \
    --max_epochs 1 \
    --gpus 1 \
    --evaluate \
    2>&1 | tee "$TRAIN_LOG"

echo "=========================================="
echo "Run complete."
echo "GPU memory log saved to: $GPU_LOG"
echo "Training log saved to: $TRAIN_LOG"
echo "=========================================="
