#!/bin/bash
#
# Single-job Slurm script: Stage 3 F2D2 / divergence training for L96 dim5 shortcut.
#
#SBATCH --job-name="div-96-dim5"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80g
#SBATCH --time=04:00:00
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

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_pnoise0p100_init3p000"
OBS_COMPONENTS="0,1,2,3,4"

# Stage 2 shortcut (dim5 FM teacher distilled)
SHORTCUT_CKPT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims/sc_dim5_fm_nomask_7854839/checkpoints/sc-epoch=037-val_loss=0.034451.ckpt"
# Same FM teacher as Stage 2 (see sc training log)
TEACHER_CKPT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims/fm_dim5_nomask_7850189/checkpoints/ema_weights.ckpt"

RUN_BASE="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${RUN_BASE}/f2d2_dim5_fm_nomask_${JOB_TAG}"

WANDB_PROJECT="rf-train-96-dim"

STATE_DIM=5
BATCH_SIZE=256
LR=1e-4
MAX_EPOCHS=100

echo "=========================================="
echo "F2D2 (div) training — L96 dim5"
echo "Project root: $(pwd)"
echo "Shortcut: $SHORTCUT_CKPT"
echo "Teacher: $TEACHER_CKPT"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

python proposals/train_div.py \
    --shortcut_ckpt "$SHORTCUT_CKPT" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --state_dim "$STATE_DIM" \
    --obs_components "$OBS_COMPONENTS" \
    --architecture resnet1d \
    --use_observations \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --denoise_timesteps 1024 \
    --gpus 1 \
    --wandb_project "$WANDB_PROJECT" \
    --teacher_div_estimator exact \
    --div_scale 1.0 \
    --evaluate \
    --eval_num_sampling_steps 4

echo "Done. Logs also under $OUTPUT_DIR/training.log"
