#!/bin/bash
#
# MeanFlow-F2D2 pipeline for L96 dim5:
#   Stage 1 — FM teacher (reuse existing checkpoint)
#   Stage 2 — MeanFlow (JVP self-consistency, no teacher)
#   Stage 3 — F2D2 divergence head (teacher div matching on MeanFlow model)
#
# Stages 2 and 3 run sequentially on a single GPU.
#
#SBATCH --job-name="mf-f2d2-96-dim5"
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80g
#SBATCH --time=48:00:00
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

# ── Dataset ──────────────────────────────────────────────────────────────
DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_pnoise0p100_init3p000"
OBS_COMPONENTS="0,1,2,3,4"

# ── Stage 1: Reuse the existing FM teacher ───────────────────────────────
TEACHER_CKPT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims/fm_dim5_nomask_7850189/checkpoints/ema_weights.ckpt"

# ── Output paths ─────────────────────────────────────────────────────────
RUN_BASE="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims"
JOB_TAG="${SLURM_JOB_ID:-manual}"

WANDB_PROJECT="rf-train-96-dim"

# ── Architecture (match existing shortcut/FM training) ───────────────────
STATE_DIM=5
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
TIME_EMBED_DIM=64

# ── Stage 2: MeanFlow hyperparams ───────────────────────────────────────
MF_BATCH_SIZE=256
MF_LR=1e-4
MF_MAX_EPOCHS=300

MF_OUTPUT_DIR="${RUN_BASE}/mf_dim5_${JOB_TAG}"

# ── Stage 3: F2D2 divergence head hyperparams ───────────────────────────
DIV_BATCH_SIZE=256
DIV_LR=1e-4
DIV_MAX_EPOCHS=50

DIV_OUTPUT_DIR="${RUN_BASE}/mf_f2d2_dim5_${JOB_TAG}"

echo "=========================================="
echo "MeanFlow-F2D2 Pipeline — L96 dim5"
echo "Project root:  $(pwd)"
echo "Teacher ckpt:  $TEACHER_CKPT"
echo "Data dir:      $DATA_DIR"
echo "MF output:     $MF_OUTPUT_DIR"
echo "F2D2 output:   $DIV_OUTPUT_DIR"
echo "Slurm Job ID:  ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

# ════════════════════════════════════════════════════════════════════════
# Stage 2: MeanFlow training (JVP self-consistency, no teacher needed)
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "[Stage 2] MeanFlow training starting ..."

python proposals/train_mf.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$MF_OUTPUT_DIR" \
    --state_dim "$STATE_DIM" \
    --obs_components "$OBS_COMPONENTS" \
    --use_observations \
    --architecture resnet1d \
    --channels "$CHANNELS" \
    --num_blocks "$NUM_BLOCKS" \
    --kernel_size "$KERNEL_SIZE" \
    --time_embed_dim "$TIME_EMBED_DIM" \
    --batch_size "$MF_BATCH_SIZE" \
    --learning_rate "$MF_LR" \
    --max_epochs "$MF_MAX_EPOCHS" \
    --gpus 1 \
    --wandb_project "$WANDB_PROJECT" \
    --tr_sampler v1 \
    --ratio 0.9 \
    --norm_p 0.75 \
    --norm_eps 1e-3 \
    --evaluate \
    --eval_num_sampling_steps 4 \

echo "[Stage 2] MeanFlow training done."

# Find best MeanFlow checkpoint (lowest val_loss from filename)
MF_BEST=$(ls "${MF_OUTPUT_DIR}/checkpoints"/mf-epoch=*.ckpt 2>/dev/null \
    | sed 's/.*val_loss=\([0-9.]*\)\.ckpt/\1 &/' \
    | sort -n | head -1 | awk '{print $2}')
if [ -z "$MF_BEST" ]; then
    echo "[ERROR] No MeanFlow best checkpoint found in ${MF_OUTPUT_DIR}/checkpoints/"
    echo "Falling back to last.ckpt ..."
    MF_BEST="${MF_OUTPUT_DIR}/checkpoints/last.ckpt"
fi
echo "[Stage 2] Best MeanFlow checkpoint: $MF_BEST"

# ════════════════════════════════════════════════════════════════════════
# Stage 3: F2D2 divergence head training (teacher for div targets only)
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "[Stage 3] F2D2 divergence training starting ..."

python proposals/train_div.py \
    --shortcut_ckpt "$MF_BEST" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$DIV_OUTPUT_DIR" \
    --state_dim "$STATE_DIM" \
    --obs_components "$OBS_COMPONENTS" \
    --use_observations \
    --architecture resnet1d \
    --batch_size "$DIV_BATCH_SIZE" \
    --learning_rate "$DIV_LR" \
    --max_epochs "$DIV_MAX_EPOCHS" \
    --denoise_timesteps 1024 \
    --gpus 1 \
    --wandb_project "$WANDB_PROJECT" \
    --teacher_div_estimator exact \
    --div_scale 1.0 \
    --evaluate \
    --eval_num_sampling_steps 4 \
    --freeze_velocity

echo "[Stage 3] F2D2 divergence training done."
echo ""
echo "=========================================="
echo "MeanFlow-F2D2 pipeline complete."
echo "  MeanFlow ckpt:  $MF_BEST"
echo "  F2D2 output:    $DIV_OUTPUT_DIR"
echo "=========================================="
