#!/bin/bash
#
#SBATCH --job-name="rf-fm96-dim5"
#SBATCH --partition=dali
#SBATCH --account=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80g
#SBATCH --time=48:00:00
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

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_pnoise0p100_init3p000"
OBS_COMPONENTS="0,1,2,3,4"

WANDB_PROJECT="rf-train-96-dim"

STATE_DIM=5
BATCH_SIZE=512
LR=3e-4
MAX_EPOCHS=500
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
COND_EMBED_DIM=128

MASK_P0=0.3
MASK_PMIN=0.05
MASK_RATIO=0.4
MASK_SIGMA=0.0

# Stage 2 shortcut (teacher = FM dim5 nomask EMA from this same job batch)
SC_BATCH_SIZE=256
SC_LR=1e-4
SC_MAX_EPOCHS=300

JOB_TAG="${SLURM_JOB_ID:-manual}"
RUN_BASE="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims"
WANDB_PROJECT_SC="${WANDB_PROJECT}"

echo "=========================================="
echo "RF/FM + SC dim5 parallel training launch (5 jobs)"
echo "Project root: $(pwd)"
echo "Data dir: $DATA_DIR"
echo "WandB project: $WANDB_PROJECT (shortcut: $WANDB_PROJECT_SC)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

COMMON_RF_ARGS=(
    --data_dir "$DATA_DIR"
    --state_dim "$STATE_DIM"
    --obs_components "$OBS_COMPONENTS"
    --architecture resnet1d
    --channels "$CHANNELS"
    --num_blocks "$NUM_BLOCKS"
    --kernel_size "$KERNEL_SIZE"
    --train-cond-method adaln
    --cond_embed_dim "$COND_EMBED_DIM"
    --use_observations
    --batch_size "$BATCH_SIZE"
    --learning_rate "$LR"
    --max_epochs "$MAX_EPOCHS"
    --gpus 1
    --wandb_project "$WANDB_PROJECT"
    --evaluate
)

COMMON_FM_ARGS=(
    --data_dir "$DATA_DIR"
    --state_dim "$STATE_DIM"
    --obs_components "$OBS_COMPONENTS"
    --architecture resnet1d
    --channels "$CHANNELS"
    --num_blocks "$NUM_BLOCKS"
    --kernel_size "$KERNEL_SIZE"
    --train_cond_method adaln
    --cond_embed_dim "$COND_EMBED_DIM"
    --use_observations
    --batch_size "$BATCH_SIZE"
    --learning_rate "$LR"
    --max_epochs "$MAX_EPOCHS"
    --gpus 1
    --wandb_project "$WANDB_PROJECT"
    --evaluate
)

MASK_ARGS=(
    --prev_state_corr_p0 "$MASK_P0"
    --prev_state_corr_p_min "$MASK_PMIN"
    --prev_state_corr_mask_ratio "$MASK_RATIO"
    --prev_state_corr_sigma "$MASK_SIGMA"
)

declare -a JOB_PIDS
declare -a JOB_NAMES

launch_job() {
    local name="$1"
    shift
    local logfile="logs/${name}_${JOB_TAG}.log"
    echo "Launching: $name"
    CUDA_VISIBLE_DEVICES=0 nohup "$@" > "$logfile" 2>&1 &
    local pid=$!
    JOB_PIDS+=("$pid")
    JOB_NAMES+=("$name")
    echo "  PID: $pid, log: $logfile"
}

# launch_job "rf_dim5_masked" \
#     python proposals/train_rf.py \
#     "${COMMON_RF_ARGS[@]}" \
#     --output_dir "$RUN_BASE/rf_dim5_masked_${JOB_TAG}" \
#     "${MASK_ARGS[@]}"

# launch_job "rf_dim5_nomask" \
#     python proposals/train_rf.py \
#     "${COMMON_RF_ARGS[@]}" \
#     --output_dir "$RUN_BASE/rf_dim5_nomask_${JOB_TAG}"

# launch_job "fm_dim5_masked" \
#     python proposals/train_fm.py \
#     "${COMMON_FM_ARGS[@]}" \
#     --output_dir "$RUN_BASE/fm_dim5_masked_${JOB_TAG}" \
#     "${MASK_ARGS[@]}"

# launch_job "fm_dim5_nomask" \
#     python proposals/train_fm.py \
#     "${COMMON_FM_ARGS[@]}" \
#     --output_dir "$RUN_BASE/fm_dim5_nomask_${JOB_TAG}"

# Waits for fm_dim5_nomask EMA (written when that run finishes), then trains shortcut + autoregressive eval.
# override that
launch_job "sc_dim5_fm_nomask" bash -c "
set -euo pipefail
cd '${PROJECT_ROOT}'
export PYTHONPATH='${PROJECT_ROOT}'
# TEACHER='${RUN_BASE}/fm_dim5_nomask_${JOB_TAG}/checkpoints/ema_weights.ckpt'
TEACHER="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims/fm_dim5_nomask_7850189/checkpoints/ema_weights.ckpt"

until [[ -f \"\$TEACHER\" ]]; do
  echo \"[sc_dim5_fm_nomask] Waiting for teacher EMA: \$TEACHER\"
  sleep 90
done
exec python proposals/train_sc.py \
  --teacher_ckpt \"\$TEACHER\" \
  --data_dir '${DATA_DIR}' \
  --output_dir '${RUN_BASE}/sc_dim5_fm_nomask_${JOB_TAG}' \
  --state_dim ${STATE_DIM} \
  --obs_components '${OBS_COMPONENTS}' \
  --use_observations \
  --architecture resnet1d \
  --channels ${CHANNELS} \
  --num_blocks ${NUM_BLOCKS} \
  --kernel_size ${KERNEL_SIZE} \
  --time_embed_dim 64 \
  --batch_size ${SC_BATCH_SIZE} \
  --learning_rate ${SC_LR} \
  --max_epochs ${SC_MAX_EPOCHS} \
  --gpus 1 \
  --wandb_project '${WANDB_PROJECT_SC}' \
  --evaluate
  --eval_num_sampling_steps 4
"

echo "=========================================="
echo "All 5 jobs launched on CUDA_VISIBLE_DEVICES=0"
echo "Waiting for completion..."
echo "=========================================="

FAILED=0
for i in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[$i]}"
    name="${JOB_NAMES[$i]}"
    if wait "$pid"; then
        echo "[OK] $name finished successfully"
    else
        echo "[FAIL] $name exited with non-zero status"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "One or more jobs failed."
    exit 1
fi

echo "All jobs completed successfully."
