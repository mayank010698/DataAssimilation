#!/bin/bash
#
#SBATCH --job-name="train-lrf-l96"
#SBATCH --partition=dali
#SBATCH --account=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
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

echo "=========================================="
echo "Training Localized Rectified Flow on L96 (4 parallel runs sharing 1 GPU)"
echo "Project root: $(pwd)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

DATA_DIR=/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_pnoise0p100_init3p000
STATE_DIM=25
ARCH=local_resnet1d
WANDB_PROJECT=lrf-train-96
TRAIN_FRACTION=0.5
JOB_ID="${SLURM_JOB_ID:-manual}"

run_experiment() {
    local radius="$1"
    local lr="$2"
    local tag="$3"

    local output_dir="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/lrf_runs/run_${JOB_ID}_${tag}"
    local train_log="logs/train_lrf_l96_${JOB_ID}_${tag}.log"
    local run_name="lrf_l96_${tag}_job${JOB_ID}"

    echo "[shared GPU] Launching ${tag}: radius=${radius}, lr=${lr} -> ${train_log}"

    python -m proposals.train_lrf \
        --data_dir "$DATA_DIR" \
        --output_dir "$output_dir" \
        --radius "$radius" \
        --state_dim "$STATE_DIM" \
        --architecture "$ARCH" \
        --hidden_dim 128 \
        --depth 4 \
        --channels 64 \
        --num_blocks 4 \
        --kernel_size 3 \
        --time_embed_dim 64 \
        --batch_size 1024 \
        --learning_rate "$lr" \
        --max_epochs 200 \
        --train_fraction "$TRAIN_FRACTION" \
        --num_workers 4 \
        --use_observations \
        --predict_delta \
        --num_sampling_steps 100 \
        --num_likelihood_steps 100 \
        --obs_dropout 0.1 \
        --early_stopping_patience 50 \
        --gpus 1 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$run_name" \
        --evaluate \
        > "$train_log" 2>&1
}

run_experiment 3 5e-4 r3_lr5e-4 &
PID0=$!
run_experiment 3 1e-3 r3_lr1e-3 &
PID1=$!
run_experiment 4 5e-4 r4_lr5e-4 &
PID2=$!
run_experiment 4 1e-3 r4_lr1e-3 &
PID3=$!

echo "Launched PIDs: $PID0 $PID1 $PID2 $PID3"

EXIT_CODE=0
for pid in $PID0 $PID1 $PID2 $PID3; do
    if ! wait "$pid"; then
        echo "Run with PID $pid failed (exit $?)"
        EXIT_CODE=1
    fi
done

echo "=========================================="
echo "All training runs complete (exit ${EXIT_CODE})."
echo "=========================================="

exit "$EXIT_CODE"
