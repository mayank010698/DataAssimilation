#!/bin/bash
#
#SBATCH --job-name="train-nasmc-l96-dim5"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

# Train NASMC baseline presets on the Lorenz96 5-dimensional dataset.
# Runs SEQUENTIALLY on a single GPU.
#
# Presets (see proposals/train_nasmc.py for details):
#   * gaussian_mle     -- supervised Gaussian-MLE baseline (phase 1 only).
#   * nasmc_warmstart  -- MLE warmstart + NASMC refinement (steelman).
#   * pure_nasmc       -- literal Gu et al. 2015 method with bootstrap warmup.
#
# Hyperparameters match the old RF proposal run so that q_phi has the same
# capacity as the RF teacher on this dataset.
#
# NOTE on batch size (phase 1 vs. phase 2):
#   Phase 1 (local MLE) is a standard one-step supervised task and runs
#   comfortably at BATCH_SIZE=512. Phase 2 (NASMC refine) unrolls an
#   SMC sweep of length SEGMENT_LENGTH with NUM_PARTICLES particles per
#   sample, so each phase-2 example keeps ~segment_length * num_particles
#   times more activations on the GPU. At 512*64*32 that OOMs a 14 GiB
#   card. We therefore use BATCH_SIZE_REFINE=128 for phase 2 only.
#   Learning rate is left at 3e-4: for Adam, a 4x smaller batch does not
#   require the linear-scaling LR drop that SGD would (and if anything,
#   more gradient steps per epoch is a stability boost here).

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
# Reduce GPU fragmentation across the three back-to-back runs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

echo "=========================================="
echo "Training NASMC (3 presets SEQUENTIALLY on 1 GPU) on L96 5-dim"
echo "Project root: $(pwd)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

DATA_DIR=/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_pnoise0p100_init3p000
STATE_DIM=5
OBS_COMPONENTS=0,1,2,3,4
ARCH=resnet1d
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
TIME_EMBED_DIM=128
BATCH_SIZE=512
BATCH_SIZE_REFINE=128
LR=3e-4

# Phase-2 / SMC config.
NUM_PARTICLES=32
SEGMENT_LENGTH=64
RESAMPLE_THRESHOLD=0.5

OUTPUT_BASE=/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/nasmc_runs_96/dim5
JOB_ID="${SLURM_JOB_ID:-manual}"

# Project lives under the ml-climate entity (matching train_rf.py).
WANDB_PROJECT=nasmc-train-96
WANDB_ENTITY=ml-climate

run_preset() {
    local preset="$1"
    local pretrain_epochs="$2"
    local refine_epochs="$3"
    local bootstrap_warmup="$4"

    local output_dir="${OUTPUT_BASE}/run_${JOB_ID}_${preset}"
    local train_log="logs/train_nasmc_l96_dim5_${JOB_ID}_${preset}.log"
    local run_name="nasmc_l96_dim5_${preset}_job${JOB_ID}"

    echo "------------------------------------------"
    echo "Running preset=${preset}  pretrain=${pretrain_epochs}  refine=${refine_epochs}  warmup=${bootstrap_warmup}"
    echo "  output_dir=${output_dir}"
    echo "  train_log =${train_log}"
    echo "------------------------------------------"

    python -m proposals.train_nasmc \
        --preset "$preset" \
        --data_dir "$DATA_DIR" \
        --output_dir "$output_dir" \
        --state_dim "$STATE_DIM" \
        --obs_components "$OBS_COMPONENTS" \
        --use_observations \
        --architecture "$ARCH" \
        --channels "$CHANNELS" \
        --num_blocks "$NUM_BLOCKS" \
        --kernel_size "$KERNEL_SIZE" \
        --time_embed_dim "$TIME_EMBED_DIM" \
        --predict_delta \
        --batch_size "$BATCH_SIZE" \
        --batch_size_refine "$BATCH_SIZE_REFINE" \
        --learning_rate "$LR" \
        --max_epochs_pretrain "$pretrain_epochs" \
        --max_epochs_refine "$refine_epochs" \
        --bootstrap_warmup_epochs "$bootstrap_warmup" \
        --num_particles "$NUM_PARTICLES" \
        --segment_length "$SEGMENT_LENGTH" \
        --resample_threshold "$RESAMPLE_THRESHOLD" \
        --num_workers 4 \
        --gpus 1 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_run_name "$run_name" \
        --seed 0 \
        > "$train_log" 2>&1
}

# Phase-count picks:
#   - gaussian_mle matches the old RF run's 500 supervised epochs.
#   - nasmc_warmstart splits a comparable compute budget between the
#     supervised warmstart (250 epochs) and the SMC refinement (100 epochs).
#   - pure_nasmc gets more refine epochs since it has no warmstart; the
#     first 10 epochs use the bootstrap proposal per Gu et al. 2015.
#
# Runs are sequential; if one fails, we continue to the next and propagate
# the failure via EXIT_CODE at the end.
#
# gaussian_mle already completed successfully in a previous submission
# (phase 1 only, BATCH_SIZE=512 was fine). Re-enable the line below if you
# need to rerun it -- the phase-1 config has not changed.
EXIT_CODE=0

# run_preset gaussian_mle     500 0   0   || { echo "gaussian_mle failed ($?)"; EXIT_CODE=1; }
run_preset nasmc_warmstart  250 100 0   || { echo "nasmc_warmstart failed ($?)"; EXIT_CODE=1; }
run_preset pure_nasmc       0   200 10  || { echo "pure_nasmc failed ($?)"; EXIT_CODE=1; }

echo "=========================================="
echo "All NASMC training runs complete (exit ${EXIT_CODE})."
echo "Output root: ${OUTPUT_BASE}"
echo "Wandb:       https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "=========================================="

exit "$EXIT_CODE"
