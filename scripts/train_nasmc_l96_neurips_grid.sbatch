#!/bin/bash
#
# 24-way job array: dims {5,10,15,20,25,50} x obs operators {arctan, quad_capped_10}
# x NASMC presets {pure_nasmc_K1, pure_nasmc} at a single obs noise level (0.2).
#
# Two proposals per dataset:
#   * pure_nasmc_K1 -- literal Gu et al. 2015 NASMC, K=1 single Gaussian head
#                      (this is what the dim=5 pilot run in
#                      train_nasmc_l96_dim5.sbatch trained; the MDN head was
#                      added to the codebase afterwards).
#   * pure_nasmc    -- same NASMC objective / schedule but with a K=3 mixture
#                      of diagonal Gaussians (the paper's "-MD-" variant,
#                      implemented in proposals/architectures/gaussian_head.py).
#
# Mirrors scripts/train_rf_l96_neurips_grid.sbatch in layout (sbatch wrapper +
# separate array driver) so tasks are independent and restartable.
#
#SBATCH --job-name=nasmc-l96-neurips
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-23
##SBATCH --array=0-23
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%A_%a.log

set -euo pipefail

# Do not derive paths from ${BASH_SOURCE[0]} under sbatch: Slurm runs a spool copy.
# Allow override for portability, but default to this project's absolute path.
PROJECT_ROOT="${PROJECT_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/DataAssimilation}"
DRIVER_SCRIPT="${PROJECT_ROOT}/scripts/train_nasmc_l96_neurips_array_driver.sh"

if [[ ! -f "${DRIVER_SCRIPT}" ]]; then
    echo "Missing array driver script: ${DRIVER_SCRIPT}" >&2
    echo "Set PROJECT_ROOT to your repo checkout before submitting." >&2
    exit 1
fi

cd "${PROJECT_ROOT}" || exit 1

module load anaconda3/2024.10
source activate
conda activate da
module load cuda/12.8

export PYTHONPATH="${PROJECT_ROOT}"
# Reduce GPU fragmentation across phase-1 / phase-2 transitions.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fixed obs noise for this grid (matches the NeurIPS subset the user asked for).
export OBS_NOISE=0.2

# Training budget matches the dim=5 pure_nasmc pilot in train_nasmc_l96_dim5.sbatch:
#   pretrain=0, refine=200, bootstrap_warmup=10.
# Set here so both presets (K=1 and K=3) get the same schedule and compute.
export MAX_EPOCHS_PRETRAIN=0
export MAX_EPOCHS_REFINE=200
export BOOTSTRAP_WARMUP_EPOCHS=10

export WANDB_PROJECT="nasmc-train-96-neurips"
export WANDB_ENTITY="ml-climate"
export DATA_ROOT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims"
export OUTPUT_ROOT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/nasmc_runs_l96_neurips"
export SEED=42

bash "${DRIVER_SCRIPT}"
