#!/bin/bash
#
# Same 60-task grid as train_rf_l96_neurips_grid.sbatch but max_epochs=1 for
# pipeline / W&B / OOM checks. Uses a separate W&B project by default.
#
#SBATCH --job-name=rf-l96-neurips-smoke
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-59
##SBATCH --array=0-59%20
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/rf_l96_neurips_smoke_%A_%a.log

set -euo pipefail

# Do not derive paths from ${BASH_SOURCE[0]} under sbatch: Slurm runs a spool copy.
# Allow override for portability, but default to this project's absolute path.
PROJECT_ROOT="${PROJECT_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/DataAssimilation}"
DRIVER_SCRIPT="${PROJECT_ROOT}/scripts/train_rf_l96_neurips_array_driver.sh"

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

export MAX_EPOCHS=1
export WANDB_PROJECT="rf-train-96-neurips-smoke"
export DATA_ROOT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims"
export OUTPUT_ROOT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_l96_neurips_smoke"
export SEED=42

bash "${DRIVER_SCRIPT}"
