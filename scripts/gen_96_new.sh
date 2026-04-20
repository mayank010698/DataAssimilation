#!/bin/bash
#
#SBATCH --job-name="gen-l96-multidim"
#SBATCH --partition=dali
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

set -euo pipefail

# In Slurm, use the original submit directory; fallback to script-relative path.
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

# Setup PYTHONPATH
export PYTHONPATH="$(pwd)"

echo "=========================================="
echo "Generating Lorenz 96 Datasets (Multi-dim)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

# Keep this list length in sync with #SBATCH --ntasks
# DIMS=(5 10 15 20 25 30 40 50)
DIMS=(5 10 15 20 25 50 100 500 1000)
OUTPUT_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims"

mkdir -p "$OUTPUT_DIR"

for DIM in "${DIMS[@]}"; do
    echo "Launching dimension ${DIM} on its own CPU task..."
    srun --exclusive -N1 -n1 -c1 --chdir "$(pwd)" python "$(pwd)/generate.py" \
        --system lorenz96 \
        --l96-dim "$DIM" \
        --process-noise-variations "0.0,0.1" \
        --obs-noise-variations "0.2,0.5,1,3,5" \
        --num-trajectories 2048 \
        --len-trajectory 200 \
        --obs-frequency 1 \
        --observation-operators "identity,arctan,quad_capped_10" \
        --output-dir "$OUTPUT_DIR" \
        --seed 42 \
        --force &
done

wait

echo ""
echo "=========================================="
echo "Dataset generation completed."
echo "=========================================="
