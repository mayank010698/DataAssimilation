#!/bin/bash
#
#SBATCH --job-name="reobs-l96-obs0p200"
#SBATCH --partition=dali
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=2
#SBATCH --mem=96g
#SBATCH --time=06:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

# Add a new obs_noise_std=0.2 variation to every existing Lorenz-96 dataset
# under l96_dims/, for the dims listed in DIMS and all 3 observation
# operators (identity, arctan, quad_capped_10) and both process-noise variants
# (deterministic + pnoise=0.1). Trajectories are reused from the existing
# obs0p100_* sources via reobserve_obs_noise.py — no dynamics integration.
# That script also dedupes the 5 sigma source variants per (d, H, pnoise)
# cell down to a single re-observe call (canonical = obs0p100_*), so each
# per-dim task does exactly 6 dataset writes.
#
# CPU-only; observation step is numpy + torch elementwise + h5py I/O.

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

export PYTHONPATH="$(pwd)"

echo "=========================================="
echo "Re-observing Lorenz-96 datasets at obs_noise_std=0.2"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

# Keep this list length in sync with #SBATCH --ntasks.
DIMS=(5 10 15 20 25 50 100 500 1000)
DATASETS_ROOT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims"
TARGET_NOISE="${TARGET_NOISE:-0.2}"
SEED="${SEED:-42}"

if [ ! -d "$DATASETS_ROOT" ]; then
    echo "ERROR: DATASETS_ROOT does not exist: $DATASETS_ROOT"; exit 1
fi

EXTRA_ARGS=()
if [ "${FORCE:-0}" = "1" ]; then EXTRA_ARGS+=(--force); fi
if [ "${DRY_RUN:-0}" = "1" ]; then EXTRA_ARGS+=(--dry-run); fi

for DIM in "${DIMS[@]}"; do
    # Match every observation operator (identity / arctan / quad_capped_10)
    # and both pnoise variants for this dim. The source-dedupe inside
    # reobserve_obs_noise.py picks the obs0p100_* dir as the canonical
    # trajectory provider, so each dim ends up writing exactly 6 new
    # destinations: 3 operators x {deterministic, pnoise=0.1}.
    PATTERN="lorenz96_*comp${DIM}of${DIM}_*"
    echo "[dim=${DIM}] launching with pattern='${PATTERN}'..."
    srun --exclusive -N1 -n1 -c"${SLURM_CPUS_PER_TASK:-2}" --chdir "$(pwd)" \
        python "$(pwd)/reobserve_obs_noise.py" \
            --src-root "$DATASETS_ROOT" \
            --target-noise "$TARGET_NOISE" \
            --pattern "$PATTERN" \
            --seed "$SEED" \
            "${EXTRA_ARGS[@]}" &
done

wait

echo ""
echo "=========================================="
echo "Re-observation completed (target_noise=${TARGET_NOISE})."
echo "=========================================="
