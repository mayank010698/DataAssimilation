#!/bin/bash
#
# Ablation 11 (headline): L-FPPF weight_type='full' vs 'likelihood_only'.
#
# Both runs use the same LocalizedRFProposal checkpoint (--proposal-type lrf)
# and the same LocalizedParticleFilter geometry. The only difference is the
# importance-weight formulation:
#   * likelihood_only : classical Farchi & Bocquet tapered observation term
#   * full            : likelihood + per-block per-dim proposal correction
#                       sum_{j in b} [log p(x_j|x_{t-1}) - ell_j]
#
# If "full" significantly improves RMSE / ESS / spatial coherence, the
# per-dim log-density decomposition is a real contribution (this is the
# paper's central claim).
#
# Usage:
#   cd /path/to/DataAssimilation
#   LRF_CHECKPOINT=/path/to/lrf.ckpt sbatch scripts/eval_lfppf_weight_type_ablation.sbatch
#
#SBATCH --job-name="ablation11-lfppf-weight"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/eval.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1

module load anaconda3/2024.10
source activate
conda activate da

export PYTHONPATH="$(pwd)"
mkdir -p logs

# ======================================================================
# Configuration
# ======================================================================

DATA_DIR="${DATA_DIR:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_pnoise0p100_init3p000}"

# REQUIRED: LRF checkpoint trained with proposals/train_lrf.py
LRF_CHECKPOINT="${LRF_CHECKPOINT:-}"
if [ -z "$LRF_CHECKPOINT" ] || [ ! -f "$LRF_CHECKPOINT" ]; then
    echo "ERROR: LRF_CHECKPOINT env var must point to a trained LocalizedRFProposal .ckpt"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data dir not found: $DATA_DIR"
    exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-lfppf-ablations-l96}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

BATCH_SIZE=5
NUM_EVAL_TRAJECTORIES=10
RESAMPLING_THRESHOLD=0.33
MAX_TIMESTEPS=1000
SEED=42

N_PARTICLES=64
PROCESS_NOISE_STD=0.1

RF_SAMPLING_STEPS=25
RF_LIKELIHOOD_STEPS=25

LPF_BLOCK_SIZE=1
LPF_LOC_RADIUS=5.0
LPF_TAPER="gaspari_cohn"
LPF_POST_REG="none"

WEIGHT_TYPES=(likelihood_only full)

echo "=========================================="
echo "Ablation 11: weight_type full vs likelihood_only"
echo "Project root     : $PROJECT_ROOT"
echo "Data             : $DATA_DIR"
echo "LRF checkpoint   : $LRF_CHECKPOINT"
echo "N particles      : $N_PARTICLES"
echo "Block size       : $LPF_BLOCK_SIZE"
echo "Loc radius       : $LPF_LOC_RADIUS"
echo "Timesteps        : $MAX_TIMESTEPS"
echo "=========================================="

pids=()
run_names=()

for WT in "${WEIGHT_TYPES[@]}"; do
    label="ablation11_${WT}_N${N_PARTICLES}_r${LPF_LOC_RADIUS}_${JOB_TAG}"
    log="logs/${label}.log"

    echo "  -> weight_type=${WT} (${label})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method lfppf \
            --proposal-type lrf \
            --rf-checkpoint "$LRF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --lpf-weight-type "$WT" \
            --n-particles "$N_PARTICLES" \
            --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --max-timesteps "$MAX_TIMESTEPS" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --block-size "$LPF_BLOCK_SIZE" \
            --localization-radius "$LPF_LOC_RADIUS" \
            --lpf-taper "$LPF_TAPER" \
            --post-regularization "$LPF_POST_REG" \
            --experiment-label "$label" \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-tags "ablation11,lfppf,weight_type=${WT}" \
            --run-name "$label" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "$log" &
    pids+=("$!")
    run_names+=("$label")
done

echo ""
echo "Waiting for ${#pids[@]} jobs …"
failures=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    run="${run_names[$idx]}"
    if wait "$pid"; then
        echo "  OK   ${run}  (pid=${pid})"
    else
        echo "  FAIL ${run}  (pid=${pid})"
        failures=$((failures + 1))
    fi
done

if [ "$failures" -gt 0 ]; then
    echo "Finished with ${failures} failure(s)."
    exit 1
fi
echo "All experiments completed."
