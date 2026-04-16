#!/bin/bash
#
# Ablation 15: Zero-shot transfer of an L-FPPF proposal trained on L96-25 to
# larger L96 state dims (N_x ∈ {50, 100, 500, 1000}).
#
# The patch-based velocity network has a bounded receptive field, so all
# parameters are independent of the global state dim. Passing
# --lrf-state-dim-override lets the wrapper carry the new state_dim through
# its API; the heavy lifting is handled by the PF using the target dataset.
#
# Target datasets are L96 pnoise variants matching the L96-25 training
# distribution (same obs_std, same process noise, just different N_x).
#
# Usage:
#   LRF_CHECKPOINT=/path/to/lrf.ckpt sbatch scripts/eval_lfppf_zeroshot.sbatch
#
#SBATCH --job-name="ablation15-zeroshot"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=08:00:00
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

DATA_ROOT="${DATA_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims}"
LRF_CHECKPOINT="${LRF_CHECKPOINT:-}"
if [ -z "$LRF_CHECKPOINT" ] || [ ! -f "$LRF_CHECKPOINT" ]; then
    echo "ERROR: LRF_CHECKPOINT must be set."; exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-lfppf-ablations-l96}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

BATCH_SIZE=2
NUM_EVAL_TRAJECTORIES=5
RESAMPLING_THRESHOLD=0.33
MAX_TIMESTEPS=500
SEED=42
N_PARTICLES=64
PROCESS_NOISE_STD=0.1
RF_SAMPLING_STEPS=25
RF_LIKELIHOOD_STEPS=25
LPF_TAPER="gaspari_cohn"
LPF_BLOCK_SIZE=1
LPF_LOC_RADIUS=5.0

TARGET_DIMS=(50 100 500 1000)

pids=()
run_names=()

for DIM in "${TARGET_DIMS[@]}"; do
    DATA_DIR="${DATA_ROOT}/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp${DIM}of${DIM}_arctan_pnoise0p100_init3p000"
    if [ ! -d "$DATA_DIR" ]; then
        echo "WARN: dataset not found for dim=${DIM} (${DATA_DIR}) — skipping"
        continue
    fi

    label="ablation15_zeroshot_N${DIM}_${JOB_TAG}"
    log="logs/${label}.log"
    echo "  -> dim=${DIM} (data: ${DATA_DIR})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method lfppf --proposal-type lrf \
            --rf-checkpoint "$LRF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --lrf-state-dim-override "$DIM" \
            --n-particles "$N_PARTICLES" --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --max-timesteps "$MAX_TIMESTEPS" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --block-size "$LPF_BLOCK_SIZE" \
            --localization-radius "$LPF_LOC_RADIUS" \
            --lpf-taper "$LPF_TAPER" \
            --experiment-label "$label" --wandb-project "$WANDB_PROJECT" \
            --wandb-tags "ablation15,lfppf,zeroshot,dim=${DIM}" \
            --run-name "$label" --seed "$SEED" --device cuda
    ) 2>&1 | tee "$log" &
    pids+=("$!")
    run_names+=("$label")
done

echo "Waiting for ${#pids[@]} jobs …"
failures=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"; run="${run_names[$idx]}"
    if wait "$pid"; then echo "  OK   ${run}"; else echo "  FAIL ${run}"; failures=$((failures+1)); fi
done
if [ "$failures" -gt 0 ]; then echo "Finished with ${failures} failure(s)."; exit 1; fi
echo "All zero-shot runs completed."
