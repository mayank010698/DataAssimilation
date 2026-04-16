#!/bin/bash
#
# Ablation 16: L-FPPF localization-radius sweep.
#
# The proposal's receptive field r (baked into the LRF checkpoint) is fixed
# per checkpoint; this script only sweeps the PF taper radius r_loc and
# the block size. For a proposal-radius sweep, train multiple LRF checkpoints
# via scripts/train_lrf_l96.sbatch (RADIUS env override).
#
# Usage:
#   LRF_CHECKPOINT=/path/to/lrf.ckpt sbatch scripts/eval_lfppf_radius_sweep.sbatch
#
#SBATCH --job-name="ablation16-radius-sweep"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=06:00:00
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

DATA_DIR="${DATA_DIR:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_pnoise0p100_init3p000}"
LRF_CHECKPOINT="${LRF_CHECKPOINT:-}"
if [ -z "$LRF_CHECKPOINT" ] || [ ! -f "$LRF_CHECKPOINT" ]; then
    echo "ERROR: LRF_CHECKPOINT must be set."; exit 1
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
LPF_TAPER="gaspari_cohn"
LPF_BLOCK_SIZE=1

LOC_RADII=(3.0 4.0 5.0 6.0 8.0)

pids=()
run_names=()

for rloc in "${LOC_RADII[@]}"; do
    label="ablation16_rloc${rloc}_${JOB_TAG}"
    log="logs/${label}.log"
    echo "  -> r_loc=${rloc}"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method lfppf --proposal-type lrf \
            --rf-checkpoint "$LRF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --n-particles "$N_PARTICLES" --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --max-timesteps "$MAX_TIMESTEPS" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --block-size "$LPF_BLOCK_SIZE" \
            --localization-radius "$rloc" \
            --lpf-taper "$LPF_TAPER" \
            --experiment-label "$label" --wandb-project "$WANDB_PROJECT" \
            --wandb-tags "ablation16,lfppf,rloc=${rloc}" \
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
echo "All experiments completed."
