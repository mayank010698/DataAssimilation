#!/bin/bash
#
# Ablation 13: L-FPPF robustness to process noise.
#
# Sweeps process_noise_std ∈ {0.01, 0.05, 0.1, 0.2} on L96-40 with the
# lfppf method + lrf proposal (weight_type=full).
#
# Usage:
#   LRF_CHECKPOINT=/path/to/lrf.ckpt sbatch scripts/eval_lfppf_pnoise_ablation.sbatch
#
# NOTE: This is distinct from the existing `eval_lfppf_pnoise_sweep.sbatch`,
# which actually runs --method lpf with a GLOBAL RF proposal and was
# misnamed. Use this script for true L-FPPF process-noise ablations.
#
#SBATCH --job-name="ablation13-pnoise"
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
RF_SAMPLING_STEPS=25
RF_LIKELIHOOD_STEPS=25
LPF_TAPER="gaspari_cohn"
LPF_BLOCK_SIZE=1
LPF_LOC_RADIUS=5.0

PROCESS_NOISES=(0.01 0.05 0.1 0.2)

pids=()
run_names=()

for PN in "${PROCESS_NOISES[@]}"; do
    label="ablation13_pnoise${PN}_${JOB_TAG}"
    log="logs/${label}.log"
    echo "  -> process_noise_std=${PN}"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method lfppf --proposal-type lrf \
            --rf-checkpoint "$LRF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --n-particles "$N_PARTICLES" --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PN" \
            --max-timesteps "$MAX_TIMESTEPS" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --block-size "$LPF_BLOCK_SIZE" \
            --localization-radius "$LPF_LOC_RADIUS" \
            --lpf-taper "$LPF_TAPER" \
            --experiment-label "$label" --wandb-project "$WANDB_PROJECT" \
            --wandb-tags "ablation13,lfppf,pnoise=${PN}" \
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
