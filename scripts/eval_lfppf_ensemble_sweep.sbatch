#!/bin/bash
#
# Ablation 12: L-FPPF vs global FPPF at matched ensemble sizes.
#
# Sweeps N_particles ∈ {16, 32, 64, 128} for:
#   * L-FPPF (method=lfppf, proposal=lrf, weight_type=full)
#   * global FPPF baseline (method=bpf, proposal=rf) -- requires --rf-checkpoint
#
# Headline claim: global FPPF collapses at small N on large N_x; L-FPPF
# maintains RMSE/ESS because local weights avoid weight collapse.
#
# Usage:
#   LRF_CHECKPOINT=/path/to/lrf.ckpt \
#   GLOBAL_RF_CHECKPOINT=/path/to/rf.ckpt \
#   sbatch scripts/eval_lfppf_ensemble_sweep.sbatch
#
#SBATCH --job-name="ablation12-ensemble-sweep"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
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
GLOBAL_RF_CHECKPOINT="${GLOBAL_RF_CHECKPOINT:-}"

if [ -z "$LRF_CHECKPOINT" ] || [ ! -f "$LRF_CHECKPOINT" ]; then
    echo "ERROR: LRF_CHECKPOINT must be set."; exit 1
fi
if [ -z "$GLOBAL_RF_CHECKPOINT" ] || [ ! -f "$GLOBAL_RF_CHECKPOINT" ]; then
    echo "WARN: GLOBAL_RF_CHECKPOINT not set — skipping global-FPPF baseline."
fi

WANDB_PROJECT="${WANDB_PROJECT:-lfppf-ablations-l96}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

BATCH_SIZE=5
NUM_EVAL_TRAJECTORIES=10
RESAMPLING_THRESHOLD=0.33
MAX_TIMESTEPS=1000
SEED=42
PROCESS_NOISE_STD=0.1
RF_SAMPLING_STEPS=25
RF_LIKELIHOOD_STEPS=25
LPF_BLOCK_SIZE=1
LPF_LOC_RADIUS=5.0
LPF_TAPER="gaspari_cohn"

ENSEMBLE_SIZES=(16 32 64 128)

pids=()
run_names=()

for N in "${ENSEMBLE_SIZES[@]}"; do
    # --- L-FPPF ---
    label="ablation12_lfppf_N${N}_${JOB_TAG}"
    log="logs/${label}.log"
    echo "  -> L-FPPF N=${N}"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method lfppf --proposal-type lrf \
            --rf-checkpoint "$LRF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --n-particles "$N" --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --max-timesteps "$MAX_TIMESTEPS" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --block-size "$LPF_BLOCK_SIZE" \
            --localization-radius "$LPF_LOC_RADIUS" \
            --lpf-taper "$LPF_TAPER" \
            --experiment-label "$label" --wandb-project "$WANDB_PROJECT" \
            --wandb-tags "ablation12,lfppf,N=${N}" \
            --run-name "$label" --seed "$SEED" --device cuda
    ) 2>&1 | tee "$log" &
    pids+=("$!")
    run_names+=("$label")

    # --- global FPPF baseline (if available) ---
    if [ -n "$GLOBAL_RF_CHECKPOINT" ] && [ -f "$GLOBAL_RF_CHECKPOINT" ]; then
        label_g="ablation12_fppf_N${N}_${JOB_TAG}"
        log_g="logs/${label_g}.log"
        echo "  -> global FPPF N=${N}"
        (
            CUDA_VISIBLE_DEVICES=0 python eval.py \
                --data-dir "$DATA_DIR" \
                --method bpf --proposal-type rf \
                --rf-checkpoint "$GLOBAL_RF_CHECKPOINT" \
                --rf-sampling-steps "$RF_SAMPLING_STEPS" \
                --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
                --n-particles "$N" --batch-size "$BATCH_SIZE" \
                --resampling-threshold "$RESAMPLING_THRESHOLD" \
                --process-noise-std "$PROCESS_NOISE_STD" \
                --max-timesteps "$MAX_TIMESTEPS" \
                --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
                --experiment-label "$label_g" --wandb-project "$WANDB_PROJECT" \
                --wandb-tags "ablation12,fppf,N=${N}" \
                --run-name "$label_g" --seed "$SEED" --device cuda
        ) 2>&1 | tee "$log_g" &
        pids+=("$!")
        run_names+=("$label_g")
    fi
done

echo "Waiting for ${#pids[@]} jobs …"
failures=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"; run="${run_names[$idx]}"
    if wait "$pid"; then echo "  OK   ${run}"; else echo "  FAIL ${run}"; failures=$((failures+1)); fi
done
if [ "$failures" -gt 0 ]; then echo "Finished with ${failures} failure(s)."; exit 1; fi
echo "All experiments completed."
