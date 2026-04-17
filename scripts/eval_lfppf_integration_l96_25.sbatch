#!/bin/bash
#
# Integration test for L-FPPF on L96-25: 1000-step run with L-FPPF and
# three baselines. Meant to validate end-to-end wiring before ablations.
#
#   1) L-FPPF              : --method lfppf --proposal-type lrf
#                            (localized proposal, weight_type='full')
#   2) LPF + global RF     : --method lpf   --proposal-type rf
#                            (global proposal, weight_type='likelihood_only')
#                            -> isolates effect of local-vs-global proposal
#   3) L-BPF               : --method lpf   --proposal-type transition
#                            (no learned proposal, weight_type='likelihood_only')
#                            -> isolates effect of learned proposal at all
#   4) global FPPF         : --method bpf   --proposal-type rf
#                            -> isolates effect of localization
#
# Required:
#   LRF_CHECKPOINT         — trained LocalizedRFProposal checkpoint
# Optional:
#   GLOBAL_RF_CHECKPOINT   — global RectifiedFlowProposal checkpoint; enables
#                            arms (2) and (4) when provided.
#
# Note on predict_delta:
#   The LRF model's predict_delta flag is restored from the checkpoint's
#   saved hyperparameters (see LocalizedRFProposalWrapper -> LocalizedRFProposal
#   .load_from_checkpoint in eval.py). There is no CLI override for it here,
#   so the mode is fixed by whichever LRF_CHECKPOINT is passed in. If the LRF
#   is trained with predict_delta=True while GLOBAL_RF_CHECKPOINT is
#   predict_delta=False (current situation), arms (1) vs (4) mix a
#   "local vs global proposal" contrast with a "delta vs absolute target"
#   contrast; interpret that comparison accordingly.
#
# Usage:
#   LRF_CHECKPOINT=/path/to/lrf.ckpt \
#   GLOBAL_RF_CHECKPOINT=/path/to/rf.ckpt \
#   sbatch scripts/eval_lfppf_integration_l96_25.sbatch
#
#SBATCH --job-name="lfppf-integration"
#SBATCH --partition=dali
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

DATA_DIR="${DATA_DIR:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_init3p000}"
LRF_CHECKPOINT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/lrf_runs/run_8092322/checkpoints/lrf-epoch=062-val_loss=0.022068.ckpt"
GLOBAL_RF_CHECKPOINT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96/l96_noise_dim25_20260109_193733/checkpoints/rf-epoch=067-val_loss=0.020633.ckpt"

if [ -z "$LRF_CHECKPOINT" ] || [ ! -f "$LRF_CHECKPOINT" ]; then
    echo "ERROR: LRF_CHECKPOINT must be set."; exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-lfppf-ablations-l96}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

BATCH_SIZE=10
NUM_EVAL_TRAJECTORIES=10
RESAMPLING_THRESHOLD=0.33
MAX_TIMESTEPS=100
SEED=42
N_PARTICLES=500
PROCESS_NOISE_STD=0.2
RF_SAMPLING_STEPS=50
RF_LIKELIHOOD_STEPS=50
LPF_TAPER="gaussian"
LPF_BLOCK_SIZE=1
LPF_LOC_RADIUS=5.0

pids=()
run_names=()

# --- L-FPPF ---
label="lfppf_integration_L96_25_N${N_PARTICLES}_${JOB_TAG}"
log="logs/${label}.log"
echo "  -> L-FPPF (weight_type=full)"
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
        --localization-radius "$LPF_LOC_RADIUS" \
        --lpf-taper "$LPF_TAPER" \
        --experiment-label "$label" --wandb-project "$WANDB_PROJECT" \
        --wandb-tags "integration,lfppf,L96_25" \
        --run-name "$label" --seed "$SEED" --device cuda
) 2>&1 | tee "$log" &
pids+=("$!")
run_names+=("$label")

# --- L-BPF baseline (no RF proposal) ---
label2="lbpf_integration_L96_25_N${N_PARTICLES}_${JOB_TAG}"
log2="logs/${label2}.log"
echo "  -> L-BPF baseline"
(
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --data-dir "$DATA_DIR" \
        --method lpf --proposal-type transition \
        --n-particles "$N_PARTICLES" --batch-size "$BATCH_SIZE" \
        --resampling-threshold "$RESAMPLING_THRESHOLD" \
        --process-noise-std "$PROCESS_NOISE_STD" \
        --max-timesteps "$MAX_TIMESTEPS" \
        --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
        --block-size "$LPF_BLOCK_SIZE" \
        --localization-radius "$LPF_LOC_RADIUS" \
        --lpf-taper "$LPF_TAPER" \
        --experiment-label "$label2" --wandb-project "$WANDB_PROJECT" \
        --wandb-tags "integration,lbpf,L96_25" \
        --run-name "$label2" --seed "$SEED" --device cuda
) 2>&1 | tee "$log2" &
pids+=("$!")
run_names+=("$label2")

# --- LPF + global RF proposal, weight_type=likelihood_only (optional) ---
# Isolates the effect of "localized vs global proposal" while keeping the
# weight formulation fixed to the standard tapered-likelihood LPF. Compare
# against L-FPPF (arm 1) to attribute any gain specifically to the learned
# local proposal, not just to "any RF proposal".
if [ -n "$GLOBAL_RF_CHECKPOINT" ] && [ -f "$GLOBAL_RF_CHECKPOINT" ]; then
    label_lpf_grf="lpf_globalRF_L96_25_N${N_PARTICLES}_${JOB_TAG}"
    log_lpf_grf="logs/${label_lpf_grf}.log"
    echo "  -> LPF + global RF (likelihood_only)"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method lpf --proposal-type rf \
            --rf-checkpoint "$GLOBAL_RF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --lpf-weight-type likelihood_only \
            --n-particles "$N_PARTICLES" --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --max-timesteps "$MAX_TIMESTEPS" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --block-size "$LPF_BLOCK_SIZE" \
            --localization-radius "$LPF_LOC_RADIUS" \
            --lpf-taper "$LPF_TAPER" \
            --experiment-label "$label_lpf_grf" --wandb-project "$WANDB_PROJECT" \
            --wandb-tags "integration,lpf_globalRF,likelihood_only,L96_25" \
            --run-name "$label_lpf_grf" --seed "$SEED" --device cuda
    ) 2>&1 | tee "$log_lpf_grf" &
    pids+=("$!")
    run_names+=("$label_lpf_grf")
fi

# --- global FPPF baseline (optional) ---
if [ -n "$GLOBAL_RF_CHECKPOINT" ] && [ -f "$GLOBAL_RF_CHECKPOINT" ]; then
    label3="fppf_integration_L96_25_N${N_PARTICLES}_${JOB_TAG}"
    log3="logs/${label3}.log"
    echo "  -> global FPPF baseline"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$DATA_DIR" \
            --method bpf --proposal-type rf \
            --rf-checkpoint "$GLOBAL_RF_CHECKPOINT" \
            --rf-sampling-steps "$RF_SAMPLING_STEPS" \
            --rf-likelihood-steps "$RF_LIKELIHOOD_STEPS" \
            --n-particles "$N_PARTICLES" --batch-size "$BATCH_SIZE" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --process-noise-std "$PROCESS_NOISE_STD" \
            --max-timesteps "$MAX_TIMESTEPS" \
            --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
            --experiment-label "$label3" --wandb-project "$WANDB_PROJECT" \
            --wandb-tags "integration,fppf,L96_25" \
            --run-name "$label3" --seed "$SEED" --device cuda
    ) 2>&1 | tee "$log3" &
    pids+=("$!")
    run_names+=("$label3")
fi

echo "Waiting for ${#pids[@]} jobs …"
failures=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"; run="${run_names[$idx]}"
    if wait "$pid"; then echo "  OK   ${run}"; else echo "  FAIL ${run}"; failures=$((failures+1)); fi
done
if [ "$failures" -gt 0 ]; then echo "Finished with ${failures} failure(s)."; exit 1; fi
echo "Integration test completed."
