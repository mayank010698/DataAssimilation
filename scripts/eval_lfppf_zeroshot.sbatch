#!/bin/bash
#
# Ablation 15: Zero-shot transfer of an L-FPPF proposal trained on L96-25 to
# larger L96 state dims (N_x ∈ {50, 100, 500, 1000}).
#
# The patch-based velocity network (local_resnet1d) has a bounded receptive
# field of (2 * radius + 1) sites, so all parameters are independent of the
# global state dim. Passing --lrf-state-dim-override lets the wrapper carry
# the new state_dim through its API; the heavy lifting is handled by the PF
# using the target dataset.
#
# Note on obs_components / zero-shot wiring:
#   The trained checkpoint stores obs_components for the *training* dim
#   (e.g. [0,...,24] for L96-25). The wrapper now overrides those with the
#   target dataset's obs_components when --lrf-state-dim-override is set
#   (see LocalizedRFProposalWrapper.__init__ in models/proposals.py); without
#   that fix the patch scatter would mismatch shapes at the new dim.
#
# Hyperparameters mirror eval_lfppf_integration_l96_25.sbatch so results are
# directly comparable to the in-distribution L-FPPF integration run.
# BATCH_SIZE is the only knob that varies per dim, because per-step memory
# scales as B * N_p * N_x: dim=100 fits at B=10, so dim=500 needs B≈2 and
# dim=1000 needs B≈1.
#
# Usage:
#   LRF_CHECKPOINT=/path/to/lrf.ckpt sbatch scripts/eval_lfppf_zeroshot.sbatch
#
#SBATCH --job-name="local-highdim"
#SBATCH --partition=dali
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=24:00:00
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

# DATA_ROOT holds the per-dim L96 datasets. Default matches the dataset
# convention used by eval_lfppf_integration_l96_25.sbatch (obs0p100, no
# pnoise suffix). To eval on the pnoise variants used at *training* time,
# override DATASET_TEMPLATE below (use the literal token __DIM__ wherever
# the per-run state dim should be substituted) or set DATA_ROOT to the
# pnoise tree.
DATA_ROOT="${DATA_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims}"
# Two-step default: a literal default that contains '{...}' inside
# ${VAR:-default} would be truncated by bash at the first '}'.
if [ -z "${DATASET_TEMPLATE:-}" ]; then
    DATASET_TEMPLATE='lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp__DIM__of__DIM___arctan_init3p000'
fi

# Defaults to the same checkpoint used by the L96-25 integration run.
LRF_CHECKPOINT="${LRF_CHECKPOINT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/lrf_runs/run_8108115_r4_lr1e-3/checkpoints/lrf-epoch=059-val_loss=0.021012.ckpt}"
if [ -z "$LRF_CHECKPOINT" ] || [ ! -f "$LRF_CHECKPOINT" ]; then
    echo "ERROR: LRF_CHECKPOINT must be set."; exit 1
fi

WANDB_PROJECT="${WANDB_PROJECT:-eval-l96-highdim}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

# Hyperparameters: match eval_lfppf_integration_l96_25.sbatch (BATCH_SIZE is
# overridden per-dim inside each run_<dim> function below).
NUM_EVAL_TRAJECTORIES=10
RESAMPLING_THRESHOLD=0.33
MAX_TIMESTEPS=100
SEED=42
N_PARTICLES=500
PROCESS_NOISE_STD=0.2
RF_SAMPLING_STEPS=100
RF_LIKELIHOOD_STEPS=100
LPF_TAPER="gaussian"
LPF_BLOCK_SIZE=1
LPF_LOC_RADIUS=5.0

# Comment out any line below to skip that experiment.
# Names must match the run_<name> functions defined further down.
EXPERIMENTS=(
    # dim50      # cheap, useful sanity check at 2x training dim
    # dim100     # known good at BATCH_SIZE=10
    dim500     # OOMs at BATCH_SIZE=10, runs at BATCH_SIZE=2
    dim1000    # OOMs at BATCH_SIZE=10, runs at BATCH_SIZE=1 (try 2 if memory permits)
)

run_labels=()
run_status=()

# --- Experiment definitions ---
# Each function runs one dim *synchronously* and records its label/status.
# Dims are executed sequentially in EXPERIMENTS order so high-dim runs
# don't fight each other for GPU memory.
# `_launch_dim DIM BATCH_SIZE` is the shared body; per-dim wrappers just
# bind the right BATCH_SIZE for that dim.

_launch_dim() {
    local DIM="$1"
    local BATCH_SIZE="$2"
    local DATA_REL="${DATASET_TEMPLATE//__DIM__/$DIM}"
    local DATA_DIR="${DATA_ROOT}/${DATA_REL}"
    if [ ! -d "$DATA_DIR" ]; then
        echo "  -> SKIP dim=${DIM}: dataset not found (${DATA_DIR})"
        run_labels+=("dim${DIM}_skipped")
        run_status+=("SKIP")
        return 0
    fi

    local label="ablation15_zeroshot_N${DIM}_B${BATCH_SIZE}_${JOB_TAG}"
    local log="logs/${label}.log"
    echo "  -> dim=${DIM} batch=${BATCH_SIZE} (data: ${DATA_DIR})"

    set +e
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
    ) 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e

    run_labels+=("$label")
    if [ "$rc" -eq 0 ]; then
        run_status+=("OK")
    else
        run_status+=("FAIL")
    fi
    return 0
}

run_dim50()   { _launch_dim 50   10; }
run_dim100()  { _launch_dim 100  10; }
run_dim500()  { _launch_dim 500   2; }
run_dim1000() { _launch_dim 1000  2; }

# --- Dispatch ---
# Sequential: one dim at a time. A failure on any single dim does not abort
# the rest of the queue — we collect statuses and exit non-zero at the end
# if any failed.
for exp in "${EXPERIMENTS[@]}"; do
    fn="run_${exp}"
    if ! declare -F "$fn" > /dev/null; then
        echo "ERROR: unknown experiment '${exp}' (no function ${fn})"; exit 1
    fi
    "$fn"
done

echo "Summary of ${#run_labels[@]} runs:"
failures=0
for idx in "${!run_labels[@]}"; do
    label="${run_labels[$idx]}"
    status="${run_status[$idx]}"
    echo "  ${status}   ${label}"
    if [ "$status" = "FAIL" ]; then failures=$((failures+1)); fi
done
if [ "$failures" -gt 0 ]; then echo "Finished with ${failures} failure(s)."; exit 1; fi
echo "All zero-shot runs completed."
