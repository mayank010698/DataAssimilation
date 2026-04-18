#!/bin/bash
#
# Auxiliary Particle Filter (APF) evaluation on the double-well system.
#
# Budget-matched to the existing BPF double-well baselines at N=1000 particles
# (same as FPPF / BPF runs), two process-noise regimes, and per the default
# APF recipe:
#   * first-stage weights lambda_t ∝ w_{t-1} * m_t (always; not ESS-gated)
#   * ancestor selection: systematic resampling
#   * propagation: transition prior (q = f), so f/q cancel and the
#     incremental weight reduces to log g(y_t|x_t) - log m_t^{k_i}
#   * ESS-triggered second resample is a post-hoc cleanup diagnostic; the
#     APF mechanism itself fires every step regardless
#
# Main runs:
#   1) APF (point,      pnoise=0.2)  — "standard_var"  dataset
#   2) APF (point,      pnoise=0.3)  — "larger_var"    dataset
# Appendix ablation (predictive multiplier; cheap in 1-D additive-Gaussian):
#   3) APF (predictive, pnoise=0.2)
#   4) APF (predictive, pnoise=0.3)
#
# Usage:  cd /path/to/DataAssimilation && sbatch scripts/eval_apf_dw.sbatch
#
#SBATCH --job-name="eval-apf-dw"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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
if [ ! -f "${PROJECT_ROOT}/eval.py" ]; then
    echo "ERROR: Cannot find project root (expected eval.py at PROJECT_ROOT)."
    echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    echo "  tried PROJECT_ROOT=$PROJECT_ROOT"
    exit 1
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

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"

DATA_DIR_PN02="${DATA_ROOT}/double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p200"
DATA_DIR_PN03="${DATA_ROOT}/double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p300"

WANDB_PROJECT="${WANDB_PROJECT:-pf-eval-dw}"
WANDB_ENTITY="${WANDB_ENTITY:-ml-climate}"
JOB_TAG="${SLURM_JOB_ID:-manual}"

# Budget-matched to the BPF double-well runs.
N_PARTICLES=1000
BATCH_SIZE=100
RESAMPLING_THRESHOLD=0.5   # only affects the post-APF ESS-triggered cleanup
SEED=42

# ======================================================================
# Sanity checks
# ======================================================================

for d in "$DATA_DIR_PN02" "$DATA_DIR_PN03"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: Data dir not found: $d"
        exit 1
    fi
done

echo "=========================================="
echo "APF double-well evaluation"
echo "Project root : $PROJECT_ROOT"
echo "Data pn=0.2  : $DATA_DIR_PN02"
echo "Data pn=0.3  : $DATA_DIR_PN03"
echo "Slurm Job ID : ${SLURM_JOB_ID:-N/A}"
echo "Wandb project: $WANDB_PROJECT"
echo ""
echo "Particles          : $N_PARTICLES"
echo "Batch size         : $BATCH_SIZE"
echo "Resamp. thresh     : $RESAMPLING_THRESHOLD"
echo "Seed               : $SEED"
echo ""
echo "Launching 4 APF runs in parallel (2 pnoise x 2 adjustments)..."
echo "=========================================="

# ======================================================================
# Launch jobs
# ======================================================================

pids=()
run_names=()

launch_apf_run() {
    local data_dir="$1"
    local process_noise="$2"
    local pn_label="$3"        # e.g. "standard_var" | "larger_var"
    local adjustment="$4"      # "point" | "predictive"
    local adj_label="$5"       # short token for run-name
    local run_name="apf_1k_${adj_label}_${pn_label}"
    local experiment_label="dw_apf_${adj_label}"
    local log_path="logs/eval_apf_${adj_label}_${pn_label}_${JOB_TAG}.log"

    echo "  -> APF  adj=${adjustment}  pnoise=${process_noise}  (${run_name})"
    (
        CUDA_VISIBLE_DEVICES=0 python eval.py \
            --data-dir "$data_dir" \
            --method apf \
            --apf-adjustment "$adjustment" \
            --proposal-type transition \
            --n-particles "$N_PARTICLES" \
            --batch-size "$BATCH_SIZE" \
            --process-noise-std "$process_noise" \
            --resampling-threshold "$RESAMPLING_THRESHOLD" \
            --experiment-label "$experiment_label" \
            --wandb-project "$WANDB_PROJECT" \
            --wandb-entity "$WANDB_ENTITY" \
            --run-name "$run_name" \
            --seed "$SEED" \
            --device cuda
    ) 2>&1 | tee "$log_path" &
    pids+=("$!")
    run_names+=("$run_name")
}

# Main APF baseline: point surrogate multiplier.
launch_apf_run "$DATA_DIR_PN02" 0.2 standard_var point point
launch_apf_run "$DATA_DIR_PN03" 0.3 larger_var   point point

# Appendix ablation: predictive multiplier.
launch_apf_run "$DATA_DIR_PN02" 0.2 standard_var predictive pred
launch_apf_run "$DATA_DIR_PN03" 0.3 larger_var   predictive pred

# ======================================================================
# Wait for all jobs
# ======================================================================

echo ""
echo "All ${#pids[@]} job(s) launched. Waiting for completion..."
echo ""

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

echo ""
if [ "$failures" -gt 0 ]; then
    echo "Finished with ${failures} failure(s)."
    exit 1
fi

echo "All ${#run_names[@]} experiments completed successfully."
