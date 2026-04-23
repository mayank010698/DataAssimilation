#!/bin/bash
#
# Head-to-head BPF evaluation of the three NASMC proposal variants against
# the existing L96 dim5 obs0p100 arctan RF proposal.
#
# Eval order (fast first, so the NASMC numbers land while RF is still running):
#   1. gaussian_mle     -- supervised Gaussian-MLE baseline (phase 1 only)
#   2. nasmc_warmstart  -- MLE warmstart + NASMC refinement (steelman)
#   3. pure_nasmc       -- literal Gu et al. 2015 w/ bootstrap warmup
#   4. rf               -- existing RectifiedFlowProposal (slowest; runs last)
#
# All four runs share the same BPF config, the same no-noise eval dataset,
# and the same seed, so the resulting summary.csv row is the paper table.
# NASMC has no eval-time hyperparameters (single-step diagonal Gaussian head);
# RF uses the same sampling/likelihood grid as eval_rf_l96_neurips_grid.sbatch.
#
# Outputs (under PROJECT_ROOT):
#   logs/eval_nasmc_vs_rf_l96_dim5_<JOB_TAG>/<run_name>.log  per-run stdout
#   logs/eval_nasmc_vs_rf_l96_dim5_<JOB_TAG>/summary.csv     one row per proposal
#   W&B project: eval-nasmc-vs-rf-l96-dim5 (overridable via WANDB_PROJECT)
#
#SBATCH --job-name="eval-nasmc-vs-rf-l96-dim5"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

set -euo pipefail

echo "=== Slurm context ==="
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-N/A}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "====================="

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
module load cuda/12.8

export PYTHONPATH="$(pwd)"

JOB_TAG="${SLURM_JOB_ID:-manual}"
RUN_DIR="logs/eval_nasmc_vs_rf_l96_dim5_${JOB_TAG}"
mkdir -p "${RUN_DIR}"
SUMMARY_CSV="${RUN_DIR}/summary.csv"

# --- Shared config ---------------------------------------------------------
# Eval dataset: no-noise variant of the L96 dim5 obs0p100 arctan training set
# (matches the convention used in eval_rf_l96_neurips_grid.sbatch).
DATA_ROOT="${DATA_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims}"
DATASET_NAME="lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_init3p000"
DATA_DIR="${DATA_ROOT}/${DATASET_NAME}"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: eval DATA_DIR not found: ${DATA_DIR}"
    exit 1
fi

# BPF / PF config (dim5 cell of eval_rf_l96_neurips_grid.sbatch).
STATE_DIM=5
OBS_TOKEN="obs0p100"
OBS_OP_TAG="arctan"
N_PARTICLES=1000
BATCH_SIZE=10
PROCESS_NOISE_STD=0.2
RESAMPLING_THRESHOLD=0.33
NUM_EVAL_TRAJECTORIES=10
MAX_TIMESTEPS=100
SEED=42

# RF-only knobs (ignored for --proposal-type nasmc; eval.py warns and skips).
SAMPLING_GRID="uniform"
LIKELIHOOD_GRID="power_back"
SAMPLING_STEPS=32
LIKELIHOOD_STEPS=32

WANDB_PROJECT="${WANDB_PROJECT:-eval-nasmc-vs-rf-l96-dim5}"

# --- Checkpoint discovery --------------------------------------------------
# NASMC training output layout (scripts/train_nasmc_l96_dim5.sbatch):
#   ${NASMC_RUN_ROOT}/run_<job>_<preset>/{pretrain,refine}/checkpoints/nasmc-*.ckpt
# RF training output layout (old run matched to the NASMC models):
#   ${RF_RUN_ROOT}/${RF_RUN_NAME}/checkpoints/rf-epoch=*-val_loss=*.ckpt
NASMC_RUN_ROOT="${NASMC_RUN_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/nasmc_runs_96/dim5}"
RF_RUN_ROOT="${RF_RUN_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96}"
RF_RUN_NAME="${RF_RUN_NAME:-l96_noise_dim5_20260109_193733}"

# Pin to the specific training jobs logged in W&B so the eval is
# reproducible and matches exactly what's plotted in the run dashboard.
# Override any of these to re-eval a different training run.
GAUSSIAN_MLE_JOB="${GAUSSIAN_MLE_JOB:-8156295}"
WARMSTART_JOB="${WARMSTART_JOB:-8192421}"
PURE_NASMC_JOB="${PURE_NASMC_JOB:-8192421}"

# NASMC ModelCheckpoint filename is `nasmc-<phase>-<NNN>-<val_nll>.ckpt` with
# `auto_insert_metric_name=False`; val_nll is negative (NLL of a density that
# comfortably exceeds 1 on this data), so the filename effectively contains
# two dashes in a row (e.g. nasmc-pretrain-021--13.447855.ckpt). Best ckpt =
# lowest val_nll = most negative. Parse with Python to be safe.
pick_best_nasmc_ckpt() {
    local pattern="$1"
    python - "$pattern" <<'PY'
import glob, re, sys
pattern = sys.argv[1]
scored = []
for f in glob.glob(pattern):
    m = re.search(r"-(-?\d+\.\d+)\.ckpt$", f)
    if m:
        scored.append((float(m.group(1)), f))
scored.sort()  # lowest val_nll first (most negative)
print(scored[0][1] if scored else "")
PY
}

# RF ModelCheckpoint filename is `rf-epoch=XXX-val_loss=YYY.ckpt` (positive).
# Parse with awk on the literal `val_loss=` separator.
pick_best_rf_ckpt() {
    local ckpt_dir="$1"
    ls -1 "${ckpt_dir}"/rf-epoch=*-val_loss=*.ckpt 2>/dev/null \
        | awk -F'val_loss=' '{print $2, $0}' \
        | sort -g \
        | head -n1 \
        | awk '{print $2}'
}

GAUSSIAN_MLE_CKPT="$(pick_best_nasmc_ckpt "${NASMC_RUN_ROOT}/run_${GAUSSIAN_MLE_JOB}_gaussian_mle/pretrain/checkpoints/nasmc-pretrain-*.ckpt")"
WARMSTART_CKPT="$(  pick_best_nasmc_ckpt "${NASMC_RUN_ROOT}/run_${WARMSTART_JOB}_nasmc_warmstart/refine/checkpoints/nasmc-refine-*.ckpt")"
PURE_NASMC_CKPT="$( pick_best_nasmc_ckpt "${NASMC_RUN_ROOT}/run_${PURE_NASMC_JOB}_pure_nasmc/refine/checkpoints/nasmc-refine-*.ckpt")"
RF_CKPT="$(         pick_best_rf_ckpt    "${RF_RUN_ROOT}/${RF_RUN_NAME}/checkpoints" )"

# Fail fast if any checkpoint is missing.
declare -A CKPT_MAP=(
    [gaussian_mle]="$GAUSSIAN_MLE_CKPT"
    [nasmc_warmstart]="$WARMSTART_CKPT"
    [pure_nasmc]="$PURE_NASMC_CKPT"
    [rf]="$RF_CKPT"
)
for label in "${!CKPT_MAP[@]}"; do
    ck="${CKPT_MAP[$label]}"
    if [ -z "$ck" ] || [ ! -f "$ck" ]; then
        echo "ERROR: missing ${label} checkpoint (resolved='${ck}')"
        exit 1
    fi
done

echo "=========================================="
echo "eval_nasmc_vs_rf_l96_dim5.sbatch"
echo "  data_dir         = ${DATA_DIR}"
echo "  n_particles      = ${N_PARTICLES}"
echo "  batch_size       = ${BATCH_SIZE}"
echo "  process_noise    = ${PROCESS_NOISE_STD}"
echo "  resampling_thr   = ${RESAMPLING_THRESHOLD}"
echo "  num_eval_trajs   = ${NUM_EVAL_TRAJECTORIES}"
echo "  seed             = ${SEED}"
echo "  wandb_project    = ${WANDB_PROJECT}"
echo "  run_dir          = ${RUN_DIR}"
echo "  checkpoints:"
echo "    gaussian_mle    = ${GAUSSIAN_MLE_CKPT}"
echo "    nasmc_warmstart = ${WARMSTART_CKPT}"
echo "    pure_nasmc      = ${PURE_NASMC_CKPT}"
echo "    rf              = ${RF_CKPT}"
echo "=========================================="

# --- Summary CSV header (once) --------------------------------------------
if [ ! -s "$SUMMARY_CSV" ]; then
    echo "label,proposal_type,wall_time_sec,mean_rmse,std_rmse,mean_crps,std_crps,mean_ess,run_name,checkpoint,log_file" > "$SUMMARY_CSV"
fi

# --- Metric extraction (same format as eval_rf_l96_neurips_grid.sbatch) ---
extract_metric() {
    local log_file="$1"
    local label="$2"
    grep -F "${label}" "${log_file}" | tail -n1 \
        | awk -F ':' '{print $2}' \
        | awk '{print $1}'
}
extract_metric_std() {
    local log_file="$1"
    local label="$2"
    grep -F "${label}" "${log_file}" | tail -n1 \
        | awk -F '±' '{print $2}' \
        | awk '{print $1}'
}

# --- Single-run driver -----------------------------------------------------
# Args: $1=label  $2=proposal_type (rf|nasmc)  $3=checkpoint
run_eval() {
    local label="$1"
    local ptype="$2"
    local ckpt="$3"
    local run_name="eval_${label}_d${STATE_DIM}_${OBS_TOKEN}_${OBS_OP_TAG}_${JOB_TAG}"
    local log_file="${RUN_DIR}/${run_name}.log"

    local -a cmd=(
        python eval.py
        --data-dir "$DATA_DIR"
        --method bpf --proposal-type "$ptype"
        --rf-checkpoint "$ckpt"
        --n-particles "$N_PARTICLES"
        --batch-size "$BATCH_SIZE"
        --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES"
        --resampling-threshold "$RESAMPLING_THRESHOLD"
        --process-noise-std "$PROCESS_NOISE_STD"
        --seed "$SEED"
        --device cuda
        --experiment-label "$run_name"
        --wandb-project "$WANDB_PROJECT"
        --wandb-tags "eval_nasmc_vs_rf,d${STATE_DIM},${OBS_TOKEN},${OBS_OP_TAG},N${N_PARTICLES},bs${BATCH_SIZE},${label}"
        --run-name "$run_name"
        --max-timesteps "$MAX_TIMESTEPS"
    )
    if [ "$ptype" = "rf" ]; then
        cmd+=(
            --rf-sampling-steps "$SAMPLING_STEPS"
            --rf-likelihood-steps "$LIKELIHOOD_STEPS"
            --sampling-grid-type "$SAMPLING_GRID"
            --likelihood-grid-type "$LIKELIHOOD_GRID"
            --likelihood-grid-param '{"gamma": 2.0}'
        )
    fi

    echo "------------------------------------------"
    echo "[${label}] proposal=${ptype}"
    echo "[${label}]   ckpt=${ckpt}"
    echo "[${label}]    log=${log_file}"
    echo "[${label}] --> running: ${cmd[*]}"
    echo "------------------------------------------"

    local t0; t0=$(date +%s)
    set +e
    CUDA_VISIBLE_DEVICES=0 "${cmd[@]}" > "$log_file" 2>&1
    local rc=$?
    set -e
    local t1; t1=$(date +%s)
    local dt=$(( t1 - t0 ))

    local mean_rmse="" std_rmse="" mean_crps="" std_crps="" mean_ess=""
    if [ "$rc" -eq 0 ]; then
        mean_rmse="$(extract_metric     "$log_file" "Mean RMSE across trajectories")"
        std_rmse="$( extract_metric_std "$log_file" "Mean RMSE across trajectories")"
        mean_crps="$(extract_metric     "$log_file" "Mean CRPS across trajectories")"
        std_crps="$( extract_metric_std "$log_file" "Mean CRPS across trajectories")"
        mean_ess="$( extract_metric     "$log_file" "Mean ESS")"
        echo "[${label}]   OK  wall=${dt}s  rmse=${mean_rmse:-?}  crps=${mean_crps:-?}  ess=${mean_ess:-?}"
    else
        echo "[${label}]   FAIL rc=${rc} wall=${dt}s (see ${log_file})"
    fi

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$label" "$ptype" "$dt" \
        "${mean_rmse:-}" "${std_rmse:-}" "${mean_crps:-}" "${std_crps:-}" \
        "${mean_ess:-}" "$run_name" "$ckpt" "$log_file" \
        >> "$SUMMARY_CSV"

    return "$rc"
}

# --- Sequential eval, fastest -> slowest ----------------------------------
# RF is ~40 min per run (~2500s on dim5 obs0p200 in the NeurIPS grid); NASMC
# is a single MLP/ResNet forward per particle per step, so it's considerably
# faster. Running NASMC first means you can watch the three NASMC rows land
# in summary.csv while RF is still going.
EXIT_CODE=0

run_eval gaussian_mle    nasmc "$GAUSSIAN_MLE_CKPT" || EXIT_CODE=1
run_eval nasmc_warmstart nasmc "$WARMSTART_CKPT"    || EXIT_CODE=1
run_eval pure_nasmc      nasmc "$PURE_NASMC_CKPT"   || EXIT_CODE=1
run_eval rf              rf    "$RF_CKPT"           || EXIT_CODE=1

echo "=========================================="
echo "All eval runs complete (exit ${EXIT_CODE})."
echo "summary.csv:"
cat "$SUMMARY_CSV"
echo "=========================================="

exit "$EXIT_CODE"
