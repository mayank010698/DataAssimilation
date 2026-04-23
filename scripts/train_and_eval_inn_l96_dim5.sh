#!/bin/bash
#
# Interactive / compute-node shell version of
# scripts/train_and_eval_inn_l96_dim5.sbatch.
#
# Use this when you've already SSH'd to a GPU compute node (e.g. ccc0482)
# and just want to run the whole IN train+eval pipeline end-to-end. Same
# logic as the sbatch version:
#
#   1. Train IN for each preset in INN_PRESETS on the L96 dim5 training
#      dataset (default: inn_rnade + inn_mog, sequential on one GPU).
#   2. For each preset, pick the best val_nll checkpoint and run
#      eval.py (BPF, 1000 particles, seed 42) on the no-noise eval
#      dataset, logging to the SAME W&B project as the NASMC/RF eval.
#
# Usage:
#
#   # Defaults (inn_rnade then inn_mog):
#   scripts/train_and_eval_inn_l96_dim5.sh
#
#   # Custom preset list:
#   INN_PRESETS="inn_mog" scripts/train_and_eval_inn_l96_dim5.sh
#   INN_PRESETS="inn_gaussian inn_mdn_k inn_mog inn_rnade" \
#       scripts/train_and_eval_inn_l96_dim5.sh
#
#   # Skip training when you already have a trained run (it will
#   # pick the latest run_*_<preset> under OUTPUT_BASE):
#   SKIP_TRAIN=1 scripts/train_and_eval_inn_l96_dim5.sh
#
#   # Pipe everything to a logfile while keeping it on your terminal:
#   scripts/train_and_eval_inn_l96_dim5.sh 2>&1 | tee logs/inn_run.log
#
# Prerequisites:
#   * Already on a node with a CUDA GPU (nvidia-smi works).
#   * `da` conda env exists and has torch/lightning installed.
#   * Data directories exist (checked at startup).

set -euo pipefail

# Resolve project root from this script's location (no SLURM_SUBMIT_DIR here).
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ ! -f "${PROJECT_ROOT}/eval.py" ]; then
    echo "ERROR: Cannot find project root (expected eval.py at ${PROJECT_ROOT})." >&2
    exit 1
fi
cd "$PROJECT_ROOT" || exit 1

# ------- Environment setup (idempotent; OK to re-run) ----------------------
# Only load modules if the module command exists (it won't on a plain
# interactive bash without the campus-cluster modulefiles sourced yet).
if command -v module &>/dev/null; then
    module load anaconda3/2024.10 2>/dev/null || true
    module load cuda/12.8 2>/dev/null || true
fi

# Activate the da conda env if not already active.
if [ "${CONDA_DEFAULT_ENV:-}" != "da" ]; then
    # Source conda so `conda activate` works in this shell.
    if [ -f /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh ]; then
        # shellcheck disable=SC1091
        source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
    elif command -v conda &>/dev/null; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        echo "ERROR: cannot locate conda. Load anaconda3/2024.10 manually and re-run." >&2
        exit 1
    fi
    conda activate da
fi

export PYTHONPATH="$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Quick sanity check: we really are on a GPU node.
if ! nvidia-smi >/dev/null 2>&1; then
    echo "WARN: nvidia-smi failed. Proceeding anyway, but GPU training may not work." >&2
fi

mkdir -p logs

# Use hostname+timestamp as the job identifier since we don't have SLURM_JOB_ID.
JOB_ID="${JOB_ID:-$(hostname -s)_$(date +%Y%m%d_%H%M%S)}"

# ==========================================================================
# Shared config (must stay in sync with train_and_eval_inn_l96_dim5.sbatch)
# ==========================================================================
STATE_DIM=5
OBS_TOKEN="obs0p100"
OBS_OP_TAG="arctan"
OBS_COMPONENTS=0,1,2,3,4

INN_PRESETS="${INN_PRESETS:-inn_rnade inn_mog}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# ------------------------- Training knobs ---------------------------------
TRAIN_DATA_DIR=/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_pnoise0p100_init3p000

NUM_MIXTURE_COMPONENTS=8
RNADE_HIDDEN_SIZE=128
RNADE_NUM_HIDDEN_LAYERS=2

ARCH=resnet1d
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
TIME_EMBED_DIM=128
FEATURE_DIM=128
BATCH_SIZE=512
LR=3e-4

MAX_EPOCHS=500
PATIENCE=40

OUTPUT_BASE=/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/inn_runs_96/dim5
TRAIN_WANDB_PROJECT="inn-train-96"

# ------------------------- Eval knobs -------------------------------------
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims}"
EVAL_DATASET_NAME="lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_init3p000"
EVAL_DATA_DIR="${EVAL_DATA_ROOT}/${EVAL_DATASET_NAME}"
for d in "$TRAIN_DATA_DIR" "$EVAL_DATA_DIR"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: data dir not found: $d" >&2
        exit 1
    fi
done

N_PARTICLES=1000
BATCH_SIZE_EVAL=10
PROCESS_NOISE_STD=0.2
RESAMPLING_THRESHOLD=0.33
NUM_EVAL_TRAJECTORIES=10
MAX_TIMESTEPS=100
SEED=42

EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-eval-nasmc-vs-rf-l96-dim5}"

RUN_DIR="logs/eval_inn_l96_dim5_${JOB_ID}"
mkdir -p "${RUN_DIR}"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
if [ ! -s "$SUMMARY_CSV" ]; then
    echo "label,proposal_type,wall_time_sec,mean_rmse,std_rmse,mean_crps,std_crps,mean_ess,run_name,checkpoint,log_file" > "$SUMMARY_CSV"
fi

echo "=========================================="
echo "train_and_eval_inn_l96_dim5.sh"
echo "  host              = $(hostname)"
echo "  job_id            = ${JOB_ID}"
echo "  inn_presets       = ${INN_PRESETS}"
echo "  skip_train        = ${SKIP_TRAIN}"
echo "  train_data_dir    = ${TRAIN_DATA_DIR}"
echo "  output_base       = ${OUTPUT_BASE}"
echo "  eval_data_dir     = ${EVAL_DATA_DIR}"
echo "  n_particles       = ${N_PARTICLES}"
echo "  batch_size_eval   = ${BATCH_SIZE_EVAL}"
echo "  process_noise_std = ${PROCESS_NOISE_STD}"
echo "  resampling_thr    = ${RESAMPLING_THRESHOLD}"
echo "  num_eval_trajs    = ${NUM_EVAL_TRAJECTORIES}"
echo "  seed              = ${SEED}"
echo "  train wandb proj  = ${TRAIN_WANDB_PROJECT}"
echo "  eval  wandb proj  = ${EVAL_WANDB_PROJECT}"
echo "  run_dir           = ${RUN_DIR}"
echo "=========================================="

# ==========================================================================
# Helpers
# ==========================================================================
pick_best_inn_ckpt() {
    local pattern="$1"
    python - "$pattern" <<'PY'
import glob, re, sys
pattern = sys.argv[1]
scored = []
for f in glob.glob(pattern):
    m = re.search(r"-(-?\d+\.\d+)\.ckpt$", f)
    if m:
        scored.append((float(m.group(1)), f))
scored.sort()
print(scored[0][1] if scored else "")
PY
}

# Pick the most-recently-modified run_*_<preset> dir under OUTPUT_BASE
# when SKIP_TRAIN=1 (so we re-evaluate without retraining).
latest_run_dir() {
    local preset="$1"
    ls -1dt "${OUTPUT_BASE}"/run_*_"${preset}" 2>/dev/null | head -n1
}

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

# ==========================================================================
# Per-preset driver: train -> best-ckpt -> eval -> summary.csv row
# ==========================================================================
run_one_preset() {
    local preset="$1"

    local output_dir="${OUTPUT_BASE}/run_${JOB_ID}_${preset}"
    local train_log="logs/train_inn_l96_dim5_${JOB_ID}_${preset}.log"
    local train_run_name="inn_l96_dim5_${preset}_job${JOB_ID}"

    echo ""
    echo "##########################################"
    echo "# Preset: ${preset}"
    echo "##########################################"

    if [ "$SKIP_TRAIN" = "1" ]; then
        local existing
        existing="$(latest_run_dir "$preset")"
        if [ -z "$existing" ] || [ ! -d "$existing" ]; then
            echo "[${preset}] ERROR: SKIP_TRAIN=1 but no existing run found under ${OUTPUT_BASE}."
            return 1
        fi
        output_dir="$existing"
        echo "[${preset}] SKIP_TRAIN=1 -- re-using existing training run: ${output_dir}"
    else
        echo "------------------------------------------"
        echo "[${preset}] Training for up to ${MAX_EPOCHS} epochs"
        echo "  output_dir=${output_dir}"
        echo "  train_log =${train_log}"
        echo "------------------------------------------"

        local train_rc=0
        set +e
        python -m proposals.train_inference_network \
            --preset "$preset" \
            --data_dir "$TRAIN_DATA_DIR" \
            --output_dir "$output_dir" \
            --state_dim "$STATE_DIM" \
            --obs_components "$OBS_COMPONENTS" \
            --use_observations \
            --architecture "$ARCH" \
            --channels "$CHANNELS" \
            --num_blocks "$NUM_BLOCKS" \
            --kernel_size "$KERNEL_SIZE" \
            --time_embed_dim "$TIME_EMBED_DIM" \
            --feature_dim "$FEATURE_DIM" \
            --num_mixture_components "$NUM_MIXTURE_COMPONENTS" \
            --rnade_hidden_size "$RNADE_HIDDEN_SIZE" \
            --rnade_num_hidden_layers "$RNADE_NUM_HIDDEN_LAYERS" \
            --predict_delta \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LR" \
            --max_epochs "$MAX_EPOCHS" \
            --patience "$PATIENCE" \
            --num_workers 4 \
            --gpus 1 \
            --wandb_project "$TRAIN_WANDB_PROJECT" \
            --wandb_entity ml-climate \
            --wandb_run_name "$train_run_name" \
            --seed 0 \
            > "$train_log" 2>&1
        train_rc=$?
        set -e

        if [ "$train_rc" -ne 0 ]; then
            echo "[${preset}] FAIL (training, rc=${train_rc}). See ${train_log}."
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
                "inn_${preset#inn_}" "inn" "" "" "" "" "" "" "${train_run_name}" "" "${train_log}" \
                >> "$SUMMARY_CSV"
            return "$train_rc"
        fi
    fi

    local inn_ckpt
    inn_ckpt="$(pick_best_inn_ckpt "${output_dir}/checkpoints/inn-*.ckpt")"
    if [ -z "$inn_ckpt" ] || [ ! -f "$inn_ckpt" ]; then
        inn_ckpt="${output_dir}/final_model.ckpt"
        if [ ! -f "$inn_ckpt" ]; then
            echo "[${preset}] ERROR: no IN checkpoint found under ${output_dir}."
            return 2
        fi
        echo "[${preset}] WARN: no best-val_nll checkpoint found, falling back to ${inn_ckpt}"
    fi
    echo "[${preset}] Selected checkpoint: ${inn_ckpt}"

    local eval_label="inn_${preset#inn_}"
    local eval_run_name="eval_${eval_label}_d${STATE_DIM}_${OBS_TOKEN}_${OBS_OP_TAG}_${JOB_ID}"
    local eval_log="${RUN_DIR}/${eval_run_name}.log"

    echo "------------------------------------------"
    echo "[${eval_label}] proposal=inn"
    echo "[${eval_label}]   ckpt=${inn_ckpt}"
    echo "[${eval_label}]    log=${eval_log}"
    echo "------------------------------------------"

    local t0 t1 dt
    t0=$(date +%s)
    local eval_rc=0
    set +e
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --data-dir "$EVAL_DATA_DIR" \
        --method bpf --proposal-type inn \
        --rf-checkpoint "$inn_ckpt" \
        --n-particles "$N_PARTICLES" \
        --batch-size "$BATCH_SIZE_EVAL" \
        --num-eval-trajectories "$NUM_EVAL_TRAJECTORIES" \
        --resampling-threshold "$RESAMPLING_THRESHOLD" \
        --process-noise-std "$PROCESS_NOISE_STD" \
        --seed "$SEED" \
        --device cuda \
        --experiment-label "$eval_run_name" \
        --wandb-project "$EVAL_WANDB_PROJECT" \
        --wandb-tags "eval_nasmc_vs_rf,d${STATE_DIM},${OBS_TOKEN},${OBS_OP_TAG},N${N_PARTICLES},bs${BATCH_SIZE_EVAL},${eval_label}" \
        --run-name "$eval_run_name" \
        --max-timesteps "$MAX_TIMESTEPS" \
        > "$eval_log" 2>&1
    eval_rc=$?
    set -e
    t1=$(date +%s)
    dt=$(( t1 - t0 ))

    local mean_rmse="" std_rmse="" mean_crps="" std_crps="" mean_ess=""
    if [ "$eval_rc" -eq 0 ]; then
        mean_rmse="$(extract_metric     "$eval_log" "Mean RMSE across trajectories")"
        std_rmse="$( extract_metric_std "$eval_log" "Mean RMSE across trajectories")"
        mean_crps="$(extract_metric     "$eval_log" "Mean CRPS across trajectories")"
        std_crps="$( extract_metric_std "$eval_log" "Mean CRPS across trajectories")"
        mean_ess="$( extract_metric     "$eval_log" "Mean ESS")"
        echo "[${eval_label}]   OK  wall=${dt}s  rmse=${mean_rmse:-?}  crps=${mean_crps:-?}  ess=${mean_ess:-?}"
    else
        echo "[${eval_label}]   FAIL rc=${eval_rc} wall=${dt}s (see ${eval_log})"
    fi

    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$eval_label" "inn" "$dt" \
        "${mean_rmse:-}" "${std_rmse:-}" "${mean_crps:-}" "${std_crps:-}" \
        "${mean_ess:-}" "$eval_run_name" "$inn_ckpt" "$eval_log" \
        >> "$SUMMARY_CSV"

    return "$eval_rc"
}

# ==========================================================================
# Main loop
# ==========================================================================
EXIT_CODE=0
for preset in $INN_PRESETS; do
    run_one_preset "$preset" || { echo "[${preset}] failed, continuing..."; EXIT_CODE=1; }
done

echo ""
echo "=========================================="
echo "Train + eval complete (exit ${EXIT_CODE})."
echo "Training root:  ${OUTPUT_BASE}"
echo "Eval summary:   ${SUMMARY_CSV}"
cat "$SUMMARY_CSV"
echo ""
echo "Wandb (eval, shared with NASMC/RF):"
echo "  https://wandb.ai/ml-climate/${EVAL_WANDB_PROJECT}"
echo "Wandb (train):"
echo "  https://wandb.ai/ml-climate/${TRAIN_WANDB_PROJECT}"
echo "=========================================="

exit "$EXIT_CODE"
