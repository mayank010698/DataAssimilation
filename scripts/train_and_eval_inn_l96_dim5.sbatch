#!/bin/bash
#
#SBATCH --job-name="train-eval-inn-l96-dim5"
#SBATCH --partition=secondary
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log
#
# Train the Paige-Wood inference-network (IN) proposal on the L96 dim5
# training dataset, then evaluate it head-to-head against the existing
# NASMC and RF proposals on the no-noise L96 dim5 eval dataset.
#
# TWO variants run SEQUENTIALLY on a single GPU:
#   1. inn_rnade -- paper-faithful conditional autoregressive mixture of
#                   Gaussians (Paige & Wood, ICML 2016).
#   2. inn_mog   -- joint mixture of K diagonal Gaussians; our recommended
#                   default for the Paige-Wood class.
# Override the list via `INN_PRESETS="inn_rnade inn_mog inn_gaussian"`
# (or a subset) when submitting.
#
# Training config matches scripts/train_nasmc_l96_dim5.sbatch so IN has
# the same feature-extractor capacity as NASMC's q_phi on this dataset.
# Eval re-uses the wandb project, dataset, and PF hyperparameters from
# scripts/eval_nasmc_vs_rf_l96_dim5.sbatch so the IN rows land in the
# same W&B dashboard as the three NASMC rows and the RF row (which we
# already ran; we deliberately skip RF here).
#
# Outputs:
#   ${OUTPUT_BASE}/run_<JOB_ID>_<preset>/              training artifacts
#   logs/train_inn_l96_dim5_<JOB_ID>_<preset>.log      training stdout (per preset)
#   logs/eval_inn_l96_dim5_<JOB_ID>/<run_name>.log     eval stdout (per preset)
#   logs/eval_inn_l96_dim5_<JOB_ID>/summary.csv        one row per preset
#   W&B project (train): inn-train-96
#   W&B project (eval):  eval-nasmc-vs-rf-l96-dim5     (shared with NASMC/RF)

set -euo pipefail

echo "=== Slurm context ==="
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-N/A}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "====================="

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/eval.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
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
# Same memory-fragmentation hint as the NASMC training script.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-manual}"

# ==========================================================================
# Shared config
# ==========================================================================
STATE_DIM=5
OBS_TOKEN="obs0p100"
OBS_OP_TAG="arctan"
OBS_COMPONENTS=0,1,2,3,4

# Which IN variants to run. Space-separated list; defaults to the two
# interesting heads (paper-faithful RNADE + recommended JointMoG).
INN_PRESETS="${INN_PRESETS:-inn_rnade inn_mog}"

# ------------------------- Training knobs ---------------------------------
# Dataset with process noise (matches the NASMC training dataset).
TRAIN_DATA_DIR=/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_pnoise0p100_init3p000

# K only matters for the mixture heads; gaussian head ignores it.
NUM_MIXTURE_COMPONENTS=8
# RNADE-specific (only used when preset=inn_rnade).
RNADE_HIDDEN_SIZE=128
RNADE_NUM_HIDDEN_LAYERS=2

# Feature-backbone capacity -- matches NASMC's L96 dim5 training run exactly.
ARCH=resnet1d
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
TIME_EMBED_DIM=128
FEATURE_DIM=128
BATCH_SIZE=512
LR=3e-4

# IN is pure supervised MLE -- no SMC in the loop -- so one "phase" of
# 500 epochs matches the total epoch budget the NASMC run spent on MLE
# warmstart + SMC refinement.
MAX_EPOCHS=500
PATIENCE=40

OUTPUT_BASE=/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/inn_runs_96/dim5
TRAIN_WANDB_PROJECT="inn-train-96"

# ------------------------- Eval knobs -------------------------------------
# No-process-noise eval dataset -- same one used by eval_nasmc_vs_rf_l96_dim5.
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims}"
EVAL_DATASET_NAME="lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_init3p000"
EVAL_DATA_DIR="${EVAL_DATA_ROOT}/${EVAL_DATASET_NAME}"
if [ ! -d "$EVAL_DATA_DIR" ]; then
    echo "ERROR: eval DATA_DIR not found: ${EVAL_DATA_DIR}"
    exit 1
fi
if [ ! -d "$TRAIN_DATA_DIR" ]; then
    echo "ERROR: train DATA_DIR not found: ${TRAIN_DATA_DIR}"
    exit 1
fi

# BPF / PF config -- identical to eval_nasmc_vs_rf_l96_dim5.sbatch so
# the IN rows are directly comparable to the NASMC / RF rows already on
# W&B.
N_PARTICLES=1000
BATCH_SIZE_EVAL=10
PROCESS_NOISE_STD=0.2
RESAMPLING_THRESHOLD=0.33
NUM_EVAL_TRAJECTORIES=10
MAX_TIMESTEPS=100
SEED=42

# Same W&B project as the NASMC/RF comparison so all rows live together.
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-eval-nasmc-vs-rf-l96-dim5}"

RUN_DIR="logs/eval_inn_l96_dim5_${JOB_ID}"
mkdir -p "${RUN_DIR}"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
if [ ! -s "$SUMMARY_CSV" ]; then
    echo "label,proposal_type,wall_time_sec,mean_rmse,std_rmse,mean_crps,std_crps,mean_ess,run_name,checkpoint,log_file" > "$SUMMARY_CSV"
fi

echo "=========================================="
echo "train_and_eval_inn_l96_dim5.sbatch"
echo "  job_id            = ${JOB_ID}"
echo "  inn_presets       = ${INN_PRESETS}"
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
# ModelCheckpoint filename is `inn-<NNN>-<val_nll>.ckpt` with
# auto_insert_metric_name=False; val_nll is negative on this data so
# the filename has two dashes in a row. Use the same Python picker as
# eval_nasmc_vs_rf_l96_dim5 (minimum val_nll = most-negative).
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
        # Write a summary row anyway so the CSV documents the failure.
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "inn_${preset#inn_}" "inn" "" "" "" "" "" "" "${train_run_name}" "" "${train_log}" \
            >> "$SUMMARY_CSV"
        return "$train_rc"
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
# Main loop: run each preset sequentially. Keep going on failures, but
# propagate a non-zero exit code at the end.
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
