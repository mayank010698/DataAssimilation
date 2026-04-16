#!/bin/bash
#
# Slurm: BPF evaluation on L96 dim5 — FM teacher vs F2D2 MeanFlow.
#
# Usage: cd .../DataAssimilation && sbatch scripts/eval_96_dim5_f2d2_meanflow.sbatch
#
#SBATCH --job-name="eval-96-mf"
#SBATCH --partition=dali
#SBATCH --account=cnagda2-ic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=ccc0424
#SBATCH --cpus-per-task=8
#SBATCH --mem=80g
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
    echo "ERROR: Cannot find project root with eval.py."
    echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
    echo "  script dir=$SCRIPT_DIR"
    echo "  tried PROJECT_ROOT=$PROJECT_ROOT"
    echo "Run: cd /path/to/DataAssimilation && sbatch scripts/eval_96_dim5_f2d2_meanflow.sbatch"
    exit 1
fi
cd "$PROJECT_ROOT" || exit 1

module load anaconda3/2024.10
source activate
conda activate da

module load cuda/12.8

export PYTHONPATH="$(pwd)"

DATA_DIR="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_init3p000"
OBS_COMPONENTS="0,1,2,3,4"

TEACHER_CKPT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims/fm_dim5_nomask_7850189/checkpoints/ema_weights.ckpt"
MEANFLOW_CKPT="/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/rf_runs_96_dims/mf_f2d2_dim5_7890621/checkpoints/meanflow_ema_weights.ckpt"

WANDB_PROJECT="eval-96-dim-f2d2test"
WANDB_ENTITY="ml-climate"

N_PARTICLES=1000
BATCH_SIZE=100
NUM_EVAL_TRAJ=10
METHOD="bpf"
SEED=0

EVAL_PY="${PROJECT_ROOT}/eval.py"

echo "=========================================="
echo "BPF eval — teacher (RF) vs F2D2 MeanFlow"
echo "Project root: $PROJECT_ROOT"
echo "Data: $DATA_DIR"
echo "Teacher: $TEACHER_CKPT"
echo "MeanFlow: $MEANFLOW_CKPT"
echo "W&B project: $WANDB_PROJECT"
echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

if [ ! -f "$EVAL_PY" ]; then
    echo "ERROR: eval.py not found at $EVAL_PY"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR does not exist: $DATA_DIR"
    exit 1
fi
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: TEACHER_CKPT not found: $TEACHER_CKPT"
    exit 1
fi
if [ ! -f "$MEANFLOW_CKPT" ]; then
    echo "ERROR: MEANFLOW_CKPT not found: $MEANFLOW_CKPT"
    exit 1
fi

run_eval() {
    local proposal_type="$1"
    local ckpt="$2"
    local samp="$3"
    local like="$4"
    local label="$5"
    local run_name="$6"

    echo ""
    echo "---- $label (sample=$samp, likelihood=$like) ----"
    python "$EVAL_PY" \
        --data-dir "$DATA_DIR" \
        --obs-components "$OBS_COMPONENTS" \
        --n-particles "$N_PARTICLES" \
        --batch-size "$BATCH_SIZE" \
        --proposal-type "$proposal_type" \
        --rf-checkpoint "$ckpt" \
        --rf-sampling-steps "$samp" \
        --rf-likelihood-steps "$like" \
        --num-eval-trajectories "$NUM_EVAL_TRAJ" \
        --method "$METHOD" \
        --experiment-label "$label" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY" \
        --run-name "$run_name" \
        --device cuda \
        --seed "$SEED"
}

# MeanFlow: (sampling steps, likelihood steps)
MF_PAIRS=(
    "25 4"
    "25 3"
    "25 2"
    "25 1"
    "20 4"
    "20 3"
    "20 2"
    "20 1"
    "10 4"
    "10 3"
    "10 2"
    "10 1"
    "5 4"
    "5 3"
    "5 2"
    "5 1"
)

# RUNS entries: proposal_type|checkpoint|samp|like|experiment_label|run_name_prefix
RUNS=(
    "rf|${TEACHER_CKPT}|100|100|eval_dim5_teacher_fm_s100_l100|eval_dim5_teacher_fm_s100_l100"
)
for pair in "${MF_PAIRS[@]}"; do
    read -r ss ls <<< "$pair"
    tag="eval_dim5_meanflow_s${ss}_l${ls}"
    RUNS+=( "meanflow|${MEANFLOW_CKPT}|${ss}|${ls}|${tag}|${tag}" )
done

NUM_RUNS=${#RUNS[@]}

LOG_ROOT="/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/eval_96_meanflow_job${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_ROOT"

echo "Launching $NUM_RUNS eval processes on one GPU (logs under $LOG_ROOT)"
pids=()
for ((i = 0; i < NUM_RUNS; i++)); do
    spec="${RUNS[$i]}"
    proposal_type="" ckpt="" samp="" like="" label="" run_prefix=""
    IFS='|' read -r proposal_type ckpt samp like label run_prefix <<< "$spec"
    run_name="${run_prefix}_${SLURM_JOB_ID:-local}"
    logfile="${LOG_ROOT}/$(printf '%02d' "$i")_${label}.log"
    echo "  [$i] $label  (pid -> log: $logfile)"
    (
        _i=$i _pt=$proposal_type _ck=$ckpt _sm=$samp _lk=$like _lb=$label _rn=$run_name
        echo "=== start $(date -Is) job=${SLURM_JOB_ID:-local} task_index=${_i} label=${_lb} ==="
        run_eval "$_pt" "$_ck" "$_sm" "$_lk" "$_lb" "$_rn"
        ec=$?
        echo "=== end exit_code=${ec} ==="
        exit "$ec"
    ) > "$logfile" 2>&1 &
    pids+=($!)
done

set +e
fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=1
    fi
done
set -e

echo ""
if [ "$fail" -ne 0 ]; then
    echo "One or more eval processes failed (see logs in $LOG_ROOT)."
    exit 1
fi
echo "All $NUM_RUNS eval processes finished OK. Logs: $LOG_ROOT"
