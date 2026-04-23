#!/bin/bash
#
# Train Paige-Wood inference-network (IN) proposals across the NeurIPS
# L96 grid at obs noise 0.2, SEQUENTIALLY on a single A100 80GB GPU.
#
# Grid: dims {5,10,15,20,25,50} x obs operators {arctan, quad_capped_10}
#       x IN presets {inn_mog, inn_rnade}   = 24 runs total.
#
# Two proposals per dataset (same pair as the dim=5 smoke test in
# scripts/train_and_eval_inn_l96_dim5.sbatch):
#   * inn_mog   -- joint mixture of K=8 diagonal Gaussians (recommended
#                  default Paige-Wood head).
#   * inn_rnade -- paper-faithful conditional autoregressive MoG, Paige
#                  & Wood ICML 2016.
#
# Mirrors scripts/train_nasmc_l96_neurips_grid.sbatch (same dataset /
# operator / dim axes and obs_noise=0.2) so the IN rows can be compared
# row-for-row against the NASMC rows. Unlike that script we DO NOT use
# a Slurm job array: the user has maxed out other partitions and we're
# running the whole grid in one sequential job on the dali A100 80GB.
#
#SBATCH --job-name=train-in-l96-neurips
#SBATCH --partition=dali
#SBATCH --account=dali
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/projects/illinois/eng/cs/arindamb/cnagda2/slurm_output/%j.log

set -euo pipefail

echo "=== Slurm context ==="
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-N/A}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "====================="

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/eval.py" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="${PROJECT_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/DataAssimilation}"
fi
if [ ! -f "${PROJECT_ROOT}/eval.py" ]; then
    echo "ERROR: Cannot find project root (expected eval.py at ${PROJECT_ROOT})." >&2
    exit 1
fi
cd "${PROJECT_ROOT}" || exit 1

module load anaconda3/2024.10
source activate
conda activate da
module load cuda/12.8

export PYTHONPATH="${PROJECT_ROOT}"
# Reduce GPU fragmentation across the 24 back-to-back runs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-manual}"

# ==========================================================================
# Grid definition
# ==========================================================================
DIMS=(5 10 15 20 25 50)
# Directory tokens under DATA_ROOT (see generate.py / data.py naming).
OPS_DIR=(arctan quad_capped_10)
# Short tags for paths / W&B run names.
OPS_TAG=(arctan quad_capped)
# Two IN proposals per dataset; tags must be filesystem-safe.
PRESETS=(inn_mog inn_rnade)
PRESETS_TAG=(mog rnade)

# Fixed obs noise for this grid (matches train_nasmc_l96_neurips_grid).
OBS_NOISE=0.2
OBS_TOKEN=obs0p200

# ==========================================================================
# Shared training config
# ==========================================================================
# Feature-backbone capacity -- identical to the dim=5 smoke test and to the
# matching NASMC runs, so q_phi capacity is comparable across methods.
ARCH=resnet1d
CHANNELS=64
NUM_BLOCKS=10
KERNEL_SIZE=5
TIME_EMBED_DIM=128
FEATURE_DIM=128

# IN is pure supervised MLE -- no SMC in the loop -- so 500 epochs with
# patience=40 matches the total epoch budget the NASMC runs spend on
# MLE warmstart + SMC refinement.
BATCH_SIZE=512
LR=3e-4
MAX_EPOCHS=500
PATIENCE=40

# K=3 matches the Paige & Wood 2016 ICML paper (nonlinear SSM experiment:
# "a mixture of 3 Gaussians as each output") and the K=3 "-MD-" variant
# used by train_nasmc_l96_neurips_grid.sbatch, so IN and NASMC rows are
# directly comparable at the same mixture order. The --preset default in
# proposals/train_inference_network.py is K=8 (a codebase choice, not
# paper-faithful); we override it here.
NUM_MIXTURE_COMPONENTS=3
# RNADE head-specific (only used when preset=inn_rnade).
RNADE_HIDDEN_SIZE=128
RNADE_NUM_HIDDEN_LAYERS=2

SEED=42

DATA_ROOT="${DATA_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/datasets/l96_dims}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/projects/illinois/eng/cs/arindamb/cnagda2/da/da_outputs/inn_runs_l96_neurips}"
mkdir -p "${OUTPUT_ROOT}"

WANDB_PROJECT="${WANDB_PROJECT:-inn-train-96-neurips}"
WANDB_ENTITY="${WANDB_ENTITY:-ml-climate}"

TOTAL=$(( ${#DIMS[@]} * ${#OPS_DIR[@]} * ${#PRESETS[@]} ))

echo "=========================================="
echo "train_in_l96_neurips_grid.sbatch"
echo "  job_id          = ${JOB_ID}"
echo "  dims            = ${DIMS[*]}"
echo "  obs operators   = ${OPS_DIR[*]}"
echo "  IN presets      = ${PRESETS[*]}"
echo "  obs_noise_std   = ${OBS_NOISE}  (${OBS_TOKEN})"
echo "  total runs      = ${TOTAL}  (sequential on 1 GPU)"
echo "  data root       = ${DATA_ROOT}"
echo "  output root     = ${OUTPUT_ROOT}"
echo "  wandb project   = ${WANDB_PROJECT}"
echo "  wandb entity    = ${WANDB_ENTITY}"
echo "=========================================="

# ==========================================================================
# Per-run driver
# ==========================================================================
run_one() {
    local state_dim="$1"
    local obs_op_dir="$2"
    local obs_op_tag="$3"
    local preset="$4"
    local preset_tag="$5"

    local dataset_name="lorenz96_n2048_len200_dt0p0100_${OBS_TOKEN}_freq1_comp${state_dim}of${state_dim}_${obs_op_dir}_pnoise0p100_init3p000"
    local data_dir="${DATA_ROOT%/}/${dataset_name}"
    if [[ ! -d "${data_dir}" ]]; then
        echo "ERROR: missing dataset directory: ${data_dir}" >&2
        return 2
    fi

    local obs_components
    obs_components="$(seq -s, 0 "$((state_dim - 1))")"

    local run_name="inn_l96_d${state_dim}_${OBS_TOKEN}_${obs_op_tag}_${preset_tag}_job${JOB_ID}"
    local output_dir="${OUTPUT_ROOT%/}/${run_name}"
    local train_log="logs/${run_name}.log"
    mkdir -p "${output_dir}"

    echo "------------------------------------------"
    echo "  state_dim      = ${state_dim}"
    echo "  obs_operator   = ${obs_op_dir}  (tag=${obs_op_tag})"
    echo "  preset         = ${preset}  (tag=${preset_tag})"
    echo "  data_dir       = ${data_dir}"
    echo "  output_dir     = ${output_dir}"
    echo "  train_log      = ${train_log}"
    echo "------------------------------------------"

    python -m proposals.train_inference_network \
        --preset "${preset}" \
        --data_dir "${data_dir}" \
        --output_dir "${output_dir}" \
        --state_dim "${state_dim}" \
        --obs_components "${obs_components}" \
        --use_observations \
        --architecture "${ARCH}" \
        --channels "${CHANNELS}" \
        --num_blocks "${NUM_BLOCKS}" \
        --kernel_size "${KERNEL_SIZE}" \
        --time_embed_dim "${TIME_EMBED_DIM}" \
        --feature_dim "${FEATURE_DIM}" \
        --num_mixture_components "${NUM_MIXTURE_COMPONENTS}" \
        --rnade_hidden_size "${RNADE_HIDDEN_SIZE}" \
        --rnade_num_hidden_layers "${RNADE_NUM_HIDDEN_LAYERS}" \
        --predict_delta \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LR}" \
        --max_epochs "${MAX_EPOCHS}" \
        --patience "${PATIENCE}" \
        --num_workers 4 \
        --gpus 1 \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_entity "${WANDB_ENTITY}" \
        --wandb_run_name "${run_name}" \
        --seed "${SEED}" \
        > "${train_log}" 2>&1
}

# ==========================================================================
# Main loop: 6 dims * 2 ops * 2 presets = 24 runs, SEQUENTIAL.
# Keep going on failures; propagate a non-zero exit code at the end.
# ==========================================================================
EXIT_CODE=0
run_count=0
FAILED_RUNS=()

for dim in "${DIMS[@]}"; do
    for op_idx in "${!OPS_DIR[@]}"; do
        op_dir="${OPS_DIR[$op_idx]}"
        op_tag="${OPS_TAG[$op_idx]}"
        for preset_idx in "${!PRESETS[@]}"; do
            preset="${PRESETS[$preset_idx]}"
            preset_tag="${PRESETS_TAG[$preset_idx]}"
            run_count=$(( run_count + 1 ))

            echo ""
            echo "##########################################"
            echo "# [${run_count}/${TOTAL}] dim=${dim}  op=${op_tag}  preset=${preset_tag}"
            echo "##########################################"

            rc=0
            run_one "${dim}" "${op_dir}" "${op_tag}" "${preset}" "${preset_tag}" || rc=$?
            if [ "${rc}" -ne 0 ]; then
                echo "  -> FAILED (rc=${rc}), continuing..."
                FAILED_RUNS+=("d${dim}/${op_tag}/${preset_tag}(rc=${rc})")
                EXIT_CODE=1
            else
                echo "  -> OK"
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "All IN training runs complete (exit ${EXIT_CODE})."
echo "  runs attempted = ${run_count} / ${TOTAL}"
echo "  runs failed    = ${#FAILED_RUNS[@]}"
if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
    echo "  failed list:"
    for f in "${FAILED_RUNS[@]}"; do
        echo "    - ${f}"
    done
fi
echo "Output root: ${OUTPUT_ROOT}"
echo "Wandb:       https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "=========================================="

exit "${EXIT_CODE}"
