#!/bin/bash

# Simple launcher for KS temporal-sparsity evals on a single dataset.
# Runs: ENSF, ENKF, FPPF (BPF+RF), BPF (transition), and EnSF+RF,
# each pinned to one of GPUs 3–7 with date-prefixed wandb run names.

set -e

# Get the directory where this script is located and cd to project root
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.." || exit 1

export PYTHONPATH=$(pwd)

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------

# GPUs to use (one run per GPU)
GPUS=(3 4 5 6 7)

# Dataset (your KS temporal sparsity setup)
DATA_DIR="/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_pnoise0p100_J64_L50p27"

# Common args
OBS_FREQ=10
PROC_NOISE=0.1
OBS_NOISE=0.1
N_TRAJ=25
BATCH_SIZE=5
WANDB_PROJECT="eval-ks-temp-sparse"
EXPERIMENT_LABEL="temporal_sparsity"
DEVICE="cuda"

# Rectified Flow checkpoint for RF-based methods
RF_CKPT="/data/da_outputs/rf_runs_ks/ks_pnoise0p1_J64_16pi_cd0p1_20260117_192257/final_model.ckpt"
RF_LIK_STEPS=100
RF_SAMP_STEPS=100

# EnSF hyperparameters (match your earlier ENSF settings, but with current CLI names)
ENSF_STEPS=1000
ENSF_EPS_A=0.5   # small time cutoff
ENSF_EPS_B=0.025 # initial noise variance

# Date prefix for wandb run names (YYYYMMDD)
DATE_PREFIX=$(date +%Y%m%d)

mkdir -p logs

echo "Running KS temporal-sparsity evals on dataset:"
echo "  $DATA_DIR"
echo "Using GPUs: ${GPUS[*]}"
echo "Date prefix: $DATE_PREFIX"

# # ------------------------------------------------------------------------
# # 1) ENSF (transition proposal)
# # ------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=${GPUS[0]} python eval.py \
#   --data-dir "$DATA_DIR" \
#   --method ensf \
#   --obs-frequency "$OBS_FREQ" \
#   --process-noise-std "$PROC_NOISE" \
#   --obs-noise-std "$OBS_NOISE" \
#   --num-eval-trajectories "$N_TRAJ" \
#   --batch-size "$BATCH_SIZE" \
#   --n-particles 100 \
#   --ensf-steps "$ENSF_STEPS" \
#   --ensf-eps-a "$ENSF_EPS_A" \
#   --ensf-eps-b "$ENSF_EPS_B" \
#   --wandb-project "$WANDB_PROJECT" \
#   --run-name "${DATE_PREFIX}_ensf_16pi_freq10" \
#   --device "$DEVICE" \
#   --experiment-label "$EXPERIMENT_LABEL" \
#   > "logs/${DATE_PREFIX}_ensf_16pi_freq10.log" 2>&1 &

# # ------------------------------------------------------------------------
# # 2) ENKF (standard)
# # ------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=${GPUS[1]} python eval.py \
#   --data-dir "$DATA_DIR" \
#   --method enkf \
#   --obs-frequency "$OBS_FREQ" \
#   --process-noise-std "$PROC_NOISE" \
#   --obs-noise-std "$OBS_NOISE" \
#   --num-eval-trajectories "$N_TRAJ" \
#   --batch-size "$BATCH_SIZE" \
#   --rf-sampling-steps 100 \
#   --rf-likelihood-steps 100 \
#   --wandb-project "$WANDB_PROJECT" \
#   --run-name "${DATE_PREFIX}_enkf_16pi_freq10" \
#   --device "$DEVICE" \
#   --experiment-label "$EXPERIMENT_LABEL" \
#   --n-particles 50 \
#   --inflation 1.0 \
#   > "logs/${DATE_PREFIX}_enkf_16pi_freq10.log" 2>&1 &

# # ------------------------------------------------------------------------
# # 3) FPPF = BPF with RF proposal
# # ------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=${GPUS[2]} python eval.py \
#   --data-dir "$DATA_DIR" \
#   --method bpf \
#   --proposal-type rf \
#   --rf-checkpoint "$RF_CKPT" \
#   --rf-likelihood-steps "$RF_LIK_STEPS" \
#   --rf-sampling-steps "$RF_SAMP_STEPS" \
#   --obs-frequency "$OBS_FREQ" \
#   --process-noise-std "$PROC_NOISE" \
#   --obs-noise-std "$OBS_NOISE" \
#   --num-eval-trajectories "$N_TRAJ" \
#   --batch-size "$BATCH_SIZE" \
#   --n-particles 5000 \
#   --wandb-project "$WANDB_PROJECT" \
#   --run-name "${DATE_PREFIX}_fppf_16pi_freq10" \
#   --device "$DEVICE" \
#   --experiment-label "$EXPERIMENT_LABEL" \
#   > "logs/${DATE_PREFIX}_fppf_16pi_freq10.log" 2>&1 &

# # ------------------------------------------------------------------------
# # 4) BPF (transition proposal)
# # ------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=${GPUS[3]} python eval.py \
#   --data-dir "$DATA_DIR" \
#   --method bpf \
#   --proposal-type transition \
#   --obs-frequency "$OBS_FREQ" \
#   --process-noise-std "$PROC_NOISE" \
#   --obs-noise-std "$OBS_NOISE" \
#   --num-eval-trajectories "$N_TRAJ" \
#   --batch-size 10 \
#   --n-particles 5000 \
#   --wandb-project "$WANDB_PROJECT" \
#   --run-name "${DATE_PREFIX}_bpf_16pi_freq10" \
#   --device "$DEVICE" \
#   --experiment-label "$EXPERIMENT_LABEL" \
#   > "logs/${DATE_PREFIX}_bpf_16pi_freq10.log" 2>&1 &

# ------------------------------------------------------------------------
# 5) EnSF with RF proposal (new)
# ------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPUS[4]} python eval.py \
  --data-dir "$DATA_DIR" \
  --method ensf \
  --proposal-type rf \
  --rf-checkpoint "$RF_CKPT" \
  --rf-likelihood-steps "$RF_LIK_STEPS" \
  --rf-sampling-steps "$RF_SAMP_STEPS" \
  --obs-frequency "$OBS_FREQ" \
  --process-noise-std "$PROC_NOISE" \
  --obs-noise-std "$OBS_NOISE" \
  --num-eval-trajectories "$N_TRAJ" \
  --batch-size "$BATCH_SIZE" \
  --n-particles 100 \
  --ensf-steps "$ENSF_STEPS" \
  --ensf-eps-a "$ENSF_EPS_A" \
  --ensf-eps-b "$ENSF_EPS_B" \
  --wandb-project "$WANDB_PROJECT" \
  --run-name "${DATE_PREFIX}_ensf_rf_16pi_freq10" \
  --device "$DEVICE" \
  --experiment-label "$EXPERIMENT_LABEL" \
  --ensf-fallback-physical \
  > "logs/${DATE_PREFIX}_ensf_rf_16pi_freq10.log" 2>&1 &

echo "Launched ENSF, ENKF, FPPF, BPF, and EnSF+RF runs."
echo "Logs in logs/${DATE_PREFIX}_*_16pi_freq10.log"

wait

echo "All KS temporal-sparsity evals completed."

