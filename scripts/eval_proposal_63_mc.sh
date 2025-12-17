#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Starting Parallel Evaluation for Lorenz 63 (MC Guidance)"
echo "=========================================="

# 1. Lorenz 63, 0 noise, 1 dim observed (GPU 0)
echo "Starting Eval 1: L63, 0 noise, 1 dim obs on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp0_arctan/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan" \
    --mc-guidance \
    --run-name "eval_l63_comp0_mc" \
    > logs/eval_l63_comp0_mc.log 2>&1 &
PID1=$!
echo "Started PID $PID1"

# 2. Lorenz 63, 0 noise, 2 dim observed (GPU 1)
echo "Starting Eval 2: L63, 0 noise, 2 dim obs on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp01_arctan/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan" \
    --mc-guidance \
    --run-name "eval_l63_comp01_mc" \
    > logs/eval_l63_comp01_mc.log 2>&1 &
PID2=$!
echo "Started PID $PID2"

# 3. Lorenz 63, 0 noise, 3 dim observed (GPU 2)
echo "Starting Eval 3: L63, 0 noise, 3 dim obs on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp012_arctan/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan" \
    --mc-guidance \
    --run-name "eval_l63_comp012_mc" \
    > logs/eval_l63_comp012_mc.log 2>&1 &
PID3=$!
echo "Started PID $PID3"

# 4. Lorenz 63, 0.25 pnoise, 1 dim observed (GPU 3)
echo "Starting Eval 4: L63, 0.25 pnoise, 1 dim obs on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp0_arctan_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan_pnoise0p250" \
    --mc-guidance \
    --run-name "eval_l63_comp0_pnoise025_mc" \
    > logs/eval_l63_comp0_pnoise025_mc.log 2>&1 &
PID4=$!
echo "Started PID $PID4"

# 5. Lorenz 63, 0.25 pnoise, 2 dim observed (GPU 4)
echo "Starting Eval 5: L63, 0.25 pnoise, 2 dim obs on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp01_arctan_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan_pnoise0p250" \
    --mc-guidance \
    --run-name "eval_l63_comp01_pnoise025_mc" \
    > logs/eval_l63_comp01_pnoise_mc.log 2>&1 &
PID5=$!
echo "Started PID $PID5"

# 6. Lorenz 63, 0.25 pnoise, 3 dim observed (GPU 5)
echo "Starting Eval 6: L63, 0.25 pnoise, 3 dim obs on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp012_arctan_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250" \
    --mc-guidance \
    --run-name "eval_l63_comp012_pnoise025_mc" \
    > logs/eval_l63_comp012_pnoise_mc.log 2>&1 &
PID6=$!
echo "Started PID $PID6"

# 7. Lorenz 63, 0 noise, NO obs (evaluated with comp0 guidance) (GPU 6)
echo "Starting Eval 7: L63, 0 noise, NO obs (guide 0) on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_no_obs/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan" \
    --mc-guidance \
    --run-name "eval_l63_no_obs_guide0_mc" \
    > logs/eval_l63_no_obs_guide0_mc.log 2>&1 &
PID7=$!
echo "Started PID $PID7"

# 8. Lorenz 63, 0.25 pnoise, NO obs (evaluated with comp0 guidance) (GPU 7)
echo "Starting Eval 8: L63, 0.25 pnoise, NO obs (guide 0) on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_no_obs_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan_pnoise0p250" \
    --mc-guidance \
    --run-name "eval_l63_no_obs_pnoise025_guide0_mc" \
    > logs/eval_l63_no_obs_pnoise025_guide0_mc.log 2>&1 &
PID8=$!
echo "Started PID $PID8"

# 9. Lorenz 63, 0 noise, NO obs (evaluated with comp0,1 guidance) (GPU 0)
echo "Starting Eval 9: L63, 0 noise, NO obs (guide 0,1) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python proposals/eval_proposal.py \
        --checkpoint "rf_runs/lorenz63_no_obs/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan" \
    --mc-guidance \
    --run-name "eval_l63_no_obs_guide01_mc" \
    > logs/eval_l63_no_obs_guide01_mc.log 2>&1 &
PID9=$!
echo "Started PID $PID9"

# 10. Lorenz 63, 0 noise, NO obs (evaluated with comp0,1,2 guidance) (GPU 1)
echo "Starting Eval 10: L63, 0 noise, NO obs (guide 0,1,2) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_no_obs/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan" \
    --mc-guidance \
    --run-name "eval_l63_no_obs_guide012_mc" \
    > logs/eval_l63_no_obs_guide012_mc.log 2>&1 &
PID10=$!
echo "Started PID $PID10"

# 11. Lorenz 63, 0.25 pnoise, NO obs (evaluated with comp0,1 guidance) (GPU 2)
echo "Starting Eval 11: L63, 0.25 pnoise, NO obs (guide 0,1) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_no_obs_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan_pnoise0p250" \
    --mc-guidance \
    --run-name "eval_l63_no_obs_pnoise025_guide01_mc" \
    > logs/eval_l63_no_obs_pnoise025_guide01_mc.log 2>&1 &
PID11=$!
echo "Started PID $PID11"

# 12. Lorenz 63, 0.25 pnoise, NO obs (evaluated with comp0,1,2 guidance) (GPU 3)
echo "Starting Eval 12: L63, 0.25 pnoise, NO obs (guide 0,1,2) on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_no_obs_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250" \
    --mc-guidance \
    --run-name "eval_l63_no_obs_pnoise025_guide012_mc" \
    > logs/eval_l63_no_obs_pnoise025_guide012_mc.log 2>&1 &
PID12=$!
echo "Started PID $PID12"

echo "=========================================="
echo "Starting Parallel Evaluation for Lorenz 63 (NO Guidance)"
echo "=========================================="

# 13. Lorenz 63, 0 noise, 1 dim observed (GPU 4)
echo "Starting Eval 13: L63, 0 noise, 1 dim obs (no guide) on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp0_arctan/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan" \
    --run-name "eval_l63_comp0_nomc" \
    > logs/eval_l63_comp0_nomc.log 2>&1 &
PID13=$!
echo "Started PID $PID13"

# 14. Lorenz 63, 0 noise, 2 dim observed (GPU 5)
echo "Starting Eval 14: L63, 0 noise, 2 dim obs (no guide) on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp01_arctan/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan" \
    --run-name "eval_l63_comp01_nomc" \
    > logs/eval_l63_comp01_nomc.log 2>&1 &
PID14=$!
echo "Started PID $PID14"

# 15. Lorenz 63, 0 noise, 3 dim observed (GPU 6)
echo "Starting Eval 15: L63, 0 noise, 3 dim obs (no guide) on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp012_arctan/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan" \
    --run-name "eval_l63_comp012_nomc" \
    > logs/eval_l63_comp012_nomc.log 2>&1 &
PID15=$!
echo "Started PID $PID15"

# 16. Lorenz 63, 0.25 pnoise, 1 dim observed (GPU 7)
echo "Starting Eval 16: L63, 0.25 pnoise, 1 dim obs (no guide) on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp0_arctan_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan_pnoise0p250" \
    --run-name "eval_l63_comp0_pnoise025_nomc" \
    > logs/eval_l63_comp0_pnoise025_nomc.log 2>&1 &
PID16=$!
echo "Started PID $PID16"

# 17. Lorenz 63, 0.25 pnoise, 2 dim observed (GPU 0)
echo "Starting Eval 17: L63, 0.25 pnoise, 2 dim obs (no guide) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp01_arctan_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1_arctan_pnoise0p250" \
    --run-name "eval_l63_comp01_pnoise025_nomc" \
    > logs/eval_l63_comp01_pnoise025_nomc.log 2>&1 &
PID17=$!
echo "Started PID $PID17"

# 18. Lorenz 63, 0.25 pnoise, 3 dim observed (GPU 1)
echo "Starting Eval 18: L63, 0.25 pnoise, 3 dim obs (no guide) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_comp012_arctan_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250" \
    --run-name "eval_l63_comp012_pnoise025_nomc" \
    > logs/eval_l63_comp012_pnoise025_nomc.log 2>&1 &
PID18=$!
echo "Started PID $PID18"

# 19. Lorenz 63, 0 noise, NO obs (no guide) (GPU 2)
echo "Starting Eval 19: L63, 0 noise, NO obs (no guide) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_no_obs/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan" \
    --run-name "eval_l63_no_obs_nomc" \
    > logs/eval_l63_no_obs_nomc.log 2>&1 &
PID19=$!
echo "Started PID $PID19"

# 20. Lorenz 63, 0.25 pnoise, NO obs (no guide) (GPU 3)
echo "Starting Eval 20: L63, 0.25 pnoise, NO obs (no guide) on GPU 3..."
CUDA_VISIBLE_DEVICES=3 nohup python proposals/eval_proposal.py \
    --checkpoint "rf_runs/lorenz63_no_obs_pnoise0p250/final_model.ckpt" \
    --data-dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan_pnoise0p250" \
    --run-name "eval_l63_no_obs_pnoise025_nomc" \
    > logs/eval_l63_no_obs_pnoise025_nomc.log 2>&1 &
PID20=$!
echo "Started PID $PID20"

echo "=========================================="
echo "All 20 evaluation experiments started in background."
echo "Logs are being written to logs/"
echo "You can monitor them with: tail -f logs/eval_*.log"
echo "=========================================="
