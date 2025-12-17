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
echo "Starting Parallel Training"
echo "=========================================="

# # 1. RF No Observations (Pure Transition Learning)
# # This model learns p(x_t | x_{t-1}) without seeing y_t
# echo "Starting Exp 1: RF No Observations on GPU 0..."
# CUDA_VISIBLE_DEVICES=1 nohup python proposals/train_rf.py \
#     --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p010_freq1_comp0_arctan" \
#     --output_dir "rf_runs/lorenz63_n1024_len100_obs0p010_freq1_comp0_arctan_no_obs_new" \
#     --max_epochs 500 \
#     --batch_size 64 \
#     --gpus 1 \
#     --evaluate \
#     > logs/train_rf_no_obs.log 2>&1 &
# PID1=$!
# echo "Started PID $PID1"

# # 2. Concat conditioning (Baseline with Observations)
# # This model learns p(x_t | x_{t-1}, y_t) using concatenation
# echo "Starting Exp 2: RF with Concat conditioning on GPU 2..."
# CUDA_VISIBLE_DEVICES=2 nohup python proposals/train_rf.py \
#     --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p010_freq1_comp0_arctan" \
#     --output_dir "rf_runs/lorenz63_n1024_len100_obs0p010_freq1_comp0_arctan_concat" \
#     --max_epochs 500 \
#     --batch_size 64 \
#     --gpus 1 \
#     --evaluate \
#     --use_observations \
#     --obs_dim 1 \
#     --conditioning_method concat \
#     > logs/train_rf_concat.log 2>&1 &
# PID2=$!
# echo "Started PID $PID2"

# # 3. FiLM conditioning
# echo "Starting Exp 3: RF with FiLM conditioning on GPU 3..."
# CUDA_VISIBLE_DEVICES=3 nohup python proposals/train_rf.py \
#     --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p010_freq1_comp0_arctan" \
#     --output_dir "rf_runs/lorenz63_n1024_len100_obs0p010_freq1_comp0_arctan_film" \
#     --max_epochs 500 \
#     --batch_size 64 \
#     --gpus 1 \
#     --evaluate \
#     --use_observations \
#     --obs_dim 1 \
#     --conditioning_method film \
#     > logs/train_rf_film.log 2>&1 &
# PID3=$!
# echo "Started PID $PID3"

# # 4. AdaLN conditioning
# echo "Starting Exp 4: RF with AdaLN conditioning on GPU 4..."
# CUDA_VISIBLE_DEVICES=5 nohup python proposals/train_rf.py \
#     --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p250_freq1_comp0_arctan_pnoise0p250" \
#     --output_dir "rf_runs/lorenz63_n1024_len100_obs0p250_freq1_comp0_arctan_adaln_pnoise0p250" \
#     --max_epochs 500 \
#     --batch_size 64 \
#     --gpus 1 \
#     --evaluate \
#     --use_observations \
#     --obs_dim 1 \
#     --conditioning_method adaln \
#     > logs/train_rf_adaln_pnoise0p250.log 2>&1 &
# PID4=$!
# echo "Started PID $PID4"

# # 5. Cross Attention conditioning
# echo "Starting Exp 5: RF with Cross Attention conditioning on GPU 5..."
# CUDA_VISIBLE_DEVICES=5 nohup python proposals/train_rf.py \
#     --data_dir "datasets/lorenz63_n1024_len100_dt0p0100_obs0p010_freq1_comp0_arctan" \
#     --output_dir "rf_runs/lorenz63_n1024_len100_obs0p010_freq1_comp0_arctan_cross_attn" \
#     --max_epochs 500 \
#     --batch_size 64 \
#     --gpus 1 \
#     --evaluate \
#     --use_observations \
#     --obs_dim 1 \
#     --conditioning_method cross_attn \
#     --num_attn_heads 4 \
#     > logs/train_rf_cross_attn.log 2>&1 &
# PID5=$!
# echo "Started PID $PID5"

# # 6. Lorenz96 ResNet1D Concat (Standard)
# echo "Starting Exp 6: Lorenz96 ResNet1D Concat on GPU 6..."
# CUDA_VISIBLE_DEVICES=4 nohup python proposals/train_rf.py \
#     --data_dir "datasets/lorenz96_n4096_len100_dt0p0100_obs0p100_freq1_comp50of50_arctan" \
#     --output_dir "rf_runs/lorenz96_resnet1d_concat_no_resid_fixinput" \
#     --state_dim 50 \
#     --obs_dim 50 \
#     --architecture resnet1d \
#     --channels 64 \
#     --num_blocks 10 \
#     --kernel_size 5 \
#     --conditioning_method concat \
#     --cond_embed_dim 128 \
#     --use_observations \
#     --batch_size 1024 \
#     --learning_rate 3e-4 \
#     --max_epochs 500 \
#     --gpus 1 \
#     --evaluate \
#     --time_embed_dim 32 \
#     > logs/train_lorenz96_resnet1d_concat.log 2>&1 &
# PID6=$!
# echo "Started PID $PID6"

# # 7. Lorenz96 ResNet1D Concat (Predict Delta)
# echo "Starting Exp 7: Lorenz96 ResNet1D Concat Predict Delta on GPU 5..."
# CUDA_VISIBLE_DEVICES=3 nohup python proposals/train_rf.py \
#     --data_dir "datasets/lorenz96_n4096_len100_dt0p0100_obs0p100_freq1_comp50of50_arctan" \
#     --output_dir "rf_runs/lorenz96_resnet1d_concat_resid_fixinput" \
#     --state_dim 50 \
#     --obs_dim 50 \
#     --architecture resnet1d \
#     --channels 64 \
#     --num_blocks 10 \
#     --kernel_size 5 \
#     --conditioning_method concat \
#     --cond_embed_dim 128 \
#     --use_observations \
#     --batch_size 1024 \
#     --learning_rate 3e-4 \
#     --max_epochs 500 \
#     --gpus 1 \
#     --predict_delta \
#     --evaluate \
#     --time_embed_dim 32 \
#     > logs/train_lorenz96_resnet1d_concat_delta.log 2>&1 &
# PID7=$!
# echo "Started PID $PID7"

# 8. Lorenz96 ResNet1D Fixed Architecture
# echo "Starting Exp 8: Lorenz96 ResNet1D Fixed Architecture on GPU 7..."
# CUDA_VISIBLE_DEVICES=7 nohup python proposals/train_rf.py \
#     --data_dir "datasets/lorenz96_n4096_len100_dt0p0100_obs0p100_freq1_comp50of50_arctan" \
#     --output_dir "rf_runs/lorenz96_resnet1d_fixed" \
#     --state_dim 50 \
#     --obs_dim 50 \
#     --architecture resnet1d_fixed \
#     --channels 64 \
#     --num_blocks 10 \
#     --kernel_size 5 \
#     --use_observations \
#     --batch_size 1024 \
#     --learning_rate 3e-4 \
#     --max_epochs 500 \
#     --gpus 1 \
#     --predict_delta \
#     --evaluate \
#     --time_embed_dim 32 \
#     > logs/train_lorenz96_resnet1d_fixed.log 2>&1 &
# PID8=$!
# echo "Started PID $PID8"

# 9. Lorenz96 ResNet1D Fixed Architecture (Partial Observations)
# First generate the dataset if it doesn't exist
# We use Python to generate the comma-separated list of even indices
OBS_INDICES_STR=$(python -c "print(','.join(map(str, range(0, 50, 2))))")

echo "Generating partial observation dataset..."
python generate.py \
    --system lorenz96 \
    --l96-dim 50 \
    --num-trajectories 4096 \
    --len-trajectory 100 \
    --dt 0.01 \
    --obs-noise-std 0.1 \
    --obs-frequency 1 \
    --obs-components "$OBS_INDICES_STR" \
    --dataset-name "lorenz96_n4096_partial_obs_every2" \
    --output-dir "datasets"

echo "Starting Exp 9: Lorenz96 ResNet1D Fixed Architecture (Partial Obs) on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup python proposals/train_rf.py \
    --data_dir "datasets/lorenz96_n4096_partial_obs_every2" \
    --output_dir "rf_runs/lorenz96_resnet1d_fixed_partial" \
    --state_dim 50 \
    --obs_dim 25 \
    --architecture resnet1d_fixed \
    --channels 64 \
    --num_blocks 10 \
    --kernel_size 5 \
    --use_observations \
    --obs-indices "$OBS_INDICES_STR" \
    --batch_size 1024 \
    --learning_rate 3e-4 \
    --max_epochs 500 \
    --gpus 1 \
    --predict_delta \
    --evaluate \
    --time_embed_dim 32 \
    > logs/train_lorenz96_resnet1d_fixed_partial.log 2>&1 &
PID9=$!
echo "Started PID $PID9"

echo "=========================================="
echo "All training experiments started in background."
echo "Logs are being written to logs/"
echo "  - logs/train_rf_no_obs.log"
echo "  - logs/train_rf_concat.log"
echo "  - logs/train_rf_film.log"
echo "  - logs/train_rf_adaln.log"
echo "You can monitor them with: tail -f logs/*.log"
echo "=========================================="