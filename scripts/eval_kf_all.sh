#!/bin/bash

# Generated script to evaluate tuned EnKF/LETKF models
mkdir -p logs

echo 'Starting KF Evaluations...'

# echo 'Starting EnKF for ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_J64_L100p53 on GPU 1...'
# CUDA_VISIBLE_DEVICES=1 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
#     --data-dir /data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_J64_L100p53 \
#     --method enkf \
#     --n-particles 50 \
#     --process-noise-std 0.1 \
#     --obs-frequency 10 \
#     --inflation 1.2 \
#     --num-eval-trajectories 50 \
#     --batch-size 32 \
#     --device cuda \
#     --wandb-project kf-eval-ks \
#     --wandb-entity ml-climate \
#     --experiment-label tuned_enkf \
#     --wandb-tags tuned,best-config > logs/eval_ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_J64_L100p53_enkf.log 2>&1 &

# echo 'Starting EnKF for ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_J64_L50p27 on GPU 2...'
# CUDA_VISIBLE_DEVICES=2 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
#     --data-dir /data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_J64_L50p27 \
#     --method enkf \
#     --n-particles 50 \
#     --process-noise-std 0.1 \
#     --obs-frequency 10 \
#     --inflation 1.2 \
#     --num-eval-trajectories 50 \
#     --batch-size 32 \
#     --device cuda \
#     --wandb-project kf-eval-ks \
#     --wandb-entity ml-climate \
#     --experiment-label tuned_enkf \
#     --wandb-tags tuned,best-config > logs/eval_ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_J64_L50p27_enkf.log 2>&1 &

echo 'Starting EnKF for ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_J64_L100p53 on GPU 4...'
CUDA_VISIBLE_DEVICES=4 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
    --data-dir /data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_J64_L100p53 \
    --method enkf \
    --n-particles 50 \
    --process-noise-std 0.1 \
    --obs-frequency 10 \
    --inflation 1.0 \
    --num-eval-trajectories 50 \
    --batch-size 32 \
    --device cuda \
    --wandb-project kf-eval-ks \
    --wandb-entity ml-climate \
    --experiment-label tuned_enkf \
    --wandb-tags tuned,best-config > logs/eval_ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_J64_L100p53_enkf.log 2>&1 &

# echo 'Starting EnKF for ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_J64_L50p27 on GPU 6...'
# CUDA_VISIBLE_DEVICES=6 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
#     --data-dir /data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_J64_L50p27 \
#     --method enkf \
#     --n-particles 50 \
#     --process-noise-std 0.1 \
#     --obs-frequency 10 \
#     --inflation 1.0 \
#     --num-eval-trajectories 50 \
#     --batch-size 32 \
#     --device cuda \
#     --wandb-project kf-eval-ks \
#     --wandb-entity ml-climate \
#     --experiment-label tuned_enkf \
#     --wandb-tags tuned,best-config > logs/eval_ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_J64_L50p27_enkf.log 2>&1 &

echo 'All evaluation jobs started in background.'
wait
echo 'All evaluations completed.'
