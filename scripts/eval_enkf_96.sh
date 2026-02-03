#!/bin/bash

# Generated script to evaluate tuned EnKF models on Lorenz96 datasets
mkdir -p logs

echo 'Starting EnKF Evaluations (L96 Only)...'

echo 'Starting batch 1...'
echo 'Starting EnKF for lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp10of10_arctan_init3p000 on GPU 0 (Job 0)...'
CUDA_VISIBLE_DEVICES=0 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
    --data-dir /data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp10of10_arctan_init3p000 \
    --method enkf \
    --n-particles 50 \
    --process-noise-std 0.1 \
    --inflation 1.0 \
    --num-eval-trajectories 50 \
    --batch-size 32 \
    --device cuda \
    --wandb-project kf-eval-96 \
    --wandb-entity ml-climate \
    --experiment-label tuned_enkf \
    --wandb-tags tuned,best-config > logs/eval_lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp10of10_arctan_init3p000_enkf.log 2>&1 &

echo 'Starting EnKF for lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp15of15_arctan_init3p000 on GPU 1 (Job 1)...'
CUDA_VISIBLE_DEVICES=1 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
    --data-dir /data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp15of15_arctan_init3p000 \
    --method enkf \
    --n-particles 50 \
    --process-noise-std 0.1 \
    --inflation 1.0 \
    --num-eval-trajectories 50 \
    --batch-size 32 \
    --device cuda \
    --wandb-project kf-eval-96 \
    --wandb-entity ml-climate \
    --experiment-label tuned_enkf \
    --wandb-tags tuned,best-config > logs/eval_lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp15of15_arctan_init3p000_enkf.log 2>&1 &

echo 'Starting EnKF for lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp20of20_arctan_init3p000 on GPU 2 (Job 2)...'
CUDA_VISIBLE_DEVICES=2 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
    --data-dir /data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp20of20_arctan_init3p000 \
    --method enkf \
    --n-particles 50 \
    --process-noise-std 0.1 \
    --inflation 1.0 \
    --num-eval-trajectories 50 \
    --batch-size 32 \
    --device cuda \
    --wandb-project kf-eval-96 \
    --wandb-entity ml-climate \
    --experiment-label tuned_enkf \
    --wandb-tags tuned,best-config > logs/eval_lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp20of20_arctan_init3p000_enkf.log 2>&1 &

echo 'Starting EnKF for lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_init3p000 on GPU 3 (Job 3)...'
CUDA_VISIBLE_DEVICES=3 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
    --data-dir /data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_init3p000 \
    --method enkf \
    --n-particles 50 \
    --process-noise-std 0.1 \
    --inflation 1.1 \
    --num-eval-trajectories 50 \
    --batch-size 32 \
    --device cuda \
    --wandb-project kf-eval-96 \
    --wandb-entity ml-climate \
    --experiment-label tuned_enkf \
    --wandb-tags tuned,best-config > logs/eval_lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_init3p000_enkf.log 2>&1 &

echo 'Starting EnKF for lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_init3p000 on GPU 4 (Job 4)...'
CUDA_VISIBLE_DEVICES=4 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
    --data-dir /data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_init3p000 \
    --method enkf \
    --n-particles 50 \
    --process-noise-std 0.1 \
    --inflation 1.1 \
    --num-eval-trajectories 50 \
    --batch-size 32 \
    --device cuda \
    --wandb-project kf-eval-96 \
    --wandb-entity ml-climate \
    --experiment-label tuned_enkf \
    --wandb-tags tuned,best-config > logs/eval_lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_init3p000_enkf.log 2>&1 &

echo 'Starting EnKF for lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_init3p000 on GPU 5 (Job 5)...'
CUDA_VISIBLE_DEVICES=5 /home/cnagda/miniconda3/envs/da/bin/python eval.py \
    --data-dir /data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_init3p000 \
    --method enkf \
    --n-particles 50 \
    --process-noise-std 0.1 \
    --inflation 1.0 \
    --num-eval-trajectories 50 \
    --batch-size 32 \
    --device cuda \
    --wandb-project kf-eval-96 \
    --wandb-entity ml-climate \
    --experiment-label tuned_enkf \
    --wandb-tags tuned,best-config > logs/eval_lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_init3p000_enkf.log 2>&1 &

wait
echo 'Batch 1 completed.'

echo 'All evaluations completed.'