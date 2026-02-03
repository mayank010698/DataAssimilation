#!/bin/bash

mkdir -p logs
mkdir -p evaluation_outputs/flowdas_dw

echo "Starting FlowDAS Double Well Evaluations..."

# Job 1: Standard noise (GPU 1)
echo "Starting dw_std on GPU 1..."
CUDA_VISIBLE_DEVICES=1 /home/cnagda/miniconda3/envs/da/bin/python scripts/eval_flowdas.py \
    --checkpoint runs/flowdas_dw_std/checkpoint_best.pth \
    --config datasets/double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p200/config.yaml \
    --data_dir datasets/double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p200 \
    --results_dir evaluation_outputs/flowdas_dw/dw_std \
    --device cuda \
    --use_wandb \
    --num_trajs 50 \
    --wandb_project flowdas-eval-dw \
    --run_name flowdas_dw_std_best > logs/eval_flowdas_dw_std.log 2>&1 &

# Job 2: Large noise (GPU 2)
echo "Starting dw_large on GPU 2..."
CUDA_VISIBLE_DEVICES=2 /home/cnagda/miniconda3/envs/da/bin/python scripts/eval_flowdas.py \
    --checkpoint runs/flowdas_dw_large/checkpoint_best.pth \
    --config datasets/double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p300/config.yaml \
    --data_dir datasets/double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p300 \
    --results_dir evaluation_outputs/flowdas_dw/dw_large \
    --device cuda \
    --use_wandb \
    --num_trajs 50 \
    --wandb_project flowdas-eval-dw \
    --run_name flowdas_dw_large_best > logs/eval_flowdas_dw_large.log 2>&1 &

wait
echo "All evaluations completed."

