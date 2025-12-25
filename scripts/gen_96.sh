#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

echo "=========================================="
echo "Generating Lorenz 96 Datasets (50D)"
echo "=========================================="

# # 1. Lorenz 96, No Process Noise
# echo ""
# echo "Dataset 1: Lorenz 96, No Process Noise, 50D, Full Obs"
# echo "----------------------------------------"
# python generate.py \
#     --system lorenz96 \
#     --l96-dim 50 \
#     --num-trajectories 2048 \
#     --len-trajectory 200 \
#     --obs-frequency 1 \
#     --observation-operator arctan \
#     --obs-noise-std 0.2 \
#     --process-noise-std 0.0 \
#     --seed 42 \
#     --force

# # 2. Lorenz 96, Process Noise 0.1
# echo ""
# echo "Dataset 2: Lorenz 96, Process Noise 0.1, 50D, Full Obs"
# echo "----------------------------------------"
# python generate.py \
#     --system lorenz96 \
#     --l96-dim 50 \
#     --num-trajectories 2048 \
#     --len-trajectory 200 \
#     --obs-frequency 1 \
#     --observation-operator arctan \
#     --obs-noise-std 0.2 \
#     --process-noise-std 0.1 \
#     --seed 42 \
#     --force

# 3. Lorenz 96, 10 dim, No Process Noise, Full Obs
echo ""
echo "Dataset 3: Lorenz 96, 10 dim, No Process Noise, Full Obs"
echo "----------------------------------------"
python generate.py \
    --system lorenz96 \
    --l96-dim 10 \
    --num-trajectories 2048 \
    --len-trajectory 200 \
    --obs-frequency 1 \
    --observation-operator arctan \
    --obs-noise-std 0.1 \
    --process-noise-std 0.1 \
    --seed 42 \
    --force

# 4. Lorenz 96, 10 dim, Process Noise 0.1, Full Obs
echo ""
echo "Dataset 4: Lorenz 96, 10 dim, Process Noise 0.1, Full Obs"
echo "----------------------------------------"
python generate.py \
    --system lorenz96 \
    --l96-dim 10 \
    --num-trajectories 2048 \
    --len-trajectory 200 \
    --obs-frequency 1 \
    --observation-operator arctan \
    --obs-noise-std 0.1 \
    --process-noise-std 0.1 \
    --seed 42 \
    --force

# 5. Lorenz 96, 100 dim, No Process Noise, Full Obs
echo ""
echo "Dataset 5: Lorenz 96, 100 dim, No Process Noise, Full Obs"
echo "----------------------------------------"
python generate.py \
    --system lorenz96 \
    --l96-dim 100 \
    --num-trajectories 2048 \
    --len-trajectory 200 \
    --obs-frequency 1 \
    --observation-operator arctan \
    --obs-noise-std 0.1 \
    --process-noise-std 0.1 \
    --seed 42 \
    --force

# 6. Lorenz 96, 100 dim, Process Noise 0.1, Full Obs
echo ""
echo "Dataset 6: Lorenz 96, 100 dim, Process Noise 0.1, Full Obs"
echo "----------------------------------------"
python generate.py \
    --system lorenz96 \
    --l96-dim 100 \
    --num-trajectories 2048 \
    --len-trajectory 200 \
    --obs-frequency 1 \
    --observation-operator arctan \
    --obs-noise-std 0.1 \
    --process-noise-std 0.1 \
    --seed 42 \
    --force

echo ""
echo "=========================================="
echo "Dataset generation completed."
echo "=========================================="
