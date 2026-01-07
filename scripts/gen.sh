#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup environment (uncomment if needed)
# source activate
# conda activate da

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

echo "=========================================="
echo "Generating 6 Lorenz 63 Datasets"
echo "=========================================="

# 1. Lorenz 63, no process noise
echo ""
echo "Dataset 1: Lorenz 63, no process noise"
echo "----------------------------------------"
python generate.py \
    --system lorenz63 \
    --num-trajectories 1024 \
    --len-trajectory 100 \
    --obs-frequency 1 \
    --observation-operator arctan \
    --process-noise-std 0.0 \
    --seed 42 \
    --force

# 2. Lorenz 63, process noise 0.25
echo ""
echo "Dataset 2: Lorenz 63, process noise 0.25"
echo "----------------------------------------"
python generate.py \
    --system lorenz63 \
    --num-trajectories 1024 \
    --len-trajectory 100 \
    --obs-frequency 1 \
    --observation-operator arctan \
    --process-noise-std 0.25 \
    --seed 42 \
    --force

