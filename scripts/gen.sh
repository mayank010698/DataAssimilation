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

# 1. Lorenz 63, no process noise, 1 dim observed (first one)
echo ""
echo "Dataset 1: Lorenz 63, no process noise, 1 dim observed (first)"
echo "----------------------------------------"
python generate.py \
    --system lorenz63 \
    --num-trajectories 1024 \
    --len-trajectory 100 \
    --obs-frequency 1 \
    --obs-components "0" \
    --observation-operator arctan \
    --process-noise-std 0.0 \
    --seed 42 \
    --force

# 2. Lorenz 63, no process noise, 2 dim observed (first two)
echo ""
echo "Dataset 2: Lorenz 63, no process noise, 2 dim observed (first two)"
echo "----------------------------------------"
python generate.py \
    --system lorenz63 \
    --num-trajectories 1024 \
    --len-trajectory 100 \
    --obs-frequency 1 \
    --obs-components "0,1" \
    --observation-operator arctan \
    --process-noise-std 0.0 \
    --seed 42 \
    --force

# 3. Lorenz 63, no process noise, all 3 dim observed
echo ""
echo "Dataset 3: Lorenz 63, no process noise, all 3 dim observed"
echo "----------------------------------------"
python generate.py \
    --system lorenz63 \
    --num-trajectories 1024 \
    --len-trajectory 100 \
    --obs-frequency 1 \
    --obs-components "0,1,2" \
    --observation-operator arctan \
    --process-noise-std 0.0 \
    --seed 42 \
    --force

# 4. Lorenz 63, process noise 0.25, 1 dim observed (first one)
echo ""
echo "Dataset 4: Lorenz 63, process noise 0.25, 1 dim observed (first)"
echo "----------------------------------------"
python generate.py \
    --system lorenz63 \
    --num-trajectories 1024 \
    --len-trajectory 100 \
    --obs-frequency 1 \
    --obs-components "0" \
    --observation-operator arctan \
    --process-noise-std 0.25 \
    --seed 42 \
    --force

# 5. Lorenz 63, process noise 0.25, 2 dim observed (first two)
echo ""
echo "Dataset 5: Lorenz 63, process noise 0.25, 2 dim observed (first two)"
echo "----------------------------------------"
python generate.py \
    --system lorenz63 \
    --num-trajectories 1024 \
    --len-trajectory 100 \
    --obs-frequency 1 \
    --obs-components "0,1" \
    --observation-operator arctan \
    --process-noise-std 0.25 \
    --seed 42 \
    --force

# 6. Lorenz 63, process noise 0.25, all 3 dim observed
echo ""
echo "Dataset 6: Lorenz 63, process noise 0.25, all 3 dim observed"
echo "----------------------------------------"
python generate.py \
    --system lorenz63 \
    --num-trajectories 1024 \
    --len-trajectory 100 \
    --obs-frequency 1 \
    --obs-components "0,1,2" \
    --observation-operator arctan \
    --process-noise-std 0.25 \
    --seed 42 \
    --force

