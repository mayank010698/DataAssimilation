#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

echo "=========================================="
echo "Generating Lorenz 96 Datasets (Multi-dim, Pairwise)"
echo "=========================================="

# Generate new datasets
# Dimensions: 5, 10, 15, 20, 25, 50
DIMS=(5 10 15 20 25)

for DIM in "${DIMS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Generating datasets for Lorenz 96, Dimension $DIM"
    echo "----------------------------------------"
    
    python generate.py \
        --system lorenz96 \
        --l96-dim $DIM \
        --process-noise-variations "0.0,0.1" \
        --obs-noise-std 0.1 \
        --num-trajectories 2048 \
        --len-trajectory 200 \
        --obs-frequency 1 \
        --observation-operator cube \
        --seed 42 \
        --force
done

echo ""
echo "=========================================="
echo "Dataset generation completed."
echo "=========================================="
