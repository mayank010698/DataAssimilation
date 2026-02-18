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
echo "Generating Lorenz 63 Datasets"
echo "=========================================="

# Generate multiple variations with shared initial states
# This will save to /data/da_outputs/datasets/ by default
python generate.py \
    --system lorenz63 \
    --num-trajectories 1024 \
    --len-trajectory 1000 \
    --obs-frequency 1 \
    --observation-operator arctan \
    --process-noise-variations "0.0,0.25" \
    --seed 42 \
    --force
