#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

echo "=========================================="
echo "Generating Kuramoto-Sivashinsky Datasets"
echo "=========================================="

# Parameters
# Trajectories: 2048
# Length: 1000
# Obs Freq: 1
# Operator: arctan
# dt: 0.25
# Process Noise: 0.0, 0.05, 0.1
# Obs Noise: 0.1, 0.2
# J: 64, 128
# L: 16*pi, 32*pi

JS=(64 128)
L_MULTS=(16 32)
OBS_NOISES=(0.1 0.2)

for J in "${JS[@]}"; do
    for L_MULT in "${L_MULTS[@]}"; do
        # Calculate L (use python for pi)
        L=$(python -c "import numpy as np; print(${L_MULT} * np.pi)")
        
        for OBS_NOISE in "${OBS_NOISES[@]}"; do
            echo ""
            echo "----------------------------------------"
            echo "Generating KS dataset: J=$J, L=${L_MULT}pi, ObsNoise=$OBS_NOISE"
            echo "----------------------------------------"
            
            python generate.py \
                --system ks \
                --ks-J $J \
                --ks-L $L \
                --process-noise-variations "0.0,0.05,0.1" \
                --obs-noise-std $OBS_NOISE \
                --num-trajectories 2048 \
                --len-trajectory 1000 \
                --obs-frequency 1 \
                --observation-operator arctan \
                --dt 0.25 \
                --seed 42 \
                --force
        done
    done
done

echo ""
echo "=========================================="
echo "KS Dataset generation completed."
echo "=========================================="

