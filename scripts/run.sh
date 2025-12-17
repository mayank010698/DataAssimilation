#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup environment (uncomment if needed)
# source activate
# conda activate da
# module load cuda/12.8

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)
# Append GraphCast_src if present (from original run.sbatch)
export PYTHONPATH=$PYTHONPATH:$(pwd)/GraphCast_src

# Define your training command
cmd="python run.py"

# Run the training
$cmd

