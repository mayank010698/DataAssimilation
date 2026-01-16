#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

PYTHON_EXEC="/home/cnagda/miniconda3/envs/da/bin/python"

echo "=========================================="
echo "Starting BPF Evaluations (Lorenz 96) - Dynamic Scheduler"
echo "Python: $PYTHON_EXEC"
echo "=========================================="

$PYTHON_EXEC scripts/run_bpf_eval_96.py

