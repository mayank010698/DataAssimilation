#!/bin/bash
# DBF-style Lorenz-96 datasets: dt=0.03, len=80, long warmup, uniform ICs [-10,10],
# obs noise 1,3,5 (same trajectories), identity and quad_capped_10 observation operators.
# Single run with --observation-operators so the same state trajectories are used for both operators.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.." || exit

export PYTHONPATH=$(pwd)

echo "=========================================="
echo "Generating DBF-style Lorenz 96 Datasets"
echo "=========================================="

# Shared settings; same state trajectories for identity and quad_capped_10
L96_OPTS=(
    --system lorenz96
    --l96-dim 40
    --l96-forcing 8
    --dt 0.03
    --len-trajectory 80
    --warmup-steps 1024
    --obs-frequency 1
    --obs-noise-variations "1,3,5"
    --l96-init-sampling uniform
    --l96-init-low -10
    --l96-init-high 10
    --num-trajectories 2048
    --process-noise-variations "0.0,0.1"
    --observation-operators "identity,quad_capped_10"
    --seed 42
    --force
)

python generate.py "${L96_OPTS[@]}"

echo ""
echo "=========================================="
echo "DBF-style dataset generation completed."
echo "=========================================="
