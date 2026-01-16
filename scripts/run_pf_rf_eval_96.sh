#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
# If script is in root (run_pf_rf_eval_96.sh), SCRIPT_DIR is root.
# If script is in scripts/ (scripts/run_pf_rf_eval_96.sh), we need to go up.
# The user asked for "run_pf_rf_eval_96.sh" (likely in root based on context of previous similar files).
# But wait, looking at file list, `scripts/train_rf_flowdas_96.sh` is in scripts/.
# The user said "write a shell file called run_pf_rf_eval_96.sh".
# I'll place it in the root directory to match the pattern of `run_experiments_96.py` callers if they exist, or just put it in root.
# Actually, `scripts/train_rf_flowdas_96.sh` content showed it cd's to `..`.
# So I should probably put this in `scripts/` or root?
# The user asked for `run_pf_rf_eval_96.sh`. I'll put it in `scripts/` to keep it clean, as `train_rf_flowdas_96.sh` is there.
# Wait, the user said "write a shell file called run_pf_rf_eval_96.sh to launch stuff".
# I'll put it in `scripts/` and make sure it works from there.

cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

echo "Starting RF-BPF Evaluation Scheduler..."
/home/cnagda/miniconda3/envs/da/bin/python scripts/run_pf_rf_eval_96.py

