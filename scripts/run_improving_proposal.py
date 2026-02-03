import os
import sys
import subprocess
import glob
import re
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset for 10-dim Lorenz 96
DATA_DIR = "/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp10of10_arctan_pnoise0p100_init3p000"
STATE_DIM = 10
OBS_COMPONENTS = "0,1,2,3,4,5,6,7,8,9"

# Training Parameters
MAX_EPOCHS = 100
SAVE_EVERY = 10
GPU_ID = 1 # Use GPU 1 as per request
BATCH_SIZE = 720
LR = 0.0003

# Evaluation Parameters
N_PARTICLES = 1000
EVAL_BATCH_SIZE = 100
NUM_EVAL_TRAJS = 50
RF_STEPS = 100

# Available GPUs for parallel evaluation
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

# Paths
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT = "/data/da_outputs/rf_runs_96"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"l96_noise_dim10_improving_{TIMESTAMP}")

# Use the specific conda environment python
PYTHON_EXEC = "/home/cnagda/miniconda3/envs/da/bin/python"

def run_command(cmd, env=None, log_file=None):
    cmd_str = ' '.join(cmd)
    print(f"Running: {cmd_str}")
    
    if log_file:
        with open(log_file, 'w') as f:
            subprocess.check_call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.check_call(cmd, env=env)

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logs_dir = os.path.join("logs", f"improving_experiment_{TIMESTAMP}")
    os.makedirs(logs_dir, exist_ok=True)

    # Prepare Environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    env["PYTHONPATH"] = os.getcwd()

    # =========================================================================
    # 1. Training
    # =========================================================================
    print("="*80)
    print("STEP 1: Training RF Proposal")
    print("="*80)
    
    train_log = os.path.join(logs_dir, "train.log")
    
    train_cmd = [
        PYTHON_EXEC, "proposals/train_rf.py",
        "--data_dir", DATA_DIR,
        "--output_dir", OUTPUT_DIR,
        "--state_dim", str(STATE_DIM),
        "--obs_components", OBS_COMPONENTS,
        "--architecture", "resnet1d",
        "--channels", "64",
        "--num_blocks", "10",
        "--kernel_size", "5",
        "--train-cond-method", "adaln",
        "--cond_embed_dim", "128",
        "--use_observations",
        "--batch_size", str(BATCH_SIZE),
        "--learning_rate", str(LR),
        "--max_epochs", str(MAX_EPOCHS),
        "--gpus", "1",
        "--save_every_n_epochs", str(SAVE_EVERY),
        "--wandb_project", "rf-improvement-10d-train"
    ]
    
    try:
        run_command(train_cmd, env=env, log_file=train_log)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        print(f"Check log at {train_log}")
        sys.exit(1)

    # =========================================================================
    # 2. Evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Evaluating Checkpoints")
    print("="*80)
    
    checkpoints_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    
    # Wait for training to finish before evaluating?
    # The user implies running this script will do training THEN evaluation.
    # But if training is already done or we want to do it in parallel?
    # The current structure runs training first (blocking), then eval.
    # To eval in parallel, we need to spawn subprocesses.

    # Find all periodic checkpoints
    all_files = os.listdir(checkpoints_dir)
    ckpt_files = [f for f in all_files if "periodic" in f and f.endswith(".ckpt")]
    
    # Parse epochs
    checkpoints = []
    for f in ckpt_files:
        match = re.search(r"epoch[=_]?(\d+)", f)
        if match:
            epoch = int(match.group(1))
            full_path = os.path.join(checkpoints_dir, f)
            checkpoints.append((epoch, full_path))
            
    # Sort by epoch
    checkpoints.sort(key=lambda x: x[0])
    
    # Filter epochs: 10, 30, 50, 60, 70, 80, 90, 100
    target_epochs = [10, 30, 50, 60, 70, 80, 90, 100]
    checkpoints = [c for c in checkpoints if c[0] in target_epochs]
    
    if not checkpoints:
        print("No matching checkpoints found!")
        sys.exit(1)
        
    print(f"Found {len(checkpoints)} checkpoints to evaluate: {[c[0] for c in checkpoints]}")
    
    running_procs = []
    
    # Distribute checkpoints across GPUs
    for i, (epoch, ckpt_path) in enumerate(checkpoints):
        gpu_idx = AVAILABLE_GPUS[i % len(AVAILABLE_GPUS)]
        
        print(f"Launching Evaluation for Epoch {epoch} on GPU {gpu_idx}...")
        
        run_name = f"eval_imp_d{STATE_DIM}_ep{epoch:03d}"
        eval_log = os.path.join(logs_dir, f"eval_ep{epoch:03d}.log")
        
        # Prepare env for this specific process
        proc_env = env.copy()
        proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        
        eval_cmd = [
            PYTHON_EXEC, "eval.py",
            "--data-dir", DATA_DIR,
            "--n-particles", str(N_PARTICLES),
            "--batch-size", str(EVAL_BATCH_SIZE),
            "--proposal-type", "rf",
            "--rf-checkpoint", ckpt_path,
            "--rf-sampling-steps", str(RF_STEPS),
            "--rf-likelihood-steps", str(RF_STEPS),
            "--num-eval-trajectories", str(NUM_EVAL_TRAJS),
            "--experiment-label", "rf_improvement_10d_eval",
            "--wandb-project", "rf-improvement-10d-eval",
            "--run-name", run_name,
            "--device", "cuda",
            "--init-mode", "climatology"
        ]
        
        # Launch in background
        with open(eval_log, 'w') as f:
            proc = subprocess.Popen(eval_cmd, env=proc_env, stdout=f, stderr=subprocess.STDOUT)
            running_procs.append((epoch, proc))
            
    print(f"\nLaunched {len(running_procs)} evaluation jobs. Waiting for completion...")
    
    # Wait for all
    for epoch, proc in running_procs:
        proc.wait()
        if proc.returncode == 0:
            print(f"Epoch {epoch}: Success")
        else:
            print(f"Epoch {epoch}: Failed (RC={proc.returncode})")
            
    print("\nAll evaluations completed.")

if __name__ == "__main__":
    main()

