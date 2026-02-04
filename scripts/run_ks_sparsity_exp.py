import os
import sys
import time
import subprocess
import argparse
from typing import List, Dict, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

# WandB Project
WANDB_PROJECT = "eval-ks-temp-sparse"

# Experiment Parameters
PROCESS_NOISE_STD = 0.1
OBS_NOISE_STD = 0.1
INFLATION = 1.1  # For EnKF
N_PARTICLES_PF = 5000  # For BPF and FPPF
N_PARTICLES_ENKF = 50  # For EnKF
NUM_EVAL_TRAJECTORIES = 25 # Default for eval, can be adjusted
BATCH_SIZE = 5
RF_SAMPLING_STEPS = 100
RF_LIKELIHOOD_STEPS = 100

# GPUs to use
AVAILABLE_GPUS = [4, 5, 6, 7]

# Frequencies to test
FREQUENCIES = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Models to test
ALL_MODELS = ["enkf", "bpf", "fppf"]

# Datasets and Checkpoints
# Format: {label: {data_dir: path, ckpt_dir: path}}
CONFIGS = {
    "16pi": {
        "data_dir": "/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_pnoise0p100_J64_L50p27",
        "rf_ckpt": "/data/da_outputs/rf_runs_ks/ks_pnoise0p1_J64_16pi_cd0p1_20260117_192257/final_model.ckpt"
    },
    "32pi": {
        "data_dir": "/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_pnoise0p100_J64_L100p53",
        "rf_ckpt": "/data/da_outputs/rf_runs_ks/ks_pnoise0p1_J64_32pi_cd0p1_20260117_192257/final_model.ckpt"
    }
}

# Logs Directory
LOGS_DIR = "logs/sparsity_exp"
os.makedirs(LOGS_DIR, exist_ok=True)

# Python Binary
PYTHON_BIN = sys.executable

# =============================================================================
# JOB GENERATION
# =============================================================================

def create_jobs(models_to_run: List[str]) -> List[Dict[str, Any]]:
    jobs = []
    
    for config_name, paths in CONFIGS.items():
        data_dir = paths["data_dir"]
        rf_ckpt = paths["rf_ckpt"]
        
        for freq in FREQUENCIES:
            for model in models_to_run:
                
                # Base run name
                run_name = f"{model}_{config_name}_freq{freq}"
                log_file = os.path.join(LOGS_DIR, f"{run_name}.log")
                
                cmd = [
                    PYTHON_BIN, "eval.py",
                    "--data-dir", data_dir,
                    "--obs-frequency", str(freq),
                    "--process-noise-std", str(PROCESS_NOISE_STD),
                    "--obs-noise-std", str(OBS_NOISE_STD), # Explicitly set obs noise
                    "--num-eval-trajectories", str(NUM_EVAL_TRAJECTORIES),
                    "--batch-size", str(BATCH_SIZE),
                    "--rf-sampling-steps", str(RF_SAMPLING_STEPS),
                    "--rf-likelihood-steps", str(RF_LIKELIHOOD_STEPS),
                    "--wandb-project", WANDB_PROJECT,
                    "--run-name", run_name,
                    "--device", "cuda",
                    "--experiment-label", "temporal_sparsity"
                ]
                
                # Model specific arguments
                if model == "enkf":
                    cmd.extend([
                        "--method", "enkf",
                        "--n-particles", str(N_PARTICLES_ENKF),
                        "--inflation", str(INFLATION)
                    ])
                elif model == "bpf":
                    cmd.extend([
                        "--method", "bpf",
                        "--proposal-type", "transition",
                        "--n-particles", str(N_PARTICLES_PF)
                    ])
                elif model == "fppf":
                    cmd.extend([
                        "--method", "bpf",
                        "--proposal-type", "rf",
                        "--rf-checkpoint", rf_ckpt,
                        "--n-particles", str(N_PARTICLES_PF),
                        "--rf-likelihood-steps", "100",
                        "--rf-sampling-steps", "100"
                    ])
                
                jobs.append({
                    "name": run_name,
                    "cmd": cmd,
                    "log_file": log_file,
                    "gpu_req": 1
                })
                
    return jobs

# =============================================================================
# SCHEDULER
# =============================================================================

def run_scheduler(models_to_run, dry_run=False):
    jobs = create_jobs(models_to_run)
    print(f"Generated {len(jobs)} jobs for models: {models_to_run}")
    
    if dry_run:
        print("\n[DRY RUN] Jobs to be executed:")
        for job in jobs:
            print(f"  {job['name']}")
            print(f"    CMD: {' '.join(job['cmd'])}")
        return

    # Queue of pending jobs
    pending_jobs = jobs.copy()
    
    # Running jobs: {gpu_id: subprocess.Popen}
    running_jobs = {}
    
    # GPU status: {gpu_id: is_free}
    gpu_status = {gpu: True for gpu in AVAILABLE_GPUS}
    
    print(f"Starting execution on GPUs: {AVAILABLE_GPUS}")
    print(f"Logs will be saved to: {LOGS_DIR}")
    
    while pending_jobs or running_jobs:
        # 1. Check completed jobs
        completed_gpus = []
        for gpu, process in list(running_jobs.items()):
            poll_result = process.poll()
            if poll_result is not None:
                # Job finished
                print(f"Job finished on GPU {gpu} (Exit Code: {poll_result})")
                del running_jobs[gpu]
                gpu_status[gpu] = True
        
        # 2. Launch new jobs
        for gpu in AVAILABLE_GPUS:
            if gpu_status[gpu] and pending_jobs:
                job = pending_jobs.pop(0)
                
                print(f"Launching {job['name']} on GPU {gpu}...")
                
                # Set environment for specific GPU
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                
                # Open log file
                with open(job['log_file'], "w") as f:
                    process = subprocess.Popen(
                        job['cmd'],
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT
                    )
                
                running_jobs[gpu] = process
                gpu_status[gpu] = False
                
        # 3. Wait before next check
        if pending_jobs or running_jobs:
            time.sleep(5)
            
    print("All jobs completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without running")
    parser.add_argument("--models", type=str, default="enkf,bpf,fppf", help="Comma-separated list of models to run (enkf,bpf,fppf)")
    args = parser.parse_args()
    
    # Parse models
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    
    # Validate
    valid_models = []
    for m in requested_models:
        if m in ALL_MODELS:
            valid_models.append(m)
        else:
            print(f"Warning: Unknown model '{m}', skipping.")
            
    if not valid_models:
        print("No valid models selected.")
        sys.exit(1)
    
    run_scheduler(valid_models, dry_run=args.dry_run)

