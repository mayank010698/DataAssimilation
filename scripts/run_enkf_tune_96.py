import argparse
import os
import sys
import time
import subprocess
import glob
import re
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPUs to use (2 through 7)
AVAILABLE_GPUS = [1, 2, 6, 7]

# Memory buffer (MiB) - EnKF is cheap
MEMORY_BUFFER = 1000 
COST_PER_JOB = 4000 # Estimated cost per tuning job (safe upper bound)

# Paths
DATASETS_ROOT = "/data/da_outputs/datasets"
LOGS_DIR = "logs/enkf_tune_scheduler"
TUNING_OUTPUT_ROOT = "tuning_results/enkf_clim_cube"

# Evaluation Parameters
TARGET_DIMS = [5, 10, 15, 20, 25]
PARTICLES = 50
BATCH_SIZE = 32
NUM_EVAL_TRAJECTORIES = 50 
PROCESS_NOISE = 0.1
INFLATION_GRID = "0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8"

PYTHON_EXEC = "/home/cnagda/miniconda3/envs/da/bin/python"

# Timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================================
# UTILITIES
# =============================================================================

def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    print(f"Logging to {LOGS_DIR}")

def get_gpu_free_memory():
    """
    Returns a dict {gpu_index: free_memory_mib}
    """
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
        result = subprocess.check_output(cmd).decode('utf-8').strip()
        
        gpu_memory = {}
        for line in result.split('\n'):
            if not line.strip(): continue
            idx, free_mem = line.split(',')
            gpu_memory[int(idx)] = int(free_mem)
            
        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return {}

# =============================================================================
# JOB GENERATION
# =============================================================================

def discover_datasets(obs_operator="arctan"):
    """
    Scans DATASETS_ROOT for relevant datasets.
    """
    all_datasets = []
    
    dataset_dirs = glob.glob(os.path.join(DATASETS_ROOT, "lorenz96_*"))
    
    for d_path in dataset_dirs:
        d_name = os.path.basename(d_path)
        
        # Check observation operator
        if obs_operator not in d_name:
            continue

        # Extract dimension
        match = re.search(r"comp(\d+)of(\d+)", d_name)
        if not match: continue
        
        dim1 = int(match.group(1))
        
        if dim1 not in TARGET_DIMS: continue
        
        # Check noise - we WANT NO noise (i.e. exclude if pnoise is present)
        is_noise = "pnoise" in d_name
        if is_noise: continue
        
        all_datasets.append({
            "path": d_path,
            "name": d_name,
            "dim": dim1
        })
        
    return sorted(all_datasets, key=lambda x: x['dim'])

def create_jobs(datasets):
    jobs = []
    
    for ds in datasets:
        dim = ds['dim']
        
        run_name = f"tune_enkf_d{dim}_clim"
        job_name = f"{run_name}_{TIMESTAMP}"
        log_file = os.path.join(LOGS_DIR, f"{job_name}.log")
        
        output_dir = os.path.join(TUNING_OUTPUT_ROOT, ds['name'])
        
        # Construct command
        cmd = [
            PYTHON_EXEC, "scripts/tune_enkf.py",
            "--data-dir", ds['path'],
            "--method", "enkf",
            "--n-particles", str(PARTICLES),
            "--process-noise-std", str(PROCESS_NOISE),
            "--inflation-values", INFLATION_GRID,
            "--batch-size", str(BATCH_SIZE),
            "--num-eval-trajectories", str(NUM_EVAL_TRAJECTORIES),
            "--output-dir", output_dir,
            "--device", "cuda",
            "--init-mode", "climatology"
        ]
        
        jobs.append({
            "name": job_name,
            "cmd": cmd,
            "log": log_file,
            "dim": dim,
            "cost": COST_PER_JOB
        })
            
    return jobs

# =============================================================================
# SCHEDULER
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Tune EnKF on L96 datasets")
    parser.add_argument("--obs-operator", type=str, default="arctan", 
                        help="Observation operator to match in dataset names (default: arctan)")
    return parser.parse_args()

def run():
    args = parse_args()
    setup_logging()
    
    print(f"Discovering datasets (obs_operator='{args.obs_operator}')...")
    datasets = discover_datasets(obs_operator=args.obs_operator)
    print(f"Found {len(datasets)} non-noisy datasets.")
    for d in datasets:
        print(f"  - {d['name']} (Dim: {d['dim']})")
        
    print("\nCreating jobs...")
    jobs = create_jobs(datasets)
    
    # Sort jobs by cost (descending)
    jobs.sort(key=lambda x: x['cost'], reverse=True)
    
    print(f"Generated {len(jobs)} jobs.")
    
    pending_jobs = jobs
    running_jobs = [] 
    finished_jobs = []
    
    print("Starting scheduler loop...")
    print(f"Available GPUs: {AVAILABLE_GPUS}")
    
    while pending_jobs or running_jobs:
        # 1. Check running jobs
        active_jobs = []
        for rj in running_jobs:
            if rj['process'].poll() is not None:
                # Job finished
                rc = rj['process'].returncode
                duration = time.time() - rj['start_time']
                print(f"Job finished: {rj['job']['name']} on GPU {rj['gpu']} (RC: {rc}, Time: {duration:.1f}s)")
                finished_jobs.append(rj)
            else:
                active_jobs.append(rj)
        running_jobs = active_jobs
        
        # 2. Get current GPU status
        gpu_free = get_gpu_free_memory()
        
        # 3. Calculate effective free memory
        effective_free = {}
        now = time.time()
        for g in AVAILABLE_GPUS:
            actual = gpu_free.get(g, 0)
            # Subtract cost of recently launched jobs (< 60s)
            recent_deduction = sum(rj['cost'] for rj in running_jobs if rj['gpu'] == g and (now - rj['start_time'] < 60))
            effective_free[g] = max(0, actual - recent_deduction)
            
        # 4. Schedule pending jobs
        
        # Try to fit jobs
        current_pending = list(pending_jobs)
        pending_jobs = [] # Will rebuild
        
        for job in current_pending:
            cost_with_buffer = job['cost'] + MEMORY_BUFFER
            
            # Find suitable GPU
            candidates = [g for g in AVAILABLE_GPUS if effective_free[g] >= cost_with_buffer]
            candidates.sort(key=lambda g: effective_free[g]) # Best fit
            
            if candidates:
                chosen_gpu = candidates[0]
                
                print(f"Launching {job['name']} on GPU {chosen_gpu} (Est: {job['cost']} MiB)")
                
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
                env["PYTHONPATH"] = os.getcwd()
                
                os.makedirs(os.path.dirname(job['log']), exist_ok=True)
                log_file = open(job['log'], 'w')
                
                proc = subprocess.Popen(
                    job['cmd'],
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
                
                running_jobs.append({
                    'job': job,
                    'process': proc,
                    'gpu': chosen_gpu,
                    'cost': job['cost'],
                    'start_time': time.time(),
                    'log_handle': log_file
                })
                
                effective_free[chosen_gpu] -= job['cost']
            else:
                pending_jobs.append(job)
        
        print(f"Status: {len(running_jobs)} running, {len(pending_jobs)} pending. GPU Free (Eff): {effective_free}")
        
        if not pending_jobs and not running_jobs:
            break
            
        time.sleep(10)
        
    print("All jobs completed.")

if __name__ == "__main__":
    run()

