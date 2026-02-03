import os
import sys
import time
import subprocess
import glob
import re
import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPUs to use (2 through 7)
AVAILABLE_GPUS = [1, 2, 6, 7]

# Memory buffer (MiB)
MEMORY_BUFFER = 1000 
COST_PER_JOB = 4000 # Estimated cost

# Paths
DATASETS_ROOT = "/data/da_outputs/datasets"
TUNING_RESULTS_ROOT = "tuning_results/enkf_clim"
LOGS_DIR = "logs/enkf_eval_scheduler"

# Evaluation Parameters
TARGET_DIMS = [5, 10, 15, 20, 25]
PARTICLES = 50
BATCH_SIZE = 32
NUM_EVAL_TRAJECTORIES = 50 
PROCESS_NOISE = 0.1

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

def discover_datasets():
    """
    Scans DATASETS_ROOT for relevant datasets.
    """
    all_datasets = []
    
    dataset_dirs = glob.glob(os.path.join(DATASETS_ROOT, "lorenz96_*"))
    
    for d_path in dataset_dirs:
        d_name = os.path.basename(d_path)
        
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

def get_best_config(dataset_name):
    """
    Reads best_config_enkf.json for the given dataset.
    """
    config_path = os.path.join(TUNING_RESULTS_ROOT, dataset_name, "best_config_enkf.json")
    if not os.path.exists(config_path):
        print(f"Warning: No tuning config found for {dataset_name} at {config_path}")
        return None
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error reading config for {dataset_name}: {e}")
        return None

def create_jobs(datasets):
    jobs = []
    
    for ds in datasets:
        dim = ds['dim']
        
        # Get tuned parameters
        config = get_best_config(ds['name'])
        if not config:
            print(f"Skipping dim {dim} due to missing config.")
            continue
            
        inflation = config.get('inflation', 1.0)
        
        run_name = f"eval_enkf_d{dim}_clim"
        job_name = f"{run_name}_{TIMESTAMP}"
        log_file = os.path.join(LOGS_DIR, f"{job_name}.log")
        
        # Construct command
        cmd = [
            PYTHON_EXEC, "eval.py",
            "--data-dir", ds['path'],
            "--method", "enkf",
            "--n-particles", str(PARTICLES),
            "--process-noise-std", str(PROCESS_NOISE),
            "--inflation", str(inflation),
            "--batch-size", str(BATCH_SIZE),
            "--num-eval-trajectories", str(NUM_EVAL_TRAJECTORIES),
            "--device", "cuda",
            "--init-mode", "climatology",
            "--wandb-project", "kf-eval-96",
            "--experiment-label", "enkf_clim_eval",
            "--run-name", run_name,
            "--wandb-tags", "tuned,best-config,climatology"
        ]
        
        jobs.append({
            "name": job_name,
            "cmd": cmd,
            "log": log_file,
            "dim": dim,
            "cost": COST_PER_JOB,
            "inflation": inflation
        })
            
    return jobs

# =============================================================================
# SCHEDULER
# =============================================================================

def run():
    setup_logging()
    
    print("Discovering datasets...")
    datasets = discover_datasets()
    print(f"Found {len(datasets)} non-noisy datasets.")
    for d in datasets:
        print(f"  - {d['name']} (Dim: {d['dim']})")
        
    print("\nCreating jobs...")
    jobs = create_jobs(datasets)
    
    # Sort jobs by cost (descending)
    jobs.sort(key=lambda x: x['cost'], reverse=True)
    
    print(f"Generated {len(jobs)} jobs.")
    for j in jobs:
        print(f"  - {j['name']} (Inflation: {j['inflation']})")
    
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

