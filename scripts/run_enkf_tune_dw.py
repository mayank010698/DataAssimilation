import argparse
import os
import sys
import time
import subprocess
import glob
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPUs to use
AVAILABLE_GPUS = [5, 6] # Adjust based on your system

# Memory buffer (MiB)
MEMORY_BUFFER = 1000 
COST_PER_JOB = 2000 # Double well is cheap

# Paths
DATASETS_ROOT = "datasets" 
LOGS_DIR = "logs/enkf_tune_dw"
TUNING_OUTPUT_ROOT = "tuning_results/enkf_dw"

# Evaluation Parameters
PARTICLES = 100
BATCH_SIZE = 100
NUM_EVAL_TRAJECTORIES = 100 
INFLATION_GRID = "0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.5,3.0"

# Uses current python
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
        # Fallback for non-NVIDIA or error: assume all GPUs have enough memory
        return {g: 24000 for g in AVAILABLE_GPUS}

# =============================================================================
# JOB GENERATION
# =============================================================================

def discover_datasets():
    """
    Finds the specific Double Well datasets.
    """
    target_datasets = [
        "double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p200",
        "double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p300"
    ]
    
    found_datasets = []
    
    for d_name in target_datasets:
        d_path = os.path.join(DATASETS_ROOT, d_name)
        if os.path.exists(d_path):
            found_datasets.append({
                "path": d_path,
                "name": d_name
            })
        else:
            print(f"Warning: Dataset not found: {d_path}")
            
    return found_datasets

def create_jobs(datasets):
    jobs = []
    
    for ds in datasets:
        # Extract process noise from name for info (not strictly needed as it's in config)
        if "pnoise0p200" in ds['name']:
            pnoise = 0.2
        elif "pnoise0p300" in ds['name']:
            pnoise = 0.3
        else:
            pnoise = 0.1 # Default?

        run_name = f"tune_enkf_{ds['name']}"
        job_name = f"{run_name}_{TIMESTAMP}"
        log_file = os.path.join(LOGS_DIR, f"{job_name}.log")
        
        output_dir = os.path.join(TUNING_OUTPUT_ROOT, ds['name'])
        
        # Construct command
        cmd = [
            PYTHON_EXEC, "scripts/tune_enkf.py",
            "--data-dir", ds['path'],
            "--method", "enkf",
            "--n-particles", str(PARTICLES),
            "--process-noise-std", str(pnoise),
            "--inflation-values", INFLATION_GRID,
            "--batch-size", str(BATCH_SIZE),
            "--num-eval-trajectories", str(NUM_EVAL_TRAJECTORIES),
            "--output-dir", output_dir,
            "--device", "cuda",
            "--init-mode", "truth" # Usually we can start from truth or climatology. Let's use truth for fair comparison if BPF/FlowDAS did? 
                                   # BPF eval doesn't specify init, likely uses truth or stationary. 
                                   # FlowDAS usually trains on data.
                                   # Let's stick to truth or check what other baselines do. 
                                   # Actually, BPF usually initializes from prior. 
                                   # But tune_enkf.py supports "truth" or "climatology". 
                                   # Let's use "truth" for now as it's standard for filter comparison if warmup is short.
        ]
        
        jobs.append({
            "name": job_name,
            "cmd": cmd,
            "log": log_file,
            "cost": COST_PER_JOB
        })
            
    return jobs

# =============================================================================
# SCHEDULER
# =============================================================================

def run():
    setup_logging()
    
    print("Discovering datasets...")
    datasets = discover_datasets()
    print(f"Found {len(datasets)} datasets.")
    for d in datasets:
        print(f"  - {d['name']}")
        
    print("\nCreating jobs...")
    jobs = create_jobs(datasets)
    
    if not jobs:
        print("No jobs created.")
        return

    print(f"Generated {len(jobs)} jobs.")
    
    pending_jobs = jobs
    running_jobs = [] 
    
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
                
                # Make sure log dir exists
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
        
        if pending_jobs or running_jobs:
            time.sleep(10)
        
    print("All jobs completed.")

if __name__ == "__main__":
    run()

