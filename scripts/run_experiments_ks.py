import os
import sys
import time
import subprocess
import glob
import re
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

try:
    from data import load_config_yaml
except ImportError:
    print("Error: Could not import load_config_yaml from data. Make sure you run this script from the project root.")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPUs to use (All 8 GPUs available)
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

# Memory buffer (MiB) - leave some headroom
MEMORY_BUFFER = 5000 

# Paths
DATASETS_ROOT = "/data/da_outputs/datasets"
OUTPUT_ROOT_RF = "/data/da_outputs/rf_runs_ks"
OUTPUT_ROOT_FLOWDAS = "/data/da_outputs/flowdas_runs_ks"
LOGS_DIR = "logs/dynamic_scheduler_ks"

# Training Parameters
BATCH_SIZE = 512
CHANNELS = 128    # Increased from 64
NUM_BLOCKS = 10   # Increased from 6
KERNEL_SIZE = 5
RF_EPOCHS = 500
FLOWDAS_EPOCHS = 500
LR = 3e-4

# WandB Projects
WANDB_PROJECT_RF = "rf-train-ks"
WANDB_PROJECT_FLOWDAS = "flowdas-train-ks"

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
        # Run nvidia-smi to get free memory
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

def estimate_job_memory(model_type, state_dim, cache={}):
    """
    Estimates memory usage for a job. Uses a cache to avoid re-running estimation.
    """
    key = (model_type, state_dim)
    if key in cache:
        return cache[key]
    
    print(f"Estimating memory for {model_type}, dim={state_dim}...")
    
    # Use the last available GPU for estimation
    est_gpu = AVAILABLE_GPUS[-1] 
    
    cmd = [
        "python", "scripts/estimate_memory.py",
        "--model_type", model_type,
        "--state_dim", str(state_dim),
        "--batch_size", str(BATCH_SIZE),
        "--channels", str(CHANNELS),
        "--num_blocks", str(NUM_BLOCKS),
        "--kernel_size", str(KERNEL_SIZE),
        "--device", str(est_gpu)
    ]
    
    try:
        # Run estimation
        result = subprocess.check_output(cmd).decode('utf-8').strip()
        # The script prints the memory in MiB as the last line (or only line)
        lines = result.split('\n')
        memory_mib = int(lines[-1])
        
        # Add a safety margin (e.g. 10%)
        safe_memory = int(memory_mib * 1.1)
        
        print(f"  -> Estimated: {memory_mib} MiB (Safe: {safe_memory} MiB)")
        cache[key] = safe_memory
        return safe_memory
    except Exception as e:
        print(f"  -> Error estimating memory: {e}")
        # Fallback values if estimation fails
        if state_dim <= 64: return 8000
        return 16000

# =============================================================================
# JOB GENERATION
# =============================================================================

def discover_datasets():
    """
    Scans DATASETS_ROOT for relevant KS datasets.
    Filters by config parameters:
    - process_noise_std approx 0.05
    - obs_noise_std approx 0.2
    - J in {64, 128}
    - L in {16pi, 32pi}
    """
    
    target_Js = [64, 128]
    # L values: 16*pi approx 50.265, 32*pi approx 100.53
    target_Ls = [16 * np.pi, 32 * np.pi]
    
    all_datasets = []
    
    dataset_dirs = glob.glob(os.path.join(DATASETS_ROOT, "ks_*"))
    
    print(f"Found {len(dataset_dirs)} potential KS dataset directories.")
    
    for d_path in dataset_dirs:
        d_name = os.path.basename(d_path)
        config_path = os.path.join(d_path, "config.yaml")
        
        if not os.path.exists(config_path):
            continue
            
        try:
            config = load_config_yaml(config_path)
            
            # Check System
            system_params = config.system_params
            
            # Check J
            J = system_params.get("J")
            if J not in target_Js:
                continue
                
            # Check L (allow small tolerance)
            L = system_params.get("L")
            L_match = False
            for target_L in target_Ls:
                if abs(L - target_L) < 1.0: # Tolerance of 1.0 is generous enough
                    L_match = True
                    break
            if not L_match:
                continue
                
            # Check Process Noise
            p_noise = config.process_noise_std
            if abs(p_noise - 0.05) > 0.01:
                continue
                
            # Check Obs Noise
            o_noise = config.obs_noise_std
            if abs(o_noise - 0.2) > 0.01:
                continue
            
            all_datasets.append({
                "path": d_path,
                "name": d_name,
                "dim": J, # State dimension is J
                "L": L,
                "p_noise": p_noise,
                "o_noise": o_noise
            })
            
        except Exception as e:
            print(f"Error reading config for {d_name}: {e}")
            continue
        
    return sorted(all_datasets, key=lambda x: (x['dim'], x['L']))

def create_jobs(datasets):
    jobs = []
    
    for ds in datasets:
        dim = ds['dim']
        
        # Determine L label for naming
        if abs(ds['L'] - 16 * np.pi) < 1.0:
            l_str = "16pi"
        elif abs(ds['L'] - 32 * np.pi) < 1.0:
            l_str = "32pi"
        else:
            l_str = f"L{int(ds['L'])}"
            
        noise_str = "pnoise0p05"
        
        # Dense observations: 0 to dim-1
        obs_components = ",".join(map(str, range(dim)))
        
        # 1. RF Job
        rf_name = f"ks_{noise_str}_J{dim}_{l_str}_{TIMESTAMP}"
        rf_out = os.path.join(OUTPUT_ROOT_RF, rf_name)
        rf_log = os.path.join(LOGS_DIR, f"rf_{rf_name}.log")
        
        rf_cmd = [
            "python", "proposals/train_rf.py",
            "--data_dir", ds['path'],
            "--output_dir", rf_out,
            "--state_dim", str(dim),
            "--obs_components", obs_components,
            "--architecture", "resnet1d",
            "--channels", str(CHANNELS),
            "--num_blocks", str(NUM_BLOCKS),
            "--kernel_size", str(KERNEL_SIZE),
            "--train-cond-method", "adaln",
            "--cond_embed_dim", "128",
            "--use_observations",
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LR),
            "--max_epochs", str(RF_EPOCHS),
            "--gpus", "1",
            "--evaluate",
            "--wandb_project", WANDB_PROJECT_RF
        ]
        
        jobs.append({
            "type": "rf",
            "dim": dim,
            "name": rf_name,
            "cmd": rf_cmd,
            "log": rf_log,
            "dataset": ds['name']
        })
        
        # 2. FlowDAS Job
        flowdas_name = f"flowdas_ks_{noise_str}_J{dim}_{l_str}_{TIMESTAMP}"
        flowdas_out = os.path.join(OUTPUT_ROOT_FLOWDAS, flowdas_name)
        flowdas_log = os.path.join(LOGS_DIR, f"flowdas_{flowdas_name}.log")
        
        flowdas_cmd = [
            "python", "scripts/train_flowdas.py",
            "--config", os.path.join(ds['path'], "config.yaml"),
            "--data_dir", ds['path'],
            "--run_dir", flowdas_out,
            "--obs_components", obs_components,
            "--architecture", "resnet1d",
            "--channels", str(CHANNELS),
            "--num_blocks", str(NUM_BLOCKS),
            "--kernel_size", str(KERNEL_SIZE),
            "--use_observations",
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR),
            "--epochs", str(FLOWDAS_EPOCHS),
            "--use_wandb",
            "--wandb_project", WANDB_PROJECT_FLOWDAS,
            "--wandb_name", flowdas_name,
            "--evaluate"
        ]
        
        jobs.append({
            "type": "flowdas",
            "dim": dim,
            "name": flowdas_name,
            "cmd": flowdas_cmd,
            "log": flowdas_log,
            "dataset": ds['name']
        })
        
    return jobs

# =============================================================================
# SCHEDULER
# =============================================================================

def run():
    setup_logging()
    
    print("Discovering KS datasets...")
    datasets = discover_datasets()
    print(f"Found {len(datasets)} matching datasets.")
    for d in datasets:
        print(f"  - {d['name']} (J: {d['dim']}, L: {d['L']:.2f}, pnoise: {d['p_noise']})")
        
    jobs = create_jobs(datasets)
    # Sort jobs by dimension descending (largest first)
    jobs.sort(key=lambda x: x['dim'], reverse=True)
    
    print(f"Generated {len(jobs)} jobs.")
    
    pending_jobs = jobs
    running_jobs = [] # List of {'job': job_dict, 'process': Popen, 'gpu': gpu_idx, 'cost': mem_est}
    finished_jobs = []
    
    memory_cache = {}
    
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
        # Filter only available GPUs
        current_gpus = {k: v for k, v in gpu_free.items() if k in AVAILABLE_GPUS}
        
        # 3. Schedule new jobs
        effective_free = {}
        now = time.time()
        for g in AVAILABLE_GPUS:
            actual = current_gpus.get(g, 0)
            # Subtract cost of recently launched jobs (start_time > now - 60s)
            recent_deduction = sum(rj['cost'] for rj in running_jobs if rj['gpu'] == g and (now - rj['start_time'] < 60))
            effective_free[g] = max(0, actual - recent_deduction)
            
        
        # Try to schedule pending jobs
        # We iterate through a copy of pending list to allow removal
        remaining_pending = []
        
        # Count running jobs per GPU
        jobs_per_gpu = {g: 0 for g in AVAILABLE_GPUS}
        for rj in running_jobs:
            jobs_per_gpu[rj['gpu']] += 1
        
        jobs_to_launch = [] # We launch strictly one by one in the loop or batch?
        # The loop logic handles one pass.
        
        for job in pending_jobs:
            if job in jobs_to_launch: continue # Skip if already decided to launch
            
            # Estimate cost
            cost = estimate_job_memory(job['type'], job['dim'], memory_cache)
            job_cost_with_buffer = cost + MEMORY_BUFFER
            
            # Find suitable GPU
            chosen_gpu = None
            candidates = [g for g in AVAILABLE_GPUS if effective_free[g] >= job_cost_with_buffer]
            # Sort candidates:
            # 1. Prefer GPUs with FEWER running jobs (balancing load)
            # 2. Tie-breaker: Best fit (least free memory) or most free? 
            #    Let's use most free as secondary to be safe, or just arbitrary.
            #    Actually, if jobs_per_gpu is 0, effective_free is max.
            candidates.sort(key=lambda g: (jobs_per_gpu[g], effective_free[g])) 
            
            if candidates:
                chosen_gpu = candidates[0]
                
                # Launch
                print(f"Launching {job['name']} on GPU {chosen_gpu} (Est: {cost} MiB)")
                
                # Construct command with CUDA_VISIBLE_DEVICES
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
                
                # Open log file
                os.makedirs(os.path.dirname(job['log']), exist_ok=True)
                log_file = open(job['log'], 'w')
                
                # Add environment setup
                env["PYTHONPATH"] = os.getcwd()
                
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
                    'cost': cost,
                    'start_time': time.time(),
                    'log_handle': log_file
                })
                
                # Update effective free
                effective_free[chosen_gpu] -= cost
                
                # Update jobs per GPU
                jobs_per_gpu[chosen_gpu] += 1
                
                # Do not add to remaining_pending
            else:
                # Could not fit this job
                remaining_pending.append(job)
        
        pending_jobs = remaining_pending
        
        # Print status summary
        print(f"Status: {len(running_jobs)} running, {len(pending_jobs)} pending. GPU Free (Eff): {effective_free}")
        
        if not pending_jobs and not running_jobs:
            break
            
        time.sleep(10)
        
    print("All jobs completed.")

if __name__ == "__main__":
    run()

