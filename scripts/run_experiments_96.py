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

# GPUs to use (excluding 2 and 3 as requested)
AVAILABLE_GPUS = [0, 1, 4, 5, 6, 7]

# Memory buffer (MiB) - leave some headroom
MEMORY_BUFFER = 5000 

# Paths
DATASETS_ROOT = "/data/da_outputs/datasets"
OUTPUT_ROOT_RF = "/data/da_outputs/rf_runs_96"
OUTPUT_ROOT_FLOWDAS = "/data/da_outputs/flowdas_runs_96"
LOGS_DIR = "logs/dynamic_scheduler"

# Training Parameters
BATCH_SIZE = 512
CHANNELS = 64
NUM_BLOCKS = 10
KERNEL_SIZE = 5
RF_EPOCHS = 500
FLOWDAS_EPOCHS = 500
LR = 3e-4

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
    
    # Use GPU 6 or 7 for estimation (usually less loaded or just pick one available)
    # We'll just try to use the first available GPU from our list
    # But for estimation, we need a GPU.
    est_gpu = AVAILABLE_GPUS[-1] # Pick the last one, maybe less utilized?
    
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
        # Fallback values if estimation fails (conservative estimates)
        # Based on typical usage for these sizes
        if state_dim <= 10: return 4000
        if state_dim <= 25: return 8000
        return 16000 # 50 dim might be heavy

# =============================================================================
# JOB GENERATION
# =============================================================================

def discover_datasets():
    """
    Scans DATASETS_ROOT for relevant datasets.
    """
    # Pattern: lorenz96_n2048_..._comp{DIM}of{DIM}_...
    # We are looking for comp5of5, 10of10, 15of15, 20of20, 25of25, 50of50
    # And variants with pnoise0p100 (process noise) and without.
    
    all_datasets = []
    
    # We want dimensions: 5, 10, 15, 20, 25, 50
    target_dims = [5, 10, 15, 20, 25, 50]
    
    dataset_dirs = glob.glob(os.path.join(DATASETS_ROOT, "lorenz96_*"))
    
    for d_path in dataset_dirs:
        d_name = os.path.basename(d_path)
        
        # Extract dimension
        match = re.search(r"comp(\d+)of(\d+)", d_name)
        if not match: continue
        
        dim1, dim2 = int(match.group(1)), int(match.group(2))
        if dim1 != dim2: continue # Partial obs, skip for now? Or did user want full obs for each dim? 
        # User said: "for each dimension from 5, 10, 15, 20, 25, 50 and one with process noise, one without process noise for each"
        # The partial obs experiments in train_96.sh were "comp50of50", "comp25of50" (indices), etc.
        # But here the user says "datasets ... for each dimension ... compXofX".
        # So we assume the dataset itself is defined by that dimension.
        
        if dim1 not in target_dims: continue
        
        # Check noise
        is_noise = "pnoise0p100" in d_name
        
        all_datasets.append({
            "path": d_path,
            "name": d_name,
            "dim": dim1,
            "noise": is_noise
        })
        
    return sorted(all_datasets, key=lambda x: (x['dim'], x['noise']))

def create_jobs(datasets):
    jobs = []
    
    for ds in datasets:
        dim = ds['dim']
        noise_str = "noise" if ds['noise'] else "nonoise"
        
        # 1. RF Job
        # Obs components: 0 to dim-1 (Full obs for that dataset)
        obs_components = ",".join(map(str, range(dim)))
        
        rf_name = f"l96_{noise_str}_dim{dim}_{TIMESTAMP}"
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
            "--cond_dropout", "0.1",
            "--gpus", "1",
            "--evaluate"
        ]
        
        jobs.append({
            "type": "rf",
            "dim": dim,
            "name": rf_name,
            "cmd": rf_cmd,
            "log": rf_log,
            "dataset": ds['name']
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
        print(f"  - {d['name']} (Dim: {d['dim']}, Noise: {d['noise']})")
        
    jobs = create_jobs(datasets)
    # Sort jobs by dimension descending (hardest first usually better for packing)
    # Or maybe smallest first to get things moving?
    # Let's do largest first.
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
        # We try to fit as many pending jobs as possible
        remaining_pending = []
        
        # We need to track allocated memory in this scheduling cycle (since nvidia-smi won't update instantly)
        # But we rely on Free memory from nvidia-smi which reflects ACTUAL usage.
        # Issue: If we launch a job, it takes time to allocate memory. 
        # So we should subtract estimates for recently launched jobs if they haven't ramped up yet?
        # A simple heuristic: Assume allocated = Free - (sum of estimates of jobs running on that GPU)
        # Actually, if we just launched a job, nvidia-smi might still say high free memory.
        # So we track 'reserved' memory for each GPU locally based on what we think is running.
        
        gpu_reserved = {g: 0 for g in AVAILABLE_GPUS}
        for rj in running_jobs:
            gpu_reserved[rj['gpu']] += rj['cost']
            
        # For GPUs, we use the MIN(Actual Free, Total - Reserved)? 
        # No, easier: We trust our reservations more than nvidia-smi for *recently* started jobs.
        # But over time, nvidia-smi is truth.
        # Let's just use: Effective Free = Actual Free (from SMI) - (Memory of jobs launched < 1 min ago?)
        # To be safe, let's just track residual capacity:
        # Capacity = Actual Free
        # But we must account for the fact that a just-launched job might show 0 usage yet.
        # So: Effective Free = Actual Free - Sum(Cost of jobs launched < 30s ago)
        
        effective_free = {}
        now = time.time()
        for g in AVAILABLE_GPUS:
            actual = current_gpus.get(g, 0)
            # Subtract cost of recently launched jobs (start_time > now - 60s)
            recent_deduction = sum(rj['cost'] for rj in running_jobs if rj['gpu'] == g and (now - rj['start_time'] < 60))
            effective_free[g] = max(0, actual - recent_deduction)
            
        
        # Try to schedule pending jobs
        # We iterate through a copy of pending list to allow removal
        jobs_to_launch = []
        
        for job in pending_jobs:
            # Estimate cost
            cost = estimate_job_memory(job['type'], job['dim'], memory_cache)
            job_cost_with_buffer = cost + MEMORY_BUFFER
            
            # Find suitable GPU
            chosen_gpu = None
            # Sort GPUs by most free memory? Or least free that fits (best fit)?
            # Best fit helps fragmentation.
            candidates = [g for g in AVAILABLE_GPUS if effective_free[g] >= job_cost_with_buffer]
            candidates.sort(key=lambda g: effective_free[g]) # Ascending order (best fit)
            
            if candidates:
                chosen_gpu = candidates[0]
                
                # Launch
                print(f"Launching {job['name']} on GPU {chosen_gpu} (Est: {cost} MiB)")
                
                # Construct command with CUDA_VISIBLE_DEVICES
                # We need to set the env var for the subprocess
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
                
                # Open log file
                os.makedirs(os.path.dirname(job['log']), exist_ok=True)
                log_file = open(job['log'], 'w')
                
                # Add environment setup if needed (like PYTHONPATH)
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

