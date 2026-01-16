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

# GPUs to use (excluding 0 and 1 as they might be reserved or busy, using 2-7)
AVAILABLE_GPUS = [0, 1, 2, 5, 6, 7]

# Memory buffer (MiB)
MEMORY_BUFFER = 2000 

# Paths
DATASETS_ROOT = "/data/da_outputs/datasets"
RF_RUNS_ROOT = "/data/da_outputs/rf_runs_96"
LOGS_DIR = "logs/rf_pf_eval_scheduler"

# Evaluation Parameters
# Per plan: 
# Dim 5, 10 -> 1k particles
# Dim 15, 20, 25 -> 5k particles
PARTICLE_SETTINGS = {
    5: 1000,
    10: 1000,
    15: 5000,
    20: 5000,
    25: 5000
    # 50 is excluded per plan
}

TARGET_BATCH_SIZE = 100 # Evaluate all 100 trajectories in one go if possible
NUM_EVAL_TRAJECTORIES = 100
RF_SAMPLING_STEPS = 100
RF_LIKELIHOOD_STEPS = 100

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

def estimate_rf_bpf_memory_usage(state_dim, n_particles, batch_size, checkpoint_path, cache={}):
    """
    Estimates memory usage for RF-BPF.
    """
    key = (state_dim, n_particles, batch_size, checkpoint_path)
    if key in cache:
        return cache[key]
    
    print(f"Estimating memory for RF-BPF: dim={state_dim}, particles={n_particles}, batch={batch_size}...")
    
    # Use the last available GPU for estimation
    est_gpu = AVAILABLE_GPUS[-1]
    
    cmd = [
        PYTHON_EXEC, "scripts/estimate_rf_pf_memory.py",
        "--state_dim", str(state_dim),
        "--n_particles", str(n_particles),
        "--batch_size", str(batch_size),
        "--checkpoint_path", checkpoint_path,
        "--device", str(est_gpu)
    ]
    
    try:
        result = subprocess.check_output(cmd).decode('utf-8').strip()
        # The script prints the memory in MiB as the last line
        lines = result.split('\n')
        memory_mib = int(lines[-1])
        
        # Add safety margin (15% for RF because it can be dynamic)
        safe_memory = int(memory_mib * 1.15)
        
        print(f"  -> Estimated: {memory_mib} MiB (Safe: {safe_memory} MiB)")
        cache[key] = safe_memory
        return safe_memory
    except Exception as e:
        print(f"  -> Error estimating memory: {e}")
        # Fallback values - RF is heavy
        if state_dim <= 10: return 20000
        return 40000 # Very conservative

def determine_optimal_batch_size(state_dim, n_particles, checkpoint_path, max_memory_mib):
    """
    Determine the largest batch size (up to TARGET_BATCH_SIZE) that fits in memory.
    """
    # Try target first
    mem = estimate_rf_bpf_memory_usage(state_dim, n_particles, TARGET_BATCH_SIZE, checkpoint_path)
    if mem < max_memory_mib:
        return TARGET_BATCH_SIZE, mem
    
    # If fails, try halving
    current_bs = TARGET_BATCH_SIZE // 2
    while current_bs >= 1:
        mem = estimate_rf_bpf_memory_usage(state_dim, n_particles, current_bs, checkpoint_path)
        if mem < max_memory_mib:
            return current_bs, mem
        current_bs //= 2
        
    return 1, 30000 # Worst case fallback

# =============================================================================
# JOB GENERATION
# =============================================================================

def find_checkpoint(dim, is_noisy):
    """
    Find the latest checkpoint for a given dimension and noise setting.
    Looking in /data/da_outputs/rf_runs_96/l96_{noise}_{dim}_{timestamp}/checkpoints/*.ckpt
    """
    noise_str = "noise" if is_noisy else "nonoise"
    pattern = f"l96_{noise_str}_dim{dim}_*"
    
    run_dirs = glob.glob(os.path.join(RF_RUNS_ROOT, pattern))
    if not run_dirs:
        return None
        
    # Sort by timestamp (in name) to get latest
    # Name format: l96_noise_dim10_20260109_184942
    # We can just sort alphabetically since YYYYMMDD_HHMMSS sorts correctly
    run_dirs.sort(reverse=True)
    
    latest_run = run_dirs[0]
    
    # Find checkpoint in run dir
    ckpt_pattern = os.path.join(latest_run, "checkpoints", "*.ckpt")
    ckpts = glob.glob(ckpt_pattern)
    
    if not ckpts:
        return None
        
    # Prefer 'last.ckpt' if exists, otherwise any
    last_ckpt = [c for c in ckpts if "last.ckpt" in c]
    if last_ckpt:
        return last_ckpt[0]
        
    # Or maybe best validation loss? Usually Lightning saves epoch=X.ckpt
    return ckpts[0]

def discover_jobs():
    """
    Creates job list based on plan.
    """
    jobs = []
    
    # Retry specific failed jobs
    RETRY_JOBS = [
        "eval_rf_noise_d15_5000k",
        "eval_rf_nonoise_d15_5000k",
        "eval_rf_nonoise_d5_1000k",
        "eval_rf_noise_d5_1000k",
        "eval_rf_noise_d10_1000k"
    ]
    
    # Dimensions to process
    dims = sorted(PARTICLE_SETTINGS.keys())
    
    # Find datasets first (non-noisy)
    datasets = {}
    dataset_dirs = glob.glob(os.path.join(DATASETS_ROOT, "lorenz96_*"))
    for d_path in dataset_dirs:
        d_name = os.path.basename(d_path)
        match = re.search(r"comp(\d+)of(\d+)", d_name)
        if not match: continue
        dim = int(match.group(1))
        
        if dim not in dims: continue
        if "pnoise" in d_name: continue # We only evaluate on NON-noisy datasets
        
        datasets[dim] = {"path": d_path, "name": d_name}

    # Now create jobs
    # 75GB limit for batch sizing (A100 80GB)
    SAFE_GPU_LIMIT = 75000 
    
    for dim in dims:
        if dim not in datasets:
            print(f"Warning: No non-noisy dataset found for dim {dim}")
            continue
            
        ds = datasets[dim]
        n_particles = PARTICLE_SETTINGS[dim]
        
        # We need two jobs: one for noisy-trained RF, one for nonoise-trained RF
        for train_noisy in [False, True]:
            noise_label = "noise" if train_noisy else "nonoise"
            
            run_name = f"eval_rf_{noise_label}_d{dim}_{n_particles}k"
            
            # Only run the failed jobs
            if run_name not in RETRY_JOBS:
                continue
            
            ckpt_path = find_checkpoint(dim, train_noisy)
            if not ckpt_path:
                print(f"Warning: No checkpoint found for dim {dim}, {noise_label}")
                continue
                
            # FORCE batch size 100 and high cost to ensure 1 job per GPU
            # as per user request to fix OOM/small batch sizes
            batch_size = 100
            est_mem = 70000 
            
            job_name = f"{run_name}_{TIMESTAMP}"
            log_file = os.path.join(LOGS_DIR, f"{job_name}.log")
            
            cmd = [
                PYTHON_EXEC, "eval.py",
                "--data-dir", ds['path'],
                "--n-particles", str(n_particles),
                "--batch-size", str(batch_size),
                "--proposal-type", "rf",
                "--rf-checkpoint", ckpt_path,
                "--rf-sampling-steps", str(RF_SAMPLING_STEPS),
                "--rf-likelihood-steps", str(RF_LIKELIHOOD_STEPS),
                "--num-eval-trajectories", str(NUM_EVAL_TRAJECTORIES),
                "--experiment-label", run_name,
                "--wandb-project", "pf-rf-eval-96",
                "--run-name", run_name,
                "--device", "cuda"
            ]
            
            jobs.append({
                "name": job_name,
                "cmd": cmd,
                "log": log_file,
                "dim": dim,
                "train_noisy": train_noisy,
                "particles": n_particles,
                "batch_size": batch_size,
                "cost": est_mem
            })
            
    return jobs

# =============================================================================
# SCHEDULER
# =============================================================================

def run():
    setup_logging()
    
    print("Generating jobs...")
    jobs = discover_jobs()
    
    # Sort jobs by cost (descending)
    jobs.sort(key=lambda x: x['cost'], reverse=True)
    
    print(f"Generated {len(jobs)} jobs.")
    for j in jobs:
        print(f"  - {j['name']} (Dim: {j['dim']}, Particles: {j['particles']}, Cost: {j['cost']} MiB)")
    
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
        remaining_pending = []
        
        # Try to fit jobs
        current_pending = list(pending_jobs)
        pending_jobs = [] 
        
        for job in current_pending:
            cost_with_buffer = job['cost'] + MEMORY_BUFFER
            
            # Find suitable GPU
            candidates = [g for g in AVAILABLE_GPUS if effective_free[g] >= cost_with_buffer]
            candidates.sort(key=lambda g: effective_free[g]) # Best fit
            
            if candidates:
                chosen_gpu = candidates[0]
                
                print(f"Launching {job['name']} on GPU {chosen_gpu} (Est: {job['cost']} MiB, Batch: {job['batch_size']})")
                
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

