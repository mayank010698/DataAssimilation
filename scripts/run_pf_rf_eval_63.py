import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPUs to use
AVAILABLE_GPUS = [0, 1, 2, 3]

# Memory buffer (MiB)
MEMORY_BUFFER = 2000 

# Paths
DATASETS_ROOT = "/data/da_outputs/datasets"
LOGS_DIR = "logs/l63_eval_scheduler"

# Evaluation Parameters
NUM_EVAL_TRAJECTORIES = 50
OBS_FREQUENCY = 50  # Temporal sparsity: observe every 50 steps

# RF (FPPF) Parameters
RF_N_PARTICLES = 100
RF_SAMPLING_STEPS = 100
RF_LIKELIHOOD_STEPS = 100

# FlowDAS Parameters
FLOWDAS_GUIDANCE_STEPS = [0.01, 0.1]  # Multiple guidance steps to test
FLOWDAS_MC_TIMES = 21
FLOWDAS_NUM_STEPS = 50  # Euler-Maruyama steps
FLOWDAS_N_SAMPLES_PER_TRAJ = 20  # For CRPS computation (needs > 1 for ensemble)

# Dataset paths (both use same datasets, just different obs components)
DATA_DIR_NOISE = "/data/da_outputs/datasets/lorenz63_n1024_len1000_dt0p0100_obs0p250_freq1_comp0,1,2_arctan_pnoise0p250"
DATA_DIR_NO_NOISE = "/data/da_outputs/datasets/lorenz63_n1024_len1000_dt0p0100_obs0p250_freq1_comp0,1,2_arctan"

# Checkpoint paths
CHECKPOINTS = {
    "flowdas_all3": "/data/da_outputs/runs_flowdas/l63_pnoise0p250_all3/checkpoint_best.pth",
    "flowdas_dim0": "/data/da_outputs/runs_flowdas/l63_pnoise0p250_dim0/checkpoint_best.pth",
    "rf_all3": "/home/cnagda/DataAssimilation/rf_runs/lorenz63_len1000_obs_all3_pnoise0p250/checkpoints/rf-epoch=004-val_loss=0.028366.ckpt",
    "rf_dim0": "/home/cnagda/DataAssimilation/rf_runs/lorenz63_len1000_obs_dim0_pnoise0p250/checkpoints/rf-epoch=176-val_loss=0.022845.ckpt",
}

PYTHON_EXEC = "/home/cnagda/miniconda3/envs/da/bin/python"

# Timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# WandB project
WANDB_PROJECT = "eval-lorenz63-temp-sparse"

# Job filtering: Specify which jobs to run (empty list = run all jobs)
# Job names: "rf_all3", "rf_dim0", "flowdas_all3", "flowdas_dim0"
# FlowDAS jobs now include guidance step: "flowdas_all3_gs0p0002", "flowdas_all3_gs0p01", "flowdas_all3_gs0p1", etc.
# Example: JOBS_TO_RUN = ["rf_all3", "flowdas_dim0_gs0p01"]  # Only run these 2 jobs
# JOBS_TO_RUN = []  # Empty = run all jobs
JOBS_TO_RUN = ["flowdas_all3_gs0p01", "flowdas_dim0_gs0p01", "flowdas_all3_gs0p1", "flowdas_dim0_gs0p1"]  # Only new FlowDAS jobs

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

def estimate_memory_usage(model_type, batch_size, cache={}):
    """
    Estimates memory usage for evaluation jobs.
    For L63, we can use simpler estimates since state_dim=3 is small.
    """
    key = (model_type, batch_size)
    if key in cache:
        return cache[key]
    
    # Conservative estimates for L63 (3D state)
    if model_type == "rf":
        # RF-BPF with 100 particles, batch_size trajectories
        # L63 is small, so memory is mostly from particles and batch processing
        base_memory = 5000  # Base for RF model + particles
        batch_memory = batch_size * 50  # Per trajectory overhead
        estimated = base_memory + batch_memory
    elif model_type == "flowdas":
        # FlowDAS evaluation (autoregressive, single sample per traj)
        base_memory = 3000  # Base for FlowDAS model
        batch_memory = batch_size * 30  # Per trajectory overhead
        estimated = base_memory + batch_memory
    else:
        estimated = 10000  # Fallback
    
    # Add safety margin (20%)
    safe_memory = int(estimated * 1.20)
    
    print(f"  -> Estimated memory for {model_type} (batch={batch_size}): {safe_memory} MiB")
    cache[key] = safe_memory
    return safe_memory

def determine_optimal_batch_size(model_type, max_memory_mib):
    """
    Determine the largest batch size (up to NUM_EVAL_TRAJECTORIES) that fits in memory.
    """
    # Try full batch first
    mem = estimate_memory_usage(model_type, NUM_EVAL_TRAJECTORIES)
    if mem < max_memory_mib:
        return NUM_EVAL_TRAJECTORIES, mem
    
    # If fails, try halving
    current_bs = NUM_EVAL_TRAJECTORIES // 2
    while current_bs >= 1:
        mem = estimate_memory_usage(model_type, current_bs)
        if mem < max_memory_mib:
            return current_bs, mem
        current_bs //= 2
        
    return 1, 10000  # Worst case fallback

# =============================================================================
# JOB GENERATION
# =============================================================================

def discover_jobs():
    """
    Creates job list for L63 evaluation.
    """
    jobs = []
    
    # Safe GPU limit (A100 80GB, leave some headroom)
    SAFE_GPU_LIMIT = 75000
    
    # FlowDAS jobs - create jobs for each guidance step
    flowdas_base_jobs = [
        {
            "name": "flowdas_all3",
            "checkpoint": CHECKPOINTS["flowdas_all3"],
            "data_dir": DATA_DIR_NOISE,
            "obs_components": "0,1,2",
            "model_type": "flowdas",
        },
        {
            "name": "flowdas_dim0",
            "checkpoint": CHECKPOINTS["flowdas_dim0"],
            "data_dir": DATA_DIR_NOISE,
            "obs_components": "0",
            "model_type": "flowdas",
        },
    ]
    
    # Expand FlowDAS jobs for each guidance step
    flowdas_jobs = []
    for guidance_step in FLOWDAS_GUIDANCE_STEPS:
        for base_job in flowdas_base_jobs:
            # Create a job config with the guidance step
            job_with_step = base_job.copy()
            job_with_step["guidance_step"] = guidance_step
            # Update name to include guidance step (format: flowdas_all3_gs0p0002)
            step_str = str(guidance_step).replace(".", "p")
            job_with_step["name"] = f"{base_job['name']}_gs{step_str}"
            flowdas_jobs.append(job_with_step)
    
    # RF (FPPF) jobs
    rf_jobs = [
        {
            "name": "rf_all3",
            "checkpoint": CHECKPOINTS["rf_all3"],
            "data_dir": DATA_DIR_NOISE,
            "obs_components": "0,1,2",
            "model_type": "rf",
        },
        {
            "name": "rf_dim0",
            "checkpoint": CHECKPOINTS["rf_dim0"],
            "data_dir": DATA_DIR_NOISE,
            "obs_components": "0",
            "model_type": "rf",
        },
    ]
    
    # Create jobs
    for job_config in flowdas_jobs + rf_jobs:
        # Filter jobs if JOBS_TO_RUN is specified
        if JOBS_TO_RUN and job_config["name"] not in JOBS_TO_RUN:
            print(f"Skipping {job_config['name']} (not in JOBS_TO_RUN list)")
            continue
        
        model_type = job_config["model_type"]
        checkpoint = job_config["checkpoint"]
        
        # Verify checkpoint exists
        if not os.path.exists(checkpoint):
            print(f"Warning: Checkpoint not found: {checkpoint}")
            continue
        
        # Determine batch size (use full batch for L63 since it's small)
        batch_size = NUM_EVAL_TRAJECTORIES  # L63 is small, can handle full batch
        est_mem = estimate_memory_usage(model_type, batch_size)
        
        # Build command
        if model_type == "flowdas":
            guidance_step = job_config.get("guidance_step", FLOWDAS_GUIDANCE_STEPS[0])  # Fallback to first if not set
            cmd = [
                PYTHON_EXEC, "scripts/eval_flowdas.py",
                "--checkpoint", checkpoint,
                "--data_dir", job_config["data_dir"],
                "--num_trajs", str(NUM_EVAL_TRAJECTORIES),
                "--num_steps", str(FLOWDAS_NUM_STEPS),
                "--sigma_obs", "0.25",  # From dataset config
                "--mc_times", str(FLOWDAS_MC_TIMES),
                "--guidance_step", str(guidance_step),
                "--n_samples_per_traj", str(FLOWDAS_N_SAMPLES_PER_TRAJ),
                "--obs_frequency", str(OBS_FREQUENCY),
                "--use_wandb",
                "--wandb_project", WANDB_PROJECT,
                "--wandb_entity", "ml-climate",
                "--run_name", f"flowdas_{job_config['name']}_freq{OBS_FREQUENCY}",
            ]
        else:  # RF
            cmd = [
                PYTHON_EXEC, "eval.py",
                "--data-dir", job_config["data_dir"],
                "--n-particles", str(RF_N_PARTICLES),
                "--batch-size", str(batch_size),
                "--proposal-type", "rf",
                "--rf-checkpoint", checkpoint,
                "--rf-sampling-steps", str(RF_SAMPLING_STEPS),
                "--rf-likelihood-steps", str(RF_LIKELIHOOD_STEPS),
                "--num-eval-trajectories", str(NUM_EVAL_TRAJECTORIES),
                "--obs-frequency", str(OBS_FREQUENCY),
                "--obs-components", job_config["obs_components"],
                "--experiment-label", f"rf_{job_config['name']}_freq{OBS_FREQUENCY}",
                "--wandb-project", WANDB_PROJECT,
                "--wandb-entity", "ml-climate",
                "--run-name", f"rf_{job_config['name']}_freq{OBS_FREQUENCY}",
                "--device", "cuda",
            ]
        
        job_name = f"{job_config['name']}_freq{OBS_FREQUENCY}_{TIMESTAMP}"
        log_file = os.path.join(LOGS_DIR, f"{job_name}.log")
        
        jobs.append({
            "name": job_name,
            "cmd": cmd,
            "log": log_file,
            "model_type": model_type,
            "cost": est_mem,
            "batch_size": batch_size,
        })
    
    return jobs

# =============================================================================
# SCHEDULER
# =============================================================================

def run():
    setup_logging()
    
    print("Generating jobs...")
    if JOBS_TO_RUN:
        print(f"Filtering: Only running jobs: {JOBS_TO_RUN}")
    else:
        print("Running all jobs")
    
    jobs = discover_jobs()
    
    # Sort jobs by cost (descending)
    jobs.sort(key=lambda x: x['cost'], reverse=True)
    
    print(f"Generated {len(jobs)} jobs.")
    for j in jobs:
        print(f"  - {j['name']} (Type: {j['model_type']}, Batch: {j['batch_size']}, Cost: {j['cost']} MiB)")
    
    pending_jobs = jobs
    running_jobs = [] 
    finished_jobs = []
    
    print("Starting scheduler loop...")
    print(f"Available GPUs: {AVAILABLE_GPUS}")
    print(f"Total jobs: {len(jobs)}, Total GPUs: {len(AVAILABLE_GPUS)}")
    
    # Determine if we should use 1 job per GPU strategy
    use_one_per_gpu = len(jobs) <= len(AVAILABLE_GPUS)
    if use_one_per_gpu:
        print("Using 1 job per GPU strategy (enough GPUs available)")
    else:
        print("Using memory-aware packing strategy (more jobs than GPUs)")
    
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
                if rj.get('log_handle'):
                    rj['log_handle'].close()
            else:
                active_jobs.append(rj)
        running_jobs = active_jobs
        
        # 2. Get current GPU status
        gpu_free = get_gpu_free_memory()
        
        # 3. Get GPUs currently in use
        gpus_in_use = {rj['gpu'] for rj in running_jobs}
        gpus_available = [g for g in AVAILABLE_GPUS if g not in gpus_in_use]
        
        # 4. Schedule pending jobs
        current_pending = list(pending_jobs)
        pending_jobs = []
        
        for job in current_pending:
            if use_one_per_gpu:
                # Strategy: 1 job per GPU - only use GPUs that are completely free
                if gpus_available:
                    chosen_gpu = gpus_available.pop(0)  # Take first available GPU
                    
                    # Still check memory as a safety check
                    cost_with_buffer = job['cost'] + MEMORY_BUFFER
                    if gpu_free.get(chosen_gpu, 0) < cost_with_buffer:
                        print(f"Warning: GPU {chosen_gpu} has insufficient memory ({gpu_free.get(chosen_gpu, 0)} MiB < {cost_with_buffer} MiB), skipping for now")
                        pending_jobs.append(job)
                        continue
                    
                    print(f"Launching {job['name']} on GPU {chosen_gpu} (1 job per GPU mode, Est: {job['cost']} MiB)")
                else:
                    # No free GPUs, wait
                    pending_jobs.append(job)
                    continue
            else:
                # Strategy: Memory-aware packing (original behavior)
                # Calculate effective free memory
                effective_free = {}
                now = time.time()
                for g in AVAILABLE_GPUS:
                    actual = gpu_free.get(g, 0)
                    # Subtract cost of recently launched jobs (< 60s)
                    recent_deduction = sum(rj['cost'] for rj in running_jobs if rj['gpu'] == g and (now - rj['start_time'] < 60))
                    effective_free[g] = max(0, actual - recent_deduction)
                
                cost_with_buffer = job['cost'] + MEMORY_BUFFER
                
                # Find suitable GPU
                candidates = [g for g in AVAILABLE_GPUS if effective_free[g] >= cost_with_buffer]
                candidates.sort(key=lambda g: effective_free[g])  # Best fit
                
                if candidates:
                    chosen_gpu = candidates[0]
                    print(f"Launching {job['name']} on GPU {chosen_gpu} (Est: {job['cost']} MiB, Batch: {job['batch_size']})")
                else:
                    pending_jobs.append(job)
                    continue
            
            # Launch job on chosen GPU
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
        
        # Status reporting
        if use_one_per_gpu:
            print(f"Status: {len(running_jobs)} running, {len(pending_jobs)} pending. Free GPUs: {len(gpus_available)}")
        else:
            effective_free = {}
            now = time.time()
            for g in AVAILABLE_GPUS:
                actual = gpu_free.get(g, 0)
                recent_deduction = sum(rj['cost'] for rj in running_jobs if rj['gpu'] == g and (now - rj['start_time'] < 60))
                effective_free[g] = max(0, actual - recent_deduction)
            print(f"Status: {len(running_jobs)} running, {len(pending_jobs)} pending. GPU Free (Eff): {effective_free}")
        
        if not pending_jobs and not running_jobs:
            break
            
        time.sleep(10)
        
    print("All jobs completed.")
    print(f"Total jobs: {len(finished_jobs)}")
    for fj in finished_jobs:
        rc = fj['process'].returncode
        status = "SUCCESS" if rc == 0 else "FAILED"
        print(f"  {fj['job']['name']}: {status} (RC: {rc})")

if __name__ == "__main__":
    run()

