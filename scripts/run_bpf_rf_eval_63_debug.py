import os
import sys
import time
import subprocess
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPUs to use (only GPU 7 available)
AVAILABLE_GPUS = [7]

# Memory buffer (MiB)
MEMORY_BUFFER = 2000 

# Paths
RF_RUNS_RELATIVE = "rf_runs"
RF_RUNS_ABSOLUTE = "/data/da_outputs/rf_runs"
LOGS_DIR = "logs/l63_debug_bpf_eval_scheduler"

# Clean dataset (no process noise)
DATA_DIR_CLEAN = "/data/da_outputs/datasets/lorenz63_n512_len500_dt0p0100_obs0p250_freq1_comp0,1,2_arctan"

# Evaluation Parameters
NUM_EVAL_TRAJECTORIES = 50
OBS_FREQUENCY = 5  # Observe every N steps
OBS_NOISE_STD = 0.25  # Observation noise std (can be overridden, default from dataset)

# RF (BPF) Parameters
RF_N_PARTICLES = 100
RF_SAMPLING_STEPS = 100
RF_LIKELIHOOD_STEPS = 100

PYTHON_EXEC = "/home/cnagda/miniconda3/envs/da/bin/python"

# Timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# WandB project
WANDB_PROJECT = "eval-63-debug"
WANDB_ENTITY = "ml-climate"

# Job filtering: Specify which jobs to run (empty list = run all jobs)
# Example: JOBS_TO_RUN = ["debug_l63_len500_delta_obs_all3"]  # Only run this job
JOBS_TO_RUN = []  # Empty = run all jobs

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
# CHECKPOINT DISCOVERY
# =============================================================================

def find_best_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    """
    Find the best checkpoint (lowest val_loss) in a checkpoint directory.
    
    Args:
        checkpoint_dir: Path to directory containing checkpoints
        
    Returns:
        Path to best checkpoint if found, None otherwise
    """
    if not checkpoint_dir.exists():
        return None
    
    # Find checkpoint files matching the pattern rf-epoch=*-val_loss=*.ckpt
    checkpoints = list(checkpoint_dir.glob("rf-*.ckpt"))
    
    if not checkpoints:
        return None
    
    def get_val_loss(path):
        try:
            stem = path.stem  # e.g., "rf-epoch=365-val_loss=0.000481"
            
            # Skip "last" checkpoint and periodic checkpoints
            if "last" in stem.lower() or "periodic" in stem.lower():
                return float('inf')
            
            # Extract val_loss from pattern: rf-epoch=*-val_loss={val_loss}
            if "-val_loss=" in stem:
                parts = stem.split("-val_loss=")
                if len(parts) == 2:
                    val_loss = float(parts[1])
                    return val_loss
        except (ValueError, IndexError):
            pass
        return float('inf')
    
    # Filter out checkpoints with invalid loss values and sort by validation loss
    valid_checkpoints = [cp for cp in checkpoints if get_val_loss(cp) != float('inf')]
    
    if not valid_checkpoints:
        return None
    
    # Sort by validation loss (lowest is best)
    valid_checkpoints.sort(key=get_val_loss)
    return str(valid_checkpoints[0])  # Best checkpoint (lowest loss)

def extract_config_from_exp_name(exp_name: str) -> Dict[str, any]:
    """
    Extract configuration from experiment name.
    
    Example: debug_l63_len500_delta_obs_all3_timestep
    -> {
        'obs_components': '0,1,2',
        'predict_delta': True,
        'use_observations': True,
        'use_time_step': True,
        ...
    }
    """
    config = {
        'obs_components': None,
        'predict_delta': False,
        'use_observations': False,
        'use_time_step': False,
        'debug_random_obs': False,
        'debug_random_prev_state': False,
    }
    
    # Parse obs_components
    if '_obs_all3' in exp_name:
        config['obs_components'] = '0,1,2'
        config['use_observations'] = True
    elif '_obs_dim0' in exp_name:
        config['obs_components'] = '0'
        config['use_observations'] = True
    elif '_noobs' in exp_name:
        config['use_observations'] = False
    
    # Parse flags
    if '_delta' in exp_name:
        config['predict_delta'] = True
    if '_timestep' in exp_name:
        config['use_time_step'] = True
    if '_randobs' in exp_name:
        config['debug_random_obs'] = True
    if '_randprev' in exp_name:
        config['debug_random_prev_state'] = True
    
    return config

def discover_checkpoints() -> List[Dict[str, any]]:
    """
    Discover all RF training runs and find best checkpoint for each unique experiment.
    
    Returns:
        List of dicts with keys: exp_name, checkpoint_path, run_dir, config, mtime
    """
    discovered_runs = []
    
    # Search in both relative and absolute paths
    search_paths = []
    if os.path.exists(RF_RUNS_RELATIVE):
        search_paths.append(Path(RF_RUNS_RELATIVE).resolve())
    if os.path.exists(RF_RUNS_ABSOLUTE):
        search_paths.append(Path(RF_RUNS_ABSOLUTE))
    
    if not search_paths:
        print(f"Warning: No RF runs directories found in {RF_RUNS_RELATIVE} or {RF_RUNS_ABSOLUTE}")
        return []
    
    # Find all directories matching debug_l63_len500_* pattern
    for search_path in search_paths:
        pattern = str(search_path / "debug_l63_len500_*")
        run_dirs = glob.glob(pattern)
        
        for run_dir in run_dirs:
            run_path = Path(run_dir)
            exp_name = run_path.name
            
            # Skip if not a directory
            if not run_path.is_dir():
                continue
            
            # Find checkpoint directory
            checkpoint_dir = run_path / "checkpoints"
            if not checkpoint_dir.exists():
                print(f"Warning: No checkpoints directory found in {run_dir}")
                continue
            
            # Find best checkpoint
            best_checkpoint = find_best_checkpoint(checkpoint_dir)
            if best_checkpoint is None:
                print(f"Warning: No valid checkpoint found in {checkpoint_dir}")
                continue
            
            # Extract config from experiment name
            config = extract_config_from_exp_name(exp_name)
            
            # Get modification time for sorting
            mtime = run_path.stat().st_mtime
            
            discovered_runs.append({
                'exp_name': exp_name,
                'checkpoint_path': best_checkpoint,
                'run_dir': str(run_path),
                'config': config,
                'mtime': mtime,
            })
    
    # Group by experiment name and select latest
    exp_groups = {}
    for run in discovered_runs:
        exp_name = run['exp_name']
        if exp_name not in exp_groups:
            exp_groups[exp_name] = []
        exp_groups[exp_name].append(run)
    
    # Select latest run for each experiment name
    selected_runs = []
    for exp_name, runs in exp_groups.items():
        # Sort by modification time (latest first)
        runs.sort(key=lambda x: x['mtime'], reverse=True)
        selected_runs.append(runs[0])
        
        if len(runs) > 1:
            print(f"Found {len(runs)} runs for {exp_name}, selecting latest (mtime: {runs[0]['mtime']})")
    
    return selected_runs

# =============================================================================
# JOB GENERATION
# =============================================================================

def discover_jobs():
    """
    Creates job list for L63 debug BPF evaluation.
    """
    jobs = []
    
    # Discover checkpoints
    print("Discovering checkpoints...")
    discovered_runs = discover_checkpoints()
    
    if not discovered_runs:
        print("No checkpoints discovered!")
        return jobs
    
    print(f"Discovered {len(discovered_runs)} unique experiments:")
    for run in discovered_runs:
        print(f"  - {run['exp_name']}: {run['checkpoint_path']}")
    
    # Safe GPU limit (A100 80GB, leave some headroom)
    SAFE_GPU_LIMIT = 75000
    
    # Create jobs for each discovered checkpoint
    for run in discovered_runs:
        exp_name = run['exp_name']
        checkpoint_path = run['checkpoint_path']
        config = run['config']
        
        # Filter jobs if JOBS_TO_RUN is specified
        if JOBS_TO_RUN and exp_name not in JOBS_TO_RUN:
            print(f"Skipping {exp_name} (not in JOBS_TO_RUN list)")
            continue
        
        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        
        # Determine batch size (use full batch for L63 since it's small)
        batch_size = NUM_EVAL_TRAJECTORIES  # L63 is small, can handle full batch
        est_mem = estimate_memory_usage("rf", batch_size)
        
        # Format obs noise for naming (0.25 -> 0p25)
        obs_noise_str = str(OBS_NOISE_STD).replace(".", "p")
        
        # Build command
        cmd = [
            PYTHON_EXEC, "eval.py",
            "--data-dir", DATA_DIR_CLEAN,
            "--n-particles", str(RF_N_PARTICLES),
            "--batch-size", str(batch_size),
            "--proposal-type", "rf",
            "--rf-checkpoint", checkpoint_path,
            "--rf-sampling-steps", str(RF_SAMPLING_STEPS),
            "--rf-likelihood-steps", str(RF_LIKELIHOOD_STEPS),
            "--num-eval-trajectories", str(NUM_EVAL_TRAJECTORIES),
            "--obs-frequency", str(OBS_FREQUENCY),
            "--experiment-label", f"fppf_{exp_name}_clean_freq{OBS_FREQUENCY}_obs{obs_noise_str}",
            "--wandb-project", WANDB_PROJECT,
            "--wandb-entity", WANDB_ENTITY,
            "--run-name", f"fppf_{exp_name}_clean_freq{OBS_FREQUENCY}_obs{obs_noise_str}",
            "--device", "cuda",
        ]
        
        # Add obs-components if observations are used
        if config['use_observations'] and config['obs_components']:
            cmd.extend(["--obs-components", config['obs_components']])
        
        # Add obs-noise-std if specified (override dataset default)
        if OBS_NOISE_STD is not None:
            cmd.extend(["--obs-noise-std", str(OBS_NOISE_STD)])
        
        job_name = f"fppf_{exp_name}_clean_freq{OBS_FREQUENCY}_obs{obs_noise_str}_{TIMESTAMP}"
        log_file = os.path.join(LOGS_DIR, f"{job_name}.log")
        
        jobs.append({
            "name": job_name,
            "cmd": cmd,
            "log": log_file,
            "model_type": "rf",
            "cost": est_mem,
            "batch_size": batch_size,
            "exp_name": exp_name,
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
    
    if not jobs:
        print("No jobs to run!")
        return
    
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
    
    # Use memory-aware packing to allow multiple jobs on same GPU
    print("Using memory-aware packing strategy")
    
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
        
        # 3. Schedule pending jobs with memory-aware packing
        current_pending = list(pending_jobs)
        pending_jobs = []
        
        for job in current_pending:
            # Calculate effective free memory (accounting for already running jobs)
            effective_free = {}
            for g in AVAILABLE_GPUS:
                actual = gpu_free.get(g, 0)
                # Subtract cost of all running jobs on this GPU
                # (nvidia-smi already accounts for older jobs, but we subtract all to be conservative)
                running_deduction = sum(rj['cost'] for rj in running_jobs if rj['gpu'] == g)
                effective_free[g] = max(0, actual - running_deduction)
            
            cost_with_buffer = job['cost'] + MEMORY_BUFFER
            
            # Find suitable GPU (one with enough effective free memory)
            candidates = [g for g in AVAILABLE_GPUS if effective_free[g] >= cost_with_buffer]
            candidates.sort(key=lambda g: effective_free[g])  # Best fit (smallest free space that fits)
            
            if candidates:
                chosen_gpu = candidates[0]
                print(f"Launching {job['name']} on GPU {chosen_gpu} (Est: {job['cost']} MiB, Effective Free: {effective_free[chosen_gpu]:.0f} MiB)")
            else:
                # No GPU has enough memory, wait
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
        effective_free = {}
        now = time.time()
        for g in AVAILABLE_GPUS:
            actual = gpu_free.get(g, 0)
            running_deduction = sum(rj['cost'] for rj in running_jobs if rj['gpu'] == g)
            effective_free[g] = max(0, actual - running_deduction)
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

