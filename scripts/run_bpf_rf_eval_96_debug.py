import os
import time
import subprocess
import glob
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# GPUs to use (adjust as needed)
AVAILABLE_GPUS = [4, 5, 6, 7]

# Memory buffer (MiB)
MEMORY_BUFFER = 2000

# Paths
RF_RUNS_RELATIVE = "rf_runs/len80_debug"
RF_RUNS_ABSOLUTE = "/data/da_outputs/rf_runs/len80_debug"
LOGS_DIR = "logs/l96_len80_debug_bpf_eval_scheduler"

# Evaluation parameters
NUM_EVAL_TRAJECTORIES = 50
OBS_FREQUENCY = 5  # Observe every N steps
# For L96 len80 we rely on dataset-stored observation noise by default
OBS_NOISE_STD = None

# RF (BPF) parameters
RF_N_PARTICLES = 5000
RF_SAMPLING_STEPS = 100
RF_LIKELIHOOD_STEPS = 100

PYTHON_EXEC = "/home/cnagda/miniconda3/envs/da/bin/python"

# Timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# WandB project
WANDB_PROJECT = "eval-96-debug"
WANDB_ENTITY = "ml-climate"

# Job filtering: Specify which jobs to run (empty list = run all jobs)
JOBS_TO_RUN: List[Tuple[str, str]] = []  # (training_slug, exp_name); empty = run all


# =============================================================================
# UTILITIES
# =============================================================================

def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    print(f"Logging to {LOGS_DIR}")


def get_gpu_free_memory() -> Dict[int, int]:
    """
    Returns a dict {gpu_index: free_memory_mib}
    """
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
        result = subprocess.check_output(cmd).decode("utf-8").strip()

        gpu_memory: Dict[int, int] = {}
        for line in result.split("\n"):
            if not line.strip():
                continue
            idx, free_mem = line.split(",")
            gpu_memory[int(idx)] = int(free_mem)

        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return {}


def estimate_memory_usage(model_type: str, batch_size: int, cache: Dict[Tuple[str, int], int] = {}) -> int:
    """
    Estimates memory usage for evaluation jobs.

    For L96 len80 with 5000 particles and 10D state. Tuned so that one job
    (batch_size=1) fits in ~80GB GPU memory with MEMORY_BUFFER headroom.
    """
    key = (model_type, batch_size)
    if key in cache:
        return cache[key]

    if model_type == "rf":
        # RF-BPF with 5000 particles, batch_size trajectories, 10D state.
        # Base + per-trajectory chosen so cost(1) + MEMORY_BUFFER fits in ~81GB.
        base_memory = 55000  # Base for RF model + 5000 particles
        batch_memory = batch_size * 800   # Per-trajectory overhead
        estimated = base_memory + batch_memory
    else:
        estimated = 80000  # Fallback

    # Add safety margin (15%)
    safe_memory = int(estimated * 1.15)

    print(f"  -> Estimated memory for {model_type} (batch={batch_size}): {safe_memory} MiB")
    cache[key] = safe_memory
    return safe_memory


def determine_optimal_batch_size(model_type: str, max_memory_mib: int) -> Tuple[int, int]:
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

    return 1, estimate_memory_usage(model_type, 1)


# =============================================================================
# CHECKPOINT DISCOVERY
# =============================================================================

def find_best_checkpoint(checkpoint_dir: Path) -> Optional[str]:
    """
    Find the best checkpoint (lowest val_loss) in a checkpoint directory.
    """
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("rf-*.ckpt"))
    if not checkpoints:
        return None

    def get_val_loss(path: Path) -> float:
        try:
            stem = path.stem  # e.g., "rf-epoch=365-val_loss=0.000481"

            # Skip "last" checkpoint and periodic checkpoints
            if "last" in stem.lower() or "periodic" in stem.lower():
                return float("inf")

            if "-val_loss=" in stem:
                parts = stem.split("-val_loss=")
                if len(parts) == 2:
                    val_loss = float(parts[1])
                    return val_loss
        except (ValueError, IndexError):
            pass
        return float("inf")

    valid_checkpoints = [cp for cp in checkpoints if get_val_loss(cp) != float("inf")]
    if not valid_checkpoints:
        return None

    valid_checkpoints.sort(key=get_val_loss)
    return str(valid_checkpoints[0])


def extract_config_from_exp_name(exp_name: str) -> Dict[str, object]:
    """
    Extract configuration from experiment name for L96 len80 debug RF runs.

    Examples:
      debug_len80_delta_obs_all10_timestep
      debug_len80_delta_obs_dim0_randobs_timestep
      debug_len80_delta_noobs_timestep
    """
    config: Dict[str, object] = {
        "obs_components": None,
        "predict_delta": False,
        "use_observations": False,
        "use_time_step": False,
        "debug_random_obs": False,
        "debug_random_prev_state": False,
    }

    # Parse obs_components
    if "_obs_all10" in exp_name:
        config["obs_components"] = ",".join(str(i) for i in range(10))
        config["use_observations"] = True
    elif "_obs_dim0" in exp_name:
        config["obs_components"] = "0"
        config["use_observations"] = True
    elif "_noobs" in exp_name:
        config["use_observations"] = False

    # Parse flags
    if "_delta" in exp_name:
        config["predict_delta"] = True
    if "_timestep" in exp_name:
        config["use_time_step"] = True
    if "_randobs" in exp_name:
        config["debug_random_obs"] = True
    if "_randprev" in exp_name:
        config["debug_random_prev_state"] = True

    return config


def discover_checkpoints() -> List[Dict[str, object]]:
    """
    Discover all RF training runs and find best checkpoint for each unique
    (training_slug, exp_name) combination under len80_debug.
    """
    discovered_runs: List[Dict[str, object]] = []

    # Search in both relative and absolute paths
    search_paths: List[Path] = []
    if os.path.exists(RF_RUNS_RELATIVE):
        search_paths.append(Path(RF_RUNS_RELATIVE).resolve())
    if os.path.exists(RF_RUNS_ABSOLUTE):
        search_paths.append(Path(RF_RUNS_ABSOLUTE))

    if not search_paths:
        print(f"Warning: No RF runs directories found in {RF_RUNS_RELATIVE} or {RF_RUNS_ABSOLUTE}")
        return []

    for search_path in search_paths:
        # Expect structure: <search_path>/<training_slug>/<exp_name>
        for training_dir in search_path.iterdir():
            if not training_dir.is_dir():
                continue
            training_slug = training_dir.name

            for run_path in training_dir.iterdir():
                if not run_path.is_dir():
                    continue

                exp_name = run_path.name

                checkpoint_dir = run_path / "checkpoints"
                if not checkpoint_dir.exists():
                    print(f"Warning: No checkpoints directory found in {run_path}")
                    continue

                best_checkpoint = find_best_checkpoint(checkpoint_dir)
                if best_checkpoint is None:
                    print(f"Warning: No valid checkpoint found in {checkpoint_dir}")
                    continue

                config = extract_config_from_exp_name(exp_name)
                mtime = run_path.stat().st_mtime

                discovered_runs.append(
                    {
                        "training_slug": training_slug,
                        "exp_name": exp_name,
                        "checkpoint_path": best_checkpoint,
                        "run_dir": str(run_path),
                        "config": config,
                        "mtime": mtime,
                    }
                )

    # Group by (training_slug, exp_name) and select latest
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for run in discovered_runs:
        key = (run["training_slug"], run["exp_name"])
        groups.setdefault(key, []).append(run)

    selected_runs: List[Dict[str, object]] = []
    for key, runs in groups.items():
        runs.sort(key=lambda x: x["mtime"], reverse=True)
        selected_runs.append(runs[0])
        if len(runs) > 1:
            training_slug, exp_name = key
            print(
                f"Found {len(runs)} runs for training_slug={training_slug}, "
                f"exp_name={exp_name}, selecting latest (mtime: {runs[0]['mtime']})"
            )

    return selected_runs


# =============================================================================
# DATASET MAPPING
# =============================================================================

def get_clean_dataset_and_tags(training_slug: str) -> Optional[Tuple[str, str]]:
    """
    Map a training dataset slug (possibly with process noise) to a clean dataset
    directory and construct a data_tag for naming.

    training_slug example:
      lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp10of10_quad_capped_10_pnoise0p010_initUnim10_10

    clean_slug:
      lorenz96_n2048_len80_dt0p0300_obs3p000_freq1_comp10of10_quad_capped_10_initUnim10_10

    data_tag example:
      obs3p000_quad_capped_10
    """
    # Strip any _pnoise... segment
    clean_slug = re.sub(r"_pnoise[0-9p]*", "", training_slug)

    eval_data_dir = f"/data/da_outputs/datasets/{clean_slug}"
    if not os.path.exists(eval_data_dir):
        print(f"Warning: Clean dataset not found for training slug {training_slug}: {eval_data_dir}")
        return None

    # Extract observation noise token, e.g. obs1p000 / obs3p000 / obs5p000
    obs_match = re.search(r"(obs[0-9p]+)", clean_slug)
    obs_token = obs_match.group(1) if obs_match else "obsUnknown"

    # Extract operator token, e.g. identity / quad_capped_10
    op_match = re.search(r"(identity|quad_capped_10)", clean_slug)
    op_token = op_match.group(1) if op_match else "opUnknown"

    data_tag = f"{obs_token}_{op_token}"
    return eval_data_dir, data_tag


# =============================================================================
# JOB GENERATION
# =============================================================================

def discover_jobs(dry_run: bool = False) -> List[Dict[str, object]]:
    """
    Creates job list for L96 len80 debug BPF evaluation (FPPF).
    """
    jobs: List[Dict[str, object]] = []

    print("Discovering checkpoints...")
    discovered_runs = discover_checkpoints()

    if not discovered_runs:
        print("No checkpoints discovered!")
        return jobs

    print(f"Discovered {len(discovered_runs)} unique (training_slug, exp_name) combinations:")
    for run in discovered_runs:
        print(
            f"  - {run['training_slug']}/{run['exp_name']}: "
            f"{run['checkpoint_path']}"
        )

    # Safe GPU limit (A100 80GB, leave some headroom)
    SAFE_GPU_LIMIT = 75000

    for run in discovered_runs:
        training_slug = run["training_slug"]
        exp_name = run["exp_name"]
        checkpoint_path = run["checkpoint_path"]
        config = run["config"]

        # Filter jobs if JOBS_TO_RUN is specified
        if JOBS_TO_RUN and (training_slug, exp_name) not in JOBS_TO_RUN:
            print(f"Skipping {training_slug}/{exp_name} (not in JOBS_TO_RUN list)")
            continue

        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue

        # Map to clean dataset
        mapped = get_clean_dataset_and_tags(training_slug)
        if mapped is None:
            continue
        eval_data_dir, data_tag = mapped

        # Determine batch size (start with full batch, limited by SAFE_GPU_LIMIT)
        batch_size = NUM_EVAL_TRAJECTORIES
        est_mem = estimate_memory_usage("rf", batch_size)

        if est_mem > SAFE_GPU_LIMIT:
            batch_size, est_mem = determine_optimal_batch_size("rf", SAFE_GPU_LIMIT)

        # Build command
        base_name = f"fppf96_{training_slug}_{exp_name}_{data_tag}_freq{OBS_FREQUENCY}"

        cmd = [
            PYTHON_EXEC,
            "eval.py",
            "--data-dir",
            eval_data_dir,
            "--n-particles",
            str(RF_N_PARTICLES),
            "--batch-size",
            str(batch_size),
            "--proposal-type",
            "rf",
            "--rf-checkpoint",
            checkpoint_path,
            "--rf-sampling-steps",
            str(RF_SAMPLING_STEPS),
            "--rf-likelihood-steps",
            str(RF_LIKELIHOOD_STEPS),
            "--num-eval-trajectories",
            str(NUM_EVAL_TRAJECTORIES),
            "--obs-frequency",
            str(OBS_FREQUENCY),
            "--experiment-label",
            base_name,
            "--wandb-project",
            WANDB_PROJECT,
            "--wandb-entity",
            WANDB_ENTITY,
            "--run-name",
            base_name,
            "--device",
            "cuda",
        ]

        # Add obs-components if observations are used
        if config.get("use_observations") and config.get("obs_components"):
            cmd.extend(["--obs-components", str(config["obs_components"])])

        # Optionally override obs-noise-std (we default to dataset noise, so skip)
        if OBS_NOISE_STD is not None:
            cmd.extend(["--obs-noise-std", str(OBS_NOISE_STD)])

        job_name = f"{base_name}_{TIMESTAMP}"
        log_file = os.path.join(LOGS_DIR, f"{job_name}.log")

        jobs.append(
            {
                "name": job_name,
                "cmd": cmd,
                "log": log_file,
                "model_type": "rf",
                "cost": est_mem,
                "batch_size": batch_size,
                "training_slug": training_slug,
                "exp_name": exp_name,
            }
        )

    if dry_run:
        print("Dry run: generated jobs (not launching):")
        for j in jobs:
            print(
                f"  - {j['name']} | data-dir={j['cmd'][j['cmd'].index('--data-dir') + 1]} | "
                f"particles={RF_N_PARTICLES} | batch={j['batch_size']}"
            )

    return jobs


# =============================================================================
# SCHEDULER
# =============================================================================

def run(dry_run: bool = False) -> None:
    setup_logging()

    print("Generating jobs...")
    if JOBS_TO_RUN:
        print(f"Filtering: Only running jobs: {JOBS_TO_RUN}")
    else:
        print("Running all jobs")

    jobs = discover_jobs(dry_run=dry_run)

    if dry_run:
        print(f"Dry run complete. Generated {len(jobs)} jobs.")
        return

    if not jobs:
        print("No jobs to run!")
        return

    # Sort jobs by cost (descending)
    jobs.sort(key=lambda x: x["cost"], reverse=True)

    print(f"Generated {len(jobs)} jobs.")
    for j in jobs:
        print(
            f"  - {j['name']} (Type: {j['model_type']}, "
            f"Batch: {j['batch_size']}, Cost: {j['cost']} MiB)"
        )

    pending_jobs = jobs
    running_jobs: List[Dict[str, object]] = []
    finished_jobs: List[Dict[str, object]] = []

    print("Starting scheduler loop...")
    print(f"Available GPUs: {AVAILABLE_GPUS}")
    print(f"Total jobs: {len(jobs)}, Total GPUs: {len(AVAILABLE_GPUS)}")

    # Use memory-aware packing to allow multiple jobs on same GPU
    print("Using memory-aware packing strategy")

    while pending_jobs or running_jobs:
        # 1. Check running jobs
        active_jobs = []
        for rj in running_jobs:
            if rj["process"].poll() is not None:
                rc = rj["process"].returncode
                duration = time.time() - rj["start_time"]
                print(
                    f"Job finished: {rj['job']['name']} on GPU {rj['gpu']} "
                    f"(RC: {rc}, Time: {duration:.1f}s)"
                )
                finished_jobs.append(rj)
                if rj.get("log_handle"):
                    rj["log_handle"].close()
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
            effective_free: Dict[int, int] = {}
            for g in AVAILABLE_GPUS:
                actual = gpu_free.get(g, 0)
                running_deduction = sum(
                    rj["cost"] for rj in running_jobs if rj["gpu"] == g
                )
                effective_free[g] = max(0, actual - running_deduction)

            cost_with_buffer = job["cost"] + MEMORY_BUFFER

            # Find suitable GPU (one with enough effective free memory)
            candidates = [
                g for g in AVAILABLE_GPUS if effective_free.get(g, 0) >= cost_with_buffer
            ]
            candidates.sort(key=lambda g: effective_free.get(g, 0))

            if candidates:
                chosen_gpu = candidates[0]
                print(
                    f"Launching {job['name']} on GPU {chosen_gpu} "
                    f"(Est: {job['cost']} MiB, Effective Free: {effective_free[chosen_gpu]:.0f} MiB)"
                )
            else:
                # No GPU has enough memory, wait
                pending_jobs.append(job)
                continue

            # Launch job on chosen GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
            env["PYTHONPATH"] = os.getcwd()

            os.makedirs(os.path.dirname(job["log"]), exist_ok=True)
            log_file = open(job["log"], "w")

            proc = subprocess.Popen(
                job["cmd"],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

            running_jobs.append(
                {
                    "job": job,
                    "process": proc,
                    "gpu": chosen_gpu,
                    "cost": job["cost"],
                    "start_time": time.time(),
                    "log_handle": log_file,
                }
            )

        # Status reporting
        effective_free: Dict[int, int] = {}
        for g in AVAILABLE_GPUS:
            actual = gpu_free.get(g, 0)
            running_deduction = sum(
                rj["cost"] for rj in running_jobs if rj["gpu"] == g
            )
            effective_free[g] = max(0, actual - running_deduction)
        print(
            f"Status: {len(running_jobs)} running, {len(pending_jobs)} pending. "
            f"GPU Free (Eff): {effective_free}"
        )

        if not pending_jobs and not running_jobs:
            break

        time.sleep(10)

    print("All jobs completed.")
    print(f"Total jobs: {len(finished_jobs)}")
    for fj in finished_jobs:
        rc = fj["process"].returncode
        status = "SUCCESS" if rc == 0 else "FAILED"
        print(f"  {fj['job']['name']}: {status} (RC: {rc})")


if __name__ == "__main__":
    # Default to dry-run when invoked directly; change to run() to launch jobs.
    run()

