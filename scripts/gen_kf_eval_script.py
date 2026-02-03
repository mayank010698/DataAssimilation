import os
import json
import stat
from pathlib import Path

def main():
    # Configuration
    TUNING_RESULTS_DIR = Path("tuning_results/enkf_clim") # Updated to match tuning script output
    BASE_DATA_DIR = "/data/da_outputs/datasets"
    OUTPUT_SCRIPT = Path("scripts/eval_kf_all.sh") # Default name or allow override
    PYTHON_EXEC = "/home/cnagda/miniconda3/envs/da/bin/python"
    
    # Common evaluation parameters
    N_PARTICLES = 50
    PROCESS_NOISE_STD = 0.1
    NUM_EVAL_TRAJECTORIES = 50
    BATCH_SIZE = 32
    DEVICE = "cuda"
    WANDB_ENTITY = "ml-climate"
    NUM_GPUS = 8
    
    lines = [
        "#!/bin/bash",
        "",
        "# Generated script to evaluate tuned EnKF models on Lorenz96 datasets",
        "mkdir -p logs",
        "",
        "echo 'Starting EnKF Evaluations (L96 Only)...'",
        ""
    ]
    
    # Iterate over datasets in tuning_results
    if not TUNING_RESULTS_DIR.exists():
        print(f"Error: {TUNING_RESULTS_DIR} not found.")
        return

    # Sort for deterministic output
    dataset_dirs = sorted([d for d in TUNING_RESULTS_DIR.iterdir() if d.is_dir()])
    
    # Filter for Lorenz96 datasets only
    l96_datasets = [d for d in dataset_dirs if "lorenz96" in d.name.lower()]
    
    jobs = []
    job_counter = 0
    for dataset_dir in l96_datasets:
        dataset_name = dataset_dir.name
        wandb_project = "kf-eval-96"
        data_path = f"{BASE_DATA_DIR}/{dataset_name}"
        
        # --- EnKF Only ---
        enkf_config_path = dataset_dir / "enkf" / "best_config_enkf.json"
        if not enkf_config_path.exists():
            continue
        
        with open(enkf_config_path, "r") as f:
            config = json.load(f)
        
        inflation = config.get("inflation", 1.0)
        
        args_lines = [
            f"{PYTHON_EXEC} eval.py \\",
            f"    --data-dir {data_path} \\",
            "    --method enkf \\",
            f"    --n-particles {N_PARTICLES} \\",
            f"    --process-noise-std {PROCESS_NOISE_STD} \\",
            f"    --inflation {inflation} \\",
            f"    --num-eval-trajectories {NUM_EVAL_TRAJECTORIES} \\",
            f"    --batch-size {BATCH_SIZE} \\",
            f"    --device {DEVICE} \\",
            f"    --wandb-project {wandb_project} \\",
            f"    --wandb-entity {WANDB_ENTITY} \\",
            "    --experiment-label tuned_enkf \\",
            "    --wandb-tags tuned,best-config",
        ]
        
        cmd = "\n".join(args_lines)
        log_file = f"logs/eval_{dataset_name}_enkf.log"
        
        jobs.append({
            "job_idx": job_counter,
            "dataset_name": dataset_name,
            "cmd": cmd,
            "log_file": log_file,
        })
        job_counter += 1

    # Execute jobs in batches: one job per GPU, wait for the batch to finish
    if not jobs:
        lines.append("echo 'No Lorenz96 EnKF jobs found.'")
    else:
        batch_num = 0
        for batch_start in range(0, len(jobs), NUM_GPUS):
            batch = jobs[batch_start:batch_start + NUM_GPUS]
            batch_num += 1
            
            lines.append(f"echo 'Starting batch {batch_num}...'")
            for gpu_idx, job in enumerate(batch):
                lines.append(
                    f"echo 'Starting EnKF for {job['dataset_name']} on GPU {gpu_idx} (Job {job['job_idx']})...'"
                )
                lines.append(
                    f"CUDA_VISIBLE_DEVICES={gpu_idx} {job['cmd']} > {job['log_file']} 2>&1 &"
                )
                lines.append("")
            
            lines.append("wait")
            lines.append(f"echo 'Batch {batch_num} completed.'")
            lines.append("")

    lines.append("echo 'All evaluations completed.'")
    
    with open(OUTPUT_SCRIPT, "w") as f:
        f.write("\n".join(lines))
    
    # Make executable
    st = os.stat(OUTPUT_SCRIPT)
    os.chmod(OUTPUT_SCRIPT, st.st_mode | stat.S_IEXEC)
    
    print(f"Generated {OUTPUT_SCRIPT} with {len(lines)} lines.")

if __name__ == "__main__":
    main()
