#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# List of available GPUs
# Edit this list to enable/disable specific GPUs
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)

# -----------------------------------------------------------------------------

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Starting Evaluations (KS) - BPF"
echo "=========================================="
echo "Using GPUs: ${AVAILABLE_GPUS[*]}"

# Dataset directory
DATASETS_DIR="/data/da_outputs/datasets"

# Check if datasets dir exists
if [ ! -d "$DATASETS_DIR" ]; then
    echo "Error: Dataset directory $DATASETS_DIR not found!"
    exit 1
fi

# Collect all KS datasets
# Note: This will only match directories starting with ks_
mapfile -t datasets < <(find "$DATASETS_DIR" -maxdepth 1 -name "ks_*" -type d | sort)

if [ ${#datasets[@]} -eq 0 ]; then
    echo "No KS datasets found in $DATASETS_DIR"
    exit 1
fi

echo "Found ${#datasets[@]} KS datasets."

# Create job list
# Each job is a string: "DATA_DIR N_PARTICLES BPF_PROCESS_NOISE"
declare -a jobs
for d in "${datasets[@]}"; do
    dir_name=$(basename "$d")
    
    # Filter: Only use datasets where process noise (during generation) was 0.
    # Datasets with pnoise > 0 have "pnoise" in their directory name.
    if [[ "$dir_name" == *"pnoise"* ]]; then
        continue
    fi

    # Filter: Only use datasets where JS is 64 and L_MULTS is 16 (L ~ 50.27)
    # if [[ "$dir_name" != *"J64"* ]] || [[ "$dir_name" != *"L50p27"* ]]; then
    #     continue
    # fi

    # For each dataset, run combinations of particles and BPF process noise
    for particles in 5000; do
        for p_noise in 0.05; do
            for n_eval in 50; do
                jobs+=("$d $particles $p_noise $n_eval")
            done
        done
    done
done

echo "Total jobs to run: ${#jobs[@]}"

if [ ${#jobs[@]} -eq 0 ]; then
    echo "No jobs created. Check filtering logic or if datasets exist."
    exit 1
fi

# Helper function to check if a value is in an array
elementIn () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

# Distribute jobs to available GPUs
# We'll use an associative array to simulate lists of lists (since bash doesn't support nested arrays directly)
# Keys will be "gpu_id", value will be space-separated indices of jobs
declare -A gpu_job_indices

num_gpus=${#AVAILABLE_GPUS[@]}
if [ $num_gpus -eq 0 ]; then
    echo "Error: No GPUs defined in AVAILABLE_GPUS."
    exit 1
fi

for ((i=0; i<${#jobs[@]}; i++)); do
    # Determine which GPU gets this job (round-robin)
    gpu_idx=$((i % num_gpus))
    gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
    
    # Append job index to that GPU's list
    if [ -z "${gpu_job_indices[$gpu_id]}" ]; then
        gpu_job_indices[$gpu_id]="$i"
    else
        gpu_job_indices[$gpu_id]="${gpu_job_indices[$gpu_id]} $i"
    fi
done

# Function to process a queue of jobs on a specific GPU
process_queue() {
    local gpu_id=$1
    shift
    local job_indices_str="$1"
    
    # Convert space-separated string back to array
    read -ra job_indices <<< "$job_indices_str"
    
    echo "GPU $gpu_id: Assigned ${#job_indices[@]} jobs."
    
    for idx in "${job_indices[@]}"; do
        local job="${jobs[$idx]}"
        read -r data_dir n_particles bpf_p_noise n_eval <<< "$job"
        
        local dir_name=$(basename "$data_dir")
        # Format noise string for filename (0.05 -> 0p05)
        local noise_str=$(echo "$bpf_p_noise" | sed 's/\./p/g')
        
        local run_name="eval_ks_bpf_${n_particles}p_pnoise${noise_str}_trajs${n_eval}_${dir_name}"
        local log_file="logs/${run_name}.log"
        
        echo "GPU $gpu_id: Starting $run_name"
        
        # Run eval.py
        # Explicitly setting --process-noise-std for the BPF
        CUDA_VISIBLE_DEVICES=$gpu_id python eval.py \
            --data-dir "$data_dir" \
            --proposal-type transition \
            --n-particles "$n_particles" \
            --process-noise-std "$bpf_p_noise" \
            --num-eval-trajectories "$n_eval" \
            --batch-size 100 \
            --obs-frequency 10 \
            --wandb-project pf-eval-ks \
            --run-name "$run_name" \
            --experiment-label "ks_bpf" \
            --device cuda \
            > "$log_file" 2>&1
            
        local status=$?
        if [ $status -eq 0 ]; then
            echo "GPU $gpu_id: Finished $run_name"
        else
            echo "GPU $gpu_id: Failed $run_name (Exit: $status)"
        fi
    done
}

# Launch background processes for each GPU
pids=()
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    job_indices="${gpu_job_indices[$gpu_id]}"
    if [ -n "$job_indices" ]; then
        (process_queue "$gpu_id" "$job_indices") &
        pids+=($!)
    else
        echo "GPU $gpu_id: No jobs assigned."
    fi
done

echo "All job queues started. PIDs: ${pids[*]}"
echo "Waiting for completion..."

# Wait for all queues to finish
for pid in "${pids[@]}"; do
    wait $pid
done

echo "=========================================="
echo "All evaluations completed."
echo "=========================================="
