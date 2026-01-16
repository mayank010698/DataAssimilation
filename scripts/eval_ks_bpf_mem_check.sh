#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Change to project root so relative paths work
cd "$SCRIPT_DIR/.." || exit

# Setup PYTHONPATH
export PYTHONPATH=$(pwd)

# Configuration
DATA_DIR="/data/da_outputs/datasets/ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_J64_L50p27"
N_PARTICLES=5000
P_NOISE=0.05
OBS_FREQ=10
GPU_ID=0  # Default to GPU 0 for testing

echo "=========================================="
echo "Starting BPF Batch Size Memory Benchmark"
echo "Dataset: $(basename "$DATA_DIR")"
echo "Using GPU: $GPU_ID"
echo "=========================================="

mkdir -p logs

# Loop through batch sizes
for BS in 100 50 20 15 10; do
    echo "------------------------------------------"
    echo "Testing Batch Size: $BS"
    
    LOG_FILE="logs/mem_check_bs${BS}.log"
    GPU_LOG="logs/gpu_mem_bs${BS}.csv"
    
    # clear previous logs
    rm -f "$LOG_FILE" "$GPU_LOG"

    # Start nvidia-smi monitoring in background
    # Log memory usage every 1 second
    nvidia-smi --id=$GPU_ID --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -l 1 > "$GPU_LOG" 2>&1 &
    SMI_PID=$!

    echo "Running eval.py for 4 minutes..."
    
    # Run eval.py with timeout
    # Using 200 trajectories to ensure it runs long enough
    start_time=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=$GPU_ID timeout 240s python eval.py \
        --data-dir "$DATA_DIR" \
        --proposal-type transition \
        --n-particles "$N_PARTICLES" \
        --process-noise-std "$P_NOISE" \
        --num-eval-trajectories 200 \
        --obs-frequency "$OBS_FREQ" \
        --batch-size "$BS" \
        --wandb-project pf-eval-ks-memcheck \
        --run-name "mem_check_bs${BS}" \
        --experiment-label "mem_check" \
        --device cuda \
        > "$LOG_FILE" 2>&1
        
    EXIT_CODE=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # Stop nvidia-smi
    kill $SMI_PID 2>/dev/null
    wait $SMI_PID 2>/dev/null

    # Analyze results
    echo "Duration: ${duration}s"
    
    if [ $EXIT_CODE -eq 124 ]; then
        echo "Status: TIMEOUT (4 minutes reached - feasible)"
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "Status: COMPLETED (Finished before timeout)"
    else
        echo "Status: FAILED (Exit Code: $EXIT_CODE)"
        echo "Check log: $LOG_FILE"
        # Print last few lines of log to see error
        tail -n 5 "$LOG_FILE"
    fi

    # Calculate Max Memory Usage
    if [ -f "$GPU_LOG" ]; then
        # Check if log is empty
        if [ ! -s "$GPU_LOG" ]; then
             echo "GPU Log is empty."
        else
            # Extract max memory used
            # Column 1 is used, Column 2 is total
            max_mem=$(awk -F', ' '{if($1>m)m=$1} END{print m}' "$GPU_LOG")
            total_mem=$(awk -F', ' 'NR==1{print $2}' "$GPU_LOG")
            
            if [ -n "$max_mem" ] && [ -n "$total_mem" ] && [ "$total_mem" -gt 0 ]; then
                percentage=$(echo "scale=2; ($max_mem / $total_mem) * 100" | bc)
                echo "Max Memory Used: $max_mem MiB / $total_mem MiB ($percentage%)"
            else
                echo "Could not calculate memory stats from $GPU_LOG"
            fi
        fi
    else
        echo "No GPU log found."
    fi
    
done

echo "=========================================="
echo "Benchmark Completed"
echo "=========================================="

