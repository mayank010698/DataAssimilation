#!/bin/bash

# Configuration
PYTHON_BIN="/home/cnagda/miniconda3/envs/da/bin/python"
LOG_DIR="logs_rerun"
mkdir -p "$LOG_DIR"

# Check if python exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python binary not found at $PYTHON_BIN"
    exit 1
fi

# Function to run a job
run_job() {
    local cmd="$1"
    local log_file="$2"
    local gpu_id="$3"
    
    echo "------------------------------------------------"
    echo "Launching on GPU $gpu_id:"
    echo "  Command: $cmd"
    echo "  Log: $log_file"
    
    # Run in background with nohup, redirecting output
    CUDA_VISIBLE_DEVICES="$gpu_id" nohup $cmd > "$log_file" 2>&1 &
    local pid=$!
    echo "  PID: $pid"
    
    # Store PID to wait for it later if needed, or we can just let them run.
    # But the requirement is "Only one job should be running on a GPU at a time."
    # So we need to manage queues per GPU.
}

# We have 18 runs. Assuming 8 GPUs (0-7).
# We can distribute them round-robin or using a queue system.
# Since this is a simple script, let's create 8 queues (arrays of commands)
# and launch 8 background sub-shells, one for each GPU, that process their queue sequentially.

# Arrays to hold commands for each GPU
declare -a GPU_QUEUES_0
declare -a GPU_QUEUES_1
declare -a GPU_QUEUES_2
declare -a GPU_QUEUES_3
declare -a GPU_QUEUES_4
declare -a GPU_QUEUES_5
declare -a GPU_QUEUES_6
declare -a GPU_QUEUES_7

# Arrays to hold log filenames corresponding to commands
declare -a LOG_FILES_0
declare -a LOG_FILES_1
declare -a LOG_FILES_2
declare -a LOG_FILES_3
declare -a LOG_FILES_4
declare -a LOG_FILES_5
declare -a LOG_FILES_6
declare -a LOG_FILES_7

# Read commands from run_commands.txt
# We need to filter out comments and empty lines
# Also, commands in the file might contain `... --device cuda` which is fine, CUDA_VISIBLE_DEVICES handles physical mapping.

COMMANDS_FILE="run_commands.txt"
current_gpu=0

while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    if [[ "$line" =~ ^#.* ]] || [[ -z "$line" ]]; then
        continue
    fi
    
    # Extract run name for log file
    # Look for --run-name argument
    run_name=$(echo "$line" | grep -oP "(?<=--run-name )[^ ]+")
    if [ -z "$run_name" ]; then
        run_name="rerun_$(date +%s)_$current_gpu"
    fi
    log_file="$LOG_DIR/${run_name}.log"
    
    # Assign to current GPU queue
    case $current_gpu in
        0) GPU_QUEUES_0+=("$line"); LOG_FILES_0+=("$log_file") ;;
        1) GPU_QUEUES_1+=("$line"); LOG_FILES_1+=("$log_file") ;;
        2) GPU_QUEUES_2+=("$line"); LOG_FILES_2+=("$log_file") ;;
        3) GPU_QUEUES_3+=("$line"); LOG_FILES_3+=("$log_file") ;;
        4) GPU_QUEUES_4+=("$line"); LOG_FILES_4+=("$log_file") ;;
        5) GPU_QUEUES_5+=("$line"); LOG_FILES_5+=("$log_file") ;;
        6) GPU_QUEUES_6+=("$line"); LOG_FILES_6+=("$log_file") ;;
        7) GPU_QUEUES_7+=("$line"); LOG_FILES_7+=("$log_file") ;;
    esac
    
    # Cycle GPU (0-7)
    current_gpu=$(( (current_gpu + 1) % 8 ))
    
done < "$COMMANDS_FILE"

# Function to process a queue for a specific GPU
process_queue() {
    local gpu_id=$1
    # Get reference to the arrays
    local -n queue="GPU_QUEUES_$gpu_id"
    local -n logs="LOG_FILES_$gpu_id"
    
    echo "GPU $gpu_id: Assigned ${#queue[@]} jobs."
    
    for i in "${!queue[@]}"; do
        cmd="${queue[$i]}"
        log="${logs[$i]}"
        
        echo "GPU $gpu_id: Starting job $((i+1))/${#queue[@]}..."
        echo "  Run: $(echo "$cmd" | grep -oP "(?<=--run-name )[^ ]+")"
        
        # Run command
        # Note: We must strip the existing python path if it's already in the command from the file?
        # The lines in run_commands.txt start with /home/cnagda/miniconda3/envs/da/bin/python
        # So we just execute the line as is.
        
        CUDA_VISIBLE_DEVICES="$gpu_id" $cmd > "$log" 2>&1
        
        status=$?
        if [ $status -eq 0 ]; then
             echo "GPU $gpu_id: Job $((i+1)) completed successfully."
        else
             echo "GPU $gpu_id: Job $((i+1)) failed with exit code $status. Check $log"
        fi
    done
    
    echo "GPU $gpu_id: All assigned jobs completed."
}

echo "Starting distribution of jobs across 8 GPUs..."

# Launch background processor for each GPU
for gpu in {0..7}; do
    process_queue $gpu &
    pids[$gpu]=$!
done

echo "All GPU queues started. Waiting for completion..."

# Wait for all background processes
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All runs completed."

