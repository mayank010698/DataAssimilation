    #!/bin/bash

    # Get the directory where this script is located
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    # Change to project root so relative paths work
    cd "$SCRIPT_DIR/.." || exit

    # Setup PYTHONPATH
    export PYTHONPATH=$(pwd)

    # Create logs directory if it doesn't exist
    mkdir -p logs

    echo "=========================================="
    echo "Lorenz96 (50D) FlowDAS Training Experiments"
    echo "=========================================="
    echo "Configuration:"
    echo "- 8 Experiments total (4 No Noise, 4 With Noise)"
    echo "- Variations: Full (50), Half (25), 1/5 (10), No Obs"
    echo "- Architecture: ResNet1D (FlowDAS)"
    echo "=========================================="

    # Common parameters
    BATCH_SIZE=512
    LR=3e-4
    EPOCHS=500
    CHANNELS=64
    NUM_BLOCKS=10
    KERNEL_SIZE=5

    # Dataset paths
    DATA_DIR_NO_NOISE="/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p200_freq1_comp50of50_arctan_init3p000"
    DATA_DIR_WITH_NOISE="/data/da_outputs/datasets/lorenz96_n2048_len200_dt0p0100_obs0p200_freq1_comp50of50_arctan_pnoise0p100_init3p000"

    # Indices strings
    COMPONENTS_50="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49"
    COMPONENTS_25="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24"
    COMPONENTS_10="0,1,2,3,4,5,6,7,8,9"

    # --- GROUP 1: NO PROCESS NOISE (GPU 4-7) ---

    # 1. Full Obs (50) - GPU 4
    echo "Starting Exp 1: No Noise, Full Obs (50) (GPU 4)..."
    CUDA_VISIBLE_DEVICES=4 nohup python scripts/train_flowdas.py \
        --config "$DATA_DIR_NO_NOISE/config.yaml" \
        --data_dir "$DATA_DIR_NO_NOISE" \
        --run_dir "/data/da_outputs/runs_flowdas/l96_nonoise_full_1230" \
        --obs_components "$COMPONENTS_50" \
        --architecture resnet1d \
        --channels $CHANNELS \
        --num_blocks $NUM_BLOCKS \
        --kernel_size $KERNEL_SIZE \
        --use_observations \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --use_wandb \
        --evaluate \
        > logs/flowdas_l96_nonoise_full.log 2>&1 &
    PID1=$!
    echo "Started PID $PID1"

    # 2. Half Obs (25) - GPU 5
    echo "Starting Exp 2: No Noise, Half Obs (25) (GPU 5)..."
    CUDA_VISIBLE_DEVICES=5 nohup python scripts/train_flowdas.py \
        --config "$DATA_DIR_NO_NOISE/config.yaml" \
        --data_dir "$DATA_DIR_NO_NOISE" \
        --run_dir "/data/da_outputs/runs_flowdas/l96_nonoise_half_1230" \
        --obs_components "$COMPONENTS_25" \
        --architecture resnet1d \
        --channels $CHANNELS \
        --num_blocks $NUM_BLOCKS \
        --kernel_size $KERNEL_SIZE \
        --use_observations \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --use_wandb \
        --evaluate \
        > logs/flowdas_l96_nonoise_half.log 2>&1 &
    PID2=$!
    echo "Started PID $PID2"

    # 3. 1/5 Obs (10) - GPU 6
    echo "Starting Exp 3: No Noise, 1/5 Obs (10) (GPU 6)..."
    CUDA_VISIBLE_DEVICES=6 nohup python scripts/train_flowdas.py \
        --config "$DATA_DIR_NO_NOISE/config.yaml" \
        --data_dir "$DATA_DIR_NO_NOISE" \
        --run_dir "/data/da_outputs/runs_flowdas/l96_nonoise_tenth_1230" \
        --obs_components "$COMPONENTS_10" \
        --architecture resnet1d \
        --channels $CHANNELS \
        --num_blocks $NUM_BLOCKS \
        --kernel_size $KERNEL_SIZE \
        --use_observations \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --use_wandb \
        --evaluate \
        > logs/flowdas_l96_nonoise_tenth.log 2>&1 &
    PID3=$!
    echo "Started PID $PID3"

    # 4. No Obs - GPU 7
    echo "Starting Exp 4: No Noise, No Obs (GPU 7)..."
    CUDA_VISIBLE_DEVICES=7 nohup python scripts/train_flowdas.py \
        --config "$DATA_DIR_NO_NOISE/config.yaml" \
        --data_dir "$DATA_DIR_NO_NOISE" \
        --run_dir "/data/da_outputs/runs_flowdas/l96_nonoise_no_obs_1230" \
        --architecture resnet1d \
        --channels $CHANNELS \
        --num_blocks $NUM_BLOCKS \
        --kernel_size $KERNEL_SIZE \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --use_wandb \
        --evaluate \
        > logs/flowdas_l96_nonoise_no_obs.log 2>&1 &
    PID4=$!
    echo "Started PID $PID4"

    # --- GROUP 2: WITH PROCESS NOISE 0.1 (GPU 4-7) ---

    # 5. Full Obs (50) - GPU 4
    echo "Starting Exp 5: Noise 0.1, Full Obs (50) (GPU 4)..."
    CUDA_VISIBLE_DEVICES=4 nohup python scripts/train_flowdas.py \
        --config "$DATA_DIR_WITH_NOISE/config.yaml" \
        --data_dir "$DATA_DIR_WITH_NOISE" \
        --run_dir "/data/da_outputs/runs_flowdas/l96_noise0p1_full_1230" \
        --obs_components "$COMPONENTS_50" \
        --architecture resnet1d \
        --channels $CHANNELS \
        --num_blocks $NUM_BLOCKS \
        --kernel_size $KERNEL_SIZE \
        --use_observations \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --use_wandb \
        --evaluate \
        > logs/flowdas_l96_noise0p1_full.log 2>&1 &
    PID5=$!
    echo "Started PID $PID5"

    # 6. Half Obs (25) - GPU 5
    echo "Starting Exp 6: Noise 0.1, Half Obs (25) (GPU 5)..."
    CUDA_VISIBLE_DEVICES=5 nohup python scripts/train_flowdas.py \
        --config "$DATA_DIR_WITH_NOISE/config.yaml" \
        --data_dir "$DATA_DIR_WITH_NOISE" \
        --run_dir "/data/da_outputs/runs_flowdas/l96_noise0p1_half_1230" \
        --obs_components "$COMPONENTS_25" \
        --architecture resnet1d \
        --channels $CHANNELS \
        --num_blocks $NUM_BLOCKS \
        --kernel_size $KERNEL_SIZE \
        --use_observations \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --use_wandb \
        --evaluate \
        > logs/flowdas_l96_noise0p1_half.log 2>&1 &
    PID6=$!
    echo "Started PID $PID6"

    # 7. 1/5 Obs (10) - GPU 6
    echo "Starting Exp 7: Noise 0.1, 1/5 Obs (10) (GPU 6)..."
    CUDA_VISIBLE_DEVICES=6 nohup python scripts/train_flowdas.py \
        --config "$DATA_DIR_WITH_NOISE/config.yaml" \
        --data_dir "$DATA_DIR_WITH_NOISE" \
        --run_dir "/data/da_outputs/runs_flowdas/l96_noise0p1_tenth_1230" \
        --obs_components "$COMPONENTS_10" \
        --architecture resnet1d \
        --channels $CHANNELS \
        --num_blocks $NUM_BLOCKS \
        --kernel_size $KERNEL_SIZE \
        --use_observations \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --use_wandb \
        --evaluate \
        > logs/flowdas_l96_noise0p1_tenth.log 2>&1 &
    PID7=$!
    echo "Started PID $PID7"

    # 8. No Obs - GPU 7
    echo "Starting Exp 8: Noise 0.1, No Obs (GPU 7)..."
    CUDA_VISIBLE_DEVICES=7 nohup python scripts/train_flowdas.py \
        --config "$DATA_DIR_WITH_NOISE/config.yaml" \
        --data_dir "$DATA_DIR_WITH_NOISE" \
        --run_dir "/data/da_outputs/runs_flowdas/l96_noise0p1_no_obs_1230" \
        --architecture resnet1d \
        --channels $CHANNELS \
        --num_blocks $NUM_BLOCKS \
        --kernel_size $KERNEL_SIZE \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --epochs $EPOCHS \
        --use_wandb \
        --evaluate \
        > logs/flowdas_l96_noise0p1_no_obs.log 2>&1 &
    PID8=$!
    echo "Started PID $PID8"

    echo "=========================================="
    echo "All 8 FlowDAS training experiments started in background."
    echo "Logs are being written to logs/"
    echo "Monitor with: tail -f logs/flowdas_l96_*.log"
    echo "=========================================="