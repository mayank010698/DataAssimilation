#!/bin/bash

# List of datasets to tune on
DATASETS=(
    "ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_J64_L100p53"
    "ks_n2048_len1000_dt0p2500_obs0p100_freq1_comp64of64_arctan_J64_L50p27"
    "ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_J64_L100p53"
    "ks_n2048_len1000_dt0p2500_obs0p200_freq1_comp64of64_arctan_J64_L50p27"
    "lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp10of10_arctan_init3p000"
    "lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp15of15_arctan_init3p000"
    "lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp20of20_arctan_init3p000"
    "lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp25of25_arctan_init3p000"
    "lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp50of50_arctan_init3p000"
    "lorenz96_n2048_len200_dt0p0100_obs0p100_freq1_comp5of5_arctan_init3p000"
)

BASE_DATA_DIR="/data/da_outputs/datasets"
BASE_OUTPUT_DIR="tuning_results"

# Loop through each dataset
for dataset in "${DATASETS[@]}"; do
    echo "================================================================="
    echo "Tuning for dataset: $dataset"
    echo "================================================================="
    
    DATA_DIR="$BASE_DATA_DIR/$dataset"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$dataset"
    
    # Check if data directory exists
    if [ ! -d "$DATA_DIR" ]; then
        echo "Warning: Dataset directory $DATA_DIR does not exist. Skipping."
        continue
    fi
    
    # Tune EnKF
    echo "--> Tuning EnKF..."
    /home/cnagda/miniconda3/envs/da/bin/python scripts/tune_enkf.py \
        --data-dir "$DATA_DIR" \
        --method enkf \
        --output-dir "$OUTPUT_DIR/enkf" \
        --num-eval-trajectories 1 \
        --batch-size 1
        
    # Tune LETKF
    echo "--> Tuning LETKF..."
    /home/cnagda/miniconda3/envs/da/bin/python scripts/tune_enkf.py \
        --data-dir "$DATA_DIR" \
        --method letkf \
        --output-dir "$OUTPUT_DIR/letkf" \
        --num-eval-trajectories 1 \
        --batch-size 1
        
    echo "Done with $dataset"
    echo ""
done

echo "All tuning runs completed."

