
import subprocess
import sys
import json
import os
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    DoubleWell,
    generate_dataset_directory_name,
    save_config_yaml
)

def run_evaluation(data_dir: Path, n_particles: int, process_noise_std: float, inflation: float, run_name: str, device: str = "cuda"):
    """Run eval.py subprocess"""
    python_executable = sys.executable
    cmd = [
        python_executable, "eval.py",
        "--data-dir", str(data_dir),
        "--method", "enkf",
        "--n-particles", str(n_particles),
        "--process-noise-std", str(process_noise_std),
        "--inflation", str(inflation),
        "--batch-size", "100",
        "--wandb-project", "enkf-eval-dw",
        "--run-name", run_name,
        "--experiment-label", "dw_enkf",
        "--device", device,
        "--num-eval-trajectories", "50",
        "--init-mode", "truth" # Initialize from truth + obs_noise (standard for comparison)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def load_best_inflation(tuning_dir: Path) -> float:
    config_path = tuning_dir / "best_config_enkf.json"
    if not config_path.exists():
        print(f"Warning: Best config not found at {config_path}. Using default 1.0.")
        return 1.0
        
    with open(config_path, "r") as f:
        config = json.load(f)
    return float(config.get("inflation", 1.0))

def main():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Define Configurations
    
    # Case 1: Standard Variance
    config_std = DataAssimilationConfig(
        num_trajectories=1000,
        len_trajectory=100,
        warmup_steps=100,
        dt=0.1,
        process_noise_std=0.2,
        obs_noise_std=0.1,
        obs_frequency=1,
        obs_components=[0],
        obs_nonlinearity="identity",
        stochastic_validation_test=True,
        system_params={"system_name": "double_well"}
    )
    
    # Case 2: Larger Variance
    config_large = DataAssimilationConfig(
        num_trajectories=1000,
        len_trajectory=100,
        warmup_steps=100,
        dt=0.1,
        process_noise_std=0.3,
        obs_noise_std=0.1,
        obs_frequency=1,
        obs_components=[0],
        obs_nonlinearity="identity",
        stochastic_validation_test=True,
        system_params={"system_name": "double_well"}
    )
    
    configs = [
        ("standard_var", config_std),
        # ("larger_var", config_large)
    ]
    
    base_data_dir = Path("datasets")
    tuning_root = Path("tuning_results/enkf_dw")
    
    for name, config in configs:
        print(f"\nProcessing {name}...")
        
        # Generate directory name
        dir_name = generate_dataset_directory_name(config, system_name="double_well")
        data_dir = base_data_dir / dir_name
        
        # Find best inflation
        tuning_dir = tuning_root / dir_name
        inflation = load_best_inflation(tuning_dir)
        print(f"Using tuned inflation: {inflation}")
        
        # Run Evaluation
        run_name = f"enkf_50_{name}_infl{inflation}_0.1"
        
        # Use GPU 1 for standard, GPU 2 for large if possible, or just current env
        # We'll just run sequentially here
        run_evaluation(
            data_dir=data_dir, 
            n_particles=50, 
            process_noise_std=0.1,
            inflation=1.0,
            run_name=run_name,
            device="cuda"
        )

if __name__ == "__main__":
    main()

