
import subprocess
import sys
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

def run_evaluation(data_dir: Path, n_particles: int, process_noise_std: float, run_name: str):
    """Run eval.py subprocess"""
    python_executable = sys.executable
    cmd = [
        python_executable, "eval.py",
        "--data-dir", str(data_dir),
        "--method", "bpf",
        "--proposal-type", "transition",
        "--n-particles", str(n_particles),
        "--process-noise-std", str(process_noise_std),
        "--batch-size", "100", # Good batch size for 1D system
        "--wandb-project", "pf-eval-dw",
        "--run-name", run_name,
        "--experiment-label", "dw_bpf",
        "--device", "cuda"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Define Configurations
    
    # Case 1: Standard Variance
    # process_noise_std: 0.2, obs_noise_std: 0.1, dt: 0.1, len: 100
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
    # process_noise_std: 0.3, obs_noise_std: 0.1, dt: 0.1, len: 100
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
        ("larger_var", config_large)
    ]
    
    # 2. Generate Data & Run Eval
    base_data_dir = Path("datasets")
    
    for name, config in configs:
        print(f"\nProcessing {name}...")
        
        # Generate directory name
        dir_name = generate_dataset_directory_name(config, system_name="double_well")
        data_dir = base_data_dir / dir_name
        
        print(f"Dataset directory: {data_dir}")
        
        # Initialize DataModule (triggers generation if needed)
        dm = DataAssimilationDataModule(
            config=config,
            system_class=DoubleWell,
            data_dir=str(data_dir)
        )
        dm.prepare_data() # Ensure data exists
        
        # Save config.yaml (required by eval.py)
        save_config_yaml(config, data_dir / "config.yaml")
        
        # Run Evaluation
        # We use the same process noise in the filter as in the system for this evaluation
        run_name = f"bpf_1k_{name}"
        run_evaluation(
            data_dir=data_dir, 
            n_particles=1000, 
            process_noise_std=config.process_noise_std, 
            run_name=run_name
        )

if __name__ == "__main__":
    main()

