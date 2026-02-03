import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    DoubleWell,
    generate_dataset_directory_name,
    save_config_yaml
)

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Define Configuration for Double Well with process noise 0.1
    config = DataAssimilationConfig(
        num_trajectories=1000,
        len_trajectory=100,
        warmup_steps=100,
        dt=0.1,
        process_noise_std=0.0,  # The requested parameter change
        obs_noise_std=0.1,
        obs_frequency=1,
        obs_components=[0],
        obs_nonlinearity="identity",
        stochastic_validation_test=True,
        system_params={"system_name": "double_well"}
    )
    
    print("Generating Double Well dataset with process_noise_std=0.1...")
    
    base_data_dir = Path("datasets")
    
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
    
    # This will generate the data if it doesn't exist
    dm.prepare_data()
    
    # Save config.yaml
    save_config_yaml(config, data_dir / "config.yaml")
    
    print("\nGeneration completed successfully!")
    print(f"Data saved to: {data_dir}")

if __name__ == "__main__":
    main()

