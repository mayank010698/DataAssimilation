import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

"""
python generate_l96.py \
  --num_trajectories 1024 \
  --len_trajectory 1024 \
  --warmup_steps 1024 \
  --dt 0.01 \
  --obs-noise-std 0.25 \
  --obs-frequency 1 \
  --obs-components \
  --observation-operator arctan \
  --state-dim 40 \
  --dataset-name lorenz96
                                                    
"""

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    Lorenz63,
    Lorenz96,
    create_projection_matrix,
    generate_dataset_directory_name,
    save_config_yaml,
)


def parse_int_list(value: str) -> list:
    if value is None or value.strip() == "":
        return []
    return [int(v) for v in value.split(",") if v.strip() != ""]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dataset for data assimilation experiments."
    )

    # Data configuration
    parser.add_argument("--num-trajectories", type=int, default=1024)
    parser.add_argument("--len-trajectory", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--dt", type=float, default=0.01)

    # Observation configuration
    parser.add_argument("--obs-noise-std", type=float, default=0.25)
    parser.add_argument("--obs-frequency", type=int, default=2)
    parser.add_argument("--obs-components", type=str, default="0,2")
    parser.add_argument(
        "--observation-operator",
        type=str,
        choices=["linear_projection", "arctan"],
        default="arctan",
    )
    parser.add_argument("--state-dim", type=int, default=3)

    # System parameters
    parser.add_argument("--system-sigma", type=float, default=10.0)
    parser.add_argument("--system-rho", type=float, default=28.0)
    parser.add_argument("--system-beta", type=float, default=8.0 / 3.0)

    # Data splits
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Base directory for saving datasets",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Custom dataset directory name (overrides auto-generated name)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset without prompting",
    )

    # Logging configuration
    parser.add_argument("--log-level", type=str, default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("Dataset Generation for Data Assimilation")
    print("=" * 80)

    obs_components = parse_int_list(args.obs_components)
    if not obs_components:
        raise ValueError("At least one observation component must be specified.")

    if args.observation_operator == "linear_projection":
        observation_operator = create_projection_matrix(args.state_dim, obs_components)
    else:
        observation_operator = np.arctan

    system_params = {
        "F": 8,
        "dim": 40
    }

    config = DataAssimilationConfig(
        num_trajectories=args.num_trajectories,
        len_trajectory=args.len_trajectory,
        warmup_steps=args.warmup_steps,
        dt=args.dt,
        obs_noise_std=args.obs_noise_std,
        obs_frequency=args.obs_frequency,
        obs_components=obs_components,
        observation_operator=observation_operator,
        system_params=system_params,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Generate directory name
    if args.dataset_name:
        dataset_dir_name = args.dataset_name
    else:
        dataset_dir_name = generate_dataset_directory_name(config, system_name="lorenz96")

    # Create output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_base / dataset_dir_name

    print(f"\nDataset will be saved to: {dataset_dir}")
    print(f"\nConfiguration:")
    print(f"  num_trajectories: {config.num_trajectories}")
    print(f"  len_trajectory: {config.len_trajectory}")
    print(f"  warmup_steps: {config.warmup_steps}")
    print(f"  dt: {config.dt}")
    print(f"  obs_noise_std: {config.obs_noise_std}")
    print(f"  obs_frequency: {config.obs_frequency}")
    print(f"  obs_components: {config.obs_components}")
    print(f"  observation_operator: {args.observation_operator}")
    print(f"  system_params: {config.system_params}")

    # Check if dataset already exists
    if dataset_dir.exists() and (dataset_dir / "data.h5").exists():
        if args.force:
            response = "y"
        else:
            response = input(
                f"\nDataset directory {dataset_dir} already exists. Overwrite? (y/N): "
            )
        
        if response.lower() != "y":
            print("Aborting.")
            return
        # Remove existing directory
        import shutil
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    print("\nGenerating dataset...")
    system = Lorenz96(config)
    data_module = DataAssimilationDataModule(
        config=config,
        system_class=Lorenz63,
        data_dir=str(dataset_dir),
        batch_size=1,  # Not used for generation
    )

    data_module.prepare_data()

    # Save config as YAML
    config_path = dataset_dir / "config.yaml"
    save_config_yaml(config, config_path)
    print(f"\nConfig saved to {config_path}")

    print("\n" + "=" * 80)
    print("Dataset generation completed!")
    print(f"Dataset directory: {dataset_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

