import argparse
import logging
import os
import sys
import random
import torch
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    Lorenz63,
    Lorenz96,
    KuramotoSivashinsky,
    generate_dataset_directory_name,
    save_config_yaml,
    generate_dataset_splits,
    save_generated_data,
    observations_from_trajectories,
)


def parse_float_list(value: str) -> list:
    if value is None or value.strip() == "":
        return []
    return [float(v) for v in value.split(",") if v.strip() != ""]


def parse_str_list(value: str) -> list:
    if value is None or value.strip() == "":
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dataset for data assimilation experiments."
    )

    # System selection
    parser.add_argument(
        "--system",
        type=str,
        choices=["lorenz63", "lorenz96", "ks", "kuramoto-sivashinsky"],
        default="lorenz63",
        help="Dynamical system to use",
    )

    # Data configuration
    parser.add_argument("--num-trajectories", type=int, default=1024)
    parser.add_argument("--len-trajectory", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--dt", type=float, default=0.01)

    # Process noise (for stochastic trajectory generation)
    parser.add_argument(
        "--process-noise-std",
        type=float,
        default=0.0,
        help="Process noise std for training trajectories (default: 0.0 = deterministic)",
    )
    
    parser.add_argument(
        "--process-noise-variations",
        type=str,
        default=None,
        help="Comma-separated list of process noise values to generate (e.g., '0.0,0.1'). If provided, overrides --process-noise-std.",
    )

    # Observation configuration
    parser.add_argument("--obs-noise-std", type=float, default=0.25)
    parser.add_argument(
        "--obs-noise-variations",
        type=str,
        default=None,
        help="Comma-separated list of observation noise std values to generate (e.g. '1,3,5'). Same trajectories (shared ICs) across datasets.",
    )
    parser.add_argument("--obs-frequency", type=int, default=2)
    # obs-components is ignored for generation (always dense) but kept for compatibility/future use
    parser.add_argument(
        "--obs-components",
        type=str,
        default=None,
        help="Ignored for generation (always dense). Use during evaluation/training.",
    )
    parser.add_argument(
        "--observation-operator",
        type=str,
        choices=["linear_projection", "arctan", "identity", "square", "cube", "quad_capped_10"],
        default="arctan",
        help="Observation nonlinearity type (used when --observation-operators is not set)",
    )
    parser.add_argument(
        "--observation-operators",
        type=str,
        default=None,
        help="Comma-separated list of observation operators (e.g. 'identity,quad_capped_10'). When set, same state trajectories are used for all; overrides --observation-operator.",
    )
    parser.add_argument("--state-dim", type=int, default=None, help="State dimension (auto-detected from system)")

    # System parameters (Lorenz 63)
    parser.add_argument("--system-sigma", type=float, default=10.0)
    parser.add_argument("--system-rho", type=float, default=28.0)
    parser.add_argument("--system-beta", type=float, default=8.0 / 3.0)

    # System parameters (Lorenz 96)
    parser.add_argument("--l96-dim", type=int, default=50, help="Dimension for Lorenz 96 system")
    parser.add_argument("--l96-forcing", type=float, default=8.0, help="Forcing parameter F for Lorenz 96")
    parser.add_argument("--l96-init-std", type=float, default=3.0, help="Initial standard deviation for Lorenz 96 (Gaussian)")
    parser.add_argument("--l96-init-sampling", type=str, choices=["gaussian", "uniform"], default="gaussian", help="Initial condition sampling for Lorenz 96")
    parser.add_argument("--l96-init-low", type=float, default=-10.0, help="Lower bound for uniform IC sampling (L96)")
    parser.add_argument("--l96-init-high", type=float, default=10.0, help="Upper bound for uniform IC sampling (L96)")

    # System parameters (Kuramoto-Sivashinsky)
    parser.add_argument("--ks-J", type=int, default=64, help="Spatial resolution J for KS")
    parser.add_argument("--ks-L", type=float, default=None, help="Domain size L for KS (default: 16*pi or 32*pi depending on usage)")
    parser.add_argument("--ks-init-std", type=float, default=1.0, help="Initial standard deviation for KS")

    # Data splits
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/da_outputs/datasets/",
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

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    # Logging configuration
    parser.add_argument("--log-level", type=str, default="INFO")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds if provided
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("Dataset Generation for Data Assimilation")
    print("=" * 80)

    # Determine system class and parameters
    if args.system == "lorenz63":
        system_class = Lorenz63
        system_name = "lorenz63"
        state_dim = 3
        system_params = {
            "sigma": args.system_sigma,
            "rho": args.system_rho,
            "beta": args.system_beta,
        }
    elif args.system == "lorenz96":
        system_class = Lorenz96
        system_name = "lorenz96"
        state_dim = args.l96_dim
        system_params = {
            "dim": args.l96_dim,
            "F": args.l96_forcing,
            "init_std": args.l96_init_std,
            "init_sampling": args.l96_init_sampling,
            "init_low": args.l96_init_low,
            "init_high": args.l96_init_high,
        }
    elif args.system in ["ks", "kuramoto-sivashinsky"]:
        system_class = KuramotoSivashinsky
        system_name = "ks"
        state_dim = args.ks_J
        
        # Default L to 16*pi if not specified, though script usually passes it
        L = args.ks_L if args.ks_L is not None else 16 * np.pi
        
        system_params = {
            "J": args.ks_J,
            "L": L,
            "init_std": args.ks_init_std,
        }
    else:
        raise ValueError(f"Unknown system: {args.system}")

    # Override state_dim if explicitly provided
    if args.state_dim is not None:
        state_dim = args.state_dim

    # FORCE DENSE OBSERVATIONS
    obs_components = list(range(state_dim))
    print(f"Generating DENSE observations for all {state_dim} components.")

    # Determine process noise levels
    if args.process_noise_variations:
        process_noise_levels = parse_float_list(args.process_noise_variations)
        print(f"Generating datasets for process noise levels: {process_noise_levels}")
    else:
        process_noise_levels = [args.process_noise_std]

    # Determine observation noise levels (same trajectories across obs noise when using variations)
    if args.obs_noise_variations:
        obs_noise_levels = parse_float_list(args.obs_noise_variations)
        print(f"Generating datasets for observation noise std levels: {obs_noise_levels}")
    else:
        obs_noise_levels = [args.obs_noise_std]

    # Observation operators: multiple => same state trajectories for all (only observations differ)
    OBS_OPERATOR_CHOICES = ["linear_projection", "arctan", "identity", "square", "cube", "quad_capped_10"]
    if args.observation_operators:
        observation_operators_list = parse_str_list(args.observation_operators)
        for op in observation_operators_list:
            if op not in OBS_OPERATOR_CHOICES:
                raise ValueError(f"Unknown observation operator '{op}'. Choices: {OBS_OPERATOR_CHOICES}")
        print(f"Generating datasets for observation operators (same trajectories): {observation_operators_list}")
    else:
        observation_operators_list = [args.observation_operator]

    # Shared initial states (lazy initialization) so trajectories match across obs-noise variations
    shared_initial_states = None
    shared_splits = {}

    for obs_noise in obs_noise_levels:
        for i, p_noise in enumerate(process_noise_levels):
            print(f"\n[obs_noise={obs_noise}, process_noise={p_noise}] Generating dataset(s)...")
            trajectories_for_this_combo = None

            for i_op, obs_op in enumerate(observation_operators_list):
                config = DataAssimilationConfig(
                    num_trajectories=args.num_trajectories,
                    len_trajectory=args.len_trajectory,
                    warmup_steps=args.warmup_steps,
                    dt=args.dt,
                    obs_noise_std=obs_noise,
                    obs_frequency=args.obs_frequency,
                    obs_components=obs_components,
                    obs_nonlinearity=obs_op,
                    process_noise_std=p_noise,
                    system_params=system_params,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                )

                system = system_class(config)

                # Generate shared initial states once (same ICs across all datasets / operators)
                if shared_initial_states is None:
                    print("Sampling SHARED initial states for all variations...")
                    n_train = int(config.train_ratio * config.num_trajectories)
                    n_val = int(config.val_ratio * config.num_trajectories)
                    n_test = config.num_trajectories - n_train - n_val

                    x0_train = system.sample_initial_state(n_train)
                    x0_val = system.sample_initial_state(n_val)
                    x0_test = system.sample_initial_state(n_test)

                    if n_train == 1 and x0_train.ndim == 1: x0_train = x0_train.unsqueeze(0)
                    if n_val == 1 and x0_val.ndim == 1: x0_val = x0_val.unsqueeze(0)
                    if n_test == 1 and x0_test.ndim == 1: x0_test = x0_test.unsqueeze(0)

                    shared_initial_states = {
                        "train": x0_train,
                        "val": x0_val,
                        "test": x0_test,
                    }

                if i_op == 0:
                    # First operator: generate trajectories + observations
                    print(f"\n  Operator '{obs_op}': generating trajectories and observations...")
                    precomputed_splits_for_call = None
                    if obs_noise == obs_noise_levels[0] and "val" in shared_splits and "test" in shared_splits:
                        precomputed_splits_for_call = {
                            "val": shared_splits["val"],
                            "test": shared_splits["test"],
                        }
                        print("  Using precomputed val/test splits for consistency.")

                    splits_unscaled, obs_mask = generate_dataset_splits(
                        system,
                        config,
                        initial_states=shared_initial_states,
                        precomputed_splits=precomputed_splits_for_call,
                    )
                    trajectories_for_this_combo = {k: v["trajectories"] for k, v in splits_unscaled.items()}
                    if obs_noise == obs_noise_levels[0] and "val" not in shared_splits:
                        shared_splits["val"] = (splits_unscaled["val"]["trajectories"], splits_unscaled["val"]["observations"])
                        shared_splits["test"] = (splits_unscaled["test"]["trajectories"], splits_unscaled["test"]["observations"])
                else:
                    # Same trajectories, different observation operator => recompute observations only
                    print(f"\n  Operator '{obs_op}': reusing trajectories, computing observations only...")
                    obs_train, obs_mask = observations_from_trajectories(
                        system, trajectories_for_this_combo["train"], config.obs_frequency, config.len_trajectory
                    )
                    obs_val, _ = observations_from_trajectories(
                        system, trajectories_for_this_combo["val"], config.obs_frequency, config.len_trajectory
                    )
                    obs_test, _ = observations_from_trajectories(
                        system, trajectories_for_this_combo["test"], config.obs_frequency, config.len_trajectory
                    )
                    splits_unscaled = {
                        "train": {"trajectories": trajectories_for_this_combo["train"], "observations": obs_train},
                        "val": {"trajectories": trajectories_for_this_combo["val"], "observations": obs_val},
                        "test": {"trajectories": trajectories_for_this_combo["test"], "observations": obs_test},
                    }

                # Directory name and save (per operator)
                if args.dataset_name:
                    if len(obs_noise_levels) > 1 or len(process_noise_levels) > 1 or len(observation_operators_list) > 1:
                        dataset_dir_name = f"{args.dataset_name}_obs{obs_noise:.3f}_pnoise{p_noise:.3f}".replace(".", "p")
                        if len(observation_operators_list) > 1:
                            dataset_dir_name = f"{dataset_dir_name}_{obs_op}"
                    else:
                        dataset_dir_name = args.dataset_name
                else:
                    dataset_dir_name = generate_dataset_directory_name(config, system_name=system_name)

                output_base = Path(args.output_dir)
                output_base.mkdir(parents=True, exist_ok=True)
                dataset_dir = output_base / dataset_dir_name

                print(f"  Saving to {dataset_dir}")

                if dataset_dir.exists() and (dataset_dir / "data.h5").exists():
                    if args.force:
                        response = "y"
                    else:
                        response = input(
                            f"\nDataset directory {dataset_dir} already exists. Overwrite? (y/N): "
                        )

                    if response.lower() != "y":
                        print("  Skipping this dataset.")
                        continue

                    import shutil
                    shutil.rmtree(dataset_dir)

                dataset_dir.mkdir(parents=True, exist_ok=True)
                save_generated_data(dataset_dir, splits_unscaled, obs_mask)
                config_path = dataset_dir / "config.yaml"
                save_config_yaml(config, config_path)
                print(f"  Config saved to {config_path}")

    print("\n" + "=" * 80)
    print("Dataset generation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
