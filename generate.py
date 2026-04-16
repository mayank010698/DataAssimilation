import argparse
import logging
import os
import subprocess
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
    LinearGaussian,
    make_lg_benchmark_b,
    generate_dataset_directory_name,
    save_config_yaml,
    save_kolmogorov_config_yaml,
    generate_dataset_splits,
    save_generated_data,
    observations_from_trajectories,
    generate_kolmogorov_data,
    generate_kolmogorov_data_subset,
    merge_kolmogorov_data_parts,
    KolmogorovConfig,
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
        choices=["lorenz63", "lorenz96", "ks", "kuramoto-sivashinsky", "kolmogorov", "linear_gaussian"],
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

    # System parameters (Linear-Gaussian) — only used when --system linear_gaussian
    parser.add_argument("--lg-state-dim", type=int, default=8,
                        help="State dimension for the LinearGaussian system (default: 8 = Benchmark B)")

    # System parameters (Kolmogorov flow) — only used when --system kolmogorov
    parser.add_argument("--kol-grid-size", type=int, default=150, help="Spatial grid resolution for Kolmogorov (size x size)")
    parser.add_argument("--kol-num-steps", type=int, default=200, help="Timesteps to save after warmup for Kolmogorov")
    parser.add_argument("--kol-warmup-steps", type=int, default=100, help="Burn-in steps for Kolmogorov")
    parser.add_argument("--kol-dt", type=float, default=0.04, help="Time step for Kolmogorov")
    parser.add_argument("--re-min", type=float, default=500.0, help="Minimum Reynolds number for Kolmogorov")
    parser.add_argument("--re-max", type=float, default=1500.0, help="Maximum Reynolds number for Kolmogorov")
    parser.add_argument("--obs-grid-size", type=int, default=150,
                        help="Observation grid resolution for Kolmogorov (obs_grid_size x obs_grid_size). "
                             "150 = fully observed; 10 = paper's sparse 100-point grid.")

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

    # Kolmogorov flow: parallel PDE rollout (JAX-based, offline generation)
    parser.add_argument(
        "--kol-mode",
        type=str,
        choices=["full", "worker", "merge"],
        default="full",
        help="Kolmogorov generation mode (internal: use worker/merge with parallel rollout).",
    )
    parser.add_argument(
        "--kol-parallel-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for the Kolmogorov PDE rollout (default: 1).",
    )
    parser.add_argument(
        "--kol-worker-start",
        type=int,
        default=None,
        help="Global trajectory start index for kol-mode=worker.",
    )
    parser.add_argument(
        "--kol-worker-end",
        type=int,
        default=None,
        help="Global trajectory end index (exclusive) for kol-mode=worker.",
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

    # -------------------------------------------------------------------------
    # Kolmogorov flow — separate pipeline (JAX-based, saves .npz)
    # -------------------------------------------------------------------------
    if args.system == "kolmogorov":
        import math

        base_seed = args.seed if args.seed is not None else 42
        g = args.kol_grid_size
        output_base = Path(args.output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        dt_str = f"{args.kol_dt:.4f}".replace(".", "p")
        re_min_str = f"{args.re_min:.1f}".replace(".", "p")
        re_max_str = f"{args.re_max:.1f}".replace(".", "p")
        pde_dir_name = (
            f"kolmogorov_pde_n{args.num_trajectories}"
            f"_len{args.kol_num_steps}"
            f"_warm{args.kol_warmup_steps}"
            f"_dt{dt_str}"
            f"_re{re_min_str}to{re_max_str}"
            f"_grid{g}"
            f"_seed{base_seed}"
        )
        pde_dir = output_base / pde_dir_name
        pde_npz_path = pde_dir / "kolmogorov_data.npz"
        parts_dir = pde_dir / "parts"

        # ---------------------------------------------------------------------
        # Internal modes: worker + merge
        # ---------------------------------------------------------------------
        if args.kol_mode == "worker":
            if args.kol_worker_start is None or args.kol_worker_end is None:
                raise ValueError("kol-mode=worker requires --kol-worker-start and --kol-worker-end")

            start = int(args.kol_worker_start)
            end = int(args.kol_worker_end)
            if start < 0 or end <= start or end > args.num_trajectories:
                raise ValueError(
                    f"Invalid worker range start={start}, end={end} for num-trajectories={args.num_trajectories}"
                )

            parts_dir.mkdir(parents=True, exist_ok=True)
            part_path = parts_dir / f"kolmogorov_data.part_{start:06d}_{end:06d}.npz"

            if part_path.exists() and not args.force:
                print(f"[worker] Part exists, skipping: {part_path}")
                return

            print(f"[worker] Generating Kolmogorov subset idx [{start}:{end}) → {part_path}")
            generate_kolmogorov_data_subset(
                size=args.kol_grid_size,
                num_trajectories=args.num_trajectories,
                num_steps=args.kol_num_steps,
                warmup_steps=args.kol_warmup_steps,
                dt=args.kol_dt,
                re_min=args.re_min,
                re_max=args.re_max,
                seed=base_seed,
                trajectory_indices=np.arange(start, end, dtype=np.int64),
                output_part_path=part_path,
            )
            return

        if args.kol_mode == "merge":
            if pde_npz_path.exists() and not args.force:
                print(f"[merge] Final npz exists, skipping: {pde_npz_path}")
                return

            if not parts_dir.exists():
                raise FileNotFoundError(f"[merge] parts directory not found: {parts_dir}")

            part_paths = sorted(parts_dir.glob("kolmogorov_data.part_*.npz"))
            if len(part_paths) == 0:
                raise FileNotFoundError(f"[merge] No part files found in {parts_dir}")

            print(f"[merge] Merging {len(part_paths)} Kolmogorov parts → {pde_npz_path}")
            merge_kolmogorov_data_parts(
                part_paths=[str(p) for p in part_paths],
                num_trajectories=args.num_trajectories,
                output_npz_path=pde_npz_path,
            )
            return

        # ---------------------------------------------------------------------
        # Full mode: generate PDE once (optionally parallel), then symlink/copy
        # into per-dataset directories for different obs-grid / obs-noise variants.
        # ---------------------------------------------------------------------
        if args.obs_noise_variations:
            obs_noise_levels = parse_float_list(args.obs_noise_variations)
            print(
                f"Generating Kolmogorov datasets for observation noise std levels: {obs_noise_levels}"
            )
        else:
            obs_noise_levels = [args.obs_noise_std]

        if not pde_npz_path.exists():
            pde_dir.mkdir(parents=True, exist_ok=True)

            if args.kol_parallel_gpus <= 1:
                print(f"Generating Kolmogorov PDE rollout (serial) → {pde_npz_path}")
                generate_kolmogorov_data(
                    size=args.kol_grid_size,
                    num_trajectories=args.num_trajectories,
                    num_steps=args.kol_num_steps,
                    warmup_steps=args.kol_warmup_steps,
                    dt=args.kol_dt,
                    re_min=args.re_min,
                    re_max=args.re_max,
                    seed=base_seed,
                    output_path=pde_dir,
                )
            else:
                # Respect an existing CUDA_VISIBLE_DEVICES mask if the user set one.
                # JAX workers will see exactly one GPU each after we set CUDA_VISIBLE_DEVICES.
                visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
                if visible_env:
                    visible_gpu_ids = [int(x) for x in visible_env.split(",") if x.strip() != ""]
                else:
                    visible_gpu_ids = list(range(int(args.kol_parallel_gpus)))

                world_size = min(len(visible_gpu_ids), int(args.kol_parallel_gpus), args.num_trajectories)
                chunk_size = int(math.ceil(args.num_trajectories / world_size))
                parts_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"Generating Kolmogorov PDE rollout (parallel across {world_size} GPUs) → {pde_npz_path}"
                )

                procs = []
                for rank in range(world_size):
                    start = rank * chunk_size
                    end = min((rank + 1) * chunk_size, args.num_trajectories)
                    if start >= end:
                        continue

                    # We assume GPU ids 0..world_size-1 exist; CUDA_VISIBLE_DEVICES
                    # remaps so each worker sees a single GPU as device 0.
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(visible_gpu_ids[rank])

                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--system",
                        "kolmogorov",
                        "--kol-mode",
                        "worker",
                        "--output-dir",
                        str(output_base),
                        "--seed",
                        str(base_seed),
                        "--num-trajectories",
                        str(args.num_trajectories),
                        "--kol-grid-size",
                        str(args.kol_grid_size),
                        "--kol-num-steps",
                        str(args.kol_num_steps),
                        "--kol-warmup-steps",
                        str(args.kol_warmup_steps),
                        "--kol-dt",
                        str(args.kol_dt),
                        "--re-min",
                        str(args.re_min),
                        "--re-max",
                        str(args.re_max),
                        "--kol-worker-start",
                        str(start),
                        "--kol-worker-end",
                        str(end),
                    ]
                    if args.force:
                        cmd.append("--force")

                    print(f"  [orchestrator] launching worker rank={rank} idx[{start}:{end})")
                    procs.append(subprocess.Popen(cmd, env=env))

                for p in procs:
                    ret = p.wait()
                    if ret != 0:
                        raise RuntimeError(f"Kolmogorov worker process failed with exit code {ret}")

                # Merge after workers finish.
                merge_cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--system",
                    "kolmogorov",
                    "--kol-mode",
                    "merge",
                    "--output-dir",
                    str(output_base),
                    "--seed",
                    str(base_seed),
                    "--num-trajectories",
                    str(args.num_trajectories),
                    "--kol-grid-size",
                    str(args.kol_grid_size),
                    "--kol-num-steps",
                    str(args.kol_num_steps),
                    "--kol-warmup-steps",
                    str(args.kol_warmup_steps),
                    "--kol-dt",
                    str(args.kol_dt),
                    "--re-min",
                    str(args.re_min),
                    "--re-max",
                    str(args.re_max),
                ]
                if args.force:
                    merge_cmd.append("--force")
                subprocess.run(merge_cmd, check=True)

        # Link/copy the shared PDE rollout into each per-observation dataset directory.
        for obs_noise_idx, obs_noise in enumerate(obs_noise_levels):
            # Keep backward-compatible directory naming for the common case:
            # single variant with obs_noise_std == 0.0.
            single_variant_default_name = (
                args.dataset_name is not None
                and len(obs_noise_levels) == 1
                and abs(obs_noise - obs_noise_levels[0]) < 1e-12
            )

            if args.dataset_name:
                if single_variant_default_name:
                    dataset_dir_name = args.dataset_name
                else:
                    dataset_dir_name = f"{args.dataset_name}_obs{obs_noise:.3f}".replace(".", "p")
            else:
                dataset_dir_name = (
                    f"kolmogorov_n{args.num_trajectories}"
                    f"_len{args.kol_num_steps}"
                    f"_dt{args.kol_dt:.4f}".replace(".", "p")
                    + f"_grid{g}"
                    f"_obs{args.obs_grid_size}x{args.obs_grid_size}"
                )
                if (len(obs_noise_levels) > 1) or (obs_noise != 0.0):
                    dataset_dir_name = (
                        f"{dataset_dir_name}_obsnoise{obs_noise:.3f}".replace(".", "p")
                    )

            dataset_dir = output_base / dataset_dir_name
            npz_path = dataset_dir / "kolmogorov_data.npz"

            kol_config = KolmogorovConfig(
                grid_size=args.kol_grid_size,
                num_trajectories=args.num_trajectories,
                num_steps=args.kol_num_steps,
                warmup_steps=args.kol_warmup_steps,
                dt=args.kol_dt,
                re_min=args.re_min,
                re_max=args.re_max,
                seed=base_seed,
                obs_frequency=args.obs_frequency,
                obs_grid_size=args.obs_grid_size,
                obs_noise_std=obs_noise,
                obs_noise_seed=base_seed + 1000 * obs_noise_idx,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )

            if dataset_dir.exists() and npz_path.exists():
                if args.force:
                    response = "y"
                else:
                    response = input(
                        f"\nDataset directory {dataset_dir} already exists. Overwrite? (y/N): "
                    )
                if response.lower() != "y":
                    print("Using existing Kolmogorov trajectory data.")
                else:
                    import shutil

                    shutil.rmtree(dataset_dir)

            dataset_dir.mkdir(parents=True, exist_ok=True)
            if not npz_path.exists():
                try:
                    # Use absolute target so relative --output-dir still works.
                    npz_path.symlink_to(pde_npz_path.resolve())
                except Exception:
                    import shutil

                    shutil.copy2(pde_npz_path, npz_path)

            config_path = dataset_dir / "config.yaml"
            save_kolmogorov_config_yaml(kol_config, config_path)
            print(f"Kolmogorov dataset ready (obs_noise_std={obs_noise}): {npz_path}")

        print("\n" + "=" * 80)
        print("Kolmogorov dataset generation completed!")
        print(f"  Output dir: {output_base}")
        print("=" * 80)
        return

    # -------------------------------------------------------------------------
    # ODE systems (Lorenz-63 / 96, Kuramoto-Sivashinsky)
    # -------------------------------------------------------------------------
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
    elif args.system == "linear_gaussian":
        # Build Benchmark-B config (d=args.lg_state_dim) and override num_trajectories /
        # len_trajectory from CLI if provided.
        _lg_config = make_lg_benchmark_b(
            d=args.lg_state_dim,
            num_trajectories=args.num_trajectories,
            len_trajectory=args.len_trajectory,
            warmup_steps=args.warmup_steps,
        )
        # Carry over split ratios from CLI
        _lg_config.train_ratio = args.train_ratio
        _lg_config.val_ratio   = args.val_ratio
        _lg_config.test_ratio  = args.test_ratio

        system_class  = LinearGaussian
        system_name   = "linear_gaussian"
        state_dim     = args.lg_state_dim
        system_params = _lg_config.system_params

        # Build the system and write the dataset using the same path as ODE systems.
        # We skip the process-noise / obs-noise variation loops (not applicable for LG)
        # and delegate directly to generate_dataset_splits + save_generated_data.
        _lg_system = LinearGaussian(_lg_config)
        _lg_dataset_name = (
            args.dataset_name
            if args.dataset_name
            else generate_dataset_directory_name(_lg_config, system_name="linear_gaussian")
        )
        output_base = Path(args.output_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        dataset_dir = output_base / _lg_dataset_name

        if dataset_dir.exists() and (dataset_dir / "data.h5").exists():
            if args.force:
                response = "y"
            else:
                response = input(
                    f"\nDataset directory {dataset_dir} already exists. Overwrite? (y/N): "
                )
            if response.lower() != "y":
                print("Skipping LinearGaussian dataset.")
                import sys; sys.exit(0)
            import shutil
            shutil.rmtree(dataset_dir)

        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating LinearGaussian (d={args.lg_state_dim}) dataset → {dataset_dir}")
        splits_unscaled, obs_mask = generate_dataset_splits(_lg_system, _lg_config)
        save_generated_data(dataset_dir, splits_unscaled, obs_mask)
        config_path = dataset_dir / "config.yaml"
        save_config_yaml(_lg_config, config_path)
        print(f"  Config saved to {config_path}")
        print("\n" + "=" * 80)
        print("LinearGaussian dataset generation completed!")
        print(f"  Output dir: {dataset_dir}")
        print("=" * 80)
        import sys; sys.exit(0)
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
