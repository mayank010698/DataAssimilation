import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    Lorenz63,
    create_projection_matrix,
)
from models.base_pf import FilteringMethod
from models.bpf import BootstrapParticleFilter
from models.proposals import RectifiedFlowProposal, TransitionProposal


def find_rf_checkpoint(config_name: str) -> Optional[str]:
    """
    Find the best RF checkpoint for a given configuration.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Path to best checkpoint if found, None otherwise
    """
    # Map configuration names to checkpoint directory patterns
    config_to_dir = {
        "Linear Projection (Preprocessing ON)": "linear_projection",
        "Linear Projection (Preprocessing OFF)": "linear_projection",
        "Arctan Nonlinear (Preprocessing OFF)": "arctan_nonlinear",
        "Arctan Nonlinear (Preprocessing ON)": "arctan_nonlinear",
    }
    
    # Get directory name for this configuration
    dir_name = config_to_dir.get(config_name)
    if not dir_name:
        # Fallback: try to derive from config name
        safe_name = config_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        dir_name = safe_name
    
    # Try to find checkpoints in rf_runs directory
    checkpoint_dir = Path("./rf_runs") / dir_name / "checkpoints"
    
    if not checkpoint_dir.exists():
        # Try alternative locations
        checkpoint_dir = Path("./rf_runs") / dir_name
        if not checkpoint_dir.exists():
            return None
    
    # Find checkpoint files matching the pattern rf-epoch={epoch}-val_loss={val_loss}.ckpt
    checkpoints = list(checkpoint_dir.glob("rf-*.ckpt"))
    if not checkpoints:
        # Try recursive search
        checkpoints = list(checkpoint_dir.glob("**/rf-*.ckpt"))
    
    if not checkpoints:
        return None
    
    # Filter out "last" checkpoint and parse validation loss from filename
    def get_val_loss(path):
        try:
            stem = path.stem
            if "last" in stem.lower():
                return float('inf')
            if "-val_loss=" in stem:
                parts = stem.split("-val_loss=")
                if len(parts) == 2:
                    val_loss = float(parts[1])
                    return val_loss
        except (ValueError, IndexError):
            pass
        return float('inf')
    
    # Filter out checkpoints with invalid loss values and sort by validation loss
    valid_checkpoints = [cp for cp in checkpoints if get_val_loss(cp) != float('inf')]
    
    if not valid_checkpoints:
        return None
    
    # Sort by validation loss (lowest is best)
    valid_checkpoints.sort(key=get_val_loss)
    return str(valid_checkpoints[0])


def get_all_trajectory_indices(dataloader) -> List[int]:
    """Get all unique trajectory indices from the dataloader"""
    trajectory_indices = set()
    for batch in dataloader:
        trajectory_indices.add(batch["trajectory_idx"].item())
    return sorted(list(trajectory_indices))


def run_particle_filter_on_trajectory(
    pf: FilteringMethod,
    dataloader,
    trajectory_idx: int,
    collect_trajectory_data: bool = False,
) -> Dict[str, Any]:
    """Run filtering method on a single trajectory and collect metrics"""
    trajectory_metrics = []
    trajectory_data = []

    for batch in dataloader:
        batch_traj_idx = batch["trajectory_idx"].item()
        if batch_traj_idx != trajectory_idx:
            continue

        metrics = pf.test_step(batch, 0)
        trajectory_metrics.append(metrics)

        if collect_trajectory_data:
            x_curr = batch["x_curr"].squeeze(0)
            y_curr = batch["y_curr"].squeeze(0) if batch["has_observation"].item() else None
            time_idx = batch["time_idx"].item()
            
            trajectory_data.append(
                {
                    "time_idx": time_idx,
                    "x_true": x_curr,
                    "x_est": metrics["x_est"],
                    "observation": y_curr if y_curr is not None else None,
                    "has_observation": batch["has_observation"].item(),
                    "rmse": metrics["rmse"],
                }
            )

    rmse_values = [m["rmse"] for m in trajectory_metrics]
    ll_values = [
        m["log_likelihood"] for m in trajectory_metrics if m["log_likelihood"] != 0.0
    ]
    
    # Collect additional metrics if available
    ess_values = [m.get("ess", 0.0) for m in trajectory_metrics if "ess" in m]
    resample_counts = [m.get("resampled", False) for m in trajectory_metrics]
    step_times = [m.get("step_time", 0.0) for m in trajectory_metrics if "step_time" in m]

    if collect_trajectory_data:
        trajectory_data.sort(key=lambda x: x["time_idx"])

    result = {
        "trajectory_idx": trajectory_idx,
        "mean_rmse": np.mean(rmse_values) if rmse_values else 0.0,
        "std_rmse": np.std(rmse_values) if rmse_values else 0.0,
        "min_rmse": np.min(rmse_values) if rmse_values else 0.0,
        "max_rmse": np.max(rmse_values) if rmse_values else 0.0,
        "mean_log_likelihood": np.mean(ll_values) if ll_values else 0.0,
        "total_log_likelihood": np.sum(ll_values) if ll_values else 0.0,
        "n_observations": len(ll_values),
        "n_steps": len(rmse_values),
        "mean_ess": np.mean(ess_values) if ess_values else 0.0,
        "min_ess": np.min(ess_values) if ess_values else 0.0,
        "n_resamples": sum(resample_counts),
        "mean_step_time": np.mean(step_times) if step_times else 0.0,
        "total_time": np.sum(step_times) if step_times else 0.0,
        "metrics": trajectory_metrics,
        "trajectory_data": trajectory_data if collect_trajectory_data else None,
    }

    return result


def aggregate_metrics_across_trajectories(trajectory_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all trajectories"""
    if not trajectory_results:
        return {}
    
    # Aggregate RMSE metrics
    all_mean_rmse = [r["mean_rmse"] for r in trajectory_results]
    all_std_rmse = [r["std_rmse"] for r in trajectory_results]
    all_min_rmse = [r["min_rmse"] for r in trajectory_results]
    all_max_rmse = [r["max_rmse"] for r in trajectory_results]
    
    # Aggregate log-likelihood metrics
    all_total_ll = [r["total_log_likelihood"] for r in trajectory_results]
    all_mean_ll = [r["mean_log_likelihood"] for r in trajectory_results]
    
    # Aggregate ESS metrics
    all_mean_ess = [r["mean_ess"] for r in trajectory_results if r["mean_ess"] > 0]
    all_min_ess = [r["min_ess"] for r in trajectory_results if r["min_ess"] > 0]
    
    # Aggregate timing metrics
    all_total_times = [r["total_time"] for r in trajectory_results if r["total_time"] > 0]
    all_mean_step_times = [r["mean_step_time"] for r in trajectory_results if r["mean_step_time"] > 0]
    
    # Aggregate resampling counts
    all_n_resamples = [r["n_resamples"] for r in trajectory_results]
    
    # Collect all per-step RMSE values for distribution analysis
    all_rmse_values = []
    for r in trajectory_results:
        for m in r["metrics"]:
            all_rmse_values.append(m["rmse"])
    
    aggregated = {
        # RMSE statistics
        "mean_rmse_across_trajectories": np.mean(all_mean_rmse),
        "std_rmse_across_trajectories": np.std(all_mean_rmse),
        "min_mean_rmse": np.min(all_mean_rmse),
        "max_mean_rmse": np.max(all_mean_rmse),
        "mean_std_rmse": np.mean(all_std_rmse),
        "mean_min_rmse": np.mean(all_min_rmse),
        "mean_max_rmse": np.mean(all_max_rmse),
        
        # Overall RMSE distribution
        "overall_mean_rmse": np.mean(all_rmse_values) if all_rmse_values else 0.0,
        "overall_std_rmse": np.std(all_rmse_values) if all_rmse_values else 0.0,
        "overall_min_rmse": np.min(all_rmse_values) if all_rmse_values else 0.0,
        "overall_max_rmse": np.max(all_rmse_values) if all_rmse_values else 0.0,
        "overall_median_rmse": np.median(all_rmse_values) if all_rmse_values else 0.0,
        
        # Log-likelihood statistics
        "mean_total_log_likelihood": np.mean(all_total_ll),
        "std_total_log_likelihood": np.std(all_total_ll),
        "mean_mean_log_likelihood": np.mean(all_mean_ll),
        
        # ESS statistics
        "mean_ess": np.mean(all_mean_ess) if all_mean_ess else 0.0,
        "std_ess": np.std(all_mean_ess) if all_mean_ess else 0.0,
        "min_ess": np.min(all_min_ess) if all_min_ess else 0.0,
        
        # Timing statistics
        "mean_total_time": np.mean(all_total_times) if all_total_times else 0.0,
        "std_total_time": np.std(all_total_times) if all_total_times else 0.0,
        "mean_step_time": np.mean(all_mean_step_times) if all_mean_step_times else 0.0,
        
        # Resampling statistics
        "mean_n_resamples": np.mean(all_n_resamples),
        "std_n_resamples": np.std(all_n_resamples),
        "total_n_resamples": np.sum(all_n_resamples),
        
        # Counts
        "n_trajectories": len(trajectory_results),
        "total_observations": sum(r["n_observations"] for r in trajectory_results),
        "total_steps": sum(r["n_steps"] for r in trajectory_results),
    }
    
    return aggregated


def log_config_to_wandb(
    config: DataAssimilationConfig,
    args,
    pf: BootstrapParticleFilter,
    rf_checkpoint: Optional[str],
):
    """Log configuration settings to wandb"""
    wandb_config = {
        # Data configuration
        "num_trajectories": config.num_trajectories,
        "len_trajectory": config.len_trajectory,
        "warmup_steps": config.warmup_steps,
        "dt": config.dt,
        "use_preprocessing": config.use_preprocessing,
        
        # Observation configuration
        "obs_noise_std": config.obs_noise_std,
        "obs_frequency": config.obs_frequency,
        "obs_components": str(config.obs_components),
        "obs_dim": pf.obs_dim,
        
        # System parameters
        "system_sigma": config.system_params.get("sigma", None),
        "system_rho": config.system_params.get("rho", None),
        "system_beta": config.system_params.get("beta", None),
        
        # Particle filter configuration
        "n_particles": pf.n_particles,
        "process_noise_std": pf.process_noise_std,
        "proposal_type": type(pf.proposal).__name__,
        "state_dim": pf.state_dim,
        "obs_dim": pf.obs_dim,
        
        # Proposal configuration
        "proposal_name": args.proposal_type,
        "device": args.device,
    }

    if isinstance(pf.proposal, RectifiedFlowProposal):
        wandb_config["rf_checkpoint"] = rf_checkpoint if rf_checkpoint else "not_found"

    wandb.config.update(wandb_config)


def parse_int_list(value: str) -> List[int]:
    if value is None or value.strip() == "":
        return []
    return [int(v) for v in value.split(",") if v.strip() != ""]


def parse_float_list(value: str) -> List[float]:
    if value is None or value.strip() == "":
        return []
    return [float(v) for v in value.split(",") if v.strip() != ""]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Bootstrap Particle Filter on Lorenz63 test trajectories."
    )

    # Data configuration
    parser.add_argument("--num-trajectories", type=int, default=1024)
    parser.add_argument("--len-trajectory", type=int, default=15)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--dt", type=float, default=0.0002)
    parser.add_argument("--use-preprocessing", action="store_true", default=False)

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
    parser.add_argument("--obs-dim", type=int, default=None)

    # System parameters
    parser.add_argument("--system-sigma", type=float, default=10.0)
    parser.add_argument("--system-rho", type=float, default=28.0)
    parser.add_argument("--system-beta", type=float, default=8.0 / 3.0)

    # Dataset/DataModule configuration
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)

    # Trajectory selection
    parser.add_argument(
        "--trajectory-ids",
        type=str,
        default=None,
        help="Comma-separated list of trajectory indices to evaluate. Defaults to all.",
    )

    # Particle filter configuration
    parser.add_argument("--n-particles", type=int, default=100)
    parser.add_argument("--process-noise-std", type=float, default=0.25)
    parser.add_argument(
        "--proposal-type", type=str, choices=["transition", "rf"], default="transition"
    )
    parser.add_argument("--rf-checkpoint", type=str, default=None)
    parser.add_argument(
        "--rf-config-name",
        type=str,
        default=None,
        help="Optional config name used for automatic RF checkpoint lookup.",
    )
    parser.add_argument("--device", type=str, default="cpu")

    # Logging configuration
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--experiment-label", type=str, default="custom_eval")
    parser.add_argument("--wandb-project", type=str, default="data-assimilation-bpf")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--disable-wandb", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("Bootstrap Particle Filter Evaluation - All Test Trajectories")
    print("=" * 80)

    obs_components = parse_int_list(args.obs_components)
    if not obs_components:
        raise ValueError("At least one observation component must be specified.")

    if args.observation_operator == "linear_projection":
        observation_operator = create_projection_matrix(args.state_dim, obs_components)
        inferred_obs_dim = len(obs_components)
    else:
        observation_operator = np.arctan
        inferred_obs_dim = 1

    obs_dim = args.obs_dim if args.obs_dim is not None else inferred_obs_dim

    system_params = {
        "sigma": args.system_sigma,
        "rho": args.system_rho,
        "beta": args.system_beta,
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
        use_preprocessing=args.use_preprocessing,
    )

    system = Lorenz63(config)

    default_data_dir = (
        f"./lorenz63_data_{args.experiment_label.lower().replace(' ', '_')}_"
        f"{args.len_trajectory}step"
    )
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir

    data_module = DataAssimilationDataModule(
        config=config,
        system_class=Lorenz63,
        data_dir=data_dir,
        batch_size=args.batch_size,
    )

    print("\nGenerating/loading data...")
    data_module.prepare_data()
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Determine which trajectories to evaluate
    if args.trajectory_ids:
        trajectory_indices = parse_int_list(args.trajectory_ids)
    else:
        trajectory_indices = get_all_trajectory_indices(test_loader)

    if not trajectory_indices:
        raise ValueError("No trajectories selected for evaluation.")

    print(f"Evaluating {len(trajectory_indices)} trajectories: {trajectory_indices}")

    print("\nInitializing Bootstrap Particle Filter...")

    if args.proposal_type == "rf":
        rf_checkpoint = args.rf_checkpoint
        if rf_checkpoint is None and args.rf_config_name:
            rf_checkpoint = find_rf_checkpoint(args.rf_config_name)

        if not rf_checkpoint or not os.path.exists(rf_checkpoint):
            raise FileNotFoundError(
                "Rectified Flow proposal selected but checkpoint not found. "
                "Provide --rf-checkpoint or a valid --rf-config-name."
            )
        proposal = RectifiedFlowProposal(rf_checkpoint, device=args.device)
    else:
        rf_checkpoint = None
        proposal = TransitionProposal(system, process_noise_std=args.process_noise_std)

    pf = BootstrapParticleFilter(
        system=system,
        proposal_distribution=proposal,
        n_particles=args.n_particles,
        state_dim=system.state_dim,
        obs_dim=obs_dim,
        process_noise_std=args.process_noise_std,
        device=args.device,
    )

    # Setup Weights & Biases
    wandb_run = None
    if not args.disable_wandb:
        safe_name = (
            args.run_name
            or f"{args.experiment_label}_{args.observation_operator}_{args.proposal_type}"
        )
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        tags.extend(
            [
                args.proposal_type,
                "preprocessing_on" if config.use_preprocessing else "preprocessing_off",
            ]
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=safe_name,
            tags=list(dict.fromkeys(tags)),
            reinit=True,
        )
        log_config_to_wandb(config, args, pf, rf_checkpoint)

    print("\nRunning particle filter on selected trajectories...")
    trajectory_results = []

    for traj_idx in trajectory_indices:
        print(f"  Processing trajectory {traj_idx}...")
        result = run_particle_filter_on_trajectory(
            pf, test_loader, traj_idx, collect_trajectory_data=False
        )
        trajectory_results.append(result)

        if wandb_run:
            wandb.log(
                {
                    f"trajectory_{traj_idx}/mean_rmse": result["mean_rmse"],
                    f"trajectory_{traj_idx}/total_log_likelihood": result[
                        "total_log_likelihood"
                    ],
                    f"trajectory_{traj_idx}/mean_ess": result["mean_ess"],
                    f"trajectory_{traj_idx}/n_resamples": result["n_resamples"],
                }
            )

    aggregated = aggregate_metrics_across_trajectories(trajectory_results)

    print("\nAggregated Results:")
    print(
        f"  Mean RMSE across trajectories: {aggregated['mean_rmse_across_trajectories']:.4f} "
        f"± {aggregated['std_rmse_across_trajectories']:.4f}"
    )
    print(
        f"  Overall mean RMSE: {aggregated['overall_mean_rmse']:.4f} "
        f"± {aggregated['overall_std_rmse']:.4f}"
    )
    print(
        f"  Mean total log-likelihood: {aggregated['mean_total_log_likelihood']:.2f} "
        f"± {aggregated['std_total_log_likelihood']:.2f}"
    )
    print(f"  Mean ESS: {aggregated['mean_ess']:.2f}")
    print(
        f"  Mean resamples per trajectory: {aggregated['mean_n_resamples']:.1f} "
        f"± {aggregated['std_n_resamples']:.1f}"
    )
    print(f"  Total trajectories: {aggregated['n_trajectories']}")
    print(f"  Total observations: {aggregated['total_observations']}")
    print(f"  Total steps: {aggregated['total_steps']}")

    if wandb_run:
        wandb.log(
            {
                "aggregated/mean_rmse_across_trajectories": aggregated[
                    "mean_rmse_across_trajectories"
                ],
                "aggregated/std_rmse_across_trajectories": aggregated[
                    "std_rmse_across_trajectories"
                ],
                "aggregated/overall_mean_rmse": aggregated["overall_mean_rmse"],
                "aggregated/overall_std_rmse": aggregated["overall_std_rmse"],
                "aggregated/overall_median_rmse": aggregated["overall_median_rmse"],
                "aggregated/overall_min_rmse": aggregated["overall_min_rmse"],
                "aggregated/overall_max_rmse": aggregated["overall_max_rmse"],
                "aggregated/mean_total_log_likelihood": aggregated[
                    "mean_total_log_likelihood"
                ],
                "aggregated/std_total_log_likelihood": aggregated[
                    "std_total_log_likelihood"
                ],
                "aggregated/mean_ess": aggregated["mean_ess"],
                "aggregated/std_ess": aggregated["std_ess"],
                "aggregated/min_ess": aggregated["min_ess"],
                "aggregated/mean_n_resamples": aggregated["mean_n_resamples"],
                "aggregated/total_n_resamples": aggregated["total_n_resamples"],
                "aggregated/mean_total_time": aggregated["mean_total_time"],
                "aggregated/mean_step_time": aggregated["mean_step_time"],
                "aggregated/n_trajectories": aggregated["n_trajectories"],
                "aggregated/total_observations": aggregated["total_observations"],
                "aggregated/total_steps": aggregated["total_steps"],
            }
        )

        per_traj_rmse = [r["mean_rmse"] for r in trajectory_results]
        wandb.log(
            {
                "histograms/per_trajectory_mean_rmse": wandb.Histogram(per_traj_rmse),
            }
        )

        all_rmse_values = []
        for r in trajectory_results:
            for m in r["metrics"]:
                all_rmse_values.append(m["rmse"])
        if all_rmse_values:
            wandb.log({"histograms/all_rmse_values": wandb.Histogram(all_rmse_values)})

        all_ess_values = []
        for r in trajectory_results:
            for m in r["metrics"]:
                if "ess" in m:
                    all_ess_values.append(m["ess"])
        if all_ess_values:
            wandb.log({"histograms/all_ess_values": wandb.Histogram(all_ess_values)})

        wandb.finish()

    print("\n" + "=" * 80)
    print("Evaluation completed")
    print("=" * 80)


if __name__ == "__main__":
    main()

