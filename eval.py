import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    Lorenz63,
    Lorenz96,
    TimeAlignedBatchSampler,
    create_projection_matrix,
    load_config_yaml,
)
from models.base_pf import FilteringMethod
from models.bpf import BootstrapParticleFilter, BootstrapParticleFilterUnbatched
from models.proposals import RectifiedFlowProposal, TransitionProposal


def compute_crps_ensemble(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """
    Compute CRPS for an ensemble.
    
    Args:
        ensemble: Shape (K, D) or (K,)
        truth: Shape (D,) or scalar
        
    Returns:
        Average CRPS over dimensions (or scalar CRPS)
    """
    # Ensure 2D (K, D)
    if ensemble.ndim == 1:
        ensemble = ensemble[:, None]
    if truth.ndim == 0:
        truth = truth[None]
        
    # Dimensions: K=ensemble_size, D=state_dim
    K, D = ensemble.shape
    
    crps_sum = 0.0
    
    for d in range(D):
        ens_d = np.sort(ensemble[:, d])
        truth_d = truth[d]
        
        # Term 1: Mean absolute error vs truth
        # E|X - y|
        mae = np.mean(np.abs(ens_d - truth_d))
        
        # Term 2: Dispersion (mean absolute difference between members)
        # 0.5 * E|X - X'|
        # Efficient calculation using sorted array:
        # sum_{i,j} |x_i - x_j| = 2 * sum_{i<j} (x_j - x_i)
        # = 2 * sum_{i=0}^{K-1} x_i * (2i - K + 1)
        
        # We want (1 / (2 * K^2)) * sum_{i,j} |x_i - x_j|
        # = (1 / K^2) * sum_{i<j} (x_j - x_i)
        
        # However, for small K, direct computation is fast enough and safer
        diffs = np.abs(ens_d[:, None] - ens_d[None, :])
        dispersion = np.sum(diffs) / (2 * K * K)
        
        crps_sum += (mae - dispersion)
        
    return crps_sum / D


def test_rf_log_probs(rf_proposal, system):
    """Test if RF is returning meaningful log probabilities"""
    print("\nTesting RF Proposal Log Probs...")
    # Create dummy data
    state_dim = system.state_dim
    x_prev = torch.randn(1, state_dim, device=rf_proposal.device)
    
    # Check if model expects observations
    obs_dim = getattr(rf_proposal.rf_model, "obs_dim", 0)
    y_curr = None
    if obs_dim > 0:
        y_curr = torch.randn(1, obs_dim, device=rf_proposal.device)
        print(f"Generated dummy observation (dim={obs_dim}) for testing.")
    
    # Sample multiple times
    samples = [rf_proposal.sample(x_prev, y_curr, 0.01) for _ in range(100)]
    samples_tensor = torch.stack(samples).squeeze(1) # (100, D)
    
    # Compute log probs
    x_prev_expanded = x_prev.repeat(100, 1)
    y_curr_expanded = y_curr.repeat(100, 1) if y_curr is not None else None
    
    log_probs = rf_proposal.log_prob(samples_tensor, x_prev_expanded, y_curr_expanded, 0.01)
    
    print(f"Log prob stats (across DIFFERENT samples):")
    print(f"  Mean: {log_probs.mean():.4f}")
    print(f"  Std: {log_probs.std():.4f}")
    print(f"  Min: {log_probs.min():.4f}")
    print(f"  Max: {log_probs.max():.4f}")
    
    # --- NEW TEST: Stochastic Variance Check ---
    print("\nTesting Stochastic Variance (SAME sample, multiple calls)...")
    # Pick the first sample and repeat it 20 times
    single_sample = samples_tensor[0:1].repeat(20, 1)
    single_prev = x_prev.repeat(20, 1)
    single_obs = y_curr.repeat(20, 1) if y_curr is not None else None
    
    # Compute log prob multiple times for the SAME input
    # Since Hutchinson trace estimator is stochastic, these might differ
    log_probs_same = rf_proposal.log_prob(single_sample, single_prev, single_obs, 0.01)
    
    print(f"Log prob stats (across SAME sample):")
    print(f"  Mean: {log_probs_same.mean():.4f}")
    print(f"  Std: {log_probs_same.std():.4f}")
    print(f"  Min: {log_probs_same.min():.4f}")
    print(f"  Max: {log_probs_same.max():.4f}")
    
    if log_probs_same.std() > 0.1:
        print("WARNING: High variance in log_prob for identical inputs! Hutchinson estimator is noisy.")
    else:
        print("Log prob is deterministic/stable for identical inputs.")
    # -------------------------------------------

    if log_probs.std() < 1e-6:
        print("WARNING: All log probs are essentially identical!")
    else:
        print("Log probs vary across samples (Good).")


def plot_trajectory_comparison(
    result: Dict[str, Any],
    config: DataAssimilationConfig,
    system: Any,
) -> plt.Figure:
    """Plot trajectory comparison (adapted from run.py for logging)"""
    trajectory_data = result["trajectory_data"]
    traj_idx = result["trajectory_idx"]

    if trajectory_data is None:
        return None

    time_steps = np.array([d["time_idx"] for d in trajectory_data])
    x_true = np.array([d["x_true"] for d in trajectory_data])
    x_est = np.array([d["x_est"] for d in trajectory_data])
    rmse_values = np.array([d["rmse"] for d in trajectory_data])
    
    # Check if CRPS is available
    has_crps = "crps" in trajectory_data[0]
    if has_crps:
        crps_values = np.array([d["crps"] for d in trajectory_data])

    # Create mapping from time_idx to array position
    time_to_idx = {t: idx for idx, t in enumerate(time_steps)}

    # Data is already in physical space
    x_true_orig = x_true
    x_est_orig = x_est
    space_label = ""

    obs_times = []
    obs_values = []
    obs_true_x = []
    obs_indices = []

    for d in trajectory_data:
        if d["has_observation"] and d["observation"] is not None:
            time_idx = d["time_idx"]
            array_idx = time_to_idx[time_idx]
            obs_times.append(time_idx)
            obs_values.append(d["observation"])
            obs_true_x.append(x_true_orig[array_idx][0])
            obs_indices.append(array_idx)

    obs_times = np.array(obs_times)
    obs_indices = np.array(obs_indices) if len(obs_indices) > 0 else np.array([])
    if len(obs_values) > 0:
        obs_values = np.array(obs_values)

    fig = plt.figure(figsize=(16, 12))

    # Determine if high dimensional (e.g. > 3)
    is_high_dim = x_true.shape[1] > 3

    if is_high_dim:
        # High-dim plotting (Heatmaps + Projections)
        
        # 1. Heatmap True
        ax1 = fig.add_subplot(3, 2, 1)
        im1 = ax1.imshow(x_true_orig.T, aspect='auto', interpolation='nearest', cmap='viridis', origin='lower')
        ax1.set_title(f"True State Heatmap (ID: {traj_idx})")
        ax1.set_ylabel("State Component")
        plt.colorbar(im1, ax=ax1)

        # 2. Heatmap Est
        ax2 = fig.add_subplot(3, 2, 2)
        im2 = ax2.imshow(x_est_orig.T, aspect='auto', interpolation='nearest', cmap='viridis', origin='lower')
        ax2.set_title(f"Estimated State Heatmap (PF)")
        ax2.set_ylabel("State Component")
        plt.colorbar(im2, ax=ax2)

        # 3. 3D Projection (0, 1, 2)
        ax3 = fig.add_subplot(3, 2, 3, projection="3d")
        ax3.plot(x_true_orig[:, 0], x_true_orig[:, 1], x_true_orig[:, 2], "b-", alpha=0.8, label="True")
        ax3.plot(x_est_orig[:, 0], x_est_orig[:, 1], x_est_orig[:, 2], "r--", alpha=0.8, label="PF")
        ax3.set_title("3D Projection (Dim 0,1,2)")
        ax3.legend()

        # 4. Time Series of Component 0
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(time_steps, x_true_orig[:, 0], "b-", label="True (Dim 0)")
        ax4.plot(time_steps, x_est_orig[:, 0], "r--", label="Est (Dim 0)")
        
        # Add observations for component 0 if present
        if 0 in config.obs_components and len(obs_times) > 0:
             valid_mask = obs_indices < len(x_true_orig)
             # Filter for obs that are actually measuring dim 0
             # obs_values is (N_obs, Obs_dim). We need to know which col of obs corresponds to state dim 0.
             # obs_components list maps obs_dim_idx -> state_dim_idx
             try:
                 obs_col_idx = config.obs_components.index(0)
                 if np.any(valid_mask):
                     # Extract specific column
                     ax4.scatter(
                        obs_times[valid_mask], 
                        obs_values[valid_mask, obs_col_idx] if obs_values.ndim > 1 else obs_values[valid_mask],
                        color="red", marker="x", s=40, label="Obs"
                     )
             except ValueError:
                 pass # Dim 0 not observed

        ax4.set_title("Component 0 Time Series")
        ax4.legend()

    else:
        # 3D Plot
        ax1 = fig.add_subplot(3, 2, 1, projection="3d")
        ax1.plot(x_true_orig[:, 0], x_true_orig[:, 1], x_true_orig[:, 2], "b-", alpha=0.8, label="True")
        ax1.plot(x_est_orig[:, 0], x_est_orig[:, 1], x_est_orig[:, 2], "r--", alpha=0.8, label="PF")
        ax1.set_title(f"3D Trajectory{space_label} (ID: {traj_idx})")
        ax1.legend()

        # State Components
        state_names = ["X", "Y", "Z"]
        colors = ["blue", "green", "orange"]
        for i, (name, color) in enumerate(zip(state_names, colors)):
            if i >= x_true.shape[1]: break
            ax = fig.add_subplot(3, 2, i + 2)
            ax.plot(time_steps, x_true_orig[:, i], color=color, label=f"{name} (true)")
            ax.plot(time_steps, x_est_orig[:, i], color="red", linestyle="--", label=f"{name} (est)")
            
            # Plot observations if available for this component
            if i in config.obs_components and len(obs_times) > 0:
                valid_mask = obs_indices < len(x_true_orig)
                if np.any(valid_mask):
                    try:
                        obs_col_idx = config.obs_components.index(i)
                        val = obs_values[valid_mask, obs_col_idx] if obs_values.ndim > 1 else obs_values[valid_mask]
                        ax.scatter(
                            obs_times[valid_mask], 
                            val, 
                            color="red", marker="x", s=40, label="Obs"
                        )
                    except ValueError:
                        pass
            ax.set_ylabel(name)
            ax.legend()

    # RMSE over time
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(time_steps, rmse_values, "purple")
    ax5.set_title("RMSE Over Time")
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("RMSE")
    
    # CRPS over time
    if has_crps:
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.plot(time_steps, crps_values, "green")
        ax6.set_title("CRPS Over Time")
        ax6.set_xlabel("Time Step")
        ax6.set_ylabel("CRPS")

    plt.tight_layout()
    return fig


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
        "mean_ess_pre": np.mean([m.get("ess_pre_resample", 0.0) for m in trajectory_metrics]),
        "min_ess_pre": np.min([m.get("ess_pre_resample", 0.0) for m in trajectory_metrics]),
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

    # Aggregate CRPS metrics
    all_mean_crps = [np.mean([m["crps"] for m in r["metrics"]]) for r in trajectory_results]
    
    # Aggregate log-likelihood metrics
    all_total_ll = [r["total_log_likelihood"] for r in trajectory_results]
    all_mean_ll = [r["mean_log_likelihood"] for r in trajectory_results]
    
    # Aggregate ESS metrics
    all_mean_ess = [r["mean_ess"] for r in trajectory_results if r["mean_ess"] > 0]
    all_min_ess = [r["min_ess"] for r in trajectory_results if r["min_ess"] > 0]
    
    # Aggregate ESS Pre-Resample metrics (if available)
    all_mean_ess_pre = []
    all_min_ess_pre = []
    if trajectory_results and "mean_ess_pre" in trajectory_results[0]:
        all_mean_ess_pre = [r["mean_ess_pre"] for r in trajectory_results]
        all_min_ess_pre = [r["min_ess_pre"] for r in trajectory_results]
    
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

        # CRPS statistics
        "mean_crps_across_trajectories": np.mean(all_mean_crps),
        "std_crps_across_trajectories": np.std(all_mean_crps),
        
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
        
        "mean_ess_pre": np.mean(all_mean_ess_pre) if all_mean_ess_pre else 0.0,
        "min_ess_pre": np.min(all_min_ess_pre) if all_min_ess_pre else 0.0,
        
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
    pf,
    rf_checkpoint: Optional[str],
):
    """Log configuration settings to wandb"""
    obs_op_type = "linear_projection" if isinstance(config.observation_operator, (np.ndarray, torch.Tensor)) else "arctan"
    
    wandb_config = {
        # Data configuration
        "num_trajectories": config.num_trajectories,
        "len_trajectory": config.len_trajectory,
        "warmup_steps": config.warmup_steps,
        "dt": config.dt,
        
        # Observation configuration
        "obs_noise_std": config.obs_noise_std,
        "obs_frequency": config.obs_frequency,
        "obs_components": str(config.obs_components),
        "observation_operator": obs_op_type,
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
        "batch_size": args.batch_size,
    }

    if isinstance(pf.proposal, RectifiedFlowProposal):
        wandb_config["rf_checkpoint"] = rf_checkpoint if rf_checkpoint else "not_found"

    wandb.config.update(wandb_config)


def parse_float_list(value: str) -> List[float]:
    if value is None or value.strip() == "":
        return []
    return [float(v) for v in value.split(",") if v.strip() != ""]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Bootstrap Particle Filter on Lorenz63 test trajectories."
    )

    # Dataset/DataModule configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory (must contain config.yaml and data.h5 or data_scaled.h5)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation. >1 uses batched evaluation.")
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--num-eval-trajectories", type=int, default=None, help="Number of trajectories to evaluate on (default: all)")

    # Particle filter configuration
    parser.add_argument("--n-particles", type=int, default=100)
    parser.add_argument("--process-noise-std", type=float, default=0.25)
    parser.add_argument("--obs-noise-std", type=float, default=None, help="Override observation noise std")
    parser.add_argument(
        "--proposal-type", type=str, choices=["transition", "rf"], default="transition"
    )
    parser.add_argument("--rf-checkpoint", type=str, default=None)
    parser.add_argument("--rf-likelihood-steps", type=int, default=None)
    parser.add_argument("--rf-sampling-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    
    # Guidance configuration
    parser.add_argument("--mc-guidance", action="store_true", help="Enable Monte Carlo guidance during inference")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Scale for Monte Carlo guidance")

    # Logging configuration
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--experiment-label", type=str, default="custom_eval")
    parser.add_argument("--wandb-project", type=str, default="data-assimilation-bpf")
    parser.add_argument("--wandb-entity", type=str, default="ml-climate")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--disable-wandb", action="store_true", default=False)

    return parser.parse_args()


def run_batched_eval(args, config, system, data_module, obs_dim, proposal, wandb_run, vis_indices):
    """Execute batched evaluation logic"""
    print(f"\nInitializing Batched Bootstrap Particle Filter (Batch Size: {args.batch_size})...")
    
    pf = BootstrapParticleFilter(
        system=system,
        proposal_distribution=proposal,
        n_particles=args.n_particles,
        state_dim=system.state_dim,
        obs_dim=obs_dim,
        process_noise_std=args.process_noise_std,
        device=args.device,
    )
    
    if wandb_run:
        log_config_to_wandb(config, args, pf, args.rf_checkpoint)

    # Setup batched dataloader with custom sampler
    test_dataset = data_module.test_dataset
    traj_len_inference = config.len_trajectory - 1
    
    batch_sampler = TimeAlignedBatchSampler(
        data_source_len=len(test_dataset),
        num_trajectories=test_dataset.n_trajectories,
        traj_len=traj_len_inference,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
    )
    
    # Storage for results
    trajectory_results_map = {}
    pbar = tqdm(test_loader, desc="Evaluating (Batches)")
    
    total_processed_steps = 0
    
    for batch in pbar:
        x_prev = batch["x_prev"]
        x_curr = batch["x_curr"]
        y_curr = batch["y_curr"] if batch["has_observation"].any() else None 
        
        traj_idxs = batch["trajectory_idx"].squeeze(-1)
        time_idxs = batch["time_idx"].squeeze(-1)
        batch_size_curr = x_prev.shape[0]
        
        # Initialize Filter for new trajectories
        is_start = (time_idxs[0] == 1)
        if is_start:
             pf.initialize_filter(x_prev)
             for tid in traj_idxs.tolist():
                 trajectory_results_map[tid] = {
                     "metrics": [], 
                     "trajectory_data": [] if tid in vis_indices else None
                 }
        
        # Step Particle Filter
        dt = config.dt
        batch_metrics = pf.step(x_prev, x_curr, y_curr, dt, traj_idxs, time_idxs)
        
        # Store Results
        for i, metrics in enumerate(batch_metrics):
            tid = traj_idxs[i].item()
            if tid not in trajectory_results_map:
                 # Should generally be handled by is_start, but safety check could go here
                 continue
            
            # Compute CRPS
            # Resample based on weights to get equally weighted ensemble
            indices = torch.multinomial(pf.weights[i], pf.n_particles, replacement=True)
            ens_i = pf.particles[i, indices].detach().cpu().numpy()
            truth_i = x_curr[i].detach().cpu().numpy()
            metrics["crps"] = compute_crps_ensemble(ens_i, truth_i)

            trajectory_results_map[tid]["metrics"].append(metrics)
            
            if tid in vis_indices:
                y_curr_i = y_curr[i] if y_curr is not None else None
                has_obs = batch["has_observation"][i].item()
                
                trajectory_results_map[tid]["trajectory_data"].append({
                    "time_idx": time_idxs[i].item(),
                    "x_true": x_curr[i].cpu().numpy(),
                    "x_est": metrics["x_est"],
                    "observation": y_curr_i.cpu().numpy() if (has_obs and y_curr_i is not None) else None,
                    "has_observation": has_obs,
                    "rmse": metrics["rmse"],
                    "crps": metrics["crps"],
                })

        total_processed_steps += batch_size_curr
        pbar.set_postfix({"batch_size": batch_size_curr})

    # Post-process results
    final_results = []
    sorted_traj_ids = sorted(trajectory_results_map.keys())
    
    for tid in sorted_traj_ids:
        data = trajectory_results_map[tid]
        metrics_list = data["metrics"]
        if not metrics_list:
            continue
            
        rmse_values = [m["rmse"] for m in metrics_list]
        ll_values = [m["log_likelihood"] for m in metrics_list if m["log_likelihood"] != 0.0]
        ess_values = [m["ess"] for m in metrics_list]
        resample_counts = [m["resampled"] for m in metrics_list]
        step_times = [m["step_time"] for m in metrics_list]
        
        res_dict = {
            "trajectory_idx": tid,
            "mean_rmse": np.mean(rmse_values),
            "std_rmse": np.std(rmse_values),
            "min_rmse": np.min(rmse_values),
            "max_rmse": np.max(rmse_values),
            "mean_log_likelihood": np.mean(ll_values) if ll_values else 0.0,
            "total_log_likelihood": np.sum(ll_values) if ll_values else 0.0,
            "n_observations": len(ll_values),
            "n_steps": len(rmse_values),
            "mean_ess": np.mean(ess_values),
            "min_ess": np.min(ess_values),
            "n_resamples": sum(resample_counts),
            "mean_step_time": np.mean(step_times),
            "total_time": np.sum(step_times),
            "metrics": metrics_list,
            "trajectory_data": data["trajectory_data"],
            "mean_ess_pre": np.mean([m.get("ess_pre_resample", 0.0) for m in metrics_list]),
            "min_ess_pre": np.min([m.get("ess_pre_resample", 0.0) for m in metrics_list]),
        }
        final_results.append(res_dict)
        
        if wandb_run and tid in vis_indices:
            fig = plot_trajectory_comparison(res_dict, config, system)
            if fig:
                wandb.log({f"visualizations/traj_{tid}": wandb.Image(fig)})
                plt.close(fig)

    return final_results


def run_sequential_eval(args, config, system, data_module, obs_dim, proposal, wandb_run, vis_indices):
    """Execute sequential evaluation logic (original behavior)"""
    print(f"\nInitializing Bootstrap Particle Filter (Sequential)...")
    
    pf = BootstrapParticleFilterUnbatched(
        system=system,
        proposal_distribution=proposal,
        n_particles=args.n_particles,
        state_dim=system.state_dim,
        obs_dim=obs_dim,
        process_noise_std=args.process_noise_std,
        device=args.device,
    )
    
    if wandb_run:
        log_config_to_wandb(config, args, pf, args.rf_checkpoint)

    test_loader = data_module.test_dataloader()
    trajectory_results = []
    current_traj_metrics = []
    current_traj_data = []
    current_traj_idx = None
    
    pbar = tqdm(test_loader, desc="Evaluating")
    running_rmse_sum = 0.0
    running_count = 0
    
    for batch in pbar:
        batch_traj_idx = batch["trajectory_idx"].item()
        
        # Check if we switched to a new trajectory
        if current_traj_idx is not None and batch_traj_idx != current_traj_idx:
            # Finalize previous trajectory
            do_vis = current_traj_idx in vis_indices
            
            rmse_values = [m["rmse"] for m in current_traj_metrics]
            ll_values = [m["log_likelihood"] for m in current_traj_metrics if m["log_likelihood"] != 0.0]
            ess_values = [m.get("ess", 0.0) for m in current_traj_metrics if "ess" in m]
            resample_counts = [m.get("resampled", False) for m in current_traj_metrics]
            step_times = [m.get("step_time", 0.0) for m in current_traj_metrics if "step_time" in m]
            
            if do_vis:
                current_traj_data.sort(key=lambda x: x["time_idx"])
                
            result = {
                "trajectory_idx": current_traj_idx,
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
                "metrics": current_traj_metrics,
                "trajectory_data": current_traj_data if do_vis else None,
                "mean_ess_pre": np.mean([m.get("ess_pre_resample", 0.0) for m in current_traj_metrics]),
                "min_ess_pre": np.min([m.get("ess_pre_resample", 0.0) for m in current_traj_metrics]),
            }
            trajectory_results.append(result)
            
            if wandb_run and do_vis:
                fig = plot_trajectory_comparison(result, config, system)
                if fig:
                    wandb.log({f"visualizations/traj_{current_traj_idx}": wandb.Image(fig)})
                    plt.close(fig)

            current_traj_metrics = []
            current_traj_data = []
            
        current_traj_idx = batch_traj_idx
        
        # Process current batch
        metrics = pf.test_step(batch, 0)
        
        # Compute CRPS
        indices = torch.multinomial(pf.weights, pf.n_particles, replacement=True)
        ens = pf.particles[indices].detach().cpu().numpy()
        truth = batch["x_curr"].squeeze(0).detach().cpu().numpy()
        metrics["crps"] = compute_crps_ensemble(ens, truth)

        current_traj_metrics.append(metrics)
        
        # Collect visualization data if needed
        if current_traj_idx in vis_indices:
             x_curr = batch["x_curr"].squeeze(0)
             y_curr = batch["y_curr"].squeeze(0) if batch["has_observation"].item() else None
             time_idx = batch["time_idx"].item()
             
             current_traj_data.append(
                {
                    "time_idx": time_idx,
                    "x_true": x_curr.cpu().numpy(),
                    "x_est": metrics["x_est"],
                    "observation": y_curr.cpu().numpy() if y_curr is not None else None,
                    "has_observation": batch["has_observation"].item(),
                    "rmse": metrics["rmse"],
                    "crps": metrics["crps"],
                }
            )
            
        running_rmse_sum += metrics["rmse"]
        running_count += 1
        avg_rmse = running_rmse_sum / running_count
        pbar.set_postfix({"avg_rmse": f"{avg_rmse:.4f}"})

    # Finalize the last trajectory
    if current_traj_idx is not None:
        do_vis = current_traj_idx in vis_indices
        rmse_values = [m["rmse"] for m in current_traj_metrics]
        ll_values = [m["log_likelihood"] for m in current_traj_metrics if m["log_likelihood"] != 0.0]
        ess_values = [m.get("ess", 0.0) for m in current_traj_metrics if "ess" in m]
        resample_counts = [m.get("resampled", False) for m in current_traj_metrics]
        step_times = [m.get("step_time", 0.0) for m in current_traj_metrics if "step_time" in m]
        
        if do_vis:
            current_traj_data.sort(key=lambda x: x["time_idx"])
            
        result = {
            "trajectory_idx": current_traj_idx,
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
            "metrics": current_traj_metrics,
            "trajectory_data": current_traj_data if do_vis else None,
            "mean_ess_pre": np.mean([m.get("ess_pre_resample", 0.0) for m in current_traj_metrics]),
            "min_ess_pre": np.min([m.get("ess_pre_resample", 0.0) for m in current_traj_metrics]),
        }
        trajectory_results.append(result)
        
        if wandb_run and do_vis:
            fig = plot_trajectory_comparison(result, config, system)
            if fig:
                wandb.log({f"visualizations/traj_{current_traj_idx}": wandb.Image(fig)})
                plt.close(fig)
                
    return trajectory_results


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("models").setLevel(logging.WARNING)

    print("=" * 80)
    print("Bootstrap Particle Filter Evaluation")
    print("=" * 80)

    # Load dataset directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")

    # Load config from YAML
    config_path = data_dir / "config.yaml"
    if not config_path.exists():
        raise ValueError(
            f"Config file not found: {config_path}. "
            "Make sure the dataset was generated with generate.py"
        )
    
    print(f"\nLoading config from {config_path}")
    config = load_config_yaml(config_path)

    if args.obs_noise_std is not None:
        config.obs_noise_std = args.obs_noise_std
        print(f"Overriding obs_noise_std to {config.obs_noise_std}")

    # Print config details
    print("\nDataset Configuration:")
    print(f"  num_trajectories: {config.num_trajectories}")
    print(f"  len_trajectory: {config.len_trajectory}")
    print(f"  dt: {config.dt}")
    obs_op_type = "linear_projection" if isinstance(config.observation_operator, (np.ndarray, torch.Tensor)) else "arctan"
    print(f"  observation_operator: {obs_op_type}")

    # Determine observation dimension
    # obs_dim = number of observed components, regardless of operator type
    # (arctan is applied element-wise, linear projection selects components)
    inferred_obs_dim = len(config.obs_components)
    obs_dim = args.obs_dim if args.obs_dim is not None else inferred_obs_dim

    # Create system
    if "dim" in config.system_params or "F" in config.system_params:
        system_class = Lorenz96
        logging.info("Detected Lorenz96 system config")
    else:
        system_class = Lorenz63
        logging.info("Detected Lorenz63 system config")
        
    system = system_class(config)

    # Load data module
    data_module = DataAssimilationDataModule(
        config=config,
        system_class=system_class,
        data_dir=str(data_dir),
        batch_size=args.batch_size,
    )

    print(f"\nLoading data...")
    data_module.setup("test")
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    
    # Optional: Limit number of trajectories for evaluation
    if args.num_eval_trajectories is not None:
        test_dataset = data_module.test_dataset
        original_n_trajs = test_dataset.n_trajectories
        
        if args.num_eval_trajectories < original_n_trajs:
            print(f"Limiting evaluation to first {args.num_eval_trajectories} trajectories (dataset had {original_n_trajs})")
            
            # Filter items to only include the first N trajectories
            # items is a list of (traj_idx, time_idx)
            test_dataset.items = [item for item in test_dataset.items if item[0] < args.num_eval_trajectories]
            
            # Update n_trajectories count
            test_dataset.n_trajectories = args.num_eval_trajectories
            
            print(f"New test dataset size: {len(test_dataset)}")
        else:
             print(f"Requested {args.num_eval_trajectories} trajectories, but dataset only has {original_n_trajs}. Using all.")

    # Use the system instance from data_module which has updated normalization stats
    system = data_module.system
    print(f"System normalization stats loaded:")
    print(f"  Mean: {system.init_mean}")
    print(f"  Std:  {system.init_std}")

    if args.proposal_type == "rf":
        rf_checkpoint = args.rf_checkpoint
        if not rf_checkpoint or not os.path.exists(rf_checkpoint):
            raise FileNotFoundError(
                "Rectified Flow proposal selected but checkpoint not found. "
                "Provide --rf-checkpoint or a valid --rf-config-name."
            )
            
        # Attempt to load observation scalers if needed
        obs_mean = None
        obs_std = None
        
        # We need to check if data_scaled.h5 exists and load from it
        import h5py
        data_scaled_path = data_dir / "data_scaled.h5"
        if data_scaled_path.exists():
            with h5py.File(data_scaled_path, "r") as f:
                if "obs_scaler_mean" in f:
                    obs_mean = torch.from_numpy(f["obs_scaler_mean"][:]).float()
                    obs_std = torch.from_numpy(f["obs_scaler_std"][:]).float()
                    print(f"Loaded observation scalers from {data_scaled_path}")

        proposal = RectifiedFlowProposal(
            rf_checkpoint,
            device=args.device,
            num_likelihood_steps=args.rf_likelihood_steps,
            num_sampling_steps=args.rf_sampling_steps,
            system=system,
            obs_mean=obs_mean,
            obs_std=obs_std,
            mc_guidance=args.mc_guidance,
            guidance_scale=args.guidance_scale,
            obs_components=config.obs_components, # Needed to construct observation_fn
        )
        
        test_rf_log_probs(proposal, system)

        # RF Proposal wrapper handles scaling internally. 
        # The filter receives unscaled particles and operates in unscaled space.
        
    else:
        rf_checkpoint = None
        proposal = TransitionProposal(system, process_noise_std=args.process_noise_std)

    # Setup Weights & Biases
    wandb_run = None
    if not args.disable_wandb:
        obs_op_type = "linear_projection" if isinstance(config.observation_operator, (np.ndarray, torch.Tensor)) else "arctan"
        safe_name = (
            args.run_name
            or f"{args.experiment_label}_{obs_op_type}_{args.proposal_type}"
        )
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        tags.extend([args.proposal_type])
        if args.batch_size > 1:
            tags.append("batched")
            
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=safe_name,
            tags=list(dict.fromkeys(tags)),
            reinit=True,
        )
        # Define x-axis for time-series plots
        wandb.define_metric("time_step")
        wandb.define_metric("mean_rmse_over_time", step_metric="time_step")
        wandb.define_metric("mean_ess_over_time", step_metric="time_step")
        wandb.define_metric("mean_ess_pre_resample_over_time", step_metric="time_step")
        wandb.define_metric("mean_cumulative_resamples", step_metric="time_step")
        wandb.define_metric("mean_proposal_log_prob", step_metric="time_step")
        wandb.define_metric("mean_obs_log_prob", step_metric="time_step")
        wandb.define_metric("mean_obs_log_prob_std", step_metric="time_step")

    # Identify first 10 trajectories for visualization
    vis_indices = set(range(10))
    
    # Choose execution path
    if args.batch_size > 1:
        trajectory_results = run_batched_eval(
            args, config, system, data_module, obs_dim, proposal, wandb_run, vis_indices
        )
    else:
        trajectory_results = run_sequential_eval(
            args, config, system, data_module, obs_dim, proposal, wandb_run, vis_indices
        )

    aggregated = aggregate_metrics_across_trajectories(trajectory_results)

    print("\nAggregated Results:")
    print(
        f"  Mean RMSE across trajectories: {aggregated['mean_rmse_across_trajectories']:.4f} "
        f"± {aggregated['std_rmse_across_trajectories']:.4f}"
    )
    print(
        f"  Mean CRPS across trajectories: {aggregated['mean_crps_across_trajectories']:.4f} "
        f"± {aggregated['std_crps_across_trajectories']:.4f}"
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
        # 1. Log Metrics Across Time
        if trajectory_results:
            max_steps = max(len(r["metrics"]) for r in trajectory_results)
            traj_cumulative_resamples = [0] * len(trajectory_results)
            
            for t in range(max_steps):
                rmse_vals = []
                crps_vals = []
                ess_vals = []
                ess_pre_vals = []
                cumulative_resample_vals = []
                prop_log_vals = []
                obs_log_vals = []
                obs_log_std_vals = []

                for i, r in enumerate(trajectory_results):
                    if t < len(r["metrics"]):
                        rmse_vals.append(r["metrics"][t]["rmse"])
                        if "crps" in r["metrics"][t]:
                            crps_vals.append(r["metrics"][t]["crps"])
                        if "ess" in r["metrics"][t]:
                            ess_vals.append(r["metrics"][t]["ess"])
                        if "ess_pre_resample" in r["metrics"][t]:
                            ess_pre_vals.append(r["metrics"][t]["ess_pre_resample"])
                        if r["metrics"][t].get("resampled", False):
                            traj_cumulative_resamples[i] += 1
                        cumulative_resample_vals.append(traj_cumulative_resamples[i])
                        if "proposal_log_prob_mean" in r["metrics"][t]:
                            prop_log_vals.append(r["metrics"][t]["proposal_log_prob_mean"])
                        if "obs_log_prob_mean" in r["metrics"][t]:
                            obs_log_vals.append(r["metrics"][t]["obs_log_prob_mean"])
                        if "obs_log_prob_std" in r["metrics"][t]:
                            obs_log_std_vals.append(r["metrics"][t]["obs_log_prob_std"])
                
                if rmse_vals:
                    wandb.log({
                        "time_step": t,
                        "mean_rmse_over_time": np.mean(rmse_vals),
                        "mean_crps_over_time": np.mean(crps_vals) if crps_vals else 0.0,
                        "mean_ess_over_time": np.mean(ess_vals) if ess_vals else 0.0,
                        "mean_ess_pre_resample_over_time": np.mean(ess_pre_vals) if ess_pre_vals else 0.0,
                        "mean_cumulative_resamples": np.mean(cumulative_resample_vals) if cumulative_resample_vals else 0.0,
                        "mean_proposal_log_prob": np.mean(prop_log_vals) if prop_log_vals else 0.0,
                        "mean_obs_log_prob": np.mean(obs_log_vals) if obs_log_vals else 0.0,
                        "mean_obs_log_prob_std": np.mean(obs_log_std_vals) if obs_log_std_vals else 0.0,
                    })

        # 2. Log Aggregated Scalars
        wandb.log(
            {
                "mean_rmse": aggregated["mean_rmse_across_trajectories"],
                "std_rmse": aggregated["std_rmse_across_trajectories"],
                "mean_crps": aggregated["mean_crps_across_trajectories"],
                "std_crps": aggregated["std_crps_across_trajectories"],
                "mean_log_likelihood": aggregated["mean_total_log_likelihood"],
                "mean_ess": aggregated["mean_ess"],
                "step_time": aggregated["mean_step_time"],
            }
        )

        wandb.finish()

    print("\n" + "=" * 80)
    print("Evaluation completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
