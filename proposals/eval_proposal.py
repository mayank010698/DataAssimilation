import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import numpy as np
import torch
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    Lorenz63,
    Lorenz96,
    TimeAlignedBatchSampler,
    load_config_yaml,
)
from proposals.rectified_flow import RFProposal

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


def plot_trajectory_comparison(
    result: Dict[str, Any],
    config: DataAssimilationConfig,
    system: Lorenz63,
) -> plt.Figure:
    """Plot trajectory comparison (True vs Generated)"""
    trajectory_data = result["trajectory_data"]
    traj_idx = result["trajectory_idx"]

    if trajectory_data is None:
        return None

    time_steps = np.array([d["time_idx"] for d in trajectory_data])
    x_true = np.array([d["x_true"] for d in trajectory_data])
    x_est = np.array([d["x_est"] for d in trajectory_data])
    rmse_values = np.array([d["rmse"] for d in trajectory_data])

    # Create mapping from time_idx to array position
    time_to_idx = {t: idx for idx, t in enumerate(time_steps)}

    # Trajectories are already stored in physical space.
    x_true_orig = x_true
    x_est_orig = x_est
    space_label = " (Physical Space)"

    obs_times = []
    obs_values = []
    obs_indices = []

    for d in trajectory_data:
        if d["has_observation"] and d["observation"] is not None:
            time_idx = d["time_idx"]
            array_idx = time_to_idx[time_idx]
            obs_times.append(time_idx)
            obs_values.append(d["observation"])
            obs_indices.append(array_idx)

    obs_times = np.array(obs_times)
    obs_indices = np.array(obs_indices) if len(obs_indices) > 0 else np.array([])
    if len(obs_values) > 0:
        obs_values = np.array(obs_values)

    fig = plt.figure(figsize=(16, 12))

    # 3D Plot
    ax1 = fig.add_subplot(3, 2, 1, projection="3d")
    ax1.plot(x_true_orig[:, 0], x_true_orig[:, 1], x_true_orig[:, 2], "b-", alpha=0.8, label="True")
    ax1.plot(x_est_orig[:, 0], x_est_orig[:, 1], x_est_orig[:, 2], "r--", alpha=0.8, label="Generated")
    ax1.set_title(f"3D Trajectory{space_label} (ID: {traj_idx})")
    ax1.legend()

    # State Components
    state_names = ["X", "Y", "Z"]
    colors = ["blue", "green", "orange"]
    for i, (name, color) in enumerate(zip(state_names, colors)):
        ax = fig.add_subplot(3, 2, i + 2)
        ax.plot(time_steps, x_true_orig[:, i], color=color, label=f"{name} (true)")
        ax.plot(time_steps, x_est_orig[:, i], color="red", linestyle="--", label=f"{name} (gen)")
        
        # Plot observations if available for this component
        if i in config.obs_components and len(obs_times) > 0:
            valid_mask = obs_indices < len(x_true_orig)
            if np.any(valid_mask):
                ax.scatter(
                    obs_times[valid_mask], 
                    x_true_orig[obs_indices[valid_mask], i], 
                    color="red", marker="x", s=40, label="Obs"
                )
        ax.set_ylabel(name)
        ax.legend()

    # RMSE over time
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(time_steps, rmse_values, "purple")
    ax5.set_title("RMSE Over Time")
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("RMSE")

    plt.tight_layout()
    return fig


def _derive_eval_run_name(checkpoint_path: str, mc_guidance: bool = False) -> str:
    """Build wandb run name like eval-<training_run>[-mc] for clarity."""
    ckpt_path = Path(checkpoint_path)
    try:
        # Expect .../<run_name>/checkpoints/<file>.ckpt
        run_dir = ckpt_path.parents[1]
        base_name = run_dir.name
    except IndexError:
        base_name = ckpt_path.stem
    
    run_name = f"eval-{base_name}"
    if mc_guidance:
        run_name += "-mc"
    return run_name


def run_proposal_eval(
    checkpoint_path: str,
    data_dir: str,
    n_trajectories: Optional[int] = None,  # None means all
    n_vis_trajectories: int = 10,
    batch_size: int = 32,
    n_samples_per_traj: int = 1,
    device: str = "cuda",
    wandb_run = None,
    num_sampling_steps: Optional[int] = None,
    num_likelihood_steps: Optional[int] = None,
    mc_guidance: bool = False,
    guidance_scale: float = 1.0,
):
    """
    Evaluate RF Proposal using batched autoregressive generation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Dataset directory
        n_trajectories: Number of trajectories to evaluate (None for all)
        n_vis_trajectories: Number of trajectories to visualize
        batch_size: Batch size for evaluation
        n_samples_per_traj: Number of independent samples per trajectory (for CRPS)
        device: Device to run on
        wandb_run: Active wandb run (optional)
        num_sampling_steps: Override number of Euler steps for sampling (None = use checkpoint value)
        num_likelihood_steps: Override number of Euler steps for likelihood computation (None = use checkpoint value)
        mc_guidance: Override Monte Carlo guidance flag
        guidance_scale: Override guidance scale
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running Proposal Evaluation (Autoregressive)")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Samples per trajectory: {n_samples_per_traj}")
    
    # Load Config
    data_path = Path(data_dir)
    config_path = data_path / "config.yaml"
    config = load_config_yaml(config_path)
    
    # Load Model
    model = RFProposal.load_from_checkpoint(checkpoint_path)

    # Determine system class
    system_class = Lorenz63
    if "dim" in config.system_params and config.system_params["dim"] > 3:
        system_class = Lorenz96
    
    # Override guidance settings
    if mc_guidance:
        logger.info(f"Overriding MC Guidance: {model.mc_guidance} -> {mc_guidance}")
        model.mc_guidance = mc_guidance
    if guidance_scale != 1.0:
        logger.info(f"Overriding Guidance Scale: {model.guidance_scale} -> {guidance_scale}")
        model.guidance_scale = guidance_scale
        
    # Override sampling/likelihood steps if provided
    if num_sampling_steps is not None:
        logger.info(f"Overriding num_sampling_steps: {model.num_sampling_steps} -> {num_sampling_steps}")
        model.num_sampling_steps = num_sampling_steps
    if num_likelihood_steps is not None:
        logger.info(f"Overriding num_likelihood_steps: {model.num_likelihood_steps} -> {num_likelihood_steps}")
        model.num_likelihood_steps = num_likelihood_steps
    
    logger.info(f"Using num_sampling_steps: {model.num_sampling_steps}")
    logger.info(f"Using num_likelihood_steps: {model.num_likelihood_steps}")

    # Extract training process noise from checkpoint path
    train_has_pnoise = False
    if "pnoise" in checkpoint_path or "noise0p1" in checkpoint_path:
        train_has_pnoise = True
    elif "nonoise" in checkpoint_path:
        train_has_pnoise = False
        
    # Determine number of observed dimensions
    num_obs_dims = 0
    if not model.hparams.get("use_observations", True):
        num_obs_dims = 0
    elif model.obs_indices is not None:
        num_obs_dims = len(model.obs_indices)
    elif model.hparams.get("obs_dim") is not None:
        num_obs_dims = model.hparams.get("obs_dim")
    elif config.obs_components is not None:
        num_obs_dims = len(config.obs_components)
    
    # Log additional config to wandb
    if wandb_run:
        wandb_config = {
            "eval/mc_guidance": model.mc_guidance,
            "eval/guidance_scale": model.guidance_scale,
            "train/has_process_noise": train_has_pnoise,
            "eval/num_obs_dims": num_obs_dims,
            "model/architecture": model.hparams.get("architecture"),
            "model/state_dim": model.hparams.get("state_dim"),
            "model/hidden_dim": model.hparams.get("hidden_dim"),
            "model/depth": model.hparams.get("depth"),
            "model/channels": model.hparams.get("channels"),
            "model/num_blocks": model.hparams.get("num_blocks"),
        }
        wandb_run.config.update(wandb_config)
    
    model.to(device)
    model.eval()
    
    # Load Data
    # Limit number of trajectories for evaluation if needed
    # but DataModule loads all. We will limit via Sampler or just stop early.
    data_module = DataAssimilationDataModule(
        config=config,
        system_class=Lorenz63,
        data_dir=str(data_dir),
        batch_size=batch_size,
    )
    data_module.setup("test")
    test_dataset = data_module.test_dataset

    # Construct observation_fn for guidance (Robust to Scaling)
    if model.mc_guidance:
        obs_components = config.obs_components
        logger.info(f"Constructing robust observation_fn with components: {obs_components}")
        
        # Get statistics from data module
        obs_scaler_mean = data_module.obs_scaler_mean.to(device) if data_module.obs_scaler_mean is not None else None
        obs_scaler_std = data_module.obs_scaler_std.to(device) if data_module.obs_scaler_std is not None else None
        
        if obs_scaler_mean is None:
             logger.warning("No observation scalers found in data module! Guidance might be incorrect if data is scaled.")
        
        def observation_fn(x_scaled):
            # x_scaled is shape (..., state_dim) in SCALED space
            
            # 1. Unscale to physical space
            # system.postprocess handles unscaling using system.init_mean/std
            # We need to make sure x_scaled is on correct device, usually it is
            x_physical = data_module.system.postprocess(x_scaled)
            
            # 2. Apply Observation Operator (Physical Space)
            y_physical = data_module.system.apply_observation_operator(x_physical)
            
            # 3. Rescale to Observation Space (if scalers exist)
            if obs_scaler_mean is not None:
                # Handle broadcasting if needed, though usually shapes align
                y_scaled = (y_physical - obs_scaler_mean) / obs_scaler_std
            else:
                y_scaled = y_physical
                
            return y_scaled
    else:
        observation_fn = None
    
    # We only want n_trajectories
    total_trajs_in_data = test_dataset.n_trajectories
    if n_trajectories is None or n_trajectories > total_trajs_in_data:
        if n_trajectories is not None:
            logger.warning(f"Requested {n_trajectories} trajectories but dataset only has {total_trajs_in_data}")
        n_trajectories = total_trajs_in_data
        
    # Use TimeAlignedBatchSampler to get synchronized batches
    # We need to run for the full length of trajectories
    traj_len_inference = config.len_trajectory - 1
    
    batch_sampler = TimeAlignedBatchSampler(
        data_source_len=len(test_dataset),
        num_trajectories=n_trajectories, # Only sample first N trajectories
        traj_len=traj_len_inference,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=batch_sampler,
        num_workers=0, # simpler for debugging/eval
    )
    
    logger.info(f"Evaluating on {n_trajectories} trajectories, length {traj_len_inference}")
    
    # System for visualization (reuse data_module system to ensure correct stats)
    system = data_module.system
    
    # Tracking state
    # Maps trajectory_idx -> current_state (tensor of shape (K, state_dim))
    current_states = {} 
    
    # Store results
    trajectory_results = {
        i: {
            "trajectory_data": [], 
            "rmse_sum": 0.0, 
            "crps_sum": 0.0,
            "count": 0
        } for i in range(n_trajectories)
    }
    
    pbar = tqdm(test_loader, desc="Generating")
    
    for batch in pbar:
        # Always use SCALED inputs for the model (it was trained on scaled data)
        # And keep UNSCALED ground truth for evaluation
        
        x_prev_input = batch["x_prev_scaled"].to(device)
        x_curr_gt = batch["x_curr"].to(device) # Physical space ground truth
        
        if batch["has_observation"].any():
            y_curr_input = batch["y_curr_scaled"].to(device)
        else:
            y_curr_input = None
        
        traj_idxs = batch["trajectory_idx"].squeeze(-1).tolist()
        time_idxs = batch["time_idx"].squeeze(-1).tolist()
        
        batch_x_prev = []
        batch_y_curr = []
        valid_mask = []
        
        # Prepare inputs
        for i, tid in enumerate(traj_idxs):
            t = time_idxs[i]
            
            # Initialize if start of trajectory (t=1 means step 1, predicting from x0)
            if t == 1:
                # Start with x0 replicated K times
                # x_prev_input[i] is (D,)
                current_states[tid] = x_prev_input[i].unsqueeze(0).repeat(n_samples_per_traj, 1)
                
            if tid in current_states:
                batch_x_prev.append(current_states[tid]) # (K, D)
                if y_curr_input is not None:
                    # Repeat y for each sample in ensemble
                    # y_curr_input[i] is (D_obs,)
                    y_expanded = y_curr_input[i].unsqueeze(0).repeat(n_samples_per_traj, 1)
                    batch_y_curr.append(y_expanded)
                valid_mask.append(i)
            else:
                pass
                
        if not batch_x_prev:
            continue
            
        # Stack for batch inference
        # List of B tensors of shape (K, D) -> Stack (B, K, D) -> Reshape (B*K, D)
        x_prev_stack = torch.stack(batch_x_prev) # (B_eff, K, D)
        B_eff, K, D = x_prev_stack.shape
        x_prev_flat = x_prev_stack.reshape(B_eff * K, D)
        
        y_curr_flat = None
        if batch_y_curr:
            y_curr_stack = torch.stack(batch_y_curr) # (B_eff, K, D_obs)
            D_obs = y_curr_stack.shape[-1]
            y_curr_flat = y_curr_stack.reshape(B_eff * K, D_obs)

            # Filter observations if model expects specific indices
            # The dataset provides full observations (e.g. 50 dims), but model might be trained on subset (e.g. 25 dims)
            if hasattr(model, 'obs_indices') and model.obs_indices is not None:
                # model.obs_indices is a tensor or list
                if isinstance(model.obs_indices, torch.Tensor):
                    indices = model.obs_indices
                else:
                    indices = torch.tensor(model.obs_indices, device=y_curr_flat.device)
                
                # Check if we need to filter
                if y_curr_flat.shape[1] > len(indices):
                    y_curr_flat = y_curr_flat[:, indices]
        
        # Sample next states
        # x_next ~ p(x_t | x_{t-1}, y_t)
        with torch.no_grad():
            # Model outputs (B*K, D)
            # Pass observation_fn if guidance is enabled
            if model.mc_guidance and observation_fn is not None:
                # IMPORTANT: sample() needs observation_fn for guidance
                # But our wrapper/model.sample signature might not accept it directly in PyTorch Lightning module
                # Let's check RFProposal.sample signature
                # It accepts: x_prev, y_curr=None, dt=None, observation_fn=None
                x_next_flat = model.sample(x_prev_flat, y_curr_flat, observation_fn=observation_fn)
            else:
                x_next_flat = model.sample(x_prev_flat, y_curr_flat)
            
        # Reshape back to (B, K, D)
        x_next_stack = x_next_flat.reshape(B_eff, K, D)
            
        # Calculate Metrics and Update
        batch_rmse = []
        batch_crps = []
        
        for k, idx in enumerate(valid_mask):
            tid = traj_idxs[idx]
            t = time_idxs[idx]
            
            # Generated ensemble (Scaled)
            x_gen_k = x_next_stack[k] # (K, D)
            
            # Ground truth (Unscaled / Physical)
            x_true = x_curr_gt[idx] # (D,)
            
            # Update current state for autoregressive generation
            current_states[tid] = x_gen_k
            
            # Metric calculation (in original space)
            # Detach and unscale for metric computation
            # process whole ensemble at once
            x_gen_orig_k = system.postprocess(x_gen_k) # (K, D)
            x_true_orig = x_true # Already physical
            
            # RMSE of the Mean
            x_mean_orig = torch.mean(x_gen_orig_k, dim=0) # (D,)
            error = x_mean_orig - x_true_orig
            rmse = torch.sqrt(torch.mean(error**2)).item()
            batch_rmse.append(rmse)
            
            # CRPS
            crps = compute_crps_ensemble(x_gen_orig_k.cpu().numpy(), x_true_orig.cpu().numpy())
            batch_crps.append(crps)
            
            # Store
            has_obs = batch["has_observation"][idx].item()
            obs = batch["y_curr"][idx].cpu().numpy() if (has_obs and "y_curr" in batch) else None
            
            # For plotting/storage, we store the mean estimate or full ensemble?
            # Storing full ensemble might be heavy if K is large.
            # For now, let's store mean for plotting compatibility, 
            # but maybe we should update plotter to show spread if K > 1.
            # For backward compatibility, x_est is mean.
            
            trajectory_results[tid]["trajectory_data"].append({
                "time_idx": t,
                "x_true": x_true_orig.cpu().numpy(),
                "x_est": x_mean_orig.cpu().numpy(), # Mean estimate
                "x_std": torch.std(x_gen_orig_k, dim=0, correction=0).cpu().numpy(), # Std dev for uncertainty
                "observation": obs,
                "has_observation": has_obs,
                "rmse": rmse,
                "crps": crps
            })
            
            trajectory_results[tid]["rmse_sum"] += rmse
            trajectory_results[tid]["crps_sum"] += crps
            trajectory_results[tid]["count"] += 1
            
        # Update pbar
        avg_rmse = sum(batch_rmse) / len(batch_rmse) if batch_rmse else 0.0
        avg_crps = sum(batch_crps) / len(batch_crps) if batch_crps else 0.0
        pbar.set_postfix({"rmse": f"{avg_rmse:.4f}", "crps": f"{avg_crps:.4f}"})

    # Aggregate and Log
    total_rmse = 0.0
    total_crps = 0.0
    total_steps = 0
    
    for tid, res in trajectory_results.items():
        if res["count"] > 0:
            mean_traj_rmse = res["rmse_sum"] / res["count"]
            mean_traj_crps = res["crps_sum"] / res["count"]
            
            res["mean_rmse"] = mean_traj_rmse
            res["mean_crps"] = mean_traj_crps
            
            total_rmse += res["rmse_sum"]
            total_crps += res["crps_sum"]
            total_steps += res["count"]
            
            # Sort data by time
            res["trajectory_data"].sort(key=lambda x: x["time_idx"])
            
            # Plot first few trajectories
            if wandb_run and tid < n_vis_trajectories:
                fig_res = {
                    "trajectory_idx": tid,
                    "trajectory_data": res["trajectory_data"]
                }
                fig = plot_trajectory_comparison(fig_res, config, system)
                if fig:
                    wandb_run.log({f"eval/trajectory_{tid}": wandb.Image(fig)})
                    plt.close(fig)

    global_mean_rmse = total_rmse / total_steps if total_steps > 0 else 0.0
    global_mean_crps = total_crps / total_steps if total_steps > 0 else 0.0
    
    logger.info(f"Global Mean RMSE (Autoregressive): {global_mean_rmse:.4f}")
    logger.info(f"Global Mean CRPS (Autoregressive): {global_mean_crps:.4f}")
    
    if wandb_run:
        wandb_run.log({
            "eval/global_mean_rmse": global_mean_rmse,
            "eval/global_mean_crps": global_mean_crps
        })
        
        # Log metrics over time
        max_t = 0
        for res in trajectory_results.values():
            if res["trajectory_data"]:
                max_t = max(max_t, res["trajectory_data"][-1]["time_idx"])
        
        for t in range(1, max_t + 1):
            rmses_at_t = []
            crps_at_t = []
            for res in trajectory_results.values():
                for d in res["trajectory_data"]:
                    if d["time_idx"] == t:
                        rmses_at_t.append(d["rmse"])
                        crps_at_t.append(d["crps"])
                        break
            
            if rmses_at_t:
                mean_rmse_t = sum(rmses_at_t) / len(rmses_at_t)
                mean_crps_t = sum(crps_at_t) / len(crps_at_t)
                wandb_run.log({
                    "eval/mean_rmse_over_time": mean_rmse_t, 
                    "eval/mean_crps_over_time": mean_crps_t,
                    "time_step": t
                })

    return global_mean_rmse

if __name__ == "__main__":
    # Basic CLI for standalone usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--n-trajectories", type=int, default=None, help="Number of trajectories to evaluate (default: all)")
    parser.add_argument("--n-vis-trajectories", type=int, default=10, help="Number of trajectories to visualize")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-samples-per-traj", type=int, default=1, help="Number of samples per trajectory (for CRPS)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--num-sampling-steps", type=int, default=None, help="Override number of Euler steps for sampling (default: use checkpoint value)")
    parser.add_argument("--num-likelihood-steps", type=int, default=None, help="Override number of Euler steps for likelihood computation (default: use checkpoint value)")
    parser.add_argument("--mc-guidance", action="store_true", help="Enable Monte Carlo guidance")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Scale for Monte Carlo guidance")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the wandb run")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    run = None
    if not args.no_wandb:
        if args.run_name:
            run_name = args.run_name
        else:
            run_name = _derive_eval_run_name(args.checkpoint, args.mc_guidance)
        run = wandb.init(project="rf-proposal-eval", name=run_name, entity="ml-climate")
        
    run_proposal_eval(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        n_trajectories=args.n_trajectories,
        n_vis_trajectories=args.n_vis_trajectories,
        batch_size=args.batch_size,
        n_samples_per_traj=args.n_samples_per_traj,
        device=args.device,
        wandb_run=run,
        num_sampling_steps=args.num_sampling_steps,
        num_likelihood_steps=args.num_likelihood_steps,
        mc_guidance=args.mc_guidance,
        guidance_scale=args.guidance_scale,
    )