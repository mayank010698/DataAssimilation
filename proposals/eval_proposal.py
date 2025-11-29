import argparse
import logging
import os
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    Lorenz63,
    TimeAlignedBatchSampler,
    load_config_yaml,
)
from proposals.rectified_flow import RFProposal

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

    if config.use_preprocessing:
        # x_est and x_true are ALREADY in original space from run_proposal_eval
        x_true_orig = x_true
        x_est_orig = x_est
        space_label = " (Original Space)"
    else:
        x_true_orig = x_true
        x_est_orig = x_est
        space_label = ""

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


def run_proposal_eval(
    checkpoint_path: str,
    data_dir: str,
    n_trajectories: Optional[int] = None,  # None means all
    n_vis_trajectories: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
    wandb_run = None,
):
    """
    Evaluate RF Proposal using batched autoregressive generation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Dataset directory
        n_trajectories: Number of trajectories to evaluate (None for all)
        n_vis_trajectories: Number of trajectories to visualize
        batch_size: Batch size for evaluation
        device: Device to run on
        wandb_run: Active wandb run (optional)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running Proposal Evaluation (Autoregressive)")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load Config
    data_path = Path(data_dir)
    config_path = data_path / "config.yaml"
    config = load_config_yaml(config_path)
    
    # Load Model
    model = RFProposal.load_from_checkpoint(checkpoint_path)
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
    # Maps trajectory_idx -> current_state (tensor)
    current_states = {} 
    
    # Store results
    trajectory_results = {i: {"trajectory_data": [], "rmse_sum": 0.0, "count": 0} for i in range(n_trajectories)}
    
    pbar = tqdm(test_loader, desc="Generating")
    
    for batch in pbar:
        # Move to device and handle preprocessing
        if config.use_preprocessing:
            x_prev_gt = batch["x_prev_scaled"].to(device)
            x_curr_gt = batch["x_curr_scaled"].to(device)
            if batch["has_observation"].any():
                y_curr = batch["y_curr_scaled"].to(device)
            else:
                y_curr = None
        else:
            x_prev_gt = batch["x_prev"].to(device)
            x_curr_gt = batch["x_curr"].to(device)
            y_curr = batch["y_curr"].to(device) if batch["has_observation"].any() else None
        
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
                current_states[tid] = x_prev_gt[i]
                
            if tid in current_states:
                batch_x_prev.append(current_states[tid])
                if y_curr is not None:
                    batch_y_curr.append(y_curr[i])
                valid_mask.append(i)
            else:
                pass
                
        if not batch_x_prev:
            continue
            
        # Stack for batch inference
        x_prev_input = torch.stack(batch_x_prev)
        y_curr_input = torch.stack(batch_y_curr) if batch_y_curr else None
        
        # Debug: Check for NaNs in input
        if torch.isnan(x_prev_input).any():
            logger.error(f"NaN detected in input x_prev! Stats: mean={x_prev_input.mean():.4f}, std={x_prev_input.std():.4f}, min={x_prev_input.min():.4f}, max={x_prev_input.max():.4f}")
        if y_curr_input is not None and torch.isnan(y_curr_input).any():
            logger.error(f"NaN detected in input y_curr! Stats: mean={y_curr_input.mean():.4f}, std={y_curr_input.std():.4f}")

        # Sample next states
        # x_next ~ p(x_t | x_{t-1}, y_t)
        with torch.no_grad():
            x_next = model.sample(x_prev_input, y_curr_input)
            
        # Debug: Check for NaNs in output
        if torch.isnan(x_next).any():
             logger.error(f"NaN produced by model! Input stats: x_mean={x_prev_input.mean():.4f}")

            
        # Calculate RMSE and Update
        batch_rmse = []
        
        for k, idx in enumerate(valid_mask):
            tid = traj_idxs[idx]
            t = time_idxs[idx]
            
            # Generated state
            x_gen = x_next[k]
            
            # Ground truth
            x_true = x_curr_gt[idx]
            
            # Update current state for autoregressive generation (always use scaled/model space)
            current_states[tid] = x_gen
            
            # Metric calculation (in original space)
            if config.use_preprocessing:
                # Detach and unscale for metric computation
                x_gen_orig = system.postprocess(x_gen)
                x_true_orig = system.postprocess(x_true)
                
                error = x_gen_orig - x_true_orig
            else:
                error = x_gen - x_true
                x_gen_orig = x_gen
                x_true_orig = x_true
                
            rmse = torch.sqrt(torch.mean(error**2)).item()
            batch_rmse.append(rmse)
            
            # Store
            has_obs = batch["has_observation"][idx].item()
            obs = y_curr[idx].cpu().numpy() if (has_obs and y_curr is not None) else None
            
            # Use original space values for storage/plotting
            trajectory_results[tid]["trajectory_data"].append({
                "time_idx": t,
                "x_true": x_true_orig.cpu().numpy(),
                "x_est": x_gen_orig.cpu().numpy(), # "x_est" used by plotter
                "observation": obs,
                "has_observation": has_obs,
                "rmse": rmse
            })
            
            trajectory_results[tid]["rmse_sum"] += rmse
            trajectory_results[tid]["count"] += 1
            
        # Update pbar
        avg_rmse = sum(batch_rmse) / len(batch_rmse) if batch_rmse else 0.0
        pbar.set_postfix({"rmse": f"{avg_rmse:.4f}"})

    # Aggregate and Log
    total_rmse = 0.0
    total_steps = 0
    
    for tid, res in trajectory_results.items():
        if res["count"] > 0:
            mean_traj_rmse = res["rmse_sum"] / res["count"]
            res["mean_rmse"] = mean_traj_rmse
            total_rmse += res["rmse_sum"]
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
    logger.info(f"Global Mean RMSE (Autoregressive): {global_mean_rmse:.4f}")
    
    if wandb_run:
        wandb_run.log({"eval/global_mean_rmse": global_mean_rmse})
        
        # Log RMSE over time (averaged across trajectories)
        # Find max time
        max_t = 0
        for res in trajectory_results.values():
            if res["trajectory_data"]:
                max_t = max(max_t, res["trajectory_data"][-1]["time_idx"])
        
        for t in range(1, max_t + 1):
            rmses_at_t = []
            for res in trajectory_results.values():
                # Find data at t (inefficient but fine for eval)
                for d in res["trajectory_data"]:
                    if d["time_idx"] == t:
                        rmses_at_t.append(d["rmse"])
                        break
            
            if rmses_at_t:
                mean_t = sum(rmses_at_t) / len(rmses_at_t)
                wandb_run.log({"eval/mean_rmse_over_time": mean_t, "time_step": t})

    return global_mean_rmse

if __name__ == "__main__":
    # Basic CLI for standalone usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--n-trajectories", type=int, default=None, help="Number of trajectories to evaluate (default: all)")
    parser.add_argument("--n-vis-trajectories", type=int, default=10, help="Number of trajectories to visualize")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    run = None
    if args.wandb:
        run = wandb.init(project="rf-proposal-eval", name="standalone-eval")
        
    run_proposal_eval(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        n_trajectories=args.n_trajectories,
        n_vis_trajectories=args.n_vis_trajectories,
        batch_size=args.batch_size,
        device=args.device,
        wandb_run=run
    )