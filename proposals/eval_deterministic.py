"""
Evaluation script for Deterministic Next-Step Prediction Model

Evaluates the deterministic model using autoregressive generation.
"""

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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    Lorenz96,
    Lorenz63,
    TimeAlignedBatchSampler,
    load_config_yaml,
)
from proposals.deterministic_model import DeterministicModel


def plot_trajectory_comparison(
    result: Dict[str, Any],
    config: DataAssimilationConfig,
    state_dim: int,
) -> plt.Figure:
    """Plot trajectory comparison (True vs Generated) for high-dimensional systems"""
    trajectory_data = result["trajectory_data"]
    traj_idx = result["trajectory_idx"]

    if trajectory_data is None or len(trajectory_data) == 0:
        return None

    time_steps = np.array([d["time_idx"] for d in trajectory_data])
    x_true = np.array([d["x_true"] for d in trajectory_data])
    x_est = np.array([d["x_est"] for d in trajectory_data])
    rmse_values = np.array([d["rmse"] for d in trajectory_data])

    # Determine number of components to plot
    n_plot = min(6, state_dim)  # Plot at most 6 components
    
    n_rows = (n_plot + 1) // 2 + 1  # +1 for RMSE plot
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
    axes = axes.flatten()

    # Plot state components
    for i in range(n_plot):
        ax = axes[i]
        ax.plot(time_steps, x_true[:, i], 'b-', label='True', alpha=0.8)
        ax.plot(time_steps, x_est[:, i], 'r--', label='Predicted', alpha=0.8)
        ax.set_ylabel(f'x[{i}]')
        ax.set_xlabel('Time Step')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'Component {i}')

    # RMSE over time
    rmse_ax = axes[n_plot]
    rmse_ax.plot(time_steps, rmse_values, 'purple', linewidth=2)
    rmse_ax.set_title('RMSE Over Time')
    rmse_ax.set_xlabel('Time Step')
    rmse_ax.set_ylabel('RMSE')
    rmse_ax.axhline(y=np.mean(rmse_values), color='orange', linestyle='--', 
                   label=f'Mean: {np.mean(rmse_values):.4f}')
    rmse_ax.legend()

    # Hide unused axes
    for i in range(n_plot + 1, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Trajectory {traj_idx} - Deterministic Model', fontsize=14)
    plt.tight_layout()
    return fig


def plot_lorenz63_trajectory(
    result: Dict[str, Any],
    config: DataAssimilationConfig,
) -> plt.Figure:
    """Plot Lorenz63-style 3D trajectory comparison"""
    trajectory_data = result["trajectory_data"]
    traj_idx = result["trajectory_idx"]

    if trajectory_data is None or len(trajectory_data) == 0:
        return None

    time_steps = np.array([d["time_idx"] for d in trajectory_data])
    x_true = np.array([d["x_true"] for d in trajectory_data])
    x_est = np.array([d["x_est"] for d in trajectory_data])
    rmse_values = np.array([d["rmse"] for d in trajectory_data])

    fig = plt.figure(figsize=(16, 12))

    # 3D Plot
    ax1 = fig.add_subplot(3, 2, 1, projection="3d")
    ax1.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], "b-", alpha=0.8, label="True")
    ax1.plot(x_est[:, 0], x_est[:, 1], x_est[:, 2], "r--", alpha=0.8, label="Predicted")
    ax1.set_title(f"3D Trajectory (ID: {traj_idx})")
    ax1.legend()

    # State Components
    state_names = ["X", "Y", "Z"]
    colors = ["blue", "green", "orange"]
    for i, (name, color) in enumerate(zip(state_names, colors)):
        ax = fig.add_subplot(3, 2, i + 2)
        ax.plot(time_steps, x_true[:, i], color=color, label=f"{name} (true)")
        ax.plot(time_steps, x_est[:, i], color="red", linestyle="--", label=f"{name} (pred)")
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


def _derive_eval_run_name(checkpoint_path: str) -> str:
    """Build wandb run name like eval-det-<training_run> for clarity."""
    ckpt_path = Path(checkpoint_path)
    try:
        run_dir = ckpt_path.parents[1]
        base_name = run_dir.name
    except IndexError:
        base_name = ckpt_path.stem
    return f"eval-det-{base_name}"


def run_deterministic_eval(
    checkpoint_path: str,
    data_dir: str,
    n_trajectories: Optional[int] = None,
    n_vis_trajectories: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
    wandb_run = None,
):
    """
    Evaluate Deterministic Model using autoregressive generation.
    
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
    logger.info(f"Running Deterministic Model Evaluation (Autoregressive)")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load Config
    data_path = Path(data_dir)
    config_path = data_path / "config.yaml"
    config = load_config_yaml(config_path)
    
    # Determine system type from config
    state_dim = config.system_params.get('dim', 3)
    is_lorenz96 = state_dim > 3
    system_class = Lorenz96 if is_lorenz96 else Lorenz63
    
    # Load Model
    model = DeterministicModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    # Load Data
    data_module = DataAssimilationDataModule(
        config=config,
        system_class=system_class,
        data_dir=str(data_dir),
        batch_size=batch_size,
    )
    data_module.setup("test")
    test_dataset = data_module.test_dataset
    
    total_trajs_in_data = test_dataset.n_trajectories
    if n_trajectories is None or n_trajectories > total_trajs_in_data:
        if n_trajectories is not None:
            logger.warning(f"Requested {n_trajectories} trajectories but dataset only has {total_trajs_in_data}")
        n_trajectories = total_trajs_in_data
        
    traj_len_inference = config.len_trajectory - 1
    
    batch_sampler = TimeAlignedBatchSampler(
        data_source_len=len(test_dataset),
        num_trajectories=n_trajectories,
        traj_len=traj_len_inference,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
    )
    
    logger.info(f"Evaluating on {n_trajectories} trajectories, length {traj_len_inference}")
    
    # System for postprocessing
    system = data_module.system
    
    # Tracking state: trajectory_idx -> current_state
    current_states = {}
    
    # Store results
    trajectory_results = {
        i: {
            "trajectory_data": [], 
            "rmse_sum": 0.0,
            "count": 0
        } for i in range(n_trajectories)
    }
    
    pbar = tqdm(test_loader, desc="Generating")
    
    for batch in pbar:
        # Use SCALED inputs for the model
        x_prev_input = batch["x_prev_scaled"].to(device)
        x_curr_gt = batch["x_curr"].to(device)  # Physical space ground truth
        
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
            
            # Initialize if start of trajectory
            if t == 1:
                current_states[tid] = x_prev_input[i].unsqueeze(0)
                
            if tid in current_states:
                batch_x_prev.append(current_states[tid])
                if y_curr_input is not None:
                    batch_y_curr.append(y_curr_input[i].unsqueeze(0))
                valid_mask.append(i)
                
        if not batch_x_prev:
            continue
            
        # Stack for batch inference
        x_prev_stack = torch.cat(batch_x_prev, dim=0)  # (B_eff, D)
        
        y_curr_flat = None
        if batch_y_curr:
            y_curr_flat = torch.cat(batch_y_curr, dim=0)  # (B_eff, D_obs)
        
        # Predict next states (deterministic)
        with torch.no_grad():
            x_next_pred = model.sample(x_prev_stack, y_curr_flat)
            
        # Calculate Metrics and Update
        batch_rmse = []
        
        for k, idx in enumerate(valid_mask):
            tid = traj_idxs[idx]
            t = time_idxs[idx]
            
            # Generated prediction (Scaled)
            x_gen = x_next_pred[k:k+1]  # (1, D)
            
            # Ground truth (Physical)
            x_true = x_curr_gt[idx]  # (D,)
            
            # Update current state for autoregressive generation
            current_states[tid] = x_gen
            
            # Metric calculation (in original space)
            x_gen_orig = system.postprocess(x_gen.squeeze(0))  # (D,)
            x_true_orig = x_true  # Already physical
            
            # RMSE
            error = x_gen_orig - x_true_orig
            rmse = torch.sqrt(torch.mean(error**2)).item()
            batch_rmse.append(rmse)
            
            # Store
            has_obs = batch["has_observation"][idx].item()
            obs = batch["y_curr"][idx].cpu().numpy() if (has_obs and "y_curr" in batch) else None
            
            trajectory_results[tid]["trajectory_data"].append({
                "time_idx": t,
                "x_true": x_true_orig.cpu().numpy(),
                "x_est": x_gen_orig.cpu().numpy(),
                "observation": obs,
                "has_observation": has_obs,
                "rmse": rmse,
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
                
                if state_dim == 3:
                    fig = plot_lorenz63_trajectory(fig_res, config)
                else:
                    fig = plot_trajectory_comparison(fig_res, config, state_dim)
                    
                if fig:
                    wandb_run.log({f"eval/trajectory_{tid}": wandb.Image(fig)})
                    plt.close(fig)

    global_mean_rmse = total_rmse / total_steps if total_steps > 0 else 0.0
    
    logger.info(f"Global Mean RMSE (Autoregressive): {global_mean_rmse:.4f}")
    
    if wandb_run:
        wandb_run.log({
            "eval/global_mean_rmse": global_mean_rmse,
        })
        
        # Log metrics over time
        max_t = 0
        for res in trajectory_results.values():
            if res["trajectory_data"]:
                max_t = max(max_t, res["trajectory_data"][-1]["time_idx"])
        
        for t in range(1, max_t + 1):
            rmses_at_t = []
            for res in trajectory_results.values():
                for d in res["trajectory_data"]:
                    if d["time_idx"] == t:
                        rmses_at_t.append(d["rmse"])
                        break
            
            if rmses_at_t:
                mean_rmse_t = sum(rmses_at_t) / len(rmses_at_t)
                wandb_run.log({
                    "eval/mean_rmse_over_time": mean_rmse_t, 
                    "time_step": t
                })

    return global_mean_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--n-trajectories", type=int, default=None)
    parser.add_argument("--n-vis-trajectories", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    run = None
    if args.wandb:
        run_name = _derive_eval_run_name(args.checkpoint)
        run = wandb.init(project="deterministic-proposal-eval", name=run_name, entity="ml-climate")
        
    run_deterministic_eval(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        n_trajectories=args.n_trajectories,
        n_vis_trajectories=args.n_vis_trajectories,
        batch_size=args.batch_size,
        device=args.device,
        wandb_run=run
    )

