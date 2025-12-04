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
from scipy.stats import norm
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    Lorenz63,
    TimeAlignedBatchSampler,
    load_config_yaml,
)
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
    if ensemble.ndim == 1:
        ensemble = ensemble[:, None]
    if truth.ndim == 0:
        truth = truth[None]
        
    K, D = ensemble.shape
    crps_sum = 0.0
    
    for d in range(D):
        ens_d = np.sort(ensemble[:, d])
        truth_d = truth[d]
        
        mae = np.mean(np.abs(ens_d - truth_d))
        
        diffs = np.abs(ens_d[:, None] - ens_d[None, :])
        dispersion = np.sum(diffs) / (2 * K * K)
        
        crps_sum += (mae - dispersion)
        
    return crps_sum / D

def plot_distribution_comparison(
    traj_idx: int,
    time_idx: int,
    x_true_prev: np.ndarray,
    x_true_curr: np.ndarray,
    rf_samples: np.ndarray,
    trans_samples: np.ndarray,
    state_dim: int = 3
) -> plt.Figure:
    """
    Plot comparison of RF and Transition distributions for a single step.
    
    Args:
        traj_idx: Trajectory ID
        time_idx: Time step
        x_true_prev: Previous true state (D,)
        x_true_curr: Current true state (D,)
        rf_samples: Samples from RF proposal (N, D)
        trans_samples: Samples from Transition proposal (N, D)
        state_dim: Dimension of state space
    """
    fig, axes = plt.subplots(1, state_dim, figsize=(18, 5))
    state_names = ["X", "Y", "Z"]
    
    for d in range(state_dim):
        ax = axes[d]
        
        # Data for this dimension
        rf_data = rf_samples[:, d]
        trans_data = trans_samples[:, d]
        true_val = x_true_curr[d]
        
        # Compute statistics
        rf_mean, rf_std = np.mean(rf_data), np.std(rf_data)
        trans_mean, trans_std = np.mean(trans_data), np.std(trans_data)

        print(f"RF Mean: {rf_mean}, RF Std: {rf_std}")
        print(f"Trans Mean: {trans_mean}, Trans Std: {trans_std}")
        
        # Plot histograms
        # ax.hist(rf_data, bins=20, density=True, alpha=0.5, color='blue', label='RF Samples')
        # ax.hist(trans_data, bins=20, density=True, alpha=0.5, color='green', label='Trans Samples')
        
        # Plot fitted Gaussians
        x_range = np.linspace(
            min(rf_data.min(), trans_data.min(), true_val) - 1,
            max(rf_data.max(), trans_data.max(), true_val) + 1,
            100
        )
        ax.plot(x_range, norm.pdf(x_range, rf_mean, rf_std), 'b-', lw=2, label=f'RF Fit (std={rf_std:.3f})')
        ax.plot(x_range, norm.pdf(x_range, trans_mean, trans_std), 'g-', lw=2, label=f'Trans Fit (std={trans_std:.3f})')
        
        # Plot Truth
        ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label='Truth')
        
        ax.set_title(f"Dimension {state_names[d]} at t={time_idx}")
        ax.legend()
        
    plt.suptitle(f"Distribution Comparison - Trajectory {traj_idx}, Time {time_idx}")
    plt.tight_layout()
    return fig

def run_comparison_eval(
    checkpoint_path: str,
    data_dir: str,
    process_noise_std: float = 0.01,
    n_trajectories: int = 10,
    batch_size: int = 32,
    dist_check_steps: List[int] = [25, 50, 100],
    n_dist_samples: int = 100,
    device: str = "cuda",
    wandb_run = None,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Running Proposal Distribution Comparison")
    logger.info(f"RF Checkpoint: {checkpoint_path}")
    logger.info(f"Transition Noise Std: {process_noise_std}")
    
    # Load Config
    data_path = Path(data_dir)
    config_path = data_path / "config.yaml"
    config = load_config_yaml(config_path)
    
    # Initialize Data Module
    data_module = DataAssimilationDataModule(
        config=config,
        system_class=Lorenz63,
        data_dir=str(data_dir),
        batch_size=batch_size,
    )
    data_module.setup("test")
    test_dataset = data_module.test_dataset
    system = data_module.system # Has stats
    
    # Load Observation Scalers for RF Wrapper if available
    obs_mean = None
    obs_std = None
    import h5py
    data_scaled_path = data_path / "data_scaled.h5"
    if data_scaled_path.exists():
        with h5py.File(data_scaled_path, "r") as f:
            if "obs_scaler_mean" in f:
                obs_mean = torch.from_numpy(f["obs_scaler_mean"][:]).float()
                obs_std = torch.from_numpy(f["obs_scaler_std"][:]).float()
                logger.info(f"Loaded observation scalers from {data_scaled_path}")

    # Initialize RF Proposal
    rf_proposal = RectifiedFlowProposal(
        checkpoint_path=checkpoint_path,
        device=device,
        system=system,
        obs_mean=obs_mean,
        obs_std=obs_std
    )
    
    # Initialize Transition Proposal
    trans_proposal = TransitionProposal(
        system=system,
        process_noise_std=process_noise_std
    )
    
    # Setup Dataloader
    traj_len_inference = config.len_trajectory - 1
    total_trajs = test_dataset.n_trajectories
    n_trajectories = min(n_trajectories, total_trajs)
    
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
    
    logger.info(f"Evaluating on {n_trajectories} trajectories")
    
    # Autoregressive States
    # Maps trajectory_idx -> current_state (1, D) - using 1 sample for autoregressive path
    current_states_rf = {}
    current_states_trans = {}
    
    # Results storage
    results = {
        "rf": {"rmse_sum": 0.0, "crps_sum": 0.0, "count": 0},
        "trans": {"rmse_sum": 0.0, "crps_sum": 0.0, "count": 0}
    }
    
    # Set of trajectories to visualize distributions for (first 5)
    vis_traj_indices = set(range(min(5, n_trajectories)))
    
    pbar = tqdm(test_loader, desc="Comparing")
    
    for batch in pbar:
        # Get Ground Truth Data (Unscaled / Physical)
        x_curr_gt = batch["x_curr"].to(device) # (B, D)
        x_prev_gt = batch["x_prev"].to(device) # (B, D) - for distribution check
        
        has_obs = batch["has_observation"]
        y_curr = batch["y_curr"].to(device) if has_obs.any() else None
        
        traj_idxs = batch["trajectory_idx"].squeeze(-1).tolist()
        time_idxs = batch["time_idx"].squeeze(-1).tolist()
        
        # Prepare Inputs for Autoregressive Step
        batch_x_prev_rf = []
        batch_x_prev_trans = []
        batch_y_curr = []
        valid_indices = [] # Indices in current batch
        
        for i, tid in enumerate(traj_idxs):
            t = time_idxs[i]
            
            # --- 1. Distribution Check (at specific steps, using GT previous state) ---
            if tid in vis_traj_indices and t in dist_check_steps:
                # Sample N points from both proposals starting from TRUE x_{t-1}
                x_prev_single = x_prev_gt[i].unsqueeze(0).repeat(n_dist_samples, 1) # (N, D)
                y_curr_single = y_curr[i].unsqueeze(0).repeat(n_dist_samples, 1) if y_curr is not None and has_obs[i] else None
                
                dt = config.dt
                
                # Sample RF
                with torch.no_grad():
                    rf_samples = rf_proposal.sample(x_prev_single, y_curr_single, dt)
                
                # Sample Trans
                with torch.no_grad():
                    trans_samples = trans_proposal.sample(x_prev_single, y_curr_single, dt)
                
                # Plot
                fig = plot_distribution_comparison(
                    tid, t,
                    x_prev_gt[i].cpu().numpy(),
                    x_curr_gt[i].cpu().numpy(),
                    rf_samples.cpu().numpy(),
                    trans_samples.cpu().numpy(),
                    state_dim=system.state_dim
                )
                
                if wandb_run:
                    wandb_run.log({f"dist_check/traj_{tid}_t_{t}": wandb.Image(fig)})
                plt.close(fig)

            # --- 2. Autoregressive Step Preparation ---
            if t == 1:
                # Initialize with x0 from GT
                x0 = batch["x_prev"][i].to(device).unsqueeze(0) # (1, D)
                current_states_rf[tid] = x0
                current_states_trans[tid] = x0.clone()
                
            if tid in current_states_rf:
                batch_x_prev_rf.append(current_states_rf[tid])
                batch_x_prev_trans.append(current_states_trans[tid])
                
                if y_curr is not None and has_obs[i]:
                    batch_y_curr.append(y_curr[i].unsqueeze(0)) # (1, obs_dim)
                else:
                    # Handle case where y_curr might be needed by structure but is None/Not available
                    # For batching, we might need a placeholder or handle separately.
                    # But our proposals handle batch processing. 
                    # If y_curr is None for the whole batch, it's fine.
                    # If mixed, it's complicated. 
                    # Assumption: In this dataset, obs are synchronous across trajectories.
                    pass
                
                valid_indices.append(i)

        if not batch_x_prev_rf:
            continue
            
        # Stack inputs
        # (B_eff, 1, D) -> (B_eff, D)
        x_prev_rf_stack = torch.stack(batch_x_prev_rf).squeeze(1)
        x_prev_trans_stack = torch.stack(batch_x_prev_trans).squeeze(1)
        
        y_curr_stack = None
        if batch_y_curr:
             y_curr_stack = torch.stack(batch_y_curr).squeeze(1)
             
        dt = config.dt

        # Sample Next Steps
        with torch.no_grad():
            x_next_rf = rf_proposal.sample(x_prev_rf_stack, y_curr_stack, dt)
            x_next_trans = trans_proposal.sample(x_prev_trans_stack, y_curr_stack, dt)
            
        # Update States and Calc Metrics
        rf_rmse_batch = []
        trans_rmse_batch = []
        
        for k, idx in enumerate(valid_indices):
            tid = traj_idxs[idx]
            
            # Update Autoregressive States
            current_states_rf[tid] = x_next_rf[k].unsqueeze(0)
            current_states_trans[tid] = x_next_trans[k].unsqueeze(0)
            
            # Ground Truth
            x_true = x_curr_gt[idx]
            
            # Calc Metrics (RF)
            rf_err = x_next_rf[k] - x_true
            rf_rmse = torch.sqrt(torch.mean(rf_err**2)).item()
            results["rf"]["rmse_sum"] += rf_rmse
            results["rf"]["count"] += 1
            rf_rmse_batch.append(rf_rmse)
            
            # Calc Metrics (Trans)
            trans_err = x_next_trans[k] - x_true
            trans_rmse = torch.sqrt(torch.mean(trans_err**2)).item()
            results["trans"]["rmse_sum"] += trans_rmse
            results["trans"]["count"] += 1
            trans_rmse_batch.append(trans_rmse)
            
        # Log batch average
        if rf_rmse_batch:
            avg_rf = sum(rf_rmse_batch)/len(rf_rmse_batch)
            avg_trans = sum(trans_rmse_batch)/len(trans_rmse_batch)
            pbar.set_postfix({"RF_RMSE": f"{avg_rf:.3f}", "Trans_RMSE": f"{avg_trans:.3f}"})

    # Report Final Results
    print("\n" + "="*50)
    print("Comparison Results (Autoregressive)")
    print("="*50)
    
    rf_mean_rmse = results["rf"]["rmse_sum"] / results["rf"]["count"]
    trans_mean_rmse = results["trans"]["rmse_sum"] / results["trans"]["count"]
    
    print(f"Rectified Flow Proposal Mean RMSE: {rf_mean_rmse:.4f}")
    print(f"Transition Proposal Mean RMSE:     {trans_mean_rmse:.4f}")
    
    if wandb_run:
        wandb_run.log({
            "eval/rf_mean_rmse": rf_mean_rmse,
            "eval/trans_mean_rmse": trans_mean_rmse
        })
        
    return rf_mean_rmse, trans_mean_rmse

def main():
    parser = argparse.ArgumentParser(description="Compare RF Proposal vs Transition Dynamics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RF checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--process-noise-std", type=float, default=0.25, help="Process noise for baseline transition")
    parser.add_argument("--n-trajectories", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project="rf-proposal-dist-check",
            entity="ml-climate",
            name=f"dist-check-{Path(args.checkpoint).stem}",
            config=vars(args)
        )
        
    run_comparison_eval(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        process_noise_std=args.process_noise_std,
        n_trajectories=args.n_trajectories,
        batch_size=args.batch_size,
        device=args.device,
        wandb_run=wandb_run
    )

if __name__ == "__main__":
    main()

