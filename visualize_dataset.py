#!/usr/bin/env python3
"""
Visualization script for sanity-checking generated datasets.
Creates 3D trajectory plots for both Lorenz 63 and Lorenz 96 systems.
For Lorenz 96 (high-dimensional), randomly selects 3 dimensions for visualization.
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import yaml


def load_dataset(data_dir: Path):
    """Load dataset from HDF5 file."""
    data_file = data_dir / "data.h5"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    with h5py.File(data_file, "r") as f:
        # Load train trajectories by default
        trajectories = f["train/trajectories"][:]
        observations = f["train/observations"][:]
        obs_mask = f["obs_mask"][:]
    
    return trajectories, observations, obs_mask


def load_config(data_dir: Path):
    """Load config from YAML file."""
    config_file = data_dir / "config.yaml"
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    return None


def plot_trajectories_3d(
    trajectories: np.ndarray,
    dims: tuple,
    title: str = "3D Trajectory Plot",
    n_trajectories: int = 5,
    figsize: tuple = (12, 10),
    cmap: str = "viridis",
    save_path: Path = None,
):
    """
    Plot trajectories in 3D space.
    
    Args:
        trajectories: Array of shape (n_traj, n_steps, state_dim)
        dims: Tuple of 3 dimension indices to plot
        title: Plot title
        n_trajectories: Number of trajectories to plot
        figsize: Figure size
        cmap: Colormap for time progression
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, min(n_trajectories, 10)))
    
    dim_x, dim_y, dim_z = dims
    
    for i in range(min(n_trajectories, trajectories.shape[0])):
        traj = trajectories[i]
        x = traj[:, dim_x]
        y = traj[:, dim_y]
        z = traj[:, dim_z]
        
        # Plot trajectory line
        ax.plot(x, y, z, alpha=0.7, linewidth=0.8, color=colors[i % 10], label=f"Traj {i}")
        
        # Mark start and end points
        ax.scatter(x[0], y[0], z[0], marker='o', s=50, color=colors[i % 10], edgecolors='black')
        ax.scatter(x[-1], y[-1], z[-1], marker='s', s=50, color=colors[i % 10], edgecolors='black')
    
    ax.set_xlabel(f"Dim {dim_x}", fontsize=12)
    ax.set_ylabel(f"Dim {dim_y}", fontsize=12)
    ax.set_zlabel(f"Dim {dim_z}", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    if n_trajectories <= 10:
        ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D plot to {save_path}")
    
    return fig, ax


def plot_time_series(
    trajectories: np.ndarray,
    dims: tuple,
    dt: float = 0.01,
    n_trajectories: int = 3,
    figsize: tuple = (14, 8),
    save_path: Path = None,
):
    """
    Plot time series for selected dimensions.
    
    Args:
        trajectories: Array of shape (n_traj, n_steps, state_dim)
        dims: Tuple of dimension indices to plot
        dt: Time step
        n_trajectories: Number of trajectories to plot
        figsize: Figure size
        save_path: Path to save the figure
    """
    n_dims = len(dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=figsize, sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, min(n_trajectories, 10)))
    
    n_steps = trajectories.shape[1]
    t = np.arange(n_steps) * dt
    
    for ax_idx, dim in enumerate(dims):
        ax = axes[ax_idx]
        for i in range(min(n_trajectories, trajectories.shape[0])):
            ax.plot(t, trajectories[i, :, dim], alpha=0.8, linewidth=1, 
                   color=colors[i % 10], label=f"Traj {i}")
        ax.set_ylabel(f"Dim {dim}", fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel("Time", fontsize=12)
    fig.suptitle("Time Series of Selected Dimensions", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved time series to {save_path}")
    
    return fig, axes


def plot_observation_vs_state(
    trajectories: np.ndarray,
    observations: np.ndarray,
    obs_mask: np.ndarray,
    obs_components: list,
    dt: float = 0.01,
    traj_idx: int = 0,
    figsize: tuple = (14, 6),
    save_path: Path = None,
):
    """
    Plot observed vs true state for a single trajectory.
    """
    n_obs_dims = min(3, observations.shape[-1])
    fig, axes = plt.subplots(n_obs_dims, 1, figsize=figsize, sharex=True)
    if n_obs_dims == 1:
        axes = [axes]
    
    n_steps = trajectories.shape[1]
    t = np.arange(n_steps) * dt
    obs_time_indices = np.where(obs_mask)[0]
    t_obs = obs_time_indices * dt
    
    for ax_idx in range(n_obs_dims):
        ax = axes[ax_idx]
        state_dim = obs_components[ax_idx] if ax_idx < len(obs_components) else ax_idx
        
        # True state (arctan applied)
        true_state = np.arctan(trajectories[traj_idx, :, state_dim])
        ax.plot(t, true_state, 'b-', linewidth=1.5, label='arctan(True State)', alpha=0.8)
        
        # Observations
        ax.scatter(t_obs, observations[traj_idx, :, ax_idx], 
                  c='red', s=20, alpha=0.7, label='Observations', zorder=5)
        
        ax.set_ylabel(f"Obs Dim {ax_idx}\n(State {state_dim})", fontsize=10)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel("Time", fontsize=12)
    fig.suptitle(f"Observations vs True State (Trajectory {traj_idx})", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved observation plot to {save_path}")
    
    return fig, axes


def plot_dataset_summary(
    trajectories: np.ndarray,
    observations: np.ndarray,
    obs_mask: np.ndarray,
    dims: tuple,
    config: dict = None,
    dt: float = 0.01,
    save_dir: Path = None,
):
    """Create a comprehensive summary visualization."""
    n_traj, n_steps, state_dim = trajectories.shape
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    dim_x, dim_y, dim_z = dims
    colors = plt.cm.viridis(np.linspace(0, 1, min(5, n_traj)))
    
    for i in range(min(5, n_traj)):
        traj = trajectories[i]
        ax1.plot(traj[:, dim_x], traj[:, dim_y], traj[:, dim_z], 
                alpha=0.7, linewidth=0.8, color=colors[i])
    
    ax1.set_xlabel(f"Dim {dim_x}")
    ax1.set_ylabel(f"Dim {dim_y}")
    ax1.set_zlabel(f"Dim {dim_z}")
    ax1.set_title("3D Trajectory (5 samples)")
    
    # Time series for selected dims
    ax2 = fig.add_subplot(2, 2, 2)
    t = np.arange(n_steps) * dt
    for dim in dims:
        ax2.plot(t, trajectories[0, :, dim], label=f"Dim {dim}", alpha=0.8)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("State")
    ax2.set_title("Time Series (Trajectory 0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Distribution of states
    ax3 = fig.add_subplot(2, 2, 3)
    for i, dim in enumerate(dims):
        data = trajectories[:, :, dim].flatten()
        ax3.hist(data, bins=50, alpha=0.5, label=f"Dim {dim}", density=True)
    ax3.set_xlabel("State Value")
    ax3.set_ylabel("Density")
    ax3.set_title("State Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Observation info
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Build info text
    info_lines = [
        "Dataset Summary",
        "=" * 40,
        f"Number of trajectories: {n_traj}",
        f"Trajectory length: {n_steps}",
        f"State dimension: {state_dim}",
        f"Observation dimension: {observations.shape[-1]}",
        f"Observation frequency: {np.sum(obs_mask)} / {len(obs_mask)} steps",
        f"Time step (dt): {dt}",
        f"Visualized dimensions: {dims}",
    ]
    
    if config:
        info_lines.append("")
        info_lines.append("Config Parameters:")
        info_lines.append("-" * 40)
        info_lines.append(f"  obs_noise_std: {config.get('obs_noise_std', 'N/A')}")
        info_lines.append(f"  obs_frequency: {config.get('obs_frequency', 'N/A')}")
        info_lines.append(f"  warmup_steps: {config.get('warmup_steps', 'N/A')}")
        if 'system_params' in config:
            info_lines.append(f"  system_params: {config['system_params']}")
    
    info_text = "\n".join(info_lines)
    ax4.text(0.1, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / "dataset_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary to {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize generated dataset")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to dataset directory containing data.h5",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default=None,
        help="Comma-separated list of 3 dimension indices to visualize (default: random for high-dim)",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=5,
        help="Number of trajectories to plot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dimension selection",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots to dataset directory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (useful for headless environments)",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    print("=" * 60)
    print("Dataset Visualization")
    print("=" * 60)
    print(f"Loading data from: {data_dir}")
    
    # Load data
    trajectories, observations, obs_mask = load_dataset(data_dir)
    config = load_config(data_dir)
    
    n_traj, n_steps, state_dim = trajectories.shape
    print(f"\nDataset shape:")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Observations: {observations.shape}")
    print(f"  State dimension: {state_dim}")
    
    # Determine dimensions to visualize
    if args.dims:
        dims = tuple(int(d) for d in args.dims.split(","))
        if len(dims) != 3:
            raise ValueError("Exactly 3 dimensions must be specified for 3D visualization")
    else:
        if state_dim <= 3:
            dims = tuple(range(state_dim))
            if state_dim < 3:
                dims = tuple(list(range(state_dim)) + [0] * (3 - state_dim))
        else:
            # Randomly select 3 dimensions for high-dimensional systems
            np.random.seed(args.seed)
            dims = tuple(sorted(np.random.choice(state_dim, size=3, replace=False)))
            print(f"\nRandomly selected dimensions for visualization: {dims}")
    
    print(f"Visualizing dimensions: {dims}")
    
    # Get dt from config
    dt = config.get("dt", 0.01) if config else 0.01
    
    # Create plots
    save_dir = data_dir if args.save else None
    
    # 1. Summary plot
    plot_dataset_summary(
        trajectories, observations, obs_mask, dims,
        config=config, dt=dt, save_dir=save_dir
    )
    
    # 2. Detailed 3D plot
    fig_3d, _ = plot_trajectories_3d(
        trajectories, dims,
        title=f"3D Trajectory Plot (Dims {dims})",
        n_trajectories=args.n_trajectories,
        save_path=save_dir / "trajectories_3d.png" if save_dir else None
    )
    
    # 3. Time series
    fig_ts, _ = plot_time_series(
        trajectories, dims, dt=dt,
        n_trajectories=min(3, args.n_trajectories),
        save_path=save_dir / "time_series.png" if save_dir else None
    )
    
    # 4. Observation comparison
    obs_components = config.get("obs_components", list(range(observations.shape[-1]))) if config else []
    fig_obs, _ = plot_observation_vs_state(
        trajectories, observations, obs_mask, obs_components,
        dt=dt, traj_idx=0,
        save_path=save_dir / "observations.png" if save_dir else None
    )
    
    if not args.no_show:
        plt.show()
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

