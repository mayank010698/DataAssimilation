import torch
import numpy as np
import logging
from typing import Dict, Any, Optional
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.base_pf import FilteringMethod
from data import (
    DataAssimilationConfig,
    Lorenz63,
    DataAssimilationDataModule,
    create_projection_matrix,
)
from models.bpf import BootstrapParticleFilter
from models.proposals import TransitionProposal, RectifiedFlowProposal
from pathlib import Path


def find_rf_checkpoint(config_name: str) -> Optional[str]:
    """
    Find the best RF checkpoint for a given configuration.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Path to best checkpoint if found, None otherwise
    """
    # Map configuration names to checkpoint directory patterns
    # This assumes checkpoints are organized by configuration
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
    # Actual format from PyTorch Lightning: rf-epoch=365-val_loss=0.000481.ckpt
    checkpoints = list(checkpoint_dir.glob("rf-*.ckpt"))
    if not checkpoints:
        # Try recursive search
        checkpoints = list(checkpoint_dir.glob("**/rf-*.ckpt"))
    
    if not checkpoints:
        return None
    
    # Filter out "last" checkpoint and parse validation loss from filename
    def get_val_loss(path):
        try:
            # Filename format: rf-epoch={epoch}-val_loss={val_loss}.ckpt
            # Example: rf-epoch=365-val_loss=0.000481.ckpt
            stem = path.stem  # e.g., "rf-epoch=365-val_loss=0.000481"
            
            # Skip "last" checkpoint
            if "last" in stem.lower():
                return float('inf')
            
            # Extract val_loss from pattern: rf-epoch=*-val_loss={val_loss}
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
    return str(valid_checkpoints[0])  # Best checkpoint (lowest loss)


def run_particle_filter(
    pf: FilteringMethod,
    dataloader,
    trajectory_idx: int,
    collect_trajectory_data: bool = True,
) -> Dict[str, Any]:
    """Run filtering method and collect metrics and trajectory data"""
    trajectory_metrics = []
    trajectory_data = []

    print(f"Running particle filter on trajectory {trajectory_idx}...")

    for batch_idx, batch in enumerate(dataloader):
        batch_traj_idx = batch["trajectory_idx"].item()
        if batch_traj_idx != trajectory_idx:
            continue

        x_prev = batch["x_prev"].squeeze(0)
        x_curr = batch["x_curr"].squeeze(0)
        y_curr = batch["y_curr"].squeeze(0) if batch["has_observation"].item() else None
        time_idx = batch["time_idx"].item()

        metrics = pf.test_step(batch, 0)
        trajectory_metrics.append(metrics)

        if collect_trajectory_data:
            trajectory_data.append(
                {
                    "time_idx": time_idx,
                    "x_true": x_curr.cpu().numpy(),
                    "x_est": metrics["x_est"],
                    "observation": y_curr.cpu().numpy() if y_curr is not None else None,
                    "has_observation": batch["has_observation"].item(),
                    "rmse": metrics["rmse"],
                }
            )

        if batch_idx % 1 == 0:
            print(f"  Step {batch_idx}: RMSE = {metrics['rmse']:.4f}")

    rmse_values = [m["rmse"] for m in trajectory_metrics]
    ll_values = [
        m["log_likelihood"] for m in trajectory_metrics if m["log_likelihood"] != 0.0
    ]

    if collect_trajectory_data:
        trajectory_data.sort(key=lambda x: x["time_idx"])

    result = {
        "trajectory_idx": trajectory_idx,
        "mean_rmse": np.mean(rmse_values),
        "std_rmse": np.std(rmse_values),
        "mean_log_likelihood": np.mean(ll_values) if ll_values else 0.0,
        "total_log_likelihood": np.sum(ll_values) if ll_values else 0.0,
        "n_observations": len(ll_values),
        "n_steps": len(rmse_values),
        "metrics": trajectory_metrics,
        "trajectory_data": trajectory_data if collect_trajectory_data else None,
    }

    print(f"Trajectory {trajectory_idx} completed:")
    print(f"  Mean RMSE: {result['mean_rmse']:.4f} ± {result['std_rmse']:.4f}")
    print(f"  Total log-likelihood: {result['total_log_likelihood']:.2f}")
    print(f"  Observations processed: {result['n_observations']}")

    return result


def plot_trajectory_comparison(
    result: Dict[str, Any],
    config: DataAssimilationConfig,
    system: Lorenz63,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot trajectory comparison with correct space handling"""

    trajectory_data = result["trajectory_data"]
    traj_idx = result["trajectory_idx"]

    if trajectory_data is None:
        raise ValueError("No trajectory data available for plotting!")

    time_steps = np.array([d["time_idx"] for d in trajectory_data])
    x_true = np.array([d["x_true"] for d in trajectory_data])
    x_est = np.array([d["x_est"] for d in trajectory_data])
    rmse_values = np.array([d["rmse"] for d in trajectory_data])

    if config.use_preprocessing:
        print(
            "Converting normalized trajectories back to original space for plotting..."
        )
        x_true_orig = np.array([system.postprocess(xt) for xt in x_true])
        x_est_orig = np.array([system.postprocess(xe) for xe in x_est])
        space_label = " (Original Space)"
    else:
        x_true_orig = x_true
        x_est_orig = x_est
        space_label = ""

    obs_times = []
    obs_values = []
    obs_true_x = []

    for d in trajectory_data:
        if d["has_observation"] and d["observation"] is not None:
            obs_times.append(d["time_idx"])
            obs_values.append(d["observation"])
            obs_true_x.append(x_true_orig[d["time_idx"]][0])

    obs_times = np.array(obs_times)
    if len(obs_values) > 0:
        obs_values = np.array(obs_values)
        obs_true_x = np.array(obs_true_x)

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(3, 2, 1, projection="3d")
    ax1.plot(
        x_true_orig[:, 0],
        x_true_orig[:, 1],
        x_true_orig[:, 2],
        "b-",
        alpha=0.8,
        linewidth=2,
        label="True trajectory",
    )
    ax1.plot(
        x_est_orig[:, 0],
        x_est_orig[:, 1],
        x_est_orig[:, 2],
        "r--",
        alpha=0.8,
        linewidth=2,
        label="PF estimate",
    )

    if len(obs_times) > 0:
        valid_obs_mask = obs_times < len(x_true_orig)
        valid_obs_times = obs_times[valid_obs_mask]
        if len(valid_obs_times) > 0:
            ax1.scatter(
                x_true_orig[valid_obs_times, 0],
                x_true_orig[valid_obs_times, 1],
                x_true_orig[valid_obs_times, 2],
                c="red",
                s=30,
                alpha=0.8,
                label="Observations",
            )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"3D Lorenz63 Trajectory{space_label} (Trajectory {traj_idx})")
    ax1.legend()

    state_names = ["X", "Y", "Z"]
    colors = ["blue", "green", "orange"]

    for i, (name, color) in enumerate(zip(state_names, colors)):
        ax = fig.add_subplot(3, 2, i + 2)

        ax.plot(
            time_steps,
            x_true_orig[:, i],
            color=color,
            alpha=0.8,
            linewidth=2,
            label=f"{name} (true)",
        )
        ax.plot(
            time_steps,
            x_est_orig[:, i],
            color="red",
            alpha=0.8,
            linestyle="--",
            linewidth=2,
            label=f"{name} (PF estimate)",
        )

        if i in config.obs_components and len(obs_times) > 0:
            valid_obs_mask = obs_times < len(time_steps)
            valid_obs_times = obs_times[valid_obs_mask]

            if len(valid_obs_times) > 0:
                obs_component_idx = config.obs_components.index(i)

                ax.scatter(
                    valid_obs_times,
                    x_true_orig[valid_obs_times, i],
                    color="red",
                    s=30,
                    alpha=0.8,
                    label="Observation times",
                    zorder=5,
                )

                if (
                    len(obs_values) > 0
                    and obs_values.ndim > 1
                    and obs_values.shape[1] > obs_component_idx
                ):
                    ax2 = ax.twinx()
                    ax2.scatter(
                        valid_obs_times,
                        obs_values[valid_obs_mask, obs_component_idx],
                        color="purple",
                        s=20,
                        alpha=0.6,
                        marker="x",
                        label="Observed values",
                    )
                    obs_label = (
                        "Observed (normalized)"
                        if config.use_preprocessing
                        else "Observed values"
                    )
                    ax2.set_ylabel(obs_label, color="purple")
                    ax2.tick_params(axis="y", labelcolor="purple")

        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if i == 0:
            ax.set_title(
                f"State Components vs Time{space_label} (Trajectory {traj_idx})"
            )
        if i == 2:
            ax.set_xlabel("Time Steps")

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(time_steps, rmse_values, "purple", linewidth=2, alpha=0.8)
    ax5.axhline(
        y=np.mean(rmse_values),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean RMSE: {np.mean(rmse_values):.4f}",
    )
    ax5.set_xlabel("Time Steps")
    ax5.set_ylabel("RMSE")
    ax5.set_title("Root Mean Square Error Over Time")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6 = fig.add_subplot(3, 2, 6)
    errors = x_est - x_true
    for i, (name, color) in enumerate(zip(state_names, colors)):
        ax6.hist(errors[:, i], bins=30, alpha=0.6, color=color, label=f"{name} error")
    ax6.set_xlabel("Estimation Error")
    ax6.set_ylabel("Frequency")
    title = "Error Distribution by State Component"
    if config.use_preprocessing:
        title += " (Normalized Space)"
    ax6.set_title(title)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()

    return fig


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 80)
    print("Bootstrap Particle Filter for Lorenz63")
    print("=" * 80)

    configurations = [
        {
            "name": "Linear Projection (Preprocessing ON) - TransitionProposal",
            "obs_components": [0, 2],
            "observation_operator": create_projection_matrix(3, [0, 2]),
            "use_preprocessing": True,
            "obs_dim": 2,
            "proposal_type": "transition",
        },
        {
            "name": "Linear Projection (Preprocessing ON) - RectifiedFlowProposal",
            "obs_components": [0, 2],
            "observation_operator": create_projection_matrix(3, [0, 2]),
            "use_preprocessing": True,
            "obs_dim": 2,
            "proposal_type": "rf",
        },
        {
            "name": "Linear Projection (Preprocessing OFF) - TransitionProposal",
            "obs_components": [0, 2],
            "observation_operator": create_projection_matrix(3, [0, 2]),
            "use_preprocessing": False,
            "obs_dim": 2,
            "proposal_type": "transition",
        },
        {
            "name": "Linear Projection (Preprocessing OFF) - RectifiedFlowProposal",
            "obs_components": [0, 2],
            "observation_operator": create_projection_matrix(3, [0, 2]),
            "use_preprocessing": False,
            "obs_dim": 2,
            "proposal_type": "rf",
        },
        {
            "name": "Arctan Nonlinear (Preprocessing OFF) - TransitionProposal",
            "obs_components": [0],
            "observation_operator": np.arctan,
            "use_preprocessing": False,
            "obs_dim": 1,
            "proposal_type": "transition",
        },
        {
            "name": "Arctan Nonlinear (Preprocessing OFF) - RectifiedFlowProposal",
            "obs_components": [0],
            "observation_operator": np.arctan,
            "use_preprocessing": False,
            "obs_dim": 1,
            "proposal_type": "rf",
        },
        {
            "name": "Arctan Nonlinear (Preprocessing ON) - TransitionProposal",
            "obs_components": [0],
            "observation_operator": np.arctan,
            "use_preprocessing": True,
            "obs_dim": 1,
            "proposal_type": "transition",
        },
        {
            "name": "Arctan Nonlinear (Preprocessing ON) - RectifiedFlowProposal",
            "obs_components": [0],
            "observation_operator": np.arctan,
            "use_preprocessing": True,
            "obs_dim": 1,
            "proposal_type": "rf",
        },
    ]

    for config_dict in configurations:
        print(f"\n{'='*60}")
        print(f"Testing: {config_dict['name']}")
        print(f"{'='*60}")

        # Extract base config name for checkpoint lookup (remove proposal type suffix)
        base_config_name = config_dict["name"].rsplit(" - ", 1)[0]

        config = DataAssimilationConfig(
            num_trajectories=50,
            len_trajectory=100,  # small for testing
            warmup_steps=1024,
            dt=0.01,
            obs_noise_std=0.25,
            obs_frequency=2,
            obs_components=config_dict["obs_components"],
            observation_operator=config_dict["observation_operator"],
            system_params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
            use_preprocessing=config_dict["use_preprocessing"],
        )

        print("Configuration:")
        print(f"  dt: {config.dt}")
        print(f"  obs_noise_std: {config.obs_noise_std}")
        print(f"  obs_frequency: {config.obs_frequency}")
        print(f"  obs_components: {config.obs_components}")
        print(f"  use_preprocessing: {config.use_preprocessing}")
        print(f"  Proposal type: {config_dict['proposal_type']}")

        system = Lorenz63(config)
        data_dir = f"./lorenz63_data_{base_config_name.lower().replace(' ', '_')}_15step"
        data_module = DataAssimilationDataModule(
            config=config,
            system_class=Lorenz63,
            data_dir=data_dir,
            batch_size=1,
        )

        print("\nGenerating/loading data...")
        data_module.prepare_data()
        data_module.setup("test")
        test_loader = data_module.test_dataloader()

        print(f"Test dataset size: {len(test_loader.dataset)}")

        print(f"\nInitializing Bootstrap Particle Filter...")
        
        # Choose proposal distribution based on proposal type
        if config_dict["proposal_type"] == "rf":
            # Try to find RF checkpoint for this configuration
            rf_checkpoint = find_rf_checkpoint(base_config_name)
            
            if rf_checkpoint and os.path.exists(rf_checkpoint):
                print(f"Using Rectified Flow proposal from: {rf_checkpoint}")
                proposal = RectifiedFlowProposal(rf_checkpoint, device="cpu")
            else:
                print(f"Warning: RF checkpoint not found for {base_config_name}")
                print(f"  Skipping this experiment (RF checkpoint required)")
                continue
        else:
            # Use standard transition proposal
            proposal = TransitionProposal(system, process_noise_std=0.25)
        
        pf = BootstrapParticleFilter(
            system=system,
            proposal_distribution=proposal,
            n_particles=100,
            state_dim=3,
            obs_dim=config_dict["obs_dim"],
            process_noise_std=0.25,
        )

        print(f"\nRunning particle filter...")
        result = run_particle_filter(
            pf, test_loader, trajectory_idx=0, collect_trajectory_data=True
        )

        print(f"\nResults for {config_dict['name']}:")
        print(f"  Mean RMSE: {result['mean_rmse']:.4f} ± {result['std_rmse']:.4f}")
        print(f"  Total log-likelihood: {result['total_log_likelihood']:.2f}")
        print(f"  Number of observations: {result['n_observations']}")
        print(f"  Number of time steps: {result['n_steps']}")

        print(f"\nCreating visualization...")
        safe_name = (
            config_dict["name"]
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        save_path = f"bpf_results_{safe_name}.png"
        fig = plot_trajectory_comparison(result, config, system, save_path=save_path)

    print("\n" + "=" * 80)
    print("All experiments completed")
    print("=" * 80)


if __name__ == "__main__":
    main()
