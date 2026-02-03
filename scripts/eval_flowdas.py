import argparse
import logging
import os
import sys
import yaml
import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataset,
    DoubleWell,
    KuramotoSivashinsky,
    load_config_yaml,
    Lorenz63,
    Lorenz96,
)
from models.flowdas import ScoreNet, Euler_Maruyama_sampler

try:
    import wandb
except ImportError:
    wandb = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def plot_trajectory_comparison(result, config):
    """
    Plot trajectory comparison (True vs Generated).
    Adapted from proposals/eval_proposal.py
    """
    trajectory_data = result["trajectory_data"]
    traj_idx = result["trajectory_idx"]
    if not trajectory_data:
        return None

    # Extract data
    time_steps = np.array([d["time_idx"] for d in trajectory_data])
    x_true = np.array([d["x_true"] for d in trajectory_data])
    x_est = np.array([d["x_est"] for d in trajectory_data])
    rmse_values = np.array([d["rmse"] for d in trajectory_data])

    # Observations
    obs_times = []
    obs_values = []
    obs_indices = []
    
    # Map time_idx to array index
    time_to_idx = {t: i for i, t in enumerate(time_steps)}

    for d in trajectory_data:
        if d.get("has_observation") and d.get("observation") is not None:
            t = d["time_idx"]
            if t in time_to_idx:
                obs_times.append(t)
                obs_values.append(d["observation"])
                obs_indices.append(time_to_idx[t])

    state_dim = x_true.shape[1]
    obs_components = config.obs_components if config.obs_components else []

    if state_dim == 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_steps, x_true[:, 0], "b-", alpha=0.8, label="True")
        ax.plot(time_steps, x_est[:, 0], "r--", alpha=0.8, label="Generated")
        if 0 in obs_components and len(obs_times) > 0:
            try:
                obs_idx = obs_components.index(0)
                obs_vals_arr = np.array(obs_values)
                ax.scatter(obs_times, obs_vals_arr[:, obs_idx], color="red", marker="x", s=40, label="Obs")
            except ValueError:
                pass
        ax.set_title(f"1D Trajectory (ID: {traj_idx})")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("State")
        ax.legend()
        ax.grid(True)
        return fig

    fig = plt.figure(figsize=(16, 12))

    if state_dim == 3:
        # 3D Plot (using first 3 dims)
        ax1 = fig.add_subplot(3, 2, 1, projection="3d")
        ax1.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], "b-", alpha=0.8, label="True")
        ax1.plot(x_est[:, 0], x_est[:, 1], x_est[:, 2], "r--", alpha=0.8, label="Generated")
        ax1.set_title(f"3D Trajectory (ID: {traj_idx})")
        ax1.legend()

        # Component Plots (first 3)
        state_names = ["X", "Y", "Z"]
        colors = ["blue", "green", "orange"]
        
        for i, (name, color) in enumerate(zip(state_names, colors)):
            if i >= x_true.shape[1]:
                break
            
            ax = fig.add_subplot(3, 2, i + 2)
            ax.plot(time_steps, x_true[:, i], color=color, label=f"{name} (true)")
            ax.plot(time_steps, x_est[:, i], color="red", linestyle="--", label=f"{name} (gen)")
            
            is_observed = i in obs_components
            if is_observed and len(obs_times) > 0:
                try:
                    obs_idx = obs_components.index(i)
                    obs_vals_arr = np.array(obs_values)
                    ax.scatter(
                        obs_times, 
                        obs_vals_arr[:, obs_idx], 
                        color="red", marker="x", s=40, label="Obs"
                    )
                except ValueError:
                    pass

            ax.set_ylabel(name)
            ax.legend()
    else:
        # High-dimensional case: plot heatmaps and component 0 time series
        ax1 = fig.add_subplot(3, 2, 1)
        im1 = ax1.imshow(x_true.T, aspect='auto', interpolation='nearest', cmap='viridis', origin='lower')
        ax1.set_title(f"True State Heatmap (ID: {traj_idx})")
        ax1.set_ylabel("State Component")
        plt.colorbar(im1, ax=ax1)

        ax2 = fig.add_subplot(3, 2, 2)
        im2 = ax2.imshow(x_est.T, aspect='auto', interpolation='nearest', cmap='viridis', origin='lower')
        ax2.set_title("Estimated State Heatmap")
        ax2.set_ylabel("State Component")
        plt.colorbar(im2, ax=ax2)

        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(time_steps, x_true[:, 0], "b-", label="True (Dim 0)")
        ax3.plot(time_steps, x_est[:, 0], "r--", label="Gen (Dim 0)")
        if 0 in obs_components and len(obs_times) > 0:
            try:
                obs_idx = obs_components.index(0)
                obs_vals_arr = np.array(obs_values)
                ax3.scatter(obs_times, obs_vals_arr[:, obs_idx], color="red", marker="x", s=40, label="Obs")
            except ValueError:
                pass
        ax3.set_title("Component 0 Time Series")
        ax3.legend()

    # RMSE over time
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(time_steps, rmse_values, "purple")
    ax5.set_title("RMSE Over Time")
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("RMSE")

    plt.tight_layout()
    return fig

def evaluate(args):
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    # Handle both old and new checkpoint formats
    if 'config' in checkpoint:
        train_config = checkpoint['config']
    else:
        # Some older checkpoints might store args separately or not at all
        logging.warning("Config not found in checkpoint, using defaults/args")
        train_config = {}
    
    # Load Data Config
    if args.config:
        config_path = Path(args.config)
    elif args.data_dir:
        config_path = Path(args.data_dir) / "config.yaml"
    else:
        # Try to find config in checkpoint or infer
        if 'config' in train_config and isinstance(train_config['config'], str): 
             config_path = Path(train_config['config']) # original config path string
        elif 'data_dir' in train_config and train_config['data_dir']:
             config_path = Path(train_config['data_dir']) / "config.yaml"
        else:
             raise ValueError("Data config not found. Please specify --config or --data_dir")
                 
    if not config_path.exists():
         # Try looking in project root/config or dataset dir
         if (Path("config") / config_path.name).exists():
             config_path = Path("config") / config_path.name
         elif args.data_dir and (Path(args.data_dir) / "config.yaml").exists():
             config_path = Path(args.data_dir) / "config.yaml"
         
    if config_path.exists():
        logging.info(f"Loading data config from {config_path}")
        da_config = load_config_yaml(config_path)
        logging.info(f"Loaded config obs_nonlinearity: {da_config.obs_nonlinearity}")
    else:
        logging.warning(f"Data config file not found at {config_path}. Using minimal defaults.")
        da_config = DataAssimilationConfig() # Defaults

    # Determine System
    data_dir_str = str(args.data_dir or train_config.get('data_dir', ''))
    
    if "ks" in str(config_path):
        system_class = KuramotoSivashinsky
        logging.info("Detected Kuramoto-Sivashinsky system")
    elif "lorenz96" in str(config_path) or "lorenz96" in data_dir_str or "96" in str(config_path):
        system_class = Lorenz96
    else:
        system_class = Lorenz63
        
    logging.info(f"Using system: {system_class.__name__}")
    system = system_class(da_config)
    logging.info(f"System observation nonlinearity: {system.observation_operator.nonlinearity}")
    
    # Load Dataset (Test Split, Full Trajectories)
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif 'data_dir' in train_config:
        data_dir = Path(train_config['data_dir'])
    else:
        raise ValueError("Data directory not specified")
    
    if not (data_dir / "data.h5").exists():
        raise FileNotFoundError(f"Data file not found at {data_dir / 'data.h5'}")

    logging.info(f"Loading data from {data_dir}")
    with h5py.File(data_dir / "data.h5", "r") as f, h5py.File(data_dir / "data_scaled.h5", "r") as f_scaled:
        test_traj = f["test/trajectories"][:]
        test_obs = f["test/observations"][:]
        obs_mask = f["obs_mask"][:]
        
        test_traj_scaled = f_scaled["test/trajectories"][:]
        test_obs_scaled = f_scaled["test/observations"][:]
        
        # Load scalers
        if "scaler_mean" in f:
            scaler_mean = torch.from_numpy(f["scaler_mean"][:]).float().to(args.device)
            scaler_std = torch.from_numpy(f["scaler_std"][:]).float().to(args.device)
        else:
            scaler_mean = None
            scaler_std = None
            logging.warning("Scaler mean and std not found in data file. Using None.")

    test_dataset = DataAssimilationDataset(
        system, test_traj, test_obs, obs_mask,
        trajectories_scaled=test_traj_scaled,
        observations_scaled=test_obs_scaled,
        mode="train" # Returns full trajectories
    )
    
    # Initialize Model
    state_dim = system.state_dim
    # Recover model params from checkpoint config
    architecture = train_config.get('architecture', 'mlp')
    width = train_config.get('width', 128)
    depth = train_config.get('depth', 2)
    use_bn = train_config.get('use_bn', False)
    
    # ResNet1D params if available
    channels = train_config.get('channels', 64)
    num_blocks = train_config.get('num_blocks', 6)
    kernel_size = train_config.get('kernel_size', 5)
    obs_components = train_config.get('obs_components', None)
    
    # Determine observation parameters
    obs_indices = None
    obs_dim = 0
    use_observations = train_config.get('use_observations', False)
    
    if use_observations:
        if obs_components is not None:
            if isinstance(obs_components, str) and obs_components.lower() != 'none':
                try:
                    obs_indices = [int(x) for x in obs_components.split(',')]
                except ValueError:
                    logging.warning(f"Could not parse obs_components: {obs_components}")
            elif isinstance(obs_components, (list, tuple)):
                obs_indices = list(obs_components)
                
        
        if obs_indices is not None:
            obs_dim = len(obs_indices)
        elif 'obs_dim' in train_config:
            obs_dim = train_config['obs_dim']
        else:
            # Fallback: if use_observations is True but no indices/dim, 
            # might be full observation or handled internally.
            # But better to be safe.
            logging.warning("use_observations is True but obs_indices/dim could not be determined. Assuming obs_dim=0.")
            obs_dim = 0
            
    logging.info(f"Model Architecture: {architecture}, Width: {width}, Depth: {depth}")
    if architecture == 'resnet1d':
        logging.info(f"ResNet Params: Channels: {channels}, Num Blocks: {num_blocks}, Kernel: {kernel_size}")
    logging.info(f"Observation Params: use_observations={use_observations}, obs_dim={obs_dim}, indices_len={len(obs_indices) if obs_indices else 0}")
    
    model = ScoreNet(
        marginal_prob_std=None,
        x_dim=state_dim,
        extra_dim=state_dim,
        hidden_depth=depth,
        embed_dim=width,
        use_bn=use_bn,
        architecture=architecture,
        # ResNet1D params
        channels=channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        # Obs params
        obs_dim=obs_dim,
        obs_indices=obs_indices,
    ).to(args.device)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Observation Operator
    def obs_op_wrapper(x):
        # x is (B, D)
        return system.apply_observation_operator(x)
        
    # Evaluation Loop
    logging.info(f"Evaluating on {len(test_dataset)} trajectories...")
    
    # WandB setup
    if args.use_wandb and wandb is not None:
        if args.run_name:
            run_name = args.run_name
        else:
            run_name = f"flowdas_eval_{system_class.__name__}_{args.num_steps}steps"
        
        # Ensure wandb dir exists
        wandb_dir = Path("/data/da_outputs/wandb")
        wandb_dir.mkdir(parents=True, exist_ok=True)
        
        wandb.init(
            project=args.wandb_project,
            config=args.__dict__,
            name=run_name,
            entity=args.wandb_entity,
            dir=str(wandb_dir),
        )
        
    # Data structure to hold detailed results for plotting
    trajectory_results = {}
    
    # Limit number of trajectories
    num_trajs = args.num_trajs if args.num_trajs > 0 else len(test_dataset)
    num_trajs = min(num_trajs, len(test_dataset))
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # H5 file for saving results
    save_path = results_dir / "eval_results.h5"
    with h5py.File(save_path, 'w') as f_out:
        
        for i in tqdm(range(num_trajs), desc="Evaluating"):
            
            item = test_dataset[i]
            
            # Init results for this traj
            trajectory_results[i] = {
                "trajectory_idx": i,
                "trajectory_data": [],
                "rmse_sum": 0.0,
                "count": 0
            }
            
            # Extract data
            gt_traj = item['trajectories'].to(args.device) # (T, D)
            gt_traj_scaled = item['trajectories_scaled'].to(args.device) # (T, D)
            observations = item['observations'].to(args.device) # (T, obs_dim)
            mask = item['obs_mask'].to(args.device) # (T,)
            
            T = gt_traj.shape[0]

            # Start with ground truth
            curr_x_scaled = gt_traj_scaled[0].unsqueeze(0) # (1, D)
            
            # Repeat for ensemble if needed
            if args.n_samples_per_traj > 1:
                curr_x_scaled = curr_x_scaled.repeat(args.n_samples_per_traj, 1) # (K, D)
            
            # Storage for trajectory (list of tensors)
            est_traj_scaled = [curr_x_scaled]
            if scaler_mean is not None:
                curr_x_phys = curr_x_scaled * scaler_std + scaler_mean
            else:
                curr_x_phys = curr_x_scaled
            est_traj_phys = [curr_x_phys]
            
            # Initial metrics
            curr_x_phys_np = curr_x_phys.detach().cpu().numpy() # (K, D)
            curr_x_mean_np = np.mean(curr_x_phys_np, axis=0) # (D,)
            curr_x_std_np = np.std(curr_x_phys_np, axis=0) if args.n_samples_per_traj > 1 else np.zeros_like(curr_x_mean_np)
            
            # Record initial state stats
            trajectory_results[i]["trajectory_data"].append({
                "time_idx": 0,
                "x_true": gt_traj[0].cpu().numpy(),
                "x_est": curr_x_mean_np,
                "x_std": curr_x_std_np,
                "rmse": 0.0, # Initial condition given
                "crps": 0.0,
                "has_observation": mask[0].item(),
                "observation": observations[0].cpu().numpy() if mask[0] else None
            })
            
            trajectory_results[i]["crps_sum"] = 0.0 # Init CRPS sum
            
            for t in range(T - 1):
                # Target time t+1
                time_idx = t + 1
                
                # Check observation availability based on dataset mask AND frequency
                is_observed = (time_idx % args.obs_frequency == 0)
                has_obs = (mask[time_idx].item() > 0.5) and is_observed
                if has_obs:
                    obs_t_plus_1 = observations[time_idx].unsqueeze(0)
                    if args.n_samples_per_traj > 1:
                        obs_t_plus_1 = obs_t_plus_1.repeat(args.n_samples_per_traj, 1)
                else:
                    obs_t_plus_1 = None
                
                # Sample
                next_x_scaled, _ = Euler_Maruyama_sampler(
                    model=model,
                    base=curr_x_scaled,
                    cond=curr_x_scaled,
                    measurement=obs_t_plus_1,
                    num_steps=args.num_steps,
                    sigma_obs=args.sigma_obs,
                    MC_times=args.mc_times,
                    step_size=args.guidance_step,
                    observation_operator=obs_op_wrapper,
                    scaler_mean=scaler_mean,
                    scaler_std=scaler_std
                )
                
                # Unscale
                if scaler_mean is not None:
                    next_x_phys = next_x_scaled * scaler_std + scaler_mean
                else:
                    next_x_phys = next_x_scaled
                    
                est_traj_scaled.append(next_x_scaled.detach())
                est_traj_phys.append(next_x_phys.detach())
                
                # Compute Step Error
                x_est_ens_np = next_x_phys.detach().cpu().numpy() # (K, D)
                x_est_mean_np = np.mean(x_est_ens_np, axis=0) # (D,)
                x_est_std_np = np.std(x_est_ens_np, axis=0) if args.n_samples_per_traj > 1 else np.zeros_like(x_est_mean_np)
                x_true_np = gt_traj[t+1].cpu().numpy() # (D,)
                
                step_mse = np.mean((x_est_mean_np - x_true_np) ** 2)
                step_rmse = np.sqrt(step_mse)
                
                step_crps = compute_crps_ensemble(x_est_ens_np, x_true_np)
                
                # Store step data
                trajectory_results[i]["trajectory_data"].append({
                    "time_idx": t + 1,
                    "x_true": x_true_np,
                    "x_est": x_est_mean_np,
                    "x_std": x_est_std_np,
                    "rmse": step_rmse,
                    "crps": step_crps,
                    "has_observation": has_obs,
                    "observation": observations[t+1].cpu().numpy() if has_obs else None
                })
                
                
                trajectory_results[i]["rmse_sum"] += step_rmse
                trajectory_results[i]["crps_sum"] += step_crps
                trajectory_results[i]["count"] += 1
                
                curr_x_scaled = next_x_scaled.detach()
                
                
            # Stack full trajectory
            est_traj_phys_stack = torch.stack(est_traj_phys, dim=0) # (T, K, D)
            if args.n_samples_per_traj == 1:
                est_traj_phys_stack = est_traj_phys_stack.squeeze(1) # (T, D)
            
            # Save to H5
            grp = f_out.create_group(f"traj_{i}")
            grp.create_dataset("gt", data=gt_traj.cpu().numpy())
            grp.create_dataset("est", data=est_traj_phys_stack.cpu().numpy())
            grp.create_dataset("obs", data=observations.cpu().numpy())
            
            # Log final traj RMSE
            traj_rmse = trajectory_results[i]["rmse_sum"] / max(1, trajectory_results[i]["count"])
            if args.use_wandb and wandb is not None:
                wandb.log({"rmse": traj_rmse, "traj_idx": i})
                
                # Visualizations for first few trajectories
            if i < args.n_vis_trajectories:
                fig = plot_trajectory_comparison(trajectory_results[i], da_config)
                
                if fig:
                    # Save locally for verification
                    local_plot_path = results_dir / f"trajectory_{i}.png"
                    try:
                        fig.savefig(local_plot_path)
                        logging.info(f"Saved plot locally to {local_plot_path}")
                    except Exception as e:
                        logging.error(f"Failed to save plot locally: {e}")

                    if args.use_wandb and wandb is not None:
                        try:
                            # Use path string for robustness
                            wandb.log({f"eval/trajectory_{i}": wandb.Image(str(local_plot_path))})
                        except Exception as e:
                            logging.error(f"Failed to log plot to wandb: {e}")
                    
                    plt.close(fig)
                else:
                    logging.warning(f"Plot generation returned None for trajectory {i}")

    # Compute Global Aggregates
    all_rmses = []
    all_crps = []
    rmse_per_timestep = {} # Map time_idx -> list of rmses
    crps_per_timestep = {} # Map time_idx -> list of crps
    
    for i in trajectory_results:
        traj_data = trajectory_results[i]["trajectory_data"]
        traj_rmse = trajectory_results[i]["rmse_sum"] / max(1, trajectory_results[i]["count"])
        traj_crps = trajectory_results[i].get("crps_sum", 0.0) / max(1, trajectory_results[i]["count"])
        
        all_rmses.append(traj_rmse)
        all_crps.append(traj_crps)
        
        for d in traj_data:
            t = d["time_idx"]
            if t not in rmse_per_timestep:
                rmse_per_timestep[t] = []
                crps_per_timestep[t] = []
            rmse_per_timestep[t].append(d["rmse"])
            crps_per_timestep[t].append(d.get("crps", 0.0))
            
    avg_rmse = np.mean(all_rmses)
    avg_crps = np.mean(all_crps)
    
    std_rmse = np.std(all_rmses)
    std_crps = np.std(all_crps)
    
    logging.info(f"Global Average RMSE: {avg_rmse:.4f}")
    logging.info(f"Global Average CRPS: {avg_crps:.4f}")
    
    if args.use_wandb and wandb is not None:
        wandb.log({
            "eval/global_mean_rmse": avg_rmse,
            "eval/global_mean_crps": avg_crps,
            "eval/global_std_rmse": std_rmse,
            "eval/global_std_crps": std_crps
        })
        
        # Log RMSE/CRPS over time
        # Sort timesteps
        sorted_times = sorted(rmse_per_timestep.keys())
        for t in sorted_times:
            mean_rmse_t = np.mean(rmse_per_timestep[t])
            mean_crps_t = np.mean(crps_per_timestep[t])
            std_rmse_t = np.std(rmse_per_timestep[t])
            std_crps_t = np.std(crps_per_timestep[t])
            wandb.log({
                "eval/mean_rmse_over_time": mean_rmse_t,
                "eval/std_rmse_over_time": std_rmse_t,
                "eval/mean_crps_over_time": mean_crps_t,
                "eval/std_crps_over_time": std_crps_t,
                "time_step": t
            })

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FlowDAS Model")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to data config (optional, inferred from checkpoint)")
    parser.add_argument("--data_dir", type=str, help="Path to dataset (optional, inferred from checkpoint)")
    parser.add_argument("--results_dir", type=str, default="/data/da_outputs/results/flowdas_eval", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    parser.add_argument("--num_trajs", type=int, default=-1, help="Number of trajectories to evaluate")
    parser.add_argument("--n_vis_trajectories", type=int, default=10, help="Number of trajectories to visualize in WandB")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of Euler steps")
    parser.add_argument("--sigma_obs", type=float, default=0.1, help="Observation noise std for guidance")
    parser.add_argument("--mc_times", type=int, default=25, help="MC samples for Taylor approximation")
    parser.add_argument("--n_samples_per_traj", type=int, default=20, help="Number of samples per trajectory (for CRPS)")
    parser.add_argument("--guidance_step", type=float, default=0.01, help="Step size for guidance gradient")
    
    parser.add_argument("--obs_frequency", type=int, default=1, help="Observation frequency (default: 1, observe every step)")
    
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="flowdas-eval", help="WandB project")
    parser.add_argument("--wandb_entity", type=str, default="ml-climate", help="WandB entity")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    
    return parser.parse_args()

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

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
