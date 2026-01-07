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

from data import DataAssimilationConfig, DataAssimilationDataset, load_config_yaml, Lorenz63, Lorenz96
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

    fig = plt.figure(figsize=(16, 12))

    # 3D Plot (using first 3 dims)
    ax1 = fig.add_subplot(3, 2, 1, projection="3d")
    ax1.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], "b-", alpha=0.8, label="True")
    ax1.plot(x_est[:, 0], x_est[:, 1], x_est[:, 2], "r--", alpha=0.8, label="Generated")
    ax1.set_title(f"3D Trajectory (ID: {traj_idx})")
    ax1.legend()

    # Component Plots (first 3)
    state_names = ["X", "Y", "Z"]
    colors = ["blue", "green", "orange"]
    
    obs_components = config.obs_components if config.obs_components else []
    
    for i, (name, color) in enumerate(zip(state_names, colors)):
        if i >= x_true.shape[1]: break
        
        ax = fig.add_subplot(3, 2, i + 2)
        ax.plot(time_steps, x_true[:, i], color=color, label=f"{name} (true)")
        ax.plot(time_steps, x_est[:, i], color="red", linestyle="--", label=f"{name} (gen)")
        
        # Plot observations if available for this component
        # Note: This simple check assumes obs_values matches state indices which might not be true for all operators
        # But for identity/subset observation it often works if we map correctly.
        # Here we simplify and just plot if we have data and it looks like identity-ish
        
        # Check if component i is observed
        # If obs_components is list of indices, check if i in it
        is_observed = i in obs_components
        
        if is_observed and len(obs_times) > 0:
            # We need to know WHICH index in obs_values corresponds to state component i
            # If obs_components = [0, 2], then obs_values[:, 0] is state 0, obs_values[:, 1] is state 2
            try:
                obs_idx = obs_components.index(i)
                # obs_values is (N_obs, Obs_Dim)
                if obs_idx < obs_values[0].shape[0]: # Check dimension validity
                    # Filter valid times
                    valid_mask = [idx < len(x_true) for idx in obs_indices]
                    # We need to array-ify obs_values first
                    obs_vals_arr = np.array(obs_values)
                    
                    ax.scatter(
                        obs_times, 
                        obs_vals_arr[:, obs_idx], 
                        color="red", marker="x", s=40, label="Obs"
                    )
            except ValueError:
                pass # Component not observed

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
    else:
        # Try to find config in checkpoint or infer
        if 'config' in train_config and isinstance(train_config['config'], str): 
             config_path = Path(train_config['config']) # original config path string
        elif 'data_dir' in train_config and train_config['data_dir']:
             config_path = Path(train_config['data_dir']) / "config.yaml"
        elif args.data_dir:
             config_path = Path(args.data_dir) / "config.yaml"
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
    else:
        logging.warning(f"Data config file not found at {config_path}. Using minimal defaults.")
        da_config = DataAssimilationConfig() # Defaults

    # Determine System
    # Heuristic:
    data_dir_str = str(args.data_dir or train_config.get('data_dir', ''))
    if "lorenz96" in str(config_path) or "lorenz96" in data_dir_str or "96" in str(config_path):
        system_class = Lorenz96
    else:
        system_class = Lorenz63
        
    logging.info(f"Using system: {system_class.__name__}")
    system = system_class(da_config)
    
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
    
    logging.info(f"Model Architecture: {architecture}, Width: {width}, Depth: {depth}")
    if architecture == 'resnet1d':
        logging.info(f"ResNet Params: Channels: {channels}, Num Blocks: {num_blocks}, Kernel: {kernel_size}")
    
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
        # Eval usually assumes no obs conditioning in the score model itself if trained without
        # But we pass what's in config
        # Note: obs_dim depends on use_observations in training
        obs_dim=0, # Assuming no_obs for these experiments, or we should infer?
        # Ideally we infer obs_dim from config 'use_observations'
        # But here we hardcode 0 if not present, or try to respect training config?
        # The 'no_obs' models were trained with use_observations=False
    ).to(args.device)
    
    # Correct obs_dim/indices if trained with observations
    if train_config.get('use_observations', False):
        # We need to reconstruct obs_dim and indices
        # This requires the training da_config usually, or just what's in train_config
        # train_config saves all args, including obs_components string
        pass # Not handling reconstruction of obs conditioning for now as we focus on no_obs models
        
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
        
        wandb.init(project="FlowDAS-Eval", config=args.__dict__, name=run_name, entity=args.wandb_entity, dir=str(wandb_dir))
        
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
            
            est_traj_scaled = [curr_x_scaled]
            if scaler_mean is not None:
                curr_x_phys = curr_x_scaled * scaler_std + scaler_mean
            else:
                curr_x_phys = curr_x_scaled
            est_traj_phys = [curr_x_phys]
            
            # Record initial state stats
            trajectory_results[i]["trajectory_data"].append({
                "time_idx": 0,
                "x_true": gt_traj[0].cpu().numpy(),
                "x_est": curr_x_phys[0].cpu().numpy(),
                "rmse": 0.0, # Initial condition given
                "has_observation": mask[0].item(),
                "observation": observations[0].cpu().numpy() if mask[0] else None
            })
            
            for t in range(T - 1):
                # Target time t+1
                has_obs = mask[t+1]
                
                if has_obs:
                    obs_t_plus_1 = observations[t+1].unsqueeze(0)
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
                x_est_np = next_x_phys[0].detach().cpu().numpy()
                x_true_np = gt_traj[t+1].cpu().numpy()
                
                step_mse = np.mean((x_est_np - x_true_np) ** 2)
                step_rmse = np.sqrt(step_mse)
                
                # Store step data
                trajectory_results[i]["trajectory_data"].append({
                    "time_idx": t + 1,
                    "x_true": x_true_np,
                    "x_est": x_est_np,
                    "rmse": step_rmse,
                    "has_observation": has_obs.item(),
                    "observation": observations[t+1].cpu().numpy() if has_obs else None
                })
                
                trajectory_results[i]["rmse_sum"] += step_rmse
                trajectory_results[i]["count"] += 1
                
                curr_x_scaled = next_x_scaled.detach()
                
            # Stack full trajectory
            est_traj_phys_stack = torch.cat(est_traj_phys, dim=0) # (T, D)
            
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
                        wandb.log({f"eval/trajectory_{i}": wandb.Image(fig)})
                        plt.close(fig)

    # Compute Global Aggregates
    all_rmses = []
    rmse_per_timestep = {} # Map time_idx -> list of rmses
    
    for i in trajectory_results:
        traj_data = trajectory_results[i]["trajectory_data"]
        traj_rmse = trajectory_results[i]["rmse_sum"] / max(1, trajectory_results[i]["count"])
        all_rmses.append(traj_rmse)
        
        for d in traj_data:
            t = d["time_idx"]
            if t not in rmse_per_timestep:
                rmse_per_timestep[t] = []
            rmse_per_timestep[t].append(d["rmse"])
            
    avg_rmse = np.mean(all_rmses)
    logging.info(f"Global Average RMSE: {avg_rmse:.4f}")
    
    if args.use_wandb and wandb is not None:
        wandb.log({"eval/global_mean_rmse": avg_rmse})
        
        # Log RMSE over time
        # Sort timesteps
        sorted_times = sorted(rmse_per_timestep.keys())
        for t in sorted_times:
            mean_rmse_t = np.mean(rmse_per_timestep[t])
            wandb.log({
                "eval/mean_rmse_over_time": mean_rmse_t,
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
    parser.add_argument("--sigma_obs", type=float, default=0.25, help="Observation noise std for guidance")
    parser.add_argument("--mc_times", type=int, default=1, help="MC samples for Taylor approximation")
    parser.add_argument("--guidance_step", type=float, default=0.1, help="Step size for guidance gradient")
    
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_entity", type=str, default="ml-climate", help="WandB entity")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
