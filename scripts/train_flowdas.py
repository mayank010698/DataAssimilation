import argparse
import logging
import os
import sys
import yaml
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data import DataAssimilationDataModule, DataAssimilationConfig, load_config_yaml, TimeAlignedBatchSampler
from models.flowdas import ScoreNet, loss_fn, prepare_batch, Euler_Maruyama_sampler
from proposals.eval_proposal import compute_crps_ensemble, plot_trajectory_comparison, _derive_eval_run_name

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

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=50, min_delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 50
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.val_loss_min = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.val_loss_min = val_loss
            self.counter = 0

class FlowDASDataModule(DataAssimilationDataModule):
    """
    Subclass to force datasets into 'inference' mode (returning pairs) for training.
    """
    def setup(self, stage=None):
        super().setup(stage)
        
        # Helper to convert dataset to pair mode
        def convert_to_pairs(dataset):
            if dataset is None: return
            dataset.mode = "inference"
            # Re-generate items list as done in __init__ for inference mode
            dataset.items = [
                (traj, t)
                for traj in range(dataset.n_trajectories)
                for t in range(1, dataset.n_steps)
            ]
            
        if stage == "fit" or stage is None:
            convert_to_pairs(self.train_dataset)
            convert_to_pairs(self.val_dataset)
            
        if stage == "test" or stage is None:
            # Keep test dataset in standard mode for autoregressive eval?
            # Actually, eval_proposal uses TimeAlignedBatchSampler which expects dataset[i] to return a step.
            # But the 'inference' mode (pairs) is what we want: (x_prev, x_curr, ...)
            convert_to_pairs(self.test_dataset)

def run_flowdas_eval(
    model,
    config,
    data_dir,
    n_trajectories=None,
    n_vis_trajectories=10,
    batch_size=32,
    device='cuda',
    wandb_run=None,
    num_steps=50,
    mc_guidance=False,
    guidance_scale=1.0,
    n_samples_per_traj=1,
):
    """
    Evaluate FlowDAS model autoregressively.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running FlowDAS Evaluation (Autoregressive)")
    
    # Load Data (fresh module for test)
    # We use Lorenz63 class for 3D, Lorenz96 for >3D
    from data import Lorenz63, Lorenz96
    
    if "dim" in config.system_params and config.system_params["dim"] > 3:
        system_class = Lorenz96
    else:
        system_class = Lorenz63
        
    data_module = DataAssimilationDataModule(
        config=config,
        system_class=system_class,
        data_dir=str(data_dir),
        batch_size=batch_size,
    )
    data_module.setup("test")
    test_dataset = data_module.test_dataset
    
    # Ensure test dataset is in 'inference' mode (returning dictionaries with x_prev, x_curr, etc)
    # The default DataAssimilationDataset might be in 'trajectory' mode if not specified?
    # Actually DataAssimilationDataset default mode is 'trajectory'.
    # But TimeAlignedBatchSampler expects to index into dataset to get time steps.
    # We need to change mode to 'inference'.
    test_dataset.mode = "inference"
    test_dataset.items = [
        (traj, t)
        for traj in range(test_dataset.n_trajectories)
        for t in range(1, test_dataset.n_steps)
    ]
    
    # Construct observation_fn for guidance
    obs_scaler_mean = data_module.obs_scaler_mean.to(device) if data_module.obs_scaler_mean is not None else None
    obs_scaler_std = data_module.obs_scaler_std.to(device) if data_module.obs_scaler_std is not None else None
    
    observation_fn = None
    if mc_guidance:
        def observation_fn(x_scaled):
            # x_scaled is shape (..., state_dim) in SCALED space
            x_physical = data_module.system.postprocess(x_scaled)
            y_physical = data_module.system.apply_observation_operator(x_physical)
            
            if obs_scaler_mean is not None:
                y_scaled = (y_physical - obs_scaler_mean) / obs_scaler_std
            else:
                y_scaled = y_physical
            return y_scaled

    # Prepare sampler
    total_trajs_in_data = test_dataset.n_trajectories
    if n_trajectories is None or n_trajectories > total_trajs_in_data:
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
        num_workers=0
    )
    
    # Tracking
    current_states = {} # trajectory_idx -> current_state (K, D)
    trajectory_results = {
        i: {
            "trajectory_data": [],
            "rmse_sum": 0.0,
            "crps_sum": 0.0,
            "count": 0
        } for i in range(n_trajectories)
    }
    
    model.eval()
    system = data_module.system
    
    pbar = tqdm(test_loader, desc="Evaluating")
    
    for batch in pbar:
        # Move to device
        x_prev_input = batch["x_prev_scaled"].to(device)
        x_curr_gt = batch["x_curr"].to(device) # Physical ground truth
        
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
                # x_prev_input[i] is (D,)
                current_states[tid] = x_prev_input[i].unsqueeze(0).repeat(n_samples_per_traj, 1)
            
            if tid in current_states:
                batch_x_prev.append(current_states[tid])
                if y_curr_input is not None:
                    # y_curr_input[i] is (D_obs,)
                    batch_y_curr.append(y_curr_input[i].unsqueeze(0).repeat(n_samples_per_traj, 1))
                valid_mask.append(i)
        
        if not batch_x_prev:
            continue
            
        # Stack
        x_prev_stack = torch.stack(batch_x_prev) # (B_eff, K, D)
        B_eff, K, D = x_prev_stack.shape
        x_prev_flat = x_prev_stack.reshape(B_eff * K, D)
        
        y_curr_flat = None
        measurement = None
        if batch_y_curr and mc_guidance:
            y_curr_stack = torch.stack(batch_y_curr)
            D_obs = y_curr_stack.shape[-1]
            y_curr_flat = y_curr_stack.reshape(B_eff * K, D_obs)
            measurement = y_curr_flat
            
        # Sample
        with torch.no_grad():
            # Euler_Maruyama_sampler(model, base, cond=None, measurement=None, ...)
            # base = x_prev, cond = x_prev
            # We condition on x_prev (flow from x_prev to x_curr)
            
            # Using x_prev as both base and condition
            # Note: ScoreNet expects extra_elements=cond
            
            x_next_flat, _ = Euler_Maruyama_sampler(
                model=model,
                base=x_prev_flat,
                cond=x_prev_flat, # Conditioning on previous state
                measurement=measurement,
                num_steps=num_steps,
                sigma_obs=0.1, # Using 0.1 as default or derive from config?
                MC_times=1, # Can be higher for better gradient est
                step_size=guidance_scale, # Using guidance_scale as step_size for updates? Or scaling grad?
                # FlowDAS EM uses: xt = ... - step_size * norm_grad
                # So step_size IS the guidance scale effectively.
                observation_operator=observation_fn,
                scaler_mean=obs_scaler_mean,
                scaler_std=obs_scaler_std
            )
            
        x_next_stack = x_next_flat.reshape(B_eff, K, D)
        
        # Metrics
        batch_rmse = []
        batch_crps = []
        
        for k, idx in enumerate(valid_mask):
            tid = traj_idxs[idx]
            t = time_idxs[idx]
            
            x_gen_k = x_next_stack[k]
            x_true = x_curr_gt[idx]
            
            current_states[tid] = x_gen_k
            
            # Unscale for metrics
            # Detach before postprocessing to avoid grad issues with numpy
            x_gen_orig_k = system.postprocess(x_gen_k.detach())
            x_true_orig = x_true.detach()
            
            # RMSE
            x_mean_orig = torch.mean(x_gen_orig_k, dim=0)
            error = x_mean_orig - x_true_orig
            rmse = torch.sqrt(torch.mean(error**2)).item()
            batch_rmse.append(rmse)
            
            # CRPS
            crps = compute_crps_ensemble(x_gen_orig_k.cpu().numpy(), x_true_orig.cpu().numpy())
            batch_crps.append(crps)
            
            has_obs = batch["has_observation"][idx].item()
            obs = batch["y_curr"][idx].cpu().numpy() if (has_obs and "y_curr" in batch) else None
            
            trajectory_results[tid]["trajectory_data"].append({
                "time_idx": t,
                "x_true": x_true_orig.cpu().numpy(),
                "x_est": x_mean_orig.cpu().numpy(),
                "x_std": torch.std(x_gen_orig_k, dim=0, correction=0).cpu().numpy(),
                "observation": obs,
                "has_observation": has_obs,
                "rmse": rmse,
                "crps": crps
            })
            
            trajectory_results[tid]["rmse_sum"] += rmse
            trajectory_results[tid]["crps_sum"] += crps
            trajectory_results[tid]["count"] += 1
            
        avg_rmse = sum(batch_rmse) / len(batch_rmse) if batch_rmse else 0.0
        pbar.set_postfix({"rmse": f"{avg_rmse:.4f}"})
        
    # Aggregate
    total_rmse = 0.0
    total_crps = 0.0
    total_steps = 0
    
    for tid, res in trajectory_results.items():
        if res["count"] > 0:
            res["mean_rmse"] = res["rmse_sum"] / res["count"]
            res["mean_crps"] = res["crps_sum"] / res["count"]
            total_rmse += res["rmse_sum"]
            total_crps += res["crps_sum"]
            total_steps += res["count"]
            
            res["trajectory_data"].sort(key=lambda x: x["time_idx"])
            
            if wandb_run and tid < n_vis_trajectories:
                fig_res = {"trajectory_idx": tid, "trajectory_data": res["trajectory_data"]}
                fig = plot_trajectory_comparison(fig_res, config, system)
                if fig:
                    wandb_run.log({f"eval/trajectory_{tid}": wandb.Image(fig)})
                    plt.close(fig)
                    
    global_mean_rmse = total_rmse / total_steps if total_steps > 0 else 0.0
    global_mean_crps = total_crps / total_steps if total_steps > 0 else 0.0
    
    logger.info(f"Global Mean RMSE: {global_mean_rmse:.4f}")
    logger.info(f"Global Mean CRPS: {global_mean_crps:.4f}")
    
    if wandb_run:
        wandb_run.log({
            "eval/global_mean_rmse": global_mean_rmse,
            "eval/global_mean_crps": global_mean_crps
        })
        
    return global_mean_rmse

def train(args):
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logging.info(f"Random seed set to {args.seed}")

    # Load config
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            # Try looking in config directory
            config_path = Path("config") / args.config
        
        if config_path.exists():
            da_config = load_config_yaml(config_path)
        else:
            raise FileNotFoundError(f"Config file not found: {args.config}")
    else:
        # Default config if none provided (not recommended)
        da_config = DataAssimilationConfig()

    # Determine system class
    from data import Lorenz63, Lorenz96, KuramotoSivashinsky, DoubleWell
    
    # Check dataset dir name or config to guess system
    path_lower = str(args.data_dir).lower()
    config_lower = str(args.config).lower()
    
    if "ks" in path_lower or "kuramoto" in path_lower or "ks" in config_lower:
        system_class = KuramotoSivashinsky
        logging.info("Detected Kuramoto-Sivashinsky system")
    elif "lorenz96" in path_lower or "96" in config_lower or "lorenz96" in config_lower:
        system_class = Lorenz96
        logging.info("Detected Lorenz 96 system")
    elif "double_well" in path_lower or "dw" in path_lower or "double_well" in config_lower:
        system_class = DoubleWell
        logging.info("Detected Double Well system")
    else:
        system_class = Lorenz63
        logging.info("Defaulting to Lorenz 63 system")

    # Handle obs_components override
    if args.obs_components:
        try:
            obs_components = [int(i) for i in args.obs_components.split(',') if i.strip()]
            da_config.obs_components = obs_components
            logging.info(f"Overriding obs_components from CLI: {obs_components}")
        except ValueError:
            raise ValueError(f"Invalid format for --obs-components: {args.obs_components}")

    # Create Data Module
    dm = FlowDASDataModule(
        config=da_config,
        system_class=system_class,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup data
    dm.prepare_data()
    dm.setup("fit")
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # Model Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Determine state dim
    state_dim = dm.system.state_dim
    
    if args.use_observations:
        obs_dim = dm.system.obs_dim
        obs_indices = dm.system.observation_operator.obs_components
    else:
        obs_dim = 0
        obs_indices = None
    
    model = ScoreNet(
        marginal_prob_std=None, # Not used in training forward pass for SI?
        x_dim=state_dim,
        extra_dim=state_dim, # We condition on previous state x_prev
        hidden_depth=args.depth,
        embed_dim=args.width,
        use_bn=args.use_bn,
        architecture=args.architecture,
        # ResNet1D / Architecture specific params
        channels=args.channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        obs_dim=obs_dim,
        obs_indices=obs_indices,
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda t: 1 - (t / args.epochs))
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training Loop
    best_val_loss = float('inf')
    save_dir = Path(args.run_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting training on {device}...")
    logging.info(f"State Dim: {state_dim}")
    logging.info(f"Batches per epoch: {len(train_loader)}")
    
    # Initialize WandB
    wandb_run = None
    if args.use_wandb and wandb is not None:
        if args.wandb_name:
            run_name = args.wandb_name
        else:
             # Use run directory name as run name (like train_rf.py)
            run_name = Path(args.run_dir).name
            
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args.__dict__,
            name=run_name,
            dir=str(save_dir)
        )
        # Watch model
        wandb.watch(model, log="all", log_freq=100)
        
        # Log extended config
        wandb_run.config.update({
            "state_dim": state_dim,
            "data_config": da_config.__dict__,
        })

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_items = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Prepare batch (moves to device)
            # prepare_batch handles dict from dataloader
            batch_si = prepare_batch(batch, device=device)
            
            loss, N = loss_fn(model, batch_si)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * N
            train_items += N
            
            pbar.set_postfix({'loss': loss.item()})
            
            if args.use_wandb and wandb is not None:
                wandb.log({"train_loss_step": loss.item()})
            
        scheduler.step()
        avg_train_loss = train_loss / train_items
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_items = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch_si = prepare_batch(batch, device=device)
                loss, N = loss_fn(model, batch_si)
                val_loss += loss.item() * N
                val_items += N
                
        avg_val_loss = val_loss / val_items
        
        logging.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        if args.use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_epoch": avg_train_loss,
                "val_loss_epoch": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # Check early stopping
        early_stopping(avg_val_loss)
        
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'config': args.__dict__
        }
        
        # Save latest
        torch.save(checkpoint, save_dir / "checkpoint_latest.pth")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, save_dir / "checkpoint_best.pth")
            logging.info(f"New best model saved! (Loss: {best_val_loss:.6f})")

    # Final Evaluation
    if args.evaluate:
        logging.info("Starting Final Evaluation...")
        
        # Load best model for eval
        best_ckpt = torch.load(save_dir / "checkpoint_best.pth", map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        
        run_flowdas_eval(
            model=model,
            config=da_config,
            data_dir=args.data_dir,
            n_trajectories=None, # All trajectories? Or subset?
            n_vis_trajectories=10,
            batch_size=args.batch_size,
            device=device.type,
            wandb_run=wandb_run,
            mc_guidance=args.mc_guidance,
            guidance_scale=args.guidance_scale
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Train FlowDAS Model")
    
    parser.add_argument("--config", type=str, required=True, help="Path to data config file (yaml)")
    parser.add_argument("--data_dir", type=str, default="datasets/lorenz63_default", help="Path to dataset directory")
    parser.add_argument("--run_dir", type=str, default="/data/da_outputs/runs_train/flowdas_test", help="Directory to save runs")
    
    # Model params
    parser.add_argument("--architecture", type=str, default="mlp", choices=["mlp", "resnet1d"], help="Model architecture")
    parser.add_argument("--width", type=int, default=128, help="Embedding/Hidden dimension")
    parser.add_argument("--depth", type=int, default=2, help="Depth (layers or blocks)")
    parser.add_argument("--use_bn", action="store_true", help="Use Batch Normalization")
    
    # ResNet1D params
    parser.add_argument("--channels", type=int, default=64, help="Channels for ResNet1D")
    parser.add_argument("--num_blocks", type=int, default=6, help="Num blocks for ResNet1D")
    parser.add_argument("--kernel_size", type=int, default=5, help="Kernel size for ResNet1D")
    parser.add_argument("--obs_components", type=str, default=None, help="Comma-separated indices of observed variables")
    parser.add_argument("--use_observations", action="store_true", help="Use observation conditioning")
    
    # Training params
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="flowdas-train", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="ml-climate", help="WandB entity")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name")
    
    # Eval params
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    parser.add_argument("--mc_guidance", action="store_true", help="Enable MC guidance during evaluation")
    parser.add_argument("--guidance_scale", type=float, default=0.1, help="Guidance scale (step size for EM)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
