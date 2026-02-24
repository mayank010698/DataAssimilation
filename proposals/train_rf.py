"""
Training script for Rectified Flow Proposal Distribution

Trains a rectified flow model to learn p(x_t | x_{t-1}) from trajectory data.
"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))
    # Also add root for data imports if needed
    sys.path.append(str(Path(__file__).parent.parent))

from rectified_flow import RFProposal
from rf_dataset import RFDataModule
try:
    from eval_proposal import run_proposal_eval
except ImportError:
    # Fallback if running from root
    from proposals.eval_proposal import run_proposal_eval


def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_file = log_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_rectified_flow(
    data_dir: str,
    output_dir: str,
    state_dim: int = 3,
    obs_dim: int = 0,
    architecture: str = 'mlp',
    hidden_dim: int = 128,
    depth: int = 4,
    channels: int = 64,
    num_blocks: int = 6,
    kernel_size: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_epochs: int = 500,
    num_workers: int = 4,
    use_observations: bool = False,
    train_cond_method: str = 'concat',
    cond_embed_dim: int = 128,
    num_attn_heads: int = 4,
    gpus: int = 1,
    predict_delta: bool = False,
    time_embed_dim: int = 64,
    mc_guidance: bool = False,
    guidance_scale: float = 1.0,
    obs_indices: list = None,
    cond_dropout: float = 0.0,
    wandb_project: str = "rf-train-96",
    save_every_n_epochs: int = None,
    debug_random_obs: bool = False,
    debug_random_prev_state: bool = False,
    use_time_step: bool = False,
    trajectory_length: int = 1000,
):
    """
    Train a rectified flow proposal distribution
    
    Args:
        data_dir: Directory containing data.h5
        output_dir: Directory to save checkpoints and logs
        state_dim: Dimension of state space
        obs_dim: Dimension of observation space
        architecture: Velocity network architecture ('mlp' or 'resnet1d')
        hidden_dim: Hidden dimension (for MLP architecture)
        depth: Network depth (for MLP architecture)
        channels: Number of channels (for ResNet1D architecture)
        num_blocks: Number of residual blocks (for ResNet1D architecture)
        kernel_size: Kernel size (for ResNet1D architecture)
        batch_size: Training batch size
        learning_rate: Learning rate
        max_epochs: Maximum number of epochs
        num_workers: Number of dataloader workers
        use_observations: Whether to condition on observations
        train_cond_method: Method for observation conditioning ['concat', 'film', 'adaln', 'cross_attn']
        cond_embed_dim: Embedding dimension for conditioning
        num_attn_heads: Number of attention heads (for cross_attn)
        gpus: Number of GPUs to use
        predict_delta: If True, learn increment (x_curr - x_prev) instead of absolute target
        time_embed_dim: Dimension of time embedding (default: 64 for MLP, typically 32 for ResNet1D)
        mc_guidance: Whether to use Monte Carlo guidance during inference (saved to model config)
        guidance_scale: Scale for Monte Carlo guidance (saved to model config)
        obs_indices: List of indices where observations occur (for sparse observations)
        cond_dropout: Probability of dropping conditioning during training
        wandb_project: WandB project name
        save_every_n_epochs: Save a checkpoint every N epochs
        debug_random_obs: If True, replace observations with random vectors (for debugging)
        debug_random_prev_state: If True, replace previous state with random vectors (for debugging)
        use_time_step: Whether to condition on trajectory time step
        trajectory_length: Length of trajectory for time step normalization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not use_observations:
        if obs_dim > 0:
            logging.warning(f"use_observations=False but obs_dim={obs_dim}. Setting obs_dim to 0.")
        obs_dim = 0
        if train_cond_method != 'concat': # 'concat' is default, so only warn if explicitly set to something else likely
             pass 
    
    logger_obj = setup_logging(output_dir)
    logger_obj.info("="*80)
    logger_obj.info("Training Rectified Flow Proposal Distribution")
    logger_obj.info("="*80)
    logger_obj.info(f"Data directory: {data_dir}")
    logger_obj.info(f"Output directory: {output_dir}")
    logger_obj.info(f"State dimension: {state_dim}")
    logger_obj.info(f"Observation dimension: {obs_dim}")
    logger_obj.info(f"Use observations: {use_observations}")
    logger_obj.info(f"Architecture: {architecture}")
    if architecture == 'mlp':
        logger_obj.info(f"  Hidden dimension: {hidden_dim}")
        logger_obj.info(f"  Network depth: {depth}")
    elif architecture == 'resnet1d':
        logger_obj.info(f"  Channels: {channels}")
        logger_obj.info(f"  Num blocks: {num_blocks}")
        logger_obj.info(f"  Kernel size: {kernel_size}")
    logger_obj.info(f"Training conditioning method: {train_cond_method}")
    if train_cond_method in ['film', 'adaln', 'cross_attn']:
        logger_obj.info(f"  Conditioning embed dim: {cond_embed_dim}")
    if train_cond_method == 'cross_attn':
        logger_obj.info(f"  Number of attention heads: {num_attn_heads}")
    logger_obj.info(f"Time embedding dimension: {time_embed_dim}")
    logger_obj.info(f"Batch size: {batch_size}")
    logger_obj.info(f"Learning rate: {learning_rate}")
    logger_obj.info(f"Max epochs: {max_epochs}")
    logger_obj.info(f"GPUs: {gpus}")
    logger_obj.info(f"Predict delta (residual mode): {predict_delta}")
    logger_obj.info(f"MC Guidance (Inference default): {mc_guidance}")
    logger_obj.info(f"Guidance Scale (Inference default): {guidance_scale}")
    logger_obj.info(f"Debug: Random observations: {debug_random_obs}")
    logger_obj.info(f"Debug: Random previous state: {debug_random_prev_state}")
    logger_obj.info(f"Use time step conditioning: {use_time_step}")
    if use_time_step:
        logger_obj.info(f"Trajectory length (for normalization): {trajectory_length}")
    
    # Create data module
    data_module = RFDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        window=1,
        use_observations=use_observations,
        obs_components=obs_indices,
    )
    
    # Create model
    model = RFProposal(
        state_dim=state_dim,
        obs_dim=obs_dim,
        architecture=architecture,
        # MLP-specific
        hidden_dim=hidden_dim,
        depth=depth,
        # ResNet1D-specific
        channels=channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        # Shared
        learning_rate=learning_rate,
        num_sampling_steps=50,
        num_likelihood_steps=50,
        use_preprocessing=True,  # ALWAYS True for RF training
        train_cond_method=train_cond_method,
        cond_embed_dim=cond_embed_dim,
        num_attn_heads=num_attn_heads,
        predict_delta=predict_delta,
        time_embed_dim=time_embed_dim,
        mc_guidance=mc_guidance,
        guidance_scale=guidance_scale,
        obs_indices=obs_indices,
        cond_dropout=cond_dropout,
        debug_random_obs=debug_random_obs,
        debug_random_prev_state=debug_random_prev_state,
        use_time_step=use_time_step,
        trajectory_length=trajectory_length,
    )
    
    logger_obj.info(f"\nModel architecture:")
    logger_obj.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger_obj.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename='rf-{epoch:03d}-{val_loss:.6f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,
        mode='min',
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    
    if save_every_n_epochs is not None and save_every_n_epochs > 0:
        periodic_checkpoint = ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename='rf-periodic-{epoch:03d}',
            every_n_epochs=save_every_n_epochs,
            save_top_k=-1, # Save all periodic checkpoints
            save_last=False, # Already handled by main checkpoint callback
            save_on_train_epoch_end=True
        )
        callbacks.append(periodic_checkpoint)
        logger_obj.info(f"Saving periodic checkpoints every {save_every_n_epochs} epochs")
    
    # Setup logger
    wandb_logger = WandbLogger(
        entity="ml-climate",
        project=wandb_project,
        name=output_dir.name,
        save_dir=str(output_dir),
    )
    
    # Log hyperparameters
    wandb_logger.log_hyperparams({
        "state_dim": state_dim,
        "obs_dim": obs_dim,
        "use_observations": use_observations,
        "architecture": architecture,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "channels": channels,
        "num_blocks": num_blocks,
        "kernel_size": kernel_size,
        "train_cond_method": train_cond_method,
        "cond_embed_dim": cond_embed_dim,
        "num_attn_heads": num_attn_heads,
        "time_embed_dim": time_embed_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "use_preprocessing": True,
        "predict_delta": predict_delta,
        "data_dir": str(data_dir),
        "mc_guidance": mc_guidance,
        "guidance_scale": guidance_scale,
        "obs_indices": obs_indices,
        "cond_dropout": cond_dropout,
        "debug_random_obs": debug_random_obs,
        "debug_random_prev_state": debug_random_prev_state,
        "use_time_step": use_time_step,
        "trajectory_length": trajectory_length,
    })
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 and torch.cuda.is_available() else 'cpu',
        devices=gpus if gpus > 0 and torch.cuda.is_available() else 1,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        precision=32,
    )
    
    logger_obj.info("\nStarting training...")
    
    # Train
    trainer.fit(model, data_module)
    
    logger_obj.info("\nTraining completed!")
    logger_obj.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    logger_obj.info(f"Best validation loss: {checkpoint_callback.best_model_score:.6f}")
    
    # Save final model
    final_model_path = output_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    logger_obj.info(f"Final model saved to: {final_model_path}")
    
    return model, checkpoint_callback.best_model_path, wandb_logger


def main():
    parser = argparse.ArgumentParser(description='Train Rectified Flow Proposal')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing data.h5')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--state_dim', type=int, default=3,
                        help='State dimension')
    parser.add_argument('--obs_dim', type=int, default=1,
                        help='Observation dimension')  # state and obs dim should come from dataset / config at some point
    parser.add_argument('--architecture', type=str, default='mlp',
                        choices=['mlp', 'resnet1d'],
                        help='Velocity network architecture')
    # MLP-specific arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (for MLP)')
    parser.add_argument('--depth', type=int, default=4,
                        help='Network depth (for MLP)')
    # ResNet1D-specific arguments
    parser.add_argument('--channels', type=int, default=64,
                        help='Number of channels (for ResNet1D)')
    parser.add_argument('--num_blocks', type=int, default=6,
                        help='Number of residual blocks (for ResNet1D)')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size (for ResNet1D)')
    # Conditioning arguments
    parser.add_argument('--train-cond-method', type=str, default='concat',
                        choices=['concat', 'film', 'adaln', 'cross_attn'],
                        help='Conditioning method used during training')
    parser.add_argument('--cond_embed_dim', type=int, default=128,
                        help='Conditioning embedding dimension (for film/adaln/cross_attn)')
    parser.add_argument('--num_attn_heads', type=int, default=4,
                        help='Number of attention heads (for cross_attn)')
    parser.add_argument('--time_embed_dim', type=int, default=64,
                        help='Dimension of time embedding (default: 64 for MLP, typically 32 for ResNet1D)')
    parser.add_argument('--predict_delta', action='store_true',
                        help='Learn increment (x_curr - x_prev) instead of absolute target (residual mode)')
    parser.add_argument('--use_time_step', action='store_true',
                        help='Condition on trajectory time step')
    parser.add_argument('--trajectory_length', type=int, default=None,
                        help='Length of trajectory (for time step normalization). Defaults to config.yaml value.')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    
    # Other arguments
    parser.add_argument('--use_observations', action='store_true',
                        help='Use observation conditioning')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate after training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to evaluate (if --evaluate is set)')
    
    # Inference arguments
    parser.add_argument('--mc-guidance', action='store_true',
                        help='Enable Monte Carlo guidance by default during inference')
    parser.add_argument('--guidance-scale', type=float, default=1.0,
                        help='Default scale for Monte Carlo guidance')
    parser.add_argument('--cond_dropout', type=float, default=0.1,
                        help='Probability of dropping conditioning during training (Classifier-Free Guidance)')

    parser.add_argument('--obs_components', type=str, default=None,
                        help='Comma-separated indices of observed variables (e.g. "0,2,4")')
    
    parser.add_argument('--wandb_project', type=str, default="rf-train-96",
                        help='WandB project name')

    parser.add_argument('--save_every_n_epochs', type=int, default=None,
                        help='Save a checkpoint every N epochs')

    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Debug arguments
    parser.add_argument('--debug_random_obs', action='store_true',
                        help='Replace observations with random vectors (for debugging)')
    parser.add_argument('--debug_random_prev_state', action='store_true',
                        help='Replace previous state with random vectors (for debugging)')

    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/data/da_outputs/rf_runs/run_{timestamp}"
    
    # Parse obs_components
    obs_components = None
    if args.obs_components is not None:
        try:
            obs_components = [int(i) for i in args.obs_components.split(',') if i.strip()]
        except ValueError:
            raise ValueError(f"Invalid format for --obs-components: {args.obs_components}. Expected comma-separated integers.")
    else:
        # Try to load from config.yaml to get default components
        config_path = Path(args.data_dir) / "config.yaml"
        if config_path.exists():
             try:
                 from data import load_config_yaml
                 config = load_config_yaml(config_path)
                 obs_components = config.obs_components
                 print(f"Loaded obs_components from config.yaml: {obs_components}")
             except Exception as e:
                 print(f"Could not load obs_components from config.yaml: {e}")

    # Infer obs_dim from obs_components if provided
    if obs_components is not None:
        args.obs_dim = len(obs_components)
        
    # Infer trajectory_length from config if not provided
    trajectory_length = args.trajectory_length
    if trajectory_length is None:
        # Try to load from config.yaml
        config_path = Path(args.data_dir) / "config.yaml"
        if config_path.exists():
             try:
                 from data import load_config_yaml
                 config = load_config_yaml(config_path)
                 trajectory_length = config.len_trajectory
                 print(f"Loaded trajectory_length from config.yaml: {trajectory_length}")
             except Exception as e:
                 print(f"Could not load trajectory_length from config.yaml: {e}")
    
    if trajectory_length is None:
        trajectory_length = 1000 # Default fallback
        print(f"Using default trajectory_length: {trajectory_length}")
    
    # Set seed if provided
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
    
    # Train model
    wandb_logger = None
    if args.checkpoint is None:
        model, best_checkpoint, wandb_logger = train_rectified_flow(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            state_dim=args.state_dim,
            obs_dim=args.obs_dim,
            architecture=args.architecture,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            channels=args.channels,
            num_blocks=args.num_blocks,
            kernel_size=args.kernel_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            use_observations=args.use_observations,
            train_cond_method=args.train_cond_method,
            cond_embed_dim=args.cond_embed_dim,
            num_attn_heads=args.num_attn_heads,
            gpus=args.gpus,
            predict_delta=args.predict_delta,
            time_embed_dim=args.time_embed_dim,
            mc_guidance=args.mc_guidance,
            guidance_scale=args.guidance_scale,
            obs_indices=obs_components,
            cond_dropout=args.cond_dropout,
            wandb_project=args.wandb_project,
            save_every_n_epochs=args.save_every_n_epochs,
            debug_random_obs=args.debug_random_obs,
            debug_random_prev_state=args.debug_random_prev_state,
            use_time_step=args.use_time_step,
            trajectory_length=trajectory_length,
        )
        checkpoint_to_eval = best_checkpoint
    else:
        checkpoint_to_eval = args.checkpoint
    
    # Evaluate model
    if args.evaluate:
        # Use the wandb run from the logger if available
        wandb_run = None
        if wandb_logger and hasattr(wandb_logger, 'experiment'):
             wandb_run = wandb_logger.experiment

        run_proposal_eval(
            checkpoint_path=checkpoint_to_eval,
            data_dir=args.data_dir,
            n_trajectories=None,  # Evaluate on ALL trajectories
            n_vis_trajectories=10, # Visualize only the first 10
            batch_size=args.batch_size,
            device='cuda' if (args.gpus > 0 and torch.cuda.is_available()) else 'cpu',
            wandb_run=wandb_run
        )


if __name__ == "__main__":
    main()