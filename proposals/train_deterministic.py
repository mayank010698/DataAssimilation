"""
Training script for Deterministic Next-Step Prediction Model

Trains a deterministic model to learn x_next from x_prev (and optionally y).
Used as a sanity check to verify the ResNet1D architecture before flow matching.
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

from deterministic_model import DeterministicModel
from rf_dataset import RFDataModule


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


def train_deterministic(
    data_dir: str,
    output_dir: str,
    state_dim: int = 10,
    obs_dim: int = 0,
    channels: int = 64,
    num_blocks: int = 6,
    kernel_size: int = 5,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    max_epochs: int = 500,
    num_workers: int = 4,
    use_observations: bool = False,
    conditioning_method: str = 'concat',
    cond_embed_dim: int = 128,
    num_attn_heads: int = 4,
    gpus: int = 1,
    output_l2_reg: float = 0.0,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 10,
    predict_residual: bool = True,
):
    """
    Train a deterministic next-step prediction model.
    
    Args:
        data_dir: Directory containing data.h5
        output_dir: Directory to save checkpoints and logs
        state_dim: Dimension of state space
        obs_dim: Dimension of observation space
        channels: Number of channels in ResNet1D
        num_blocks: Number of residual blocks
        kernel_size: Kernel size for conv layers
        batch_size: Training batch size
        learning_rate: Learning rate
        max_epochs: Maximum number of epochs
        num_workers: Number of dataloader workers
        use_observations: Whether to condition on observations
        conditioning_method: Method for observation conditioning
        cond_embed_dim: Embedding dimension for conditioning
        num_attn_heads: Number of attention heads (for cross_attn)
        gpus: Number of GPUs to use
        output_l2_reg: L2 regularization on output layer (0 to disable, recommended: 1e-4)
        weight_decay: Weight decay for optimizer
        scheduler_patience: Patience for LR scheduler
        predict_residual: If True, learn delta (x_next = x_prev + delta).
                          If False, directly predict x_next.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not use_observations:
        obs_dim = 0
    
    logger_obj = setup_logging(output_dir)
    logger_obj.info("="*80)
    logger_obj.info("Training Deterministic Next-Step Prediction Model")
    logger_obj.info("="*80)
    logger_obj.info(f"Data directory: {data_dir}")
    logger_obj.info(f"Output directory: {output_dir}")
    logger_obj.info(f"State dimension: {state_dim}")
    logger_obj.info(f"Observation dimension: {obs_dim}")
    logger_obj.info(f"Use observations: {use_observations}")
    logger_obj.info(f"Architecture: ResNet1D (Deterministic)")
    logger_obj.info(f"  Channels: {channels}")
    logger_obj.info(f"  Num blocks: {num_blocks}")
    logger_obj.info(f"  Kernel size: {kernel_size}")
    logger_obj.info(f"Conditioning method: {conditioning_method}")
    if conditioning_method in ['film', 'adaln', 'cross_attn']:
        logger_obj.info(f"  Conditioning embed dim: {cond_embed_dim}")
    if conditioning_method == 'cross_attn':
        logger_obj.info(f"  Number of attention heads: {num_attn_heads}")
    logger_obj.info(f"Batch size: {batch_size}")
    logger_obj.info(f"Learning rate: {learning_rate}")
    logger_obj.info(f"Weight decay: {weight_decay}")
    logger_obj.info(f"Scheduler patience: {scheduler_patience}")
    logger_obj.info(f"Output L2 regularization: {output_l2_reg}")
    logger_obj.info(f"Predict residual: {predict_residual}")
    logger_obj.info(f"Max epochs: {max_epochs}")
    logger_obj.info(f"GPUs: {gpus}")
    
    # Create data module (reuse RF data module - same data format)
    data_module = RFDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        window=1,
        use_observations=use_observations,
    )
    
    # Create model
    model = DeterministicModel(
        state_dim=state_dim,
        obs_dim=obs_dim,
        channels=channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        learning_rate=learning_rate,
        conditioning_method=conditioning_method,
        cond_embed_dim=cond_embed_dim,
        num_attn_heads=num_attn_heads,
        output_l2_reg=output_l2_reg,
        weight_decay=weight_decay,
        scheduler_patience=scheduler_patience,
        predict_residual=predict_residual,
    )
    
    logger_obj.info(f"\nModel architecture:")
    logger_obj.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger_obj.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename='det-{epoch:03d}-{val_loss:.6f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    wandb_logger = WandbLogger(
        entity="ml-climate",
        project="deterministic-proposal",
        name=output_dir.name,
        save_dir=str(output_dir),
    )
    
    # Log hyperparameters
    wandb_logger.log_hyperparams({
        "model_type": "deterministic",
        "state_dim": state_dim,
        "obs_dim": obs_dim,
        "use_observations": use_observations,
        "channels": channels,
        "num_blocks": num_blocks,
        "kernel_size": kernel_size,
        "conditioning_method": conditioning_method,
        "cond_embed_dim": cond_embed_dim,
        "num_attn_heads": num_attn_heads,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler_patience": scheduler_patience,
        "output_l2_reg": output_l2_reg,
        "predict_residual": predict_residual,
        "max_epochs": max_epochs,
        "data_dir": str(data_dir),
    })
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 and torch.cuda.is_available() else 'cpu',
        devices=gpus if gpus > 0 and torch.cuda.is_available() else 1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
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
    parser = argparse.ArgumentParser(description='Train Deterministic Next-Step Prediction Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing data.h5')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--state_dim', type=int, default=40,
                        help='State dimension')
    parser.add_argument('--obs_dim', type=int, default=40,
                        help='Observation dimension')
    parser.add_argument('--channels', type=int, default=64,
                        help='Number of channels')
    parser.add_argument('--num_blocks', type=int, default=6,
                        help='Number of residual blocks')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Kernel size')
    
    # Conditioning arguments
    parser.add_argument('--conditioning_method', type=str, default='adaln',
                        choices=['concat', 'film', 'adaln', 'cross_attn'],
                        help='Conditioning method')
    parser.add_argument('--cond_embed_dim', type=int, default=128,
                        help='Conditioning embedding dimension')
    parser.add_argument('--num_attn_heads', type=int, default=4,
                        help='Number of attention heads (for cross_attn)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (default lowered to 3e-4 for stability)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                        help='Patience for LR scheduler')
    parser.add_argument('--output_l2_reg', type=float, default=0.0,
                        help='L2 regularization on output layer (recommended: 1e-4, 0 to disable)')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Maximum number of epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    
    # Residual prediction (default: True, use --no_predict_residual to disable)
    parser.add_argument('--no_predict_residual', action='store_false', dest='predict_residual',
                        default=True,
                        help='Disable residual prediction (directly predict x_next instead of learning delta). Default: residual prediction enabled')
    
    # Other arguments
    parser.add_argument('--use_observations', action='store_true',
                        help='Use observation conditioning')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate after training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to evaluate (if --evaluate is set)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./deterministic_runs/run_{timestamp}"
    
    # Train model
    wandb_logger = None
    if args.checkpoint is None:
        model, best_checkpoint, wandb_logger = train_deterministic(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            state_dim=args.state_dim,
            obs_dim=args.obs_dim,
            channels=args.channels,
            num_blocks=args.num_blocks,
            kernel_size=args.kernel_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_patience=args.scheduler_patience,
            output_l2_reg=args.output_l2_reg,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            use_observations=args.use_observations,
            conditioning_method=args.conditioning_method,
            cond_embed_dim=args.cond_embed_dim,
            num_attn_heads=args.num_attn_heads,
            gpus=args.gpus,
            predict_residual=args.predict_residual,
        )
        checkpoint_to_eval = best_checkpoint
    else:
        checkpoint_to_eval = args.checkpoint
    
    # Evaluate model
    if args.evaluate:
        # Import here to avoid circular imports
        try:
            from eval_deterministic import run_deterministic_eval
        except ImportError:
            from proposals.eval_deterministic import run_deterministic_eval
        
        # Use the wandb run from the logger if available
        wandb_run = None
        if wandb_logger and hasattr(wandb_logger, 'experiment'):
            wandb_run = wandb_logger.experiment

        run_deterministic_eval(
            checkpoint_path=checkpoint_to_eval,
            data_dir=args.data_dir,
            n_trajectories=None,  # Evaluate on ALL trajectories
            n_vis_trajectories=10,
            batch_size=args.batch_size,
            device='cuda' if (args.gpus > 0 and torch.cuda.is_available()) else 'cpu',
            wandb_run=wandb_run
        )


if __name__ == "__main__":
    main()

