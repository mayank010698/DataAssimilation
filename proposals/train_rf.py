"""
Training script for Rectified Flow Proposal Distribution

Trains a rectified flow model to learn p(x_t | x_{t-1}) from trajectory data.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))

from rectified_flow import RFProposal
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


def train_rectified_flow(
    data_dir: str,
    output_dir: str,
    state_dim: int = 3,
    hidden_dim: int = 128,
    depth: int = 4,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_epochs: int = 500,
    num_workers: int = 4,
    use_preprocessing: bool = True,
    gpus: int = 1,
):
    """
    Train a rectified flow proposal distribution
    
    Args:
        data_dir: Directory containing data.h5
        output_dir: Directory to save checkpoints and logs
        state_dim: Dimension of state space
        hidden_dim: Hidden dimension of velocity network
        depth: Depth of velocity network
        batch_size: Training batch size
        learning_rate: Learning rate
        max_epochs: Maximum number of epochs
        num_workers: Number of dataloader workers
        use_preprocessing: Whether data is preprocessed/normalized
        gpus: Number of GPUs to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger_obj = setup_logging(output_dir)
    logger_obj.info("="*80)
    logger_obj.info("Training Rectified Flow Proposal Distribution")
    logger_obj.info("="*80)
    logger_obj.info(f"Data directory: {data_dir}")
    logger_obj.info(f"Output directory: {output_dir}")
    logger_obj.info(f"State dimension: {state_dim}")
    logger_obj.info(f"Hidden dimension: {hidden_dim}")
    logger_obj.info(f"Network depth: {depth}")
    logger_obj.info(f"Batch size: {batch_size}")
    logger_obj.info(f"Learning rate: {learning_rate}")
    logger_obj.info(f"Max epochs: {max_epochs}")
    logger_obj.info(f"Use preprocessing: {use_preprocessing}")
    logger_obj.info(f"GPUs: {gpus}")
    
    # Create data module
    data_module = RFDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        window=1,
        use_preprocessing=use_preprocessing,
    )
    
    # Create model
    model = RFProposal(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        depth=depth,
        learning_rate=learning_rate,
        num_sampling_steps=50,
        num_likelihood_steps=50,
        use_preprocessing=use_preprocessing,
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
    
    # Setup logger
    wandb_logger = WandbLogger(
        project="rectified-flow-proposal",
        name=output_dir.name,
        save_dir=str(output_dir),
    )
    
    # Log hyperparameters
    wandb_logger.log_hyperparams({
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "use_preprocessing": use_preprocessing,
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
    
    return model, checkpoint_callback.best_model_path


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    n_trajectories: int = 5,
    n_steps: int = 100,
):
    """
    Evaluate a trained RF model by generating trajectories
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing data for initial states
        n_trajectories: Number of trajectories to generate
        n_steps: Number of steps per trajectory
    """
    import matplotlib.pyplot as plt
    import h5py
    
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*80)
    logger.info("Evaluating Trained Model")
    logger.info("="*80)
    
    # Load model
    model = RFProposal.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Load some test trajectories for initial states
    data_file = Path(data_dir) / "data.h5"
    with h5py.File(data_file, "r") as f:
        test_traj = f["test/trajectories"][:]
    
    logger.info(f"Generating {n_trajectories} trajectories with {n_steps} steps each...")
    
    # Generate trajectories
    generated_trajs = []
    true_trajs = []
    
    for i in range(min(n_trajectories, len(test_traj))):
        x_init = torch.FloatTensor(test_traj[i, 0])
        
        # Generate trajectory
        gen_traj = model.generate_trajectory(x_init, n_steps=n_steps, return_all=True)
        generated_trajs.append(gen_traj)
        
        # Get corresponding true trajectory
        true_traj = test_traj[i, :n_steps+1]
        true_trajs.append(true_traj)
    
    # Compute metrics
    generated_trajs = torch.tensor(generated_trajs)  # (n_traj, n_steps+1, state_dim)
    true_trajs = torch.tensor(true_trajs)
    
    rmse = torch.sqrt(torch.mean((generated_trajs - true_trajs) ** 2))
    logger.info(f"Average RMSE: {rmse:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, (gen_traj, true_traj) in enumerate(zip(generated_trajs, true_trajs)):
        for dim in range(3):
            axes[dim].plot(
                true_traj[:, dim].numpy(),
                label=f'True {i+1}' if dim == 0 else '',
                alpha=0.7,
                linewidth=2,
            )
            axes[dim].plot(
                gen_traj[:, dim].numpy(),
                label=f'Generated {i+1}' if dim == 0 else '',
                alpha=0.7,
                linestyle='--',
                linewidth=2,
            )
            axes[dim].set_ylabel(f'Dimension {dim+1}')
            axes[dim].grid(True, alpha=0.3)
    
    axes[0].set_title('Generated vs True Trajectories')
    axes[0].legend()
    axes[2].set_xlabel('Time Step')
    
    plt.tight_layout()
    save_path = Path(checkpoint_path).parent.parent / "trajectory_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison plot saved to: {save_path}")
    plt.close()
    
    logger.info("Evaluation completed!")


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
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=4,
                        help='Network depth')
    
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
    parser.add_argument('--use_preprocessing', action='store_true',
                        help='Whether data is preprocessed/normalized')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate after training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to evaluate (if --evaluate is set)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./rf_runs/run_{timestamp}"
    
    # Train model
    if args.checkpoint is None:
        model, best_checkpoint = train_rectified_flow(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            state_dim=args.state_dim,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            use_preprocessing=args.use_preprocessing,
            gpus=args.gpus,
        )
        checkpoint_to_eval = best_checkpoint
    else:
        checkpoint_to_eval = args.checkpoint
    
    # Evaluate model
    if args.evaluate:
        evaluate_model(
            checkpoint_path=checkpoint_to_eval,
            data_dir=args.data_dir,
            n_trajectories=5,
            n_steps=100,
        )


if __name__ == "__main__":
    main()