import argparse
import logging
import os
import sys
import yaml
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data import DataAssimilationDataModule, DataAssimilationConfig, load_config_yaml
from models.flowdas import ScoreNet, loss_fn, prepare_batch

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
            convert_to_pairs(self.test_dataset)

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
    from data import Lorenz63, Lorenz96
    
    # Check dataset dir name or config to guess system
    if "lorenz96" in str(args.data_dir).lower() or "96" in str(args.config) or "lorenz96" in str(args.config).lower():
        system_class = Lorenz96
    else:
        system_class = Lorenz63

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
    
    model = ScoreNet(
        marginal_prob_std=None, # Not used in training forward pass for SI?
        x_dim=state_dim,
        extra_dim=state_dim, # We condition on previous state x_prev
        hidden_depth=args.depth,
        embed_dim=args.width,
        use_bn=args.use_bn,
        architecture=args.architecture
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda t: 1 - (t / args.epochs))
    
    # Training Loop
    best_val_loss = float('inf')
    save_dir = Path(args.run_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting training on {device}...")
    logging.info(f"State Dim: {state_dim}")
    logging.info(f"Batches per epoch: {len(train_loader)}")
    
    # Initialize WandB
    if args.use_wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args.__dict__,
            name=f"flowdas_{args.architecture}_d{args.depth}_w{args.width}"
        )
        # Watch model
        wandb.watch(model, log="all", log_freq=100)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Train FlowDAS Model")
    
    parser.add_argument("--config", type=str, required=True, help="Path to data config file (yaml)")
    parser.add_argument("--data_dir", type=str, default="datasets/lorenz63_default", help="Path to dataset directory")
    parser.add_argument("--run_dir", type=str, default="runs_train/flowdas_test", help="Directory to save runs")
    
    # Model params
    parser.add_argument("--architecture", type=str, default="mlp", choices=["mlp", "resnet1d"], help="Model architecture")
    parser.add_argument("--width", type=int, default=128, help="Embedding/Hidden dimension")
    parser.add_argument("--depth", type=int, default=2, help="Depth (layers or blocks)")
    parser.add_argument("--use_bn", action="store_true", help="Use Batch Normalization")
    
    # Training params
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="flowdas-train", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="ml-climate", help="WandB entity")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)


