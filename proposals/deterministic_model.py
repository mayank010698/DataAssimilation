"""
Deterministic Next-Step Prediction Model

A simple regression model that predicts x_next from x_prev (and optionally y).
Used as a sanity check to verify the architecture before flow matching.
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
import numpy as np
from typing import Optional, Tuple
import logging
import sys
from pathlib import Path

# Handle both package import and direct script execution
try:
    from .architectures.resnet1d_deterministic import ResNet1DDeterministic
except ImportError:
    # Add parent directories to path for direct script execution
    file_path = Path(__file__).resolve()
    parent_dir = file_path.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from proposals.architectures.resnet1d_deterministic import ResNet1DDeterministic


class DeterministicModel(pl.LightningModule):
    """
    Deterministic next-step prediction model.
    
    Learns p(x_t | x_{t-1}, y_t) as a point estimate using MSE loss.
    
    Args:
        state_dim: Dimension of state space
        channels: Number of channels in conv layers
        num_blocks: Number of residual blocks
        kernel_size: Kernel size for conv layers
        learning_rate: Learning rate for optimizer
        obs_dim: Dimension of observations
        conditioning_method: One of 'concat', 'film', 'adaln', 'cross_attn'
        cond_embed_dim: Embedding dimension for conditioning
        num_attn_heads: Number of attention heads (for cross_attn)
        output_l2_reg: L2 regularization coefficient for output layer (0 to disable).
                       Recommended: 1e-4 (from Brajard et al. 2020)
        weight_decay: Weight decay for optimizer
        scheduler_patience: Patience for ReduceLROnPlateau scheduler
        predict_residual: If True, network learns delta and returns x_prev + delta.
                          If False, network directly predicts x_next. Default True.
    """
    
    def __init__(
        self,
        state_dim: int,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        learning_rate: float = 3e-4,  # Lowered from 1e-3
        obs_dim: int = 0,
        conditioning_method: str = 'concat',
        cond_embed_dim: int = 128,
        num_attn_heads: int = 4,
        output_l2_reg: float = 0.0,  # Optional L2 reg on output layer
        weight_decay: float = 1e-4,  # Increased from 1e-5
        scheduler_patience: int = 10,  # Reduced from 20
        predict_residual: bool = True,  # If True, learn delta; if False, learn x_next directly
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.learning_rate = learning_rate
        self.conditioning_method = conditioning_method
        self.output_l2_reg = output_l2_reg
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.predict_residual = predict_residual
        
        # Create the network
        self.network = ResNet1DDeterministic(
            state_dim=state_dim,
            channels=channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            obs_dim=obs_dim,
            conditioning_method=conditioning_method,
            cond_embed_dim=cond_embed_dim,
            num_attn_heads=num_attn_heads,
            predict_residual=predict_residual,
        )
        
        # For tracking training progress
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(
        self, 
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict next state from previous state."""
        return self.network(x_prev, y)
    
    def compute_loss(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute MSE loss for next-step prediction.
        
        Args:
            x_prev: Previous states, shape (batch, state_dim)
            x_curr: Current states (targets), shape (batch, state_dim)
            y_curr: Current observations, shape (batch, obs_dim) or None
            
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics for logging
        """
        # Predict next state
        x_pred = self.network(x_prev, y_curr)
        
        # MSE loss
        mse_loss = torch.mean((x_pred - x_curr) ** 2)
        
        # Optional L2 regularization on output layer (like Brajard et al. 2020)
        l2_reg_loss = torch.tensor(0.0, device=x_pred.device)
        if self.output_l2_reg > 0:
            for param in self.network.output_proj.parameters():
                l2_reg_loss = l2_reg_loss + param.pow(2).sum()
            l2_reg_loss = self.output_l2_reg * l2_reg_loss
        
        loss = mse_loss + l2_reg_loss
        
        # RMSE for logging
        rmse = torch.sqrt(mse_loss)
        
        # Additional metrics
        metrics = {
            'loss': loss.item(),
            'mse_loss': mse_loss.item(),
            'l2_reg_loss': l2_reg_loss.item(),
            'rmse': rmse.item(),
            'pred_norm': torch.mean(torch.norm(x_pred, dim=-1)).item(),
            'target_norm': torch.mean(torch.norm(x_curr, dim=-1)).item(),
        }
        
        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x_prev = batch['x_prev']
        x_curr = batch['x_curr']
        y_curr = batch.get('y_curr', None)
        
        loss, metrics = self.compute_loss(x_prev, x_curr, y_curr)
        
        # NaN/Inf detection
        if torch.isnan(loss) or torch.isinf(loss):
            logging.error(f"NaN/Inf loss detected at epoch {self.current_epoch}, batch {batch_idx}!")
            logging.error(f"  x_prev range: [{x_prev.min():.4f}, {x_prev.max():.4f}]")
            logging.error(f"  x_curr range: [{x_curr.min():.4f}, {x_curr.max():.4f}]")
            logging.error(f"  pred_norm: {metrics['pred_norm']:.4f}")
            # Return a large but finite loss to allow recovery
            loss = torch.tensor(100.0, device=loss.device, requires_grad=True)
        
        # Log metrics
        self.log('train_loss', metrics['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_rmse', metrics['rmse'], on_step=False, on_epoch=True)
        if self.output_l2_reg > 0:
            self.log('train_l2_reg', metrics['l2_reg_loss'], on_step=False, on_epoch=True)
        
        self.training_step_outputs.append(metrics)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x_prev = batch['x_prev']
        x_curr = batch['x_curr']
        y_curr = batch.get('y_curr', None)
        
        loss, metrics = self.compute_loss(x_prev, x_curr, y_curr)
        
        # Log metrics
        self.log('val_loss', metrics['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_rmse', metrics['rmse'], on_step=False, on_epoch=True)
        
        self.validation_step_outputs.append(metrics)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        if len(self.training_step_outputs) > 0:
            avg_loss = np.mean([x['loss'] for x in self.training_step_outputs])
            avg_rmse = np.mean([x['rmse'] for x in self.training_step_outputs])
            logging.info(f"Epoch {self.current_epoch}: Train Loss = {avg_loss:.6f}, RMSE = {avg_rmse:.6f}")
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        if len(self.validation_step_outputs) > 0:
            avg_loss = np.mean([x['loss'] for x in self.validation_step_outputs])
            avg_rmse = np.mean([x['rmse'] for x in self.validation_step_outputs])
            logging.info(f"Epoch {self.current_epoch}: Val Loss = {avg_loss:.6f}, RMSE = {avg_rmse:.6f}")
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=self.scheduler_patience,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
    
    @torch.no_grad()
    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample (predict) next state. 
        
        For deterministic model, this is just a forward pass.
        Signature matches RFProposal.sample() for compatibility.
        
        Args:
            x_prev: Previous state, shape (state_dim,) or (batch, state_dim)
            y_curr: Current observation, shape (obs_dim,) or (batch, obs_dim) or None
            
        Returns:
            Predicted next state, shape same as x_prev
        """
        was_1d = x_prev.dim() == 1
        if was_1d:
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        
        x_next = self.network(x_prev, y_curr)
        
        if was_1d:
            x_next = x_next.squeeze(0)
        
        return x_next


def test_deterministic_model():
    """Test the deterministic model"""
    print("Testing Deterministic Model...")
    
    state_dim = 40
    obs_dim = 40
    batch_size = 16
    
    model = DeterministicModel(
        state_dim=state_dim,
        obs_dim=obs_dim,
        channels=32,
        num_blocks=4,
        kernel_size=5,
        conditioning_method='adaln',
    )
    
    # Test forward pass
    x_prev = torch.randn(batch_size, state_dim)
    y = torch.randn(batch_size, obs_dim)
    
    x_pred = model(x_prev, y)
    print(f"✓ Forward pass: {x_pred.shape}")
    assert x_pred.shape == (batch_size, state_dim)
    
    # Test loss computation
    x_curr = torch.randn(batch_size, state_dim)
    loss, metrics = model.compute_loss(x_prev, x_curr, y)
    print(f"✓ Loss computation: {loss.item():.6f}")
    
    # Test sampling (single)
    x_prev_single = torch.randn(state_dim)
    y_single = torch.randn(obs_dim)
    x_sample = model.sample(x_prev_single, y_single)
    print(f"✓ Sampling (single): {x_sample.shape}")
    assert x_sample.shape == (state_dim,)
    
    # Test sampling (batch)
    x_sample_batch = model.sample(x_prev, y)
    print(f"✓ Sampling (batch): {x_sample_batch.shape}")
    assert x_sample_batch.shape == (batch_size, state_dim)
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total parameters: {total_params:,}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_deterministic_model()

