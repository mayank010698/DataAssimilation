"""
1D ResNet velocity network with AdaLN conditioning and inpainting-style observation handling.

Designed for Lorenz-96 and similar systems where observations are partial and spatial structure is important.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union

from .base import BaseVelocityNetwork


class AdaLN1d(nn.Module):
    """
    Adaptive Layer Normalization for 1D data.
    
    Regresses scale and shift parameters from a global embedding (e.g., time)
    to modulate the normalized spatial feature map.
    """
    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        # GroupNorm with 1 group is equivalent to LayerNorm over channels
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Linear(embed_dim, 2 * channels)
        
        # Initialize projection to identity (scale=1, shift=0) for stability
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)
        # We'll add 1 to scale in forward to make it identity by default

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch, channels, spatial_dim)
            embed: Global embedding, shape (batch, embed_dim)
            
        Returns:
            Modulated features, shape (batch, channels, spatial_dim)
        """
        # 1. Normalize
        x_norm = self.norm(x)
        
        # 2. Regress parameters
        params = self.proj(embed)  # (batch, 2*channels)
        scale, shift = params.chunk(2, dim=1)  # (batch, channels) each
        
        # 3. Modulate (add 1 to scale so initialization at 0 results in identity)
        scale = scale.unsqueeze(-1) + 1.0
        shift = shift.unsqueeze(-1)
        
        return scale * x_norm + shift


class ResBlock1DAdaLN(nn.Module):
    """
    Residual block with AdaLN conditioning and 1D circular convolutions.
    
    Structure (Pre-activation):
        AdaLN(x, t) -> SiLU -> Conv -> AdaLN(x, t) -> SiLU -> Conv + Residual
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        time_embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Padding for 'same' convolution with circular mode
        padding = (kernel_size - 1) // 2
        
        # First sub-block
        self.adaln1 = AdaLN1d(channels, time_embed_dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            padding_mode='circular'
        )
        
        # Second sub-block
        self.adaln2 = AdaLN1d(channels, time_embed_dim)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            padding_mode='circular'
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First sub-block
        h = self.adaln1(x, t_embed)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Second sub-block
        h = self.adaln2(h, t_embed)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + residual


class ResNet1DFixed(BaseVelocityNetwork):
    """
    Fixed 1D ResNet Architecture for Data Assimilation.
    
    Features:
    - Inpainting-style conditioning: [x, x_prev, y, mask] stacked at input.
    - Global time conditioning via AdaLN.
    - Sparse observation mapping.
    - Circular convolutions for periodic boundary conditions.
    
    Args:
        state_dim: Dimension of state space (spatial dimension).
        obs_dim: Dimension of observations (can be < state_dim).
        obs_indices: List of indices where observations occur. 
                     If None and obs_dim < state_dim, assumes first obs_dim indices? 
                     Better: Must be provided if obs_dim < state_dim for correct mapping.
                     If None and obs_dim == state_dim, assumes full observation.
        channels: Number of hidden channels.
        num_blocks: Number of residual blocks.
        kernel_size: Convolution kernel size (should be odd).
        time_embed_dim: Dimension of time embedding.
    """
    def __init__(
        self,
        state_dim: int,
        obs_dim: int = 0,
        obs_indices: Optional[List[int]] = None,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
    ):
        # We don't use the base class conditioning_method argument since this architecture is fixed
        super().__init__(state_dim, obs_dim, conditioning_method='adaln')
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        
        # Handle Observation Indices
        if obs_indices is not None:
            # Register as buffer so it moves to device and saves with model
            self.register_buffer('obs_indices', torch.tensor(obs_indices, dtype=torch.long))
        else:
            # Set attribute to None directly (register_buffer expects Tensor)
            self.obs_indices = None

        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input Projection
        # Channels: 
        # 1 (x_s) 
        # + 1 (x_prev) 
        # + 1 (y_full) 
        # + 1 (mask)
        # = 4 input channels
        self.input_channels = 4
        
        self.input_proj = nn.Conv1d(
            self.input_channels, channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            padding_mode='circular'
        )
        
        # Residual Blocks
        self.blocks = nn.ModuleList([
            ResBlock1DAdaLN(channels, kernel_size, time_embed_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output Projection (Initialize to zero for better convergence?)
        self.final_norm = nn.GroupNorm(1, channels)
        self.final_act = nn.SiLU()
        self.output_proj = nn.Conv1d(channels, 1, kernel_size=1)
        
        # Zero-init output projection for "start from identity/zero-velocity" behavior
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor, 
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute velocity v(x, s | x_prev, y).
        
        Args:
            x: Current state (batch, state_dim)
            s: Time (batch, 1) or (batch,)
            x_prev: Previous state (batch, state_dim)
            y: Observations (batch, obs_dim) or None
            
        Returns:
            Velocity (batch, state_dim)
        """
        B = x.shape[0]
        
        # 1. Time Embedding
        if s.dim() == 1:
            s = s.unsqueeze(1)
        t_embed = self.time_embed(s)  # (B, time_embed_dim)
        
        # 2. Prepare Observation Channels
        # We need to construct y_full and mask
        
        # Initialize containers on correct device
        y_full = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        mask = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        
        if self.obs_dim > 0 and y is not None:
            if self.obs_indices is not None:
                # Sparse observations with known indices
                # y is (B, obs_dim)
                # We scatter y into y_full at obs_indices
                
                # Check dimensions match
                if y.shape[1] != self.obs_dim:
                    # If passed y doesn't match expected obs_dim, try to handle or warn
                    pass

                # Scatter logic:
                # y_full[:, indices] = y
                y_full[:, self.obs_indices] = y
                mask[:, self.obs_indices] = 1.0
                
            elif y.shape[1] == self.state_dim:
                # Dense observations (or pre-padded)
                y_full = y
                # If dense, assume all observed (mask=1) unless y contains specific missing values (not handled here)
                mask = torch.ones_like(mask)
            else:
                # obs_indices is None BUT y is smaller than state_dim
                # Fallback: assume first obs_dim indices (e.g. truncated state)
                y_full[:, :y.shape[1]] = y
                mask[:, :y.shape[1]] = 1.0

        # 3. Stack Inputs
        # Reshape inputs to (B, 1, L)
        x_in = x.unsqueeze(1)          # (B, 1, L)
        x_prev_in = x_prev.unsqueeze(1)# (B, 1, L)
        y_in = y_full.unsqueeze(1)     # (B, 1, L)
        mask_in = mask.unsqueeze(1)    # (B, 1, L)
        
        # Stack channels: (B, 4, L)
        net_input = torch.cat([x_in, x_prev_in, y_in, mask_in], dim=1)
        
        # 4. Forward Pass
        h = self.input_proj(net_input)
        
        for block in self.blocks:
            h = block(h, t_embed)
            
        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.output_proj(h) # (B, 1, L)
        
        return out.squeeze(1) # (B, L)

