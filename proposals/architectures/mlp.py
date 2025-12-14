"""
Fixed MLP velocity network with concatenation conditioning.

Designed for low-dimensional systems like Lorenz-63.
Treats all inputs as flat vectors and concatenates them at the input.
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .base import BaseVelocityNetwork


class MLPVelocityNetwork(BaseVelocityNetwork):
    """
    Fixed MLP Architecture for Data Assimilation (Lorenz-63).
    
    Features:
    - Input flattening: Concatenates [x, x_prev, y_full, mask, time_embed]
    - Sparse observation handling via mapping and mask
    - Simple concatenated conditioning
    
    Args:
        state_dim: Dimension of state space
        obs_dim: Dimension of observations
        obs_indices: List of indices where observations occur
        hidden_dim: Hidden dimension size
        depth: Number of hidden layers
        time_embed_dim: Dimension of time embedding
        dropout: Dropout probability
    """
    def __init__(
        self,
        state_dim: int,
        obs_dim: int = 0,
        obs_indices: Optional[List[int]] = None,
        hidden_dim: int = 128,
        depth: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
    ):
        # We don't use the generic conditioning_method here
        super().__init__(state_dim, obs_dim, conditioning_method='concat')
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.time_embed_dim = time_embed_dim
        
        # Handle Observation Indices
        if obs_indices is not None:
            self.register_buffer('obs_indices', torch.tensor(obs_indices, dtype=torch.long))
        else:
            self.obs_indices = None
            
        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input Dimension Calculation
        # If obs_dim > 0: x (state_dim) + x_prev (state_dim) + y_full (state_dim) + mask (state_dim) + time (time_embed_dim)
        # If obs_dim == 0: x (state_dim) + x_prev (state_dim) + time (time_embed_dim)
        if obs_dim > 0:
            input_dim = 4 * state_dim + time_embed_dim
        else:
            input_dim = 2 * state_dim + time_embed_dim
        
        # Build MLP
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
        # Output layer
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.net = nn.Sequential(*layers)
        
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
        
        # 2. Prepare Input based on Observations
        if self.obs_dim > 0:
            y_full = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
            mask = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
            
            if y is not None:
                if self.obs_indices is not None:
                    # Sparse observations
                    y_full[:, self.obs_indices] = y
                    mask[:, self.obs_indices] = 1.0
                elif y.shape[1] == self.state_dim:
                    # Dense observations
                    y_full = y
                    mask = torch.ones_like(mask)
                else:
                    # Fallback: assume first obs_dim indices
                    y_full[:, :y.shape[1]] = y
                    mask[:, :y.shape[1]] = 1.0
            
            # [x, x_prev, y_full, mask, t_embed]
            net_input = torch.cat([x, x_prev, y_full, mask, t_embed], dim=-1)
        else:
            # [x, x_prev, t_embed]
            net_input = torch.cat([x, x_prev, t_embed], dim=-1)
        
        # 4. Forward Pass
        out = self.net(net_input)
        
        return out
