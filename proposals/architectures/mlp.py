"""
MLP-based velocity network for rectified flow.

Suitable for low-dimensional systems like Lorenz63.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseVelocityNetwork
from .conditioning import create_conditioning


class MLPVelocityNetwork(BaseVelocityNetwork):
    """
    MLP-based velocity network v_θ(x, s | x_prev, y)
    
    Supports multiple conditioning methods:
    - 'concat': Concatenate [x, time_embed, x_prev, y]
    - 'film': Feature-wise Linear Modulation
    - 'adaln': Adaptive Layer Normalization  
    - 'cross_attn': Cross-attention
    
    Args:
        state_dim: Dimension of state space
        hidden_dim: Hidden layer dimension
        depth: Number of hidden layers
        time_embed_dim: Dimension of time embedding
        obs_dim: Dimension of observations
        conditioning_method: One of ['concat', 'film', 'adaln', 'cross_attn']
        cond_embed_dim: Embedding dimension for conditioning (for film/adaln/cross_attn)
        num_attn_heads: Number of attention heads (for cross_attn only)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        depth: int = 4,
        time_embed_dim: int = 64,
        obs_dim: int = 0,
        conditioning_method: str = 'concat',
        cond_embed_dim: int = 128,
        num_attn_heads: int = 4,
    ):
        super().__init__(state_dim, obs_dim, conditioning_method)
        
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.depth = depth
        
        # Time embedding: map s ∈ [0,1] to higher dimensional space
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Create conditioning module
        cond_dim = state_dim + obs_dim  # [x_prev, y]
        
        self.conditioning = create_conditioning(
            method=conditioning_method,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            num_layers=depth,
            cond_embed_dim=cond_embed_dim,
            num_attn_heads=num_attn_heads,
        )
        
        # Input dimension: [x, time_embed] + optional conditioning (for concat only)
        input_dim = state_dim + time_embed_dim + self.conditioning.get_input_dim_adjustment()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_activation = nn.SiLU()
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(depth - 1)
        ])
        
        self.hidden_activations = nn.ModuleList([
            nn.SiLU()
            for _ in range(depth - 1)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, state_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor, 
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute velocity v_θ(x, s | x_prev, y)
        
        Args:
            x: Current position in flow, shape (batch, state_dim)
            s: Flow time ∈ [0,1], shape (batch, 1) or (batch,)
            x_prev: Conditioning (previous state), shape (batch, state_dim)
            y: Observation conditioning, shape (batch, obs_dim) or None
            
        Returns:
            Velocity vector, shape (batch, state_dim)
        """
        if s.dim() == 1:
            s = s.unsqueeze(1)
        
        # Time embedding
        s_embed = self.time_embed(s)
        
        # Prepare conditioning: [x_prev, y] or just x_prev if y is None
        if self.obs_dim > 0 and y is not None:
            cond = torch.cat([x_prev, y], dim=-1)
        else:
            # If no observations, pad with zeros or just use x_prev
            if self.obs_dim > 0:
                # Expected observations but got None - pad with zeros
                y_dummy = torch.zeros(x_prev.shape[0], self.obs_dim, 
                                     device=x_prev.device, dtype=x_prev.dtype)
                cond = torch.cat([x_prev, y_dummy], dim=-1)
            else:
                cond = x_prev
        
        # Build input based on conditioning method
        if self.conditioning_method == 'concat':
            # Concatenate everything at input
            net_input = torch.cat([x, s_embed, cond], dim=-1)
        else:
            # For other methods, just concatenate x and time embedding
            net_input = torch.cat([x, s_embed], dim=-1)
        
        # Input layer
        h = self.input_layer(net_input)
        h = self.input_activation(h)
        
        # Apply conditioning at input layer if not concat
        if self.conditioning_method != 'concat':
            h = self.conditioning(h, cond, layer_idx=0)
        
        # Hidden layers with conditioning
        for i, (layer, activation) in enumerate(zip(self.hidden_layers, self.hidden_activations)):
            h = layer(h)
            h = activation(h)
            
            # Apply conditioning at each hidden layer if not concat
            if self.conditioning_method != 'concat':
                h = self.conditioning(h, cond, layer_idx=i+1)
        
        # Output layer
        return self.output_layer(h)

