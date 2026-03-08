"""
Gated velocity network wrapper.

Decomposes velocity into a prior branch and a gated observation-correction branch.
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .base import BaseVelocityNetwork

class GatedVelocityNetwork(BaseVelocityNetwork):
    """
    Wrapper that combines a prior velocity network and an observation-correction network
    with a learnable gate.
    
    v(x, s | x_prev, y) = v_prior(x, s | x_prev) + g(x_prev, y) * v_obs(x, s | x_prev, y)
    """
    def __init__(
        self,
        architecture: str,
        state_dim: int,
        obs_dim: int,
        obs_indices: Optional[List[int]] = None,
        conditioning_method: str = 'concat',
        use_time_step: bool = False,
        # Gating args
        gate_type: str = 'scalar',
        gate_hidden_dim: int = 64,
        gate_init_bias: float = 0.0,
        prior_zero_init: bool = True,
        obs_zero_init: bool = False,
        **kwargs
    ):
        super().__init__(state_dim, obs_dim, conditioning_method)
        
        self.gate_type = gate_type
        
        # Handle Observation Indices
        if obs_indices is not None:
            self.register_buffer('obs_indices', torch.tensor(obs_indices, dtype=torch.long))
        else:
            self.obs_indices = None
        
        # Import factory locally to avoid circular import
        from . import create_velocity_network
        
        # 1. Prior Network (Unconditional on y)
        # We force obs_dim=0.
        # For ResNet1D, we pass zero_init_output=prior_zero_init
        self.prior_net = create_velocity_network(
            architecture=architecture,
            state_dim=state_dim,
            obs_dim=0, # No observations
            conditioning_method=conditioning_method,
            use_time_step=use_time_step,
            zero_init_output=prior_zero_init,
            **kwargs
        )
        
        # 2. Observation Correction Network
        # Uses full obs_dim.
        # For ResNet1D, we pass zero_init_output=obs_zero_init
        self.obs_net = create_velocity_network(
            architecture=architecture,
            state_dim=state_dim,
            obs_dim=obs_dim,
            obs_indices=obs_indices,
            conditioning_method=conditioning_method,
            use_time_step=use_time_step,
            zero_init_output=obs_zero_init,
            **kwargs
        )
        
        # 3. Gate Network
        if gate_type == 'scalar':
            # Scalar gate: g(x_prev, y) -> (1,)
            # Input is flat concatenation of x_prev and y
            gate_input_dim = state_dim + obs_dim
            
            self.gate_net = nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(gate_hidden_dim, 1)
            )
            
            # Initialize bias
            nn.init.constant_(self.gate_net[-1].bias, gate_init_bias)
            
        elif gate_type == 'spatial':
            # Spatial gate: g(x_prev, y) -> (state_dim,)
            # Only for ResNet1D / 1D data
            
            if architecture != 'resnet1d':
                raise ValueError("gate_type='spatial' is only supported for 'resnet1d' architecture.")
                
            # We reuse the "inpainting" logic: x_prev, y_full, mask
            # Channels: 1 (x_prev) + 1 (y_full) + 1 (mask) = 3
            gate_in_channels = 3
            gate_channels = kwargs.get('channels', 64) # Reuse main channels
            kernel_size = kwargs.get('kernel_size', 5)
            
            # Use circular padding for consistency with ResNet1D
            padding = (kernel_size - 1) // 2
            
            self.gate_net = nn.Sequential(
                nn.Conv1d(gate_in_channels, gate_channels, kernel_size, padding=padding, padding_mode='circular'),
                nn.SiLU(),
                nn.Conv1d(gate_channels, 1, kernel_size=1)
            )
            
            # Initialize bias
            nn.init.constant_(self.gate_net[-1].bias, gate_init_bias)
            
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
            
    def forward(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor, 
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Prior
        v_prior = self.prior_net(x, s, x_prev, None, t)
        
        # If no observation, return prior
        if y is None:
            return v_prior
            
        # 2. Obs Correction
        v_obs = self.obs_net(x, s, x_prev, y, t)
        
        # 3. Gate
        if self.gate_type == 'scalar':
            # Flatten inputs if needed (ResNet inputs are (B, L))
            x_prev_flat = x_prev.reshape(x_prev.shape[0], -1)
            y_flat = y.reshape(y.shape[0], -1)
            
            gate_in = torch.cat([x_prev_flat, y_flat], dim=-1)
            gate_logit = self.gate_net(gate_in) # (B, 1)
            gate = torch.sigmoid(gate_logit)
            
            # Broadcast gate to match v shape (B, state_dim)
            # gate is (B, 1), v_obs is (B, state_dim)
            if v_obs.dim() > 1:
                # Expand dims to match v_obs for broadcasting
                # e.g. if v_obs is (B, L), gate (B, 1) broadcasts fine
                pass
            
        elif self.gate_type == 'spatial':
            # Construct spatial inputs: x_prev, y_full, mask
            B, L = x_prev.shape
            device = x_prev.device
            
            # x_prev channel
            x_prev_in = x_prev.unsqueeze(1) # (B, 1, L)
            
            # y_full, mask channels
            y_full = torch.zeros(B, L, device=device, dtype=x_prev.dtype)
            mask = torch.zeros(B, L, device=device, dtype=x_prev.dtype)
            
            if self.obs_indices is not None:
                # Ensure obs_indices is on correct device
                if self.obs_indices.device != device:
                     # This shouldn't happen if registered as buffer in parent, but here it's a list or tensor?
                     # In GatedVelocityNetwork, self.obs_indices is passed as arg.
                     # If it's a list, we need to convert.
                     pass
                
                # We need to handle obs_indices carefully.
                # In ResNet1DVelocityNetwork, it's a buffer. Here it's stored as attribute.
                # Let's convert to tensor if list
                indices = self.obs_indices
                if isinstance(indices, list):
                    indices = torch.tensor(indices, device=device, dtype=torch.long)
                elif isinstance(indices, torch.Tensor):
                    indices = indices.to(device)
                
                y_full[:, indices] = y
                mask[:, indices] = 1.0
                
            elif y.shape[1] == L:
                y_full = y
                mask = torch.ones_like(mask)
            else:
                y_full[:, :y.shape[1]] = y
                mask[:, :y.shape[1]] = 1.0
                
            y_in = y_full.unsqueeze(1)
            mask_in = mask.unsqueeze(1)
            
            gate_in = torch.cat([x_prev_in, y_in, mask_in], dim=1) # (B, 3, L)
            gate_logit = self.gate_net(gate_in) # (B, 1, L)
            gate = torch.sigmoid(gate_logit).squeeze(1) # (B, L)
            
        # Combine
        return v_prior + gate * v_obs
