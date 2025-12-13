"""
1D ResNet velocity network with periodic (circular) convolutions.

Designed for systems with periodic boundary conditions like Lorenz96.
The circular padding naturally respects the wrap-around structure of such systems.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseVelocityNetwork


class Conv1dConditioning(nn.Module):
    """
    Base class for conditioning in 1D convolutional networks.
    
    Unlike the MLP conditioning which operates on (batch, hidden_dim),
    this operates on (batch, channels, spatial_dim).
    """
    
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
    
    def forward(
        self, 
        h: torch.Tensor, 
        cond: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        """
        Apply conditioning to conv hidden activations.
        
        Args:
            h: Hidden activations, shape (batch, channels, spatial_dim)
            cond: Conditioning input [x_prev, y], shape (batch, cond_dim)
            layer_idx: Which layer this is being applied to (0-indexed)
            
        Returns:
            Conditioned hidden activations, shape (batch, channels, spatial_dim)
        """
        raise NotImplementedError


class Conv1dConcatConditioning(Conv1dConditioning):
    """
    Concatenation-based conditioning for conv networks.
    
    Expands conditioning to spatial dimension and concatenates with input channels.
    """
    
    def __init__(self, channels: int, cond_dim: int):
        super().__init__(channels, cond_dim)
    
    def get_extra_channels(self) -> int:
        """Returns number of extra input channels needed for conditioning."""
        return self.cond_dim
    
    def forward(
        self, 
        h: torch.Tensor, 
        cond: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        # For concat, conditioning is handled at input level
        # This is a no-op for intermediate layers
        return h


class Conv1dFiLMConditioning(Conv1dConditioning):
    """
    Feature-wise Linear Modulation for 1D conv networks.
    
    Applies channel-wise affine transformation: h_out = scale * h + shift
    where scale and shift are (batch, channels, 1) and broadcast over spatial dim.
    """
    
    def __init__(
        self, 
        channels: int, 
        cond_dim: int, 
        num_layers: int,
        cond_embed_dim: int = 128,
    ):
        super().__init__(channels, cond_dim)
        
        # Encode conditioning into a shared embedding
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        
        # FiLM generators for each layer (scale and shift per channel)
        self.film_layers = nn.ModuleList([
            nn.Linear(cond_embed_dim, 2 * channels)
            for _ in range(num_layers)
        ])
    
    def forward(
        self, 
        h: torch.Tensor, 
        cond: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        # h: (batch, channels, spatial_dim)
        cond_embed = self.cond_encoder(cond)  # (batch, cond_embed_dim)
        
        film_params = self.film_layers[layer_idx](cond_embed)  # (batch, 2*channels)
        scale, shift = torch.chunk(film_params, 2, dim=-1)  # Each (batch, channels)
        
        # Reshape for broadcasting: (batch, channels, 1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        
        return scale * h + shift


class Conv1dAdaLNConditioning(Conv1dConditioning):
    """
    Adaptive Layer Normalization for 1D conv networks.
    
    Applies LayerNorm over channels, then channel-wise modulation.
    """
    
    def __init__(
        self, 
        channels: int, 
        cond_dim: int, 
        num_layers: int,
        cond_embed_dim: int = 128,
    ):
        super().__init__(channels, cond_dim)
        
        # Encode conditioning
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        
        # Layer normalization for each layer (normalize over channels)
        # Using GroupNorm with num_groups=1 is equivalent to LayerNorm over channels
        self.layer_norms = nn.ModuleList([
            nn.GroupNorm(num_groups=1, num_channels=channels)
            for _ in range(num_layers)
        ])
        
        # AdaLN parameters for each layer
        self.adaLN_layers = nn.ModuleList([
            nn.Linear(cond_embed_dim, 2 * channels)
            for _ in range(num_layers)
        ])
    
    def forward(
        self, 
        h: torch.Tensor, 
        cond: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        # h: (batch, channels, spatial_dim)
        cond_embed = self.cond_encoder(cond)  # (batch, cond_embed_dim)
        
        # Apply layer normalization
        h_norm = self.layer_norms[layer_idx](h)
        
        # Get scale and shift
        adaLN_params = self.adaLN_layers[layer_idx](cond_embed)  # (batch, 2*channels)
        scale, shift = torch.chunk(adaLN_params, 2, dim=-1)  # Each (batch, channels)
        
        # Reshape for broadcasting: (batch, channels, 1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        
        return scale * h_norm + shift


class Conv1dCrossAttentionConditioning(Conv1dConditioning):
    """
    Cross-attention conditioning for 1D conv networks.
    
    Pools spatial dimension, applies cross-attention with conditioning,
    then broadcasts result back to spatial dimension.
    """
    
    def __init__(
        self, 
        channels: int, 
        cond_dim: int, 
        num_layers: int,
        num_heads: int = 4,
        cond_embed_dim: int = 128,
    ):
        super().__init__(channels, cond_dim)
        
        # Ensure cond_embed_dim is divisible by num_heads
        if cond_embed_dim % num_heads != 0:
            cond_embed_dim = ((cond_embed_dim + num_heads - 1) // num_heads) * num_heads
        
        self.num_heads = num_heads
        self.head_dim = cond_embed_dim // num_heads
        self.cond_embed_dim = cond_embed_dim
        
        # Project conditioning to key/value space
        self.cond_to_kv = nn.Linear(cond_dim, 2 * cond_embed_dim)
        
        # Query projection from pooled spatial features
        self.query_projections = nn.ModuleList([
            nn.Linear(channels, cond_embed_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection back to channels
        self.output_projections = nn.ModuleList([
            nn.Linear(cond_embed_dim, channels)
            for _ in range(num_layers)
        ])
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        h: torch.Tensor, 
        cond: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        # h: (batch, channels, spatial_dim)
        batch_size, channels, spatial_dim = h.shape
        
        # Global average pooling over spatial dimension
        h_pooled = h.mean(dim=-1)  # (batch, channels)
        
        # Project conditioning to key and value
        kv = self.cond_to_kv(cond)  # (batch, 2 * cond_embed_dim)
        k, v = torch.chunk(kv, 2, dim=-1)
        
        # Reshape for multi-head attention
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project pooled h to query
        q = self.query_projections[layer_idx](h_pooled)  # (batch, cond_embed_dim)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.cond_embed_dim)
        
        # Output projection
        output = self.output_projections[layer_idx](attn_output)  # (batch, channels)
        
        # Broadcast over spatial dimension and add residually
        output = output.unsqueeze(-1)  # (batch, channels, 1)
        
        return h + output


class ResBlock1D(nn.Module):
    """
    Residual block with 1D circular convolutions.
    
    Structure:
        h -> Conv1d -> Activation -> Conv1d -> + h (skip)
                                    ^
                                    |-- Conditioning
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        
        # Compute padding for 'same' output size
        # With circular padding, padding = (kernel_size - 1) // 2 works for odd kernels
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(
            channels, channels, 
            kernel_size=kernel_size, 
            padding=padding,
            padding_mode='circular'
        )
        self.activation1 = nn.SiLU()
        
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode='circular'
        )
        self.activation2 = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (batch, channels, spatial_dim)
            
        Returns:
            Output tensor, shape (batch, channels, spatial_dim)
        """
        residual = x
        
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.conv2(out)
        out = out + residual
        out = self.activation2(out)
        
        return out


class ResNet1DVelocityNetwork(BaseVelocityNetwork):
    """
    1D ResNet velocity network with periodic (circular) convolutions.
    
    Designed for systems with periodic boundary conditions like Lorenz96.
    
    Architecture:
        1. Project state to channels: (batch, D) -> (batch, C, D)
        2. Stack of ResBlock1D with conditioning after each block
        3. Project back: (batch, C, D) -> (batch, D)
    
    Args:
        state_dim: Dimension of state space (also spatial dimension for conv)
        channels: Number of channels in conv layers
        num_blocks: Number of residual blocks
        kernel_size: Kernel size for conv layers (should be odd). Default 5 to capture
                     Lorenz96's x_{i-2} dependency (dynamics depend on i-2, i-1, i+1).
        time_embed_dim: Dimension of time embedding
        obs_dim: Dimension of observations
        conditioning_method: One of ['concat', 'film', 'adaln', 'cross_attn']
        cond_embed_dim: Embedding dimension for conditioning
        num_attn_heads: Number of attention heads (for cross_attn only)
    """
    
    def __init__(
        self,
        state_dim: int,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,  # Changed from 3 to 5 to capture x_{i-2} dependency in Lorenz96
        time_embed_dim: int = 32,
        obs_dim: int = 0,
        conditioning_method: str = 'adaln',
        cond_embed_dim: int = 128,
        num_attn_heads: int = 4,
    ):
        super().__init__(state_dim, obs_dim, conditioning_method)
        
        self.channels = channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Conditioning dimension: y + time_embedding (x_prev is now input)
        cond_dim = obs_dim + time_embed_dim
        
        # Create conditioning module based on method
        if conditioning_method == 'concat':
            self.conditioning = Conv1dConcatConditioning(channels, cond_dim)
            # For concat, we add cond as extra channels at input
            input_channels = 2 + cond_dim  # 1 for x, 1 for x_prev, cond_dim for conditioning
        elif conditioning_method == 'film':
            self.conditioning = Conv1dFiLMConditioning(
                channels, cond_dim, num_blocks, cond_embed_dim
            )
            input_channels = 2  # x and x_prev
        elif conditioning_method == 'adaln':
            self.conditioning = Conv1dAdaLNConditioning(
                channels, cond_dim, num_blocks, cond_embed_dim
            )
            input_channels = 2  # x and x_prev
        elif conditioning_method == 'cross_attn':
            self.conditioning = Conv1dCrossAttentionConditioning(
                channels, cond_dim, num_blocks, num_attn_heads, cond_embed_dim
            )
            input_channels = 2  # x and x_prev
        else:
            raise ValueError(f"Unknown conditioning method: {conditioning_method}")
        
        # Input projection: lift to channel dimension
        # Input shape: (batch, input_channels, state_dim)
        self.input_proj = nn.Conv1d(
            input_channels, channels,
            kernel_size=1,
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock1D(channels, kernel_size)
            for _ in range(num_blocks)
        ])
        
        # Output projection: back to single channel
        self.output_proj = nn.Conv1d(
            channels, 1,
            kernel_size=1,
        )
    
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
        batch_size = x.shape[0]
        
        if s.dim() == 1:
            s = s.unsqueeze(1)
        
        # Time embedding
        s_embed = self.time_embed(s)  # (batch, time_embed_dim)
        
        # Build conditioning vector: [y, s_embed] (x_prev moved to input)
        if self.obs_dim > 0 and y is not None:
            cond = torch.cat([y, s_embed], dim=-1)
        elif self.obs_dim > 0:
            # Expected observations but got None - pad with zeros
            y_dummy = torch.zeros(batch_size, self.obs_dim, 
                                 device=x.device, dtype=x.dtype)
            cond = torch.cat([y_dummy, s_embed], dim=-1)
        else:
            # cond = torch.cat([x_prev, s_embed], dim=-1) # Old way
            cond = s_embed
        
        # Reshape x for conv: (batch, state_dim) -> (batch, 1, state_dim)
        x_conv = x.unsqueeze(1)
        
        # Reshape x_prev for conv: (batch, state_dim) -> (batch, 1, state_dim)
        x_prev_conv = x_prev.unsqueeze(1)
        
        # Stack x and x_prev as input channels: (batch, 2, state_dim)
        x_input = torch.cat([x_conv, x_prev_conv], dim=1)
        
        # Handle input based on conditioning method
        if self.conditioning_method == 'concat':
            # Expand conditioning to spatial dimension and concatenate as channels
            cond_spatial = cond.unsqueeze(-1).expand(-1, -1, self.state_dim)
            # cond_spatial: (batch, cond_dim, state_dim)
            x_input = torch.cat([x_input, cond_spatial], dim=1)
            # x_input: (batch, 2 + cond_dim, state_dim)
        
        # Input projection
        h = self.input_proj(x_input)  # (batch, channels, state_dim)
        
        # Residual blocks with conditioning
        for i, block in enumerate(self.res_blocks):
            h = block(h)
            # Apply conditioning after each block (for non-concat methods)
            if self.conditioning_method != 'concat':
                h = self.conditioning(h, cond, layer_idx=i)
        
        # Output projection
        out = self.output_proj(h)  # (batch, 1, state_dim)
        
        # Reshape back: (batch, 1, state_dim) -> (batch, state_dim)
        out = out.squeeze(1)
        
        return out

