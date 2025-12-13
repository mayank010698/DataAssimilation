"""
Conditioning modules for velocity networks.

Provides multiple conditioning mechanisms:
- ConcatConditioning: Concatenate conditioning to input
- FiLMConditioning: Feature-wise Linear Modulation
- AdaLNConditioning: Adaptive Layer Normalization
- CrossAttentionConditioning: Cross-attention based conditioning
"""

import torch
import torch.nn as nn
from typing import Optional


class BaseConditioning(nn.Module):
    """
    Base class for conditioning mechanisms.
    All conditioning methods must implement this interface.
    """
    
    def __init__(self, hidden_dim: int, cond_dim: int):
        """
        Args:
            hidden_dim: Dimension of hidden activations
            cond_dim: Dimension of conditioning (state_dim + obs_dim for x_prev + y)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
    
    def get_input_dim_adjustment(self) -> int:
        """
        Returns how much to adjust the input dimension.
        
        For concat: return cond_dim (we add conditioning to input)
        For FiLM/AdaLN/Attention: return 0 (conditioning applied separately)
        """
        raise NotImplementedError
    
    def forward(self, h: torch.Tensor, cond: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Apply conditioning to hidden activations.
        
        Args:
            h: Hidden activations, shape (batch, hidden_dim)
            cond: Conditioning input [x_prev, y], shape (batch, cond_dim)
            layer_idx: Which layer this is being applied to (0-indexed)
            
        Returns:
            Conditioned hidden activations, shape (batch, hidden_dim)
        """
        raise NotImplementedError


class ConcatConditioning(BaseConditioning):
    """
    Concatenation-based conditioning.
    Concatenates conditioning to the input before the first layer.
    """
    
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__(hidden_dim, cond_dim)
        # No additional parameters needed
    
    def get_input_dim_adjustment(self) -> int:
        return self.cond_dim
    
    def forward(self, h: torch.Tensor, cond: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # Concatenation happens at input, so this is a no-op for hidden layers
        return h


class FiLMConditioning(BaseConditioning):
    """
    Feature-wise Linear Modulation (FiLM).
    
    Applies affine transformation: h_out = scale * h + shift
    where scale and shift are predicted from the conditioning.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        cond_dim: int, 
        num_layers: int,
        cond_embed_dim: int = 128,
    ):
        super().__init__(hidden_dim, cond_dim)
        
        # Encode conditioning into a shared embedding
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        
        # Create separate FiLM generators for each layer
        # Each outputs scale and shift parameters
        self.film_layers = nn.ModuleList([
            nn.Linear(cond_embed_dim, 2 * hidden_dim)
            for _ in range(num_layers)
        ])
    
    def get_input_dim_adjustment(self) -> int:
        return 0  # Conditioning applied separately
    
    def forward(self, h: torch.Tensor, cond: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # Encode conditioning once (will be the same across calls in same forward pass)
        # Note: In practice, we'll cache this in the main network or recompute
        # For simplicity here we recompute, but it's cheap
        cond_embed = self.cond_encoder(cond)
        
        # Get scale and shift for this specific layer
        film_params = self.film_layers[layer_idx](cond_embed)
        scale, shift = torch.chunk(film_params, 2, dim=-1)
        
        # Apply affine transformation
        return scale * h + shift


class AdaLNConditioning(BaseConditioning):
    """
    Adaptive Layer Normalization (AdaLN).
    
    Applies layer normalization followed by FiLM-style modulation:
    h_out = scale * LayerNorm(h) + shift
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        cond_dim: int, 
        num_layers: int,
        cond_embed_dim: int = 128,
    ):
        super().__init__(hidden_dim, cond_dim)
        
        # Encode conditioning
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # AdaLN parameters (scale and shift) for each layer
        self.adaLN_layers = nn.ModuleList([
            nn.Linear(cond_embed_dim, 2 * hidden_dim)
            for _ in range(num_layers)
        ])
    
    def get_input_dim_adjustment(self) -> int:
        return 0  # Conditioning applied separately
    
    def forward(self, h: torch.Tensor, cond: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # Encode conditioning
        cond_embed = self.cond_encoder(cond)
        
        # Apply layer normalization
        h_norm = self.layer_norms[layer_idx](h)
        
        # Get scale and shift
        adaLN_params = self.adaLN_layers[layer_idx](cond_embed)
        scale, shift = torch.chunk(adaLN_params, 2, dim=-1)
        
        # Apply modulation
        return scale * h_norm + shift


class CrossAttentionConditioning(BaseConditioning):
    """
    Cross-attention based conditioning.
    
    Uses multi-head attention where:
    - Query: current hidden state h
    - Key/Value: conditioning (x_prev, y)
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        cond_dim: int, 
        num_layers: int,
        num_heads: int = 4,
        cond_embed_dim: int = 128,
    ):
        super().__init__(hidden_dim, cond_dim)
        
        # Ensure cond_embed_dim is divisible by num_heads
        if cond_embed_dim % num_heads != 0:
            # Adjust cond_embed_dim to be divisible
            new_dim = ((cond_embed_dim + num_heads - 1) // num_heads) * num_heads
            if new_dim != cond_embed_dim:
                # Ideally we'd warn, but for now just use the adjusted value
                cond_embed_dim = new_dim
        
        self.num_heads = num_heads
        self.head_dim = cond_embed_dim // num_heads
        self.cond_embed_dim = cond_embed_dim
        
        # Project conditioning to key/value space (shared across layers)
        self.cond_to_kv = nn.Linear(cond_dim, 2 * cond_embed_dim)
        
        # Query projection for each layer
        self.query_projections = nn.ModuleList([
            nn.Linear(hidden_dim, cond_embed_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection for each layer
        self.output_projections = nn.ModuleList([
            nn.Linear(cond_embed_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Scale factor for attention
        self.scale = self.head_dim ** -0.5
    
    def get_input_dim_adjustment(self) -> int:
        return 0  # Conditioning applied separately
    
    def forward(self, h: torch.Tensor, cond: torch.Tensor, layer_idx: int) -> torch.Tensor:
        batch_size = h.shape[0]
        
        # Project conditioning to key and value
        kv = self.cond_to_kv(cond)  # (batch, 2 * cond_embed_dim)
        k, v = torch.chunk(kv, 2, dim=-1)  # Each is (batch, cond_embed_dim)
        
        # Reshape for multi-head attention: (batch, num_heads, 1, head_dim)
        # We have 1 "token" for the conditioning
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Project h to query
        q = self.query_projections[layer_idx](h)  # (batch, cond_embed_dim)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (batch, num_heads, 1, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn_scores: (batch, num_heads, 1, 1)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: (batch, num_heads, 1, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.cond_embed_dim)
        
        # Output projection
        output = self.output_projections[layer_idx](attn_output)
        
        # Add to original hidden state (residual connection)
        return h + output


def create_conditioning(
    method: str,
    hidden_dim: int,
    cond_dim: int,
    num_layers: int,
    cond_embed_dim: int = 128,
    num_attn_heads: int = 4,
) -> BaseConditioning:
    """
    Factory function to create conditioning module.
    
    Args:
        method: One of 'concat', 'film', 'adaln', 'cross_attn'
        hidden_dim: Dimension of hidden activations
        cond_dim: Dimension of conditioning (state_dim + obs_dim)
        num_layers: Number of layers that need conditioning
        cond_embed_dim: Embedding dimension for conditioning
        num_attn_heads: Number of attention heads (for cross_attn only)
        
    Returns:
        Conditioning module instance
    """
    if method == 'concat':
        return ConcatConditioning(hidden_dim, cond_dim)
    elif method == 'film':
        return FiLMConditioning(
            hidden_dim, cond_dim, num_layers, cond_embed_dim
        )
    elif method == 'adaln':
        return AdaLNConditioning(
            hidden_dim, cond_dim, num_layers, cond_embed_dim
        )
    elif method == 'cross_attn':
        return CrossAttentionConditioning(
            hidden_dim, cond_dim, num_layers, num_attn_heads, cond_embed_dim
        )
    else:
        raise ValueError(f"Unknown conditioning method: {method}")

