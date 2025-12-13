"""
Velocity network architectures for rectified flow.

Available architectures:
- MLPVelocityNetwork: MLP-based, suitable for low-dimensional systems (e.g., Lorenz63)
- ResNet1DVelocityNetwork: 1D ResNet with circular padding, suitable for periodic systems (e.g., Lorenz96)
"""

from .base import BaseVelocityNetwork
from .mlp import MLPVelocityNetwork
from .resnet1d import ResNet1DVelocityNetwork
from .resnet1d_deterministic import ResNet1DDeterministic
from .resnet1d_fixed import ResNet1DFixed
from .conditioning import (
    BaseConditioning,
    ConcatConditioning,
    FiLMConditioning,
    AdaLNConditioning,
    CrossAttentionConditioning,
    create_conditioning,
)


def create_velocity_network(
    architecture: str,
    state_dim: int,
    obs_dim: int = 0,
    conditioning_method: str = 'concat',
    **kwargs,
) -> BaseVelocityNetwork:
    """
    Factory function to create velocity network.
    
    Args:
        architecture: One of 'mlp', 'resnet1d', 'resnet1d_fixed'
        state_dim: Dimension of state space
        obs_dim: Dimension of observations (0 for unconditional)
        conditioning_method: One of 'concat', 'film', 'adaln', 'cross_attn' (renamed to train_cond_method in caller)
        **kwargs: Additional architecture-specific arguments
        
    Returns:
        Velocity network instance
        
    Architecture-specific kwargs:
        MLP:
            hidden_dim: int = 128
            depth: int = 4
            time_embed_dim: int = 64
            cond_embed_dim: int = 128
            num_attn_heads: int = 4
            
        ResNet1D:
            channels: int = 64
            num_blocks: int = 6
            kernel_size: int = 3
            time_embed_dim: int = 64
            cond_embed_dim: int = 128
            num_attn_heads: int = 4
            
        ResNet1DFixed:
            channels: int = 64
            num_blocks: int = 6
            kernel_size: int = 5
            time_embed_dim: int = 64
            obs_indices: Optional[List[int]] = None
            dropout: float = 0.0
    """
    if architecture == 'mlp':
        return MLPVelocityNetwork(
            state_dim=state_dim,
            obs_dim=obs_dim,
            conditioning_method=conditioning_method,
            hidden_dim=kwargs.get('hidden_dim', 128),
            depth=kwargs.get('depth', 4),
            time_embed_dim=kwargs.get('time_embed_dim', 64),
            cond_embed_dim=kwargs.get('cond_embed_dim', 128),
            num_attn_heads=kwargs.get('num_attn_heads', 4),
        )
    elif architecture == 'resnet1d':
        return ResNet1DVelocityNetwork(
            state_dim=state_dim,
            obs_dim=obs_dim,
            conditioning_method=conditioning_method,
            channels=kwargs.get('channels', 64),
            num_blocks=kwargs.get('num_blocks', 6),
            kernel_size=kwargs.get('kernel_size', 3),
            time_embed_dim=kwargs.get('time_embed_dim', 64),
            cond_embed_dim=kwargs.get('cond_embed_dim', 128),
            num_attn_heads=kwargs.get('num_attn_heads', 4),
        )
    elif architecture == 'resnet1d_fixed':
        return ResNet1DFixed(
            state_dim=state_dim,
            obs_dim=obs_dim,
            obs_indices=kwargs.get('obs_indices', None),
            channels=kwargs.get('channels', 64),
            num_blocks=kwargs.get('num_blocks', 6),
            kernel_size=kwargs.get('kernel_size', 5),
            time_embed_dim=kwargs.get('time_embed_dim', 64),
            dropout=kwargs.get('dropout', 0.0),
        )
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: 'mlp', 'resnet1d', 'resnet1d_fixed'"
        )


__all__ = [
    # Base classes
    'BaseVelocityNetwork',
    'BaseConditioning',
    # Velocity networks
    'MLPVelocityNetwork',
    'ResNet1DVelocityNetwork',
    'ResNet1DFixed',
    # Deterministic networks
    'ResNet1DDeterministic',
    # Conditioning modules
    'ConcatConditioning',
    'FiLMConditioning',
    'AdaLNConditioning',
    'CrossAttentionConditioning',
    # Factory functions
    'create_velocity_network',
    'create_conditioning',
]

