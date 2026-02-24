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
    use_time_step: bool = False,
    **kwargs,
) -> BaseVelocityNetwork:
    """
    Factory function to create velocity network.
    
    Args:
        architecture: One of 'mlp', 'resnet1d'
        state_dim: Dimension of state space
        obs_dim: Dimension of observations (0 for unconditional)
        conditioning_method: Ignored for current fixed architectures (they use specific methods internally)
        use_time_step: Whether to condition on trajectory time step
        **kwargs: Additional architecture-specific arguments
        
    Returns:
        Velocity network instance
        
    Architecture-specific kwargs:
        MLP:
            hidden_dim: int = 128
            depth: int = 4
            time_embed_dim: int = 64
            obs_indices: Optional[List[int]] = None
            dropout: float = 0.0
            
        ResNet1D:
            channels: int = 64
            num_blocks: int = 6
            kernel_size: int = 5
            time_embed_dim: int = 64
            obs_indices: Optional[List[int]] = None
            dropout: float = 0.0
    """
    if architecture == 'mlp' or architecture == 'mlp_fixed':
        return MLPVelocityNetwork(
            state_dim=state_dim,
            obs_dim=obs_dim,
            obs_indices=kwargs.get('obs_indices', None),
            hidden_dim=kwargs.get('hidden_dim', 128),
            depth=kwargs.get('depth', 4),
            time_embed_dim=kwargs.get('time_embed_dim', 64),
            dropout=kwargs.get('dropout', 0.0),
            use_time_step=use_time_step,
        )
    elif architecture == 'resnet1d':
        return ResNet1DVelocityNetwork(
            state_dim=state_dim,
            obs_dim=obs_dim,
            obs_indices=kwargs.get('obs_indices', None),
            channels=kwargs.get('channels', 64),
            num_blocks=kwargs.get('num_blocks', 6),
            kernel_size=kwargs.get('kernel_size', 5),
            time_embed_dim=kwargs.get('time_embed_dim', 64),
            dropout=kwargs.get('dropout', 0.0),
            use_time_step=use_time_step,
        )
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: 'mlp', 'resnet1d'"
        )


__all__ = [
    # Base classes
    'BaseVelocityNetwork',
    'BaseConditioning',
    # Velocity networks
    'MLPVelocityNetwork',
    'ResNet1DVelocityNetwork',
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
