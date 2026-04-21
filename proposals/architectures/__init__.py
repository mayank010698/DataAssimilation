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
from .gated import GatedVelocityNetwork
from .shortcut_mlp import ShortcutMLPVelocityNetwork
from .shortcut_resnet1d import ShortcutResNet1DVelocityNetwork
from .local_mlp import LocalMLPVelocityNetwork
from .local_resnet1d import LocalResNet1DVelocityNetwork
from .gaussian_head import MLPGaussianHead, ResNet1DGaussianHead
from .feature_backbones import MLPFeatureBackbone, ResNet1DFeatureBackbone
from .cde_heads import (
    BaseCDEHead,
    GaussianHead,
    MDNHead,
    JointMoGHead,
    RNADEHead,
    create_cde_head,
)
from .conditioning import (
    BaseConditioning,
    ConcatConditioning,
    FiLMConditioning,
    AdaLNConditioning,
    CrossAttentionConditioning,
    create_conditioning,
)


def create_local_velocity_network(
    architecture: str,
    radius: int,
    use_obs: bool = True,
    use_time_step: bool = False,
    **kwargs,
):
    """
    Factory for patch-based local velocity networks.

    The network returns the scalar velocity at the centre of a 2*radius+1 window.

    Args:
        architecture: 'local_mlp' or 'local_resnet1d'.
        radius: Spatial radius r.
        use_obs: Whether observations are fed into the net.
        use_time_step: Whether trajectory-time conditioning is used.
        **kwargs: architecture-specific hyperparameters.

    Architecture-specific kwargs:
        local_mlp:
            hidden_dim: int = 128
            depth: int = 4
            time_embed_dim: int = 64
            dropout: float = 0.0
            zero_init_output: bool = True

        local_resnet1d:
            channels: int = 32
            num_blocks: int = 2
            kernel_size: int = 3
            time_embed_dim: int = 64
            zero_init_output: bool = True
    """
    if architecture == "local_mlp":
        return LocalMLPVelocityNetwork(
            radius=radius,
            hidden_dim=kwargs.get("hidden_dim", 128),
            depth=kwargs.get("depth", 4),
            time_embed_dim=kwargs.get("time_embed_dim", 64),
            use_obs=use_obs,
            use_time_step=use_time_step,
            dropout=kwargs.get("dropout", 0.0),
            zero_init_output=kwargs.get("zero_init_output", True),
        )
    elif architecture == "local_resnet1d":
        return LocalResNet1DVelocityNetwork(
            radius=radius,
            channels=kwargs.get("channels", 32),
            num_blocks=kwargs.get("num_blocks", 2),
            kernel_size=kwargs.get("kernel_size", 3),
            time_embed_dim=kwargs.get("time_embed_dim", 64),
            use_obs=use_obs,
            use_time_step=use_time_step,
            zero_init_output=kwargs.get("zero_init_output", True),
        )
    else:
        raise ValueError(
            f"Unknown local architecture: {architecture}. "
            f"Available: 'local_mlp', 'local_resnet1d'"
        )


def create_shortcut_network(
    architecture: str,
    state_dim: int,
    obs_dim: int = 0,
    use_time_step: bool = False,
    div_head_active: bool = False,
    **kwargs,
):
    """
    Factory function for shortcut velocity networks (Stage 2 / Stage 3 of F2D2).

    Args:
        architecture: 'mlp' or 'resnet1d'
        state_dim: State dimension.
        obs_dim: Observation dimension (0 = unconditional).
        use_time_step: Whether to accept trajectory-time conditioning.
        div_head_active: If True the divergence head is live (Stage 3).
        **kwargs: Architecture-specific hyper-parameters (same as create_velocity_network).

    Returns:
        ShortcutMLPVelocityNetwork or ShortcutResNet1DVelocityNetwork
    """
    common = dict(
        state_dim=state_dim,
        obs_dim=obs_dim,
        obs_indices=kwargs.get("obs_indices", None),
        time_embed_dim=kwargs.get("time_embed_dim", 64),
        dropout=kwargs.get("dropout", 0.0),
        use_time_step=use_time_step,
        div_head_active=div_head_active,
        div_hidden_dim=kwargs.get("div_hidden_dim", 64),
    )
    if architecture in ("mlp", "mlp_fixed"):
        return ShortcutMLPVelocityNetwork(
            hidden_dim=kwargs.get("hidden_dim", 128),
            depth=kwargs.get("depth", 4),
            **common,
        )
    elif architecture == "resnet1d":
        return ShortcutResNet1DVelocityNetwork(
            channels=kwargs.get("channels", 64),
            num_blocks=kwargs.get("num_blocks", 6),
            kernel_size=kwargs.get("kernel_size", 5),
            zero_init_output=kwargs.get("zero_init_output", True),
            **common,
        )
    else:
        raise ValueError(
            f"Unknown architecture for shortcut network: {architecture}. "
            "Available: 'mlp', 'resnet1d'"
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
            zero_init_output=kwargs.get('zero_init_output', True),
        )
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: 'mlp', 'resnet1d'"
        )


def create_feature_backbone(
    architecture: str,
    state_dim: int,
    obs_dim: int = 0,
    use_time_step: bool = False,
    **kwargs,
):
    """Factory for the inference-network feature backbones.

    Constructs a network that maps ``(x_prev, y, [t])`` to a pooled
    feature vector in ``R^feature_dim``, ready to feed a CDE head
    (see :func:`create_cde_head`).

    Args:
        architecture: 'mlp' or 'resnet1d'.
        state_dim: State dimension.
        obs_dim: Observation dimension (0 = unconditional).
        use_time_step: Whether to condition on trajectory time.
        **kwargs: Architecture-specific hyperparameters. Accepts the same
            options as :func:`create_velocity_network` plus an MLP-only
            ``hidden_dim`` (which doubles as the feature dim) and a
            ResNet1D-only ``feature_dim`` override (default 128).

    Returns:
        :class:`MLPFeatureBackbone` or :class:`ResNet1DFeatureBackbone`.
        The returned module exposes a ``feature_dim`` attribute used by
        the CDE-head factory to size its linear projections.
    """
    if architecture in ("mlp", "mlp_fixed"):
        return MLPFeatureBackbone(
            state_dim=state_dim,
            obs_dim=obs_dim,
            obs_indices=kwargs.get("obs_indices", None),
            hidden_dim=kwargs.get("hidden_dim", 128),
            depth=kwargs.get("depth", 4),
            time_embed_dim=kwargs.get("time_embed_dim", 64),
            dropout=kwargs.get("dropout", 0.0),
            use_time_step=use_time_step,
        )
    elif architecture == "resnet1d":
        return ResNet1DFeatureBackbone(
            state_dim=state_dim,
            obs_dim=obs_dim,
            obs_indices=kwargs.get("obs_indices", None),
            channels=kwargs.get("channels", 64),
            num_blocks=kwargs.get("num_blocks", 6),
            kernel_size=kwargs.get("kernel_size", 5),
            time_embed_dim=kwargs.get("time_embed_dim", 64),
            dropout=kwargs.get("dropout", 0.0),
            use_time_step=use_time_step,
            feature_dim=kwargs.get("feature_dim", 128),
        )
    else:
        raise ValueError(
            f"Unknown architecture for feature backbone: {architecture}. "
            "Available: 'mlp', 'resnet1d'"
        )


def create_gaussian_head_network(
    architecture: str,
    state_dim: int,
    obs_dim: int = 0,
    use_time_step: bool = False,
    **kwargs,
):
    """Factory for the NASMC Gaussian-head backbones.

    Constructs a network that maps ``(x_prev, y, [t])`` to
    ``(B, 2 * state_dim)``, interpreted as ``[mu_raw, log_sigma]`` per
    site.

    Args:
        architecture: 'mlp' or 'resnet1d'.
        state_dim: State dimension.
        obs_dim: Observation dimension (0 = unconditional).
        use_time_step: Whether the network conditions on trajectory time.
        **kwargs: Architecture-specific hyperparameters. Accepts the same
            options as :func:`create_velocity_network` plus
            ``init_log_sigma`` which controls the initial value of the
            log-std bias (default -1.0, i.e. sigma ~ 0.37).
    """
    init_log_sigma = float(kwargs.pop("init_log_sigma", -1.0))
    zero_init_output = bool(kwargs.pop("zero_init_output", True))

    if architecture in ("mlp", "mlp_fixed"):
        return MLPGaussianHead(
            state_dim=state_dim,
            obs_dim=obs_dim,
            obs_indices=kwargs.get("obs_indices", None),
            hidden_dim=kwargs.get("hidden_dim", 128),
            depth=kwargs.get("depth", 4),
            time_embed_dim=kwargs.get("time_embed_dim", 64),
            dropout=kwargs.get("dropout", 0.0),
            use_time_step=use_time_step,
            zero_init_output=zero_init_output,
            init_log_sigma=init_log_sigma,
        )
    elif architecture == "resnet1d":
        return ResNet1DGaussianHead(
            state_dim=state_dim,
            obs_dim=obs_dim,
            obs_indices=kwargs.get("obs_indices", None),
            channels=kwargs.get("channels", 64),
            num_blocks=kwargs.get("num_blocks", 6),
            kernel_size=kwargs.get("kernel_size", 5),
            time_embed_dim=kwargs.get("time_embed_dim", 64),
            dropout=kwargs.get("dropout", 0.0),
            use_time_step=use_time_step,
            zero_init_output=zero_init_output,
            init_log_sigma=init_log_sigma,
        )
    else:
        raise ValueError(
            f"Unknown architecture for gaussian head: {architecture}. "
            "Available: 'mlp', 'resnet1d'"
        )


__all__ = [
    # Base classes
    'BaseVelocityNetwork',
    'BaseConditioning',
    # Velocity networks (teacher / standard RF)
    'MLPVelocityNetwork',
    'ResNet1DVelocityNetwork',
    'GatedVelocityNetwork',
    # Shortcut / F2D2 networks (Stages 2 & 3)
    'ShortcutMLPVelocityNetwork',
    'ShortcutResNet1DVelocityNetwork',
    # Localized (patch-based) networks
    'LocalMLPVelocityNetwork',
    'LocalResNet1DVelocityNetwork',
    # Deterministic networks
    'ResNet1DDeterministic',
    # Gaussian-head networks (NASMC)
    'MLPGaussianHead',
    'ResNet1DGaussianHead',
    # Inference-network feature backbones & CDE heads (Paige-Wood)
    'MLPFeatureBackbone',
    'ResNet1DFeatureBackbone',
    'BaseCDEHead',
    'GaussianHead',
    'MDNHead',
    'JointMoGHead',
    'RNADEHead',
    'create_cde_head',
    'create_feature_backbone',
    # Conditioning modules
    'ConcatConditioning',
    'FiLMConditioning',
    'AdaLNConditioning',
    'CrossAttentionConditioning',
    # Factory functions
    'create_velocity_network',
    'create_shortcut_network',
    'create_local_velocity_network',
    'create_gaussian_head_network',
    'create_conditioning',
]
