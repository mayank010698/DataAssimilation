"""
Base class for velocity network architectures.

All velocity networks must implement this interface for use with RFProposal.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class BaseVelocityNetwork(nn.Module, ABC):
    """
    Abstract base class for velocity networks v_θ(x, s | x_prev, y)
    
    All velocity network architectures must inherit from this class and
    implement the forward method.
    
    Attributes:
        state_dim: Dimension of state space
        obs_dim: Dimension of observations (0 if unconditional on observations)
        conditioning_method: Method used for conditioning ('concat', 'film', 'adaln', 'cross_attn')
    """
    
    def __init__(
        self,
        state_dim: int,
        obs_dim: int = 0,
        conditioning_method: str = 'concat',
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.conditioning_method = conditioning_method
        
        # Validate conditioning method
        valid_methods = ['concat', 'film', 'adaln', 'cross_attn']
        if conditioning_method not in valid_methods:
            raise ValueError(
                f"conditioning_method must be one of {valid_methods}, got {conditioning_method}"
            )
    
    @abstractmethod
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
        pass

