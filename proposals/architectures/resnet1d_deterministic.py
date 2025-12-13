"""
1D ResNet for deterministic next-step prediction with periodic (circular) convolutions.

This is a simplified version of ResNet1DVelocityNetwork without flow-time dependency.
Used for sanity-checking the architecture before flow matching.

Designed for systems with periodic boundary conditions like Lorenz96.
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
from pathlib import Path

# Handle both package import and direct script execution
try:
    from .resnet1d import (
        ResBlock1D,
        Conv1dConcatConditioning,
        Conv1dFiLMConditioning,
        Conv1dAdaLNConditioning,
        Conv1dCrossAttentionConditioning,
    )
except ImportError:
    # Add parent directories to path for direct script execution
    file_path = Path(__file__).resolve()
    parent_dir = file_path.parent.parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from proposals.architectures.resnet1d import (
        ResBlock1D,
        Conv1dConcatConditioning,
        Conv1dFiLMConditioning,
        Conv1dAdaLNConditioning,
        Conv1dCrossAttentionConditioning,
    )


class ResNet1DDeterministic(nn.Module):
    """
    1D ResNet for deterministic next-step prediction with outer residual connection.
    
    Predicts x_next = x_prev + f_nn(x_prev) where f_nn learns the INCREMENT.
    This residual structure is critical for learning dynamics (Brajard et al. 2020).
    No flow-time s input - this is a direct regression model.
    
    Architecture:
        1. Project x_prev to channels: (batch, D) -> (batch, C, D)
        2. Stack of ResBlock1D with optional observation conditioning
        3. Project back: (batch, C, D) -> (batch, D)
        4. Add outer residual: output = x_prev + f_nn(x_prev)
    
    Args:
        state_dim: Dimension of state space (also spatial dimension for conv)
        channels: Number of channels in conv layers
        num_blocks: Number of residual blocks
        kernel_size: Kernel size for conv layers (should be odd). Default 5 to capture
                     Lorenz96's x_{i-2} dependency.
        obs_dim: Dimension of observations (0 for unconditional)
        conditioning_method: One of ['concat', 'film', 'adaln', 'cross_attn']
        cond_embed_dim: Embedding dimension for conditioning
        num_attn_heads: Number of attention heads (for cross_attn only)
        predict_residual: If True, network learns delta and returns x_prev + delta.
                          If False, network directly predicts x_next.
    """
    
    def __init__(
        self,
        state_dim: int,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        obs_dim: int = 0,
        conditioning_method: str = 'adaln',
        cond_embed_dim: int = 128,
        num_attn_heads: int = 4,
        predict_residual: bool = True,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.channels = channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.conditioning_method = conditioning_method
        self.predict_residual = predict_residual
        
        # Conditioning dimension: just obs_dim (no time embedding, x_prev is input)
        # If no observations, we still need conditioning for the modules that expect it
        cond_dim = obs_dim if obs_dim > 0 else 1  # Minimum dim for conditioning modules
        self.cond_dim = cond_dim
        self.has_observations = obs_dim > 0
        
        # Create conditioning module based on method
        if conditioning_method == 'concat':
            self.conditioning = Conv1dConcatConditioning(channels, cond_dim)
            # For concat, we add cond as extra channels at input
            input_channels = 1 + (cond_dim if self.has_observations else 0)
        elif conditioning_method == 'film':
            self.conditioning = Conv1dFiLMConditioning(
                channels, cond_dim, num_blocks, cond_embed_dim
            )
            input_channels = 1
        elif conditioning_method == 'adaln':
            self.conditioning = Conv1dAdaLNConditioning(
                channels, cond_dim, num_blocks, cond_embed_dim
            )
            input_channels = 1
        elif conditioning_method == 'cross_attn':
            self.conditioning = Conv1dCrossAttentionConditioning(
                channels, cond_dim, num_blocks, num_attn_heads, cond_embed_dim
            )
            input_channels = 1
        else:
            raise ValueError(f"Unknown conditioning method: {conditioning_method}")
        
        # Input projection: lift to channel dimension
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
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict next state.
        
        If predict_residual=True (default):
            x_next = x_prev + f_nn(x_prev)
            The network learns the INCREMENT (delta), not the full next state.
            This residual structure makes learning easier because:
            - At initialization, output ≈ x_prev (good starting point)
            - Network only needs to learn the small correction term
        
        If predict_residual=False:
            x_next = f_nn(x_prev)
            The network directly predicts the full next state.
        
        Args:
            x_prev: Previous state, shape (batch, state_dim)
            y: Observation conditioning, shape (batch, obs_dim) or None
            
        Returns:
            Predicted next state, shape (batch, state_dim)
        """
        batch_size = x_prev.shape[0]
        
        # Build conditioning vector (just observations, or dummy if none)
        if self.has_observations and y is not None:
            cond = y
        elif self.has_observations:
            # Expected observations but got None - pad with zeros
            cond = torch.zeros(batch_size, self.obs_dim, 
                              device=x_prev.device, dtype=x_prev.dtype)
        else:
            # No observations expected - create dummy conditioning
            cond = torch.zeros(batch_size, 1, 
                              device=x_prev.device, dtype=x_prev.dtype)
        
        # Reshape x_prev for conv: (batch, state_dim) -> (batch, 1, state_dim)
        x_conv = x_prev.unsqueeze(1)
        
        # Handle input based on conditioning method
        if self.conditioning_method == 'concat' and self.has_observations:
            # Expand conditioning to spatial dimension and concatenate as channels
            cond_spatial = cond.unsqueeze(-1).expand(-1, -1, self.state_dim)
            x_conv = torch.cat([x_conv, cond_spatial], dim=1)
        
        # Input projection
        h = self.input_proj(x_conv)  # (batch, channels, state_dim)
        
        # Residual blocks with conditioning
        for i, block in enumerate(self.res_blocks):
            h = block(h)
            # Apply conditioning after each block (for non-concat methods)
            if self.conditioning_method != 'concat' and self.has_observations:
                h = self.conditioning(h, cond, layer_idx=i)
        
        # Output projection
        output = self.output_proj(h)  # (batch, 1, state_dim)
        
        # Reshape back: (batch, 1, state_dim) -> (batch, state_dim)
        output = output.squeeze(1)
        
        if self.predict_residual:
            # OUTER RESIDUAL: x_next = x_prev + delta
            # This is the key architectural choice from Brajard et al. (2020)
            # Network learns the increment/delta, not the full next state
            return x_prev + output
        else:
            # Direct prediction: x_next = f_nn(x_prev)
            # Network directly predicts the full next state
            return output


def test_resnet1d_deterministic():
    """Test the deterministic ResNet1D implementation"""
    print("Testing ResNet1D Deterministic...")
    
    state_dim = 40  # Lorenz96 dimension
    obs_dim = 40
    batch_size = 16
    
    # Test with observations and AdaLN conditioning
    print("\n=== Testing with observations (AdaLN) ===")
    model = ResNet1DDeterministic(
        state_dim=state_dim,
        obs_dim=obs_dim,
        channels=32,
        num_blocks=4,
        kernel_size=5,
        conditioning_method='adaln',
    )
    
    x_prev = torch.randn(batch_size, state_dim)
    y = torch.randn(batch_size, obs_dim)
    
    x_next = model(x_prev, y)
    print(f"✓ Forward pass: {x_next.shape}")
    assert x_next.shape == (batch_size, state_dim)
    
    # Test the outer residual: verify that delta is being computed
    # x_next = x_prev + delta, so delta = x_next - x_prev
    delta = x_next - x_prev
    delta_norm = torch.norm(delta, dim=-1).mean().item()
    x_prev_norm = torch.norm(x_prev, dim=-1).mean().item()
    print(f"✓ Outer residual check: ||delta||={delta_norm:.4f}, ||x_prev||={x_prev_norm:.4f}")
    print(f"  (Network learns increment, not full state)")
    
    # Test without residual prediction (direct prediction)
    print("\n=== Testing without residual prediction ===")
    model_direct = ResNet1DDeterministic(
        state_dim=state_dim,
        obs_dim=obs_dim,
        channels=32,
        num_blocks=4,
        kernel_size=5,
        conditioning_method='adaln',
        predict_residual=False,
    )
    x_next_direct = model_direct(x_prev, y)
    print(f"✓ Direct prediction: {x_next_direct.shape}")
    assert x_next_direct.shape == (batch_size, state_dim)
    
    # Test without observations
    print("\n=== Testing without observations ===")
    model_uncond = ResNet1DDeterministic(
        state_dim=state_dim,
        obs_dim=0,
        channels=32,
        num_blocks=4,
        kernel_size=5,
        conditioning_method='adaln',
    )
    
    x_next_uncond = model_uncond(x_prev)
    print(f"✓ Forward pass (uncond): {x_next_uncond.shape}")
    assert x_next_uncond.shape == (batch_size, state_dim)
    
    # Test all conditioning methods
    print("\n=== Testing all conditioning methods ===")
    for method in ['concat', 'film', 'adaln', 'cross_attn']:
        model_test = ResNet1DDeterministic(
            state_dim=state_dim,
            obs_dim=obs_dim,
            channels=32,
            num_blocks=4,
            conditioning_method=method,
        )
        out = model_test(x_prev, y)
        print(f"✓ {method}: {out.shape}")
        assert out.shape == (batch_size, state_dim)
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total parameters: {total_params:,}")
    
    # Test that residual structure helps at initialization
    print("\n=== Testing residual initialization behavior ===")
    # With zero-initialized output layer, delta should be ~0, so x_next ≈ x_prev
    model_zero = ResNet1DDeterministic(
        state_dim=state_dim,
        obs_dim=0,
        channels=32,
        num_blocks=4,
    )
    # Zero out the output projection weights
    nn.init.zeros_(model_zero.output_proj.weight)
    nn.init.zeros_(model_zero.output_proj.bias)
    
    x_test = torch.randn(batch_size, state_dim)
    x_next_zero = model_zero(x_test)
    diff = torch.norm(x_next_zero - x_test, dim=-1).mean().item()
    print(f"✓ With zero output layer: ||x_next - x_prev|| = {diff:.6f}")
    assert diff < 1e-5, "With zeroed output, x_next should equal x_prev"
    print(f"  (This confirms outer residual: x_next = x_prev + 0)")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_resnet1d_deterministic()

