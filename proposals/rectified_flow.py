"""
Rectified Flow Proposal Distribution for Particle Filtering

Implements a learned proposal q(x_t | x_{t-1}) using Rectified Flow with:
- Gaussian source distribution N(0, I)
- Conditional vector field v_θ(x, s | x_{t-1}) where s ∈ [0,1]
- Exact likelihood computation via continuous normalizing flows
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
import numpy as np
from typing import Optional, Tuple
from torchdiffeq import odeint, odeint_adjoint
import logging

# Handle both package import and direct script execution
try:
    from .architectures import create_velocity_network, BaseVelocityNetwork
except ImportError:
    from architectures import create_velocity_network, BaseVelocityNetwork


class RFProposal(pl.LightningModule):
    """
    Rectified Flow Proposal Distribution
    
    Learns p(x_t | x_{t-1}) using rectified flow from N(0,I) to the data distribution.
    Can be used as a proposal distribution in particle filters.
    
    Args:
        state_dim: Dimension of state space
        architecture: Velocity network architecture ('mlp' or 'resnet1d')
        hidden_dim: Hidden dimension (for MLP)
        depth: Number of layers (for MLP)
        channels: Number of channels (for ResNet1D)
        num_blocks: Number of residual blocks (for ResNet1D)
        learning_rate: Learning rate for optimizer
        num_sampling_steps: Number of Euler steps for sampling
        num_likelihood_steps: Number of Euler steps for likelihood computation
        use_preprocessing: Whether to use preprocessing (unused, for compatibility)
        obs_dim: Dimension of observations
        conditioning_method: One of 'concat', 'film', 'adaln', 'cross_attn'
        cond_embed_dim: Embedding dimension for conditioning
        num_attn_heads: Number of attention heads (for cross_attn)
        predict_delta: If True, learn increment (x_curr - x_prev) instead of absolute x_curr.
                       This provides a residual structure that can improve learning.
        time_embed_dim: Dimension of time embedding (default: 64 for MLP, typically 32 for ResNet1D)
    """
    
    def __init__(
        self,
        state_dim: int,
        architecture: str = 'mlp',
        hidden_dim: int = 128,
        depth: int = 4,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 3,
        learning_rate: float = 1e-3,
        num_sampling_steps: int = 10,
        num_likelihood_steps: int = 10,
        use_preprocessing: bool = False,
        obs_dim: int = 0,
        train_cond_method: str = 'concat',
        cond_embed_dim: int = 128,
        num_attn_heads: int = 4,
        predict_delta: bool = False,
        time_embed_dim: int = 64,
        mc_guidance: bool = False,
        guidance_scale: float = 1.0,
        obs_indices: Optional[list] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.num_sampling_steps = num_sampling_steps
        self.num_likelihood_steps = num_likelihood_steps
        self.use_preprocessing = use_preprocessing
        self.train_cond_method = train_cond_method
        self.predict_delta = predict_delta
        self.mc_guidance = mc_guidance
        self.guidance_scale = guidance_scale
        self.obs_indices = obs_indices
        
        # Create velocity network using factory
        self.velocity_net = create_velocity_network(
            architecture=architecture,
            state_dim=state_dim,
            obs_dim=obs_dim,
            conditioning_method=train_cond_method,
            # MLP-specific
            hidden_dim=hidden_dim,
            depth=depth,
            # ResNet1D-specific
            channels=channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            # Shared
            cond_embed_dim=cond_embed_dim,
            num_attn_heads=num_attn_heads,
            time_embed_dim=time_embed_dim,
            obs_indices=obs_indices,
        )
        
        # For tracking training progress
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor, 
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity field"""
        return self.velocity_net(x, s, x_prev, y)
    
    def compute_rf_loss(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Rectified Flow training loss
        
        The loss is: E_{t~U(0,1), z~N(0,I)} ||v_θ(x(t), t | x_prev, y) - (target - z)||^2
        where x(t) = (1-t)*z + t*target is the straight-line interpolation
        
        If predict_delta=False (default): target = x_curr
        If predict_delta=True: target = x_curr - x_prev (learn increment)
        
        Args:
            x_prev: Previous states, shape (batch, state_dim)
            x_curr: Current states, shape (batch, state_dim)
            y_curr: Current observations, shape (batch, obs_dim) or None
            
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics for logging
        """
        batch_size = x_prev.shape[0]
        
        # Sample random times t ~ U(0,1)
        s = torch.rand(batch_size, 1, device=self.device)
        
        # Sample noise z ~ N(0, I)
        z = torch.randn_like(x_curr)
        
        # Determine target based on mode
        if self.predict_delta:
            # Learn the increment (residual structure)
            target = x_curr - x_prev
        else:
            # Learn absolute target (original behavior)
            target = x_curr
        
        # Linear interpolation: x(s) = (1-s)*z + s*target
        x_s = (1 - s) * z + s * target
        
        # Target velocity (constant along straight line)
        target_velocity = target - z
        
        # Predicted velocity
        pred_velocity = self.velocity_net(x_s, s, x_prev, y_curr)
        
        # MSE loss
        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        
        # Additional metrics
        metrics = {
            'loss': loss.item(),
            'velocity_norm': torch.mean(torch.norm(pred_velocity, dim=-1)).item(),
            'target_velocity_norm': torch.mean(torch.norm(target_velocity, dim=-1)).item(),
        }
        
        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Extract pairs from batch
        x_prev = batch['x_prev']
        x_curr = batch['x_curr']
        y_curr = batch.get('y_curr', None)
        
        # Compute loss
        loss, metrics = self.compute_rf_loss(x_prev, x_curr, y_curr)
        
        # Log metrics
        self.log('train_loss', metrics['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_velocity_norm', metrics['velocity_norm'], on_step=False, on_epoch=True)
        
        self.training_step_outputs.append(metrics)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x_prev = batch['x_prev']
        x_curr = batch['x_curr']
        y_curr = batch.get('y_curr', None)
        
        loss, metrics = self.compute_rf_loss(x_prev, x_curr, y_curr)
        
        # Log metrics
        self.log('val_loss', metrics['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_velocity_norm', metrics['velocity_norm'], on_step=False, on_epoch=True)
        
        self.validation_step_outputs.append(metrics)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        if len(self.training_step_outputs) > 0:
            avg_loss = np.mean([x['loss'] for x in self.training_step_outputs])
            logging.info(f"Epoch {self.current_epoch}: Train Loss = {avg_loss:.6f}")
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        if len(self.validation_step_outputs) > 0:
            avg_loss = np.mean([x['loss'] for x in self.validation_step_outputs])
            logging.info(f"Epoch {self.current_epoch}: Val Loss = {avg_loss:.6f}")
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
    
    @torch.enable_grad()
    def compute_guidance_grad(
        self,
        x_s: torch.Tensor,
        s: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: torch.Tensor,
        observation_fn: callable,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute guidance gradient for Monte Carlo guidance
        
        Args:
            x_s: Current state at time s
            s: Current time
            x_prev: Previous state (conditioning)
            y_curr: Current observation
            observation_fn: Function mapping state to observation space
            create_graph: Whether to create graph for higher-order derivatives
            
        Returns:
            Gradient of loss with respect to x_s
        """
        # Enable gradient computation for x_s if needed
        if not x_s.requires_grad:
            x_s = x_s.detach().requires_grad_(True)
        
        # 1. Lookahead Prediction: Predict final state \hat{x}_1
        # \hat{x}_1^{delta} = x_s + (1-s) * v_\theta(x_s, s)
        v = self.velocity_net(x_s, s, x_prev, y_curr)
        x_1_delta = x_s + (1.0 - s) * v
        
        # If predict_delta is True, convert to absolute state
        if self.predict_delta:
            x_1 = x_prev + x_1_delta
        else:
            x_1 = x_1_delta
            
        # 2. Observation: Apply observation_fn
        y_hat = observation_fn(x_1)
        
        # 3. Loss: Compute MSE loss
        # Handle batch dimensions correctly
        diff = y_hat - y_curr
        loss = torch.sum(diff ** 2, dim=-1).sum()  # Sum over batch for gradient
        
        # 4. Gradient: Compute \nabla_{x_s} J
        grad = torch.autograd.grad(loss, x_s, create_graph=create_graph)[0]
        
        return grad

    @torch.no_grad()
    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        observation_fn: Optional[callable] = None,
    ) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_{t-1}, y_t) using Euler method
        
        Args:
            x_prev: Previous state, shape (state_dim,) or (batch, state_dim)
            y_curr: Current observation, shape (obs_dim,) or (batch, obs_dim) or None
            dt: Time step (unused, for interface compatibility)
            observation_fn: Optional observation function for guidance
            
        Returns:
            Sampled next state, shape same as x_prev
        """
        was_1d = x_prev.dim() == 1
        if was_1d:
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        
        batch_size = x_prev.shape[0]
        
        # Start from noise z ~ N(0, I)
        x = torch.randn(batch_size, self.state_dim, device=self.device)
        
        # Euler steps from s=0 to s=1
        ds = 1.0 / self.num_sampling_steps
        
        for i in range(self.num_sampling_steps):
            s = i * ds
            s_tensor = torch.full((batch_size, 1), s, device=self.device)
            v = self.velocity_net(x, s_tensor, x_prev, y_curr)
            
            # Apply Monte Carlo Guidance if enabled
            if self.mc_guidance and observation_fn is not None and y_curr is not None:
                # We need to compute gradient, so we temporarily enable grads
                grad = self.compute_guidance_grad(x, s_tensor, x_prev, y_curr, observation_fn)
                v = v - self.guidance_scale * grad
                
            x = x + v * ds  # Euler step
        
        # If predict_delta mode, x is the delta - convert to absolute state
        if self.predict_delta:
            x = x_prev + x  # x_curr = x_prev + delta
        
        if was_1d:
            x = x.squeeze(0)
        
        return x

    @torch.no_grad()
    def sample_and_log_prob(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        observation_fn: Optional[callable] = None,
        use_exact_trace: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t and compute its log probability, optionally with guidance.
        
        Args:
            x_prev: Previous state
            y_curr: Current observation
            observation_fn: Observation function (required for guidance)
            use_exact_trace: Whether to use exact trace or Hutchinson estimator
            
        Returns:
            Tuple of (sampled_state, log_prob)
        """
        was_1d = x_prev.dim() == 1
        if was_1d:
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        
        batch_size = x_prev.shape[0]
        
        # Start from noise z ~ N(0, I)
        x = torch.randn(batch_size, self.state_dim, device=self.device)
        
        # Initial log prob is Gaussian density of x(0)
        log_prob = -0.5 * torch.sum(x ** 2, dim=-1) - 0.5 * self.state_dim * np.log(2 * np.pi)
        
        # Euler steps from s=0 to s=1
        ds = 1.0 / self.num_sampling_steps
        
        for i in range(self.num_sampling_steps):
            s = i * ds
            s_tensor = torch.full((batch_size, 1), s, device=self.device)
            
            # Use enable_grad for divergence computation
            with torch.enable_grad():
                x_for_grad = x.detach().requires_grad_(True)
                
                # 1. Compute Network Velocity & Divergence
                if use_exact_trace:
                    # EXACT TRACE
                    def v_func(x_in):
                        return self.velocity_net(x_in, s_tensor, x_prev, y_curr)
                        
                    v = v_func(x_for_grad)
                    
                    divergence = torch.zeros(batch_size, device=self.device)
                    for k in range(self.state_dim):
                        grad_k = torch.autograd.grad(
                            v[:, k], x_for_grad, 
                            grad_outputs=torch.ones_like(v[:, k]),
                            create_graph=False, retain_graph=True
                        )[0]
                        divergence += grad_k[:, k]
                else:
                    # HUTCHINSON TRACE
                    v = self.velocity_net(x_for_grad, s_tensor, x_prev, y_curr)
                    eps = torch.randn_like(x)
                    vjp = torch.autograd.grad(
                        v, x_for_grad, grad_outputs=eps, create_graph=False
                    )[0]
                    divergence = torch.sum(vjp * eps, dim=-1)
                
                # 2. Compute Guidance & Divergence (if enabled)
                if self.mc_guidance and observation_fn is not None and y_curr is not None:
                    # Compute guidance gradient
                    # IMPORTANT: create_graph=True for second derivative
                    grad = self.compute_guidance_grad(
                        x_for_grad, s_tensor, x_prev, y_curr, observation_fn, create_graph=True
                    )
                    
                    # Compute divergence of guidance (Laplacian of Loss)
                    if use_exact_trace:
                        grad_div = torch.zeros(batch_size, device=self.device)
                        for k in range(self.state_dim):
                             g_k_grad = torch.autograd.grad(
                                grad[:, k], x_for_grad,
                                grad_outputs=torch.ones_like(grad[:, k]),
                                create_graph=False, retain_graph=True
                            )[0]
                             grad_div += g_k_grad[:, k]
                    else:
                        # Hutchinson
                        # Reuse eps if available (consistent estimator)
                        # otherwise sample new eps
                        if 'eps' not in locals():
                             eps = torch.randn_like(x)
                        
                        grad_jvp = torch.autograd.grad(
                            grad, x_for_grad,
                            grad_outputs=eps,
                            create_graph=False
                        )[0]
                        grad_div = torch.sum(grad_jvp * eps, dim=-1)
                        
                    # Update velocity and divergence
                    v = v - self.guidance_scale * grad
                    divergence = divergence - self.guidance_scale * grad_div

            # Update state
            x = x + v.detach() * ds
            
            # Update log_prob
            log_prob = log_prob - divergence.detach() * ds

        # If predict_delta mode, x is the delta - convert to absolute state
        if self.predict_delta:
            x = x_prev + x
            
        if was_1d:
            x = x.squeeze(0)
            log_prob = log_prob.squeeze(0)
            
        return x, log_prob

    @torch.no_grad()
    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        use_exact_trace: bool = True,
    ) -> torch.Tensor:
        """
        Compute log prob using Euler method.
        
        Args:
            x_curr: Current state, shape (state_dim,) or (batch, state_dim)
            x_prev: Previous state (conditioning), shape same as x_curr
            y_curr: Observation conditioning, shape (obs_dim,) or (batch, obs_dim) or None
            dt: Time step (unused, for interface compatibility)
            use_exact_trace: If True, computes exact Jacobian trace (deterministic, O(D^2)).
                             If False, uses Hutchinson trace estimator (stochastic, O(D)).
        
        Returns:
            Log probability, shape () or (batch,)
        """
        was_1d = x_curr.dim() == 1
        if was_1d:
            x_curr = x_curr.unsqueeze(0)
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        
        batch_size = x_curr.shape[0]
        
        # Start from target at s=1, integrate backward to s=0
        # If predict_delta mode, we start from the delta, not x_curr
        if self.predict_delta:
            x = (x_curr - x_prev).clone()  # Start from delta
        else:
            x = x_curr.clone()  # Original: start from x_curr
        log_prob_correction = torch.zeros(batch_size, device=self.device)
        
        ds = 1.0 / self.num_likelihood_steps
        
        # Backward integration: s goes from 1 to 0
        for i in range(self.num_likelihood_steps):
            s = 1.0 - i * ds  # Start at s=1, go to s=0
            s_tensor = torch.full((batch_size, 1), s, device=self.device)
            
            # Compute velocity and divergence
            with torch.enable_grad():
                x_for_grad = x.detach().requires_grad_(True)
                
                if use_exact_trace:
                    # EXACT TRACE COMPUTATION
                    def v_func(x_in):
                        return self.velocity_net(x_in, s_tensor, x_prev, y_curr)

                    # Compute velocity for the step
                    v = v_func(x_for_grad)
                    
                    # Compute exact trace
                    divergence = torch.zeros(batch_size, device=self.device)
                    
                    # Iterate over state dimensions to compute diagonal of Jacobian
                    for k in range(self.state_dim):
                        grad_k = torch.autograd.grad(
                            v[:, k], 
                            x_for_grad, 
                            grad_outputs=torch.ones_like(v[:, k]),
                            create_graph=False,
                            retain_graph=True
                        )[0]
                        
                        divergence += grad_k[:, k]
                        
                else:
                    # HUTCHINSON TRACE ESTIMATOR (Stochastic)
                    v = self.velocity_net(x_for_grad, s_tensor, x_prev, y_curr)
                    
                    eps = torch.randn_like(x)
                    vjp = torch.autograd.grad(
                        v, x_for_grad,
                        grad_outputs=eps,
                        create_graph=False,
                    )[0]
                    divergence = torch.sum(vjp * eps, dim=-1)
            
            # Backward Euler step
            x = x - v.detach() * ds
            log_prob_correction = log_prob_correction + divergence.detach() * ds
        
        # Base log prob of final z
        log_prob_base = -0.5 * torch.sum(x ** 2, dim=-1) - 0.5 * self.state_dim * np.log(2 * np.pi)
        log_prob = log_prob_base - log_prob_correction
        
        if was_1d:
            log_prob = log_prob.squeeze(0)
        
        return log_prob
    
    @torch.no_grad()
    def generate_trajectory(
        self,
        x_initial: torch.Tensor,
        n_steps: int,
        return_all: bool = True,
    ) -> torch.Tensor:
        """
        Generate a trajectory by chaining samples
        
        Args:
            x_initial: Initial state, shape (state_dim,)
            n_steps: Number of steps to generate
            return_all: If True, return full trajectory; if False, return only final state
            
        Returns:
            If return_all: trajectory of shape (n_steps+1, state_dim)
            If not return_all: final state of shape (state_dim,)
        """
        trajectory = [x_initial.cpu().numpy()]
        x_current = x_initial
        
        for _ in range(n_steps):
            x_next = self.sample(x_current)
            trajectory.append(x_next.cpu().numpy())
            x_current = x_next
        
        if return_all:
            return np.array(trajectory)
        else:
            return trajectory[-1]


def test_rectified_flow():
    """Test the rectified flow implementation with both architectures"""
    print("Testing Rectified Flow Proposal...")
    
    # Test MLP architecture (for Lorenz63-like systems)
    print("\n=== Testing MLP Architecture ===")
    state_dim = 3
    batch_size = 16
    
    rf_mlp = RFProposal(
        state_dim=state_dim,
        architecture='mlp',
        hidden_dim=64,
        depth=3,
        num_sampling_steps=20,
        num_likelihood_steps=20,
    )
    
    # Test forward pass
    x = torch.randn(batch_size, state_dim)
    s = torch.rand(batch_size, 1)
    x_prev = torch.randn(batch_size, state_dim)
    
    v = rf_mlp(x, s, x_prev)
    print(f"✓ MLP Forward pass: {v.shape}")
    assert v.shape == (batch_size, state_dim)
    
    # Test loss computation
    x_curr = torch.randn(batch_size, state_dim)
    loss, metrics = rf_mlp.compute_rf_loss(x_prev, x_curr)
    print(f"✓ MLP Loss computation: {loss.item():.6f}")
    
    # Test sampling
    x_prev_single = torch.randn(state_dim)
    x_sample = rf_mlp.sample(x_prev_single)
    print(f"✓ MLP Sampling (single): {x_sample.shape}")
    assert x_sample.shape == (state_dim,)
    
    # Test ResNet1D architecture (for Lorenz96-like systems)
    print("\n=== Testing ResNet1D Architecture ===")
    state_dim_l96 = 40  # Lorenz96 dimension
    
    rf_resnet = RFProposal(
        state_dim=state_dim_l96,
        architecture='resnet1d',
        channels=32,
        num_blocks=4,
        kernel_size=3,
        conditioning_method='adaln',
        num_sampling_steps=20,
        num_likelihood_steps=20,
    )
    
    # Test forward pass
    x = torch.randn(batch_size, state_dim_l96)
    s = torch.rand(batch_size, 1)
    x_prev = torch.randn(batch_size, state_dim_l96)
    
    v = rf_resnet(x, s, x_prev)
    print(f"✓ ResNet1D Forward pass: {v.shape}")
    assert v.shape == (batch_size, state_dim_l96)
    
    # Test loss computation
    x_curr = torch.randn(batch_size, state_dim_l96)
    loss, metrics = rf_resnet.compute_rf_loss(x_prev, x_curr)
    print(f"✓ ResNet1D Loss computation: {loss.item():.6f}")
    
    # Test sampling
    x_prev_single = torch.randn(state_dim_l96)
    x_sample = rf_resnet.sample(x_prev_single)
    print(f"✓ ResNet1D Sampling (single): {x_sample.shape}")
    assert x_sample.shape == (state_dim_l96,)
    
    # Test batch sampling
    x_prev_batch = torch.randn(batch_size, state_dim_l96)
    x_sample_batch = rf_resnet.sample(x_prev_batch)
    print(f"✓ ResNet1D Sampling (batch): {x_sample_batch.shape}")
    assert x_sample_batch.shape == (batch_size, state_dim_l96)
    
    # Test log probability (small batch, as it's slow)
    print("\n=== Testing Log Probability ===")
    x_prev_small = torch.randn(2, state_dim)
    x_curr_small = torch.randn(2, state_dim)
    log_prob = rf_mlp.log_prob(x_curr_small, x_prev_small)
    print(f"✓ MLP Log probability: {log_prob.shape}, values: {log_prob}")
    assert log_prob.shape == (2,)
    
    # Test trajectory generation
    x_init = torch.randn(state_dim)
    trajectory = rf_mlp.generate_trajectory(x_init, n_steps=10, return_all=True)
    print(f"✓ Trajectory generation: {trajectory.shape}")
    assert trajectory.shape == (11, state_dim)
    
    # Test all conditioning methods with ResNet1D
    print("\n=== Testing Conditioning Methods (ResNet1D) ===")
    for method in ['concat', 'film', 'adaln', 'cross_attn']:
        rf_test = RFProposal(
            state_dim=state_dim_l96,
            architecture='resnet1d',
            channels=32,
            num_blocks=4,
            conditioning_method=method,
            num_sampling_steps=5,
        )
        x = torch.randn(4, state_dim_l96)
        s = torch.rand(4, 1)
        x_prev = torch.randn(4, state_dim_l96)
        v = rf_test(x, s, x_prev)
        print(f"✓ ResNet1D with {method}: {v.shape}")
        assert v.shape == (4, state_dim_l96)
    
    # Test predict_delta mode (residual learning)
    print("\n=== Testing predict_delta Mode (Residual Learning) ===")
    
    # Test with MLP
    rf_delta_mlp = RFProposal(
        state_dim=state_dim,
        architecture='mlp',
        hidden_dim=64,
        depth=3,
        num_sampling_steps=20,
        num_likelihood_steps=20,
        predict_delta=True,
    )
    
    x_prev = torch.randn(batch_size, state_dim)
    x_curr = torch.randn(batch_size, state_dim)
    
    # Test loss computation in delta mode
    loss_delta, metrics_delta = rf_delta_mlp.compute_rf_loss(x_prev, x_curr)
    print(f"✓ MLP (predict_delta) Loss computation: {loss_delta.item():.6f}")
    
    # Test sampling in delta mode
    x_prev_single = torch.randn(state_dim)
    x_sample_delta = rf_delta_mlp.sample(x_prev_single)
    print(f"✓ MLP (predict_delta) Sampling: {x_sample_delta.shape}")
    assert x_sample_delta.shape == (state_dim,)
    
    # Test batch sampling in delta mode
    x_sample_batch_delta = rf_delta_mlp.sample(x_prev)
    print(f"✓ MLP (predict_delta) Batch sampling: {x_sample_batch_delta.shape}")
    assert x_sample_batch_delta.shape == (batch_size, state_dim)
    
    # Test log probability in delta mode
    x_prev_small = torch.randn(2, state_dim)
    x_curr_small = torch.randn(2, state_dim)
    log_prob_delta = rf_delta_mlp.log_prob(x_curr_small, x_prev_small)
    print(f"✓ MLP (predict_delta) Log probability: {log_prob_delta.shape}, values: {log_prob_delta}")
    assert log_prob_delta.shape == (2,)
    
    # Test with ResNet1D in delta mode
    rf_delta_resnet = RFProposal(
        state_dim=state_dim_l96,
        architecture='resnet1d',
        channels=32,
        num_blocks=4,
        conditioning_method='adaln',
        num_sampling_steps=10,
        predict_delta=True,
    )
    
    x_prev_l96 = torch.randn(batch_size, state_dim_l96)
    x_curr_l96 = torch.randn(batch_size, state_dim_l96)
    
    loss_delta_resnet, _ = rf_delta_resnet.compute_rf_loss(x_prev_l96, x_curr_l96)
    print(f"✓ ResNet1D (predict_delta) Loss: {loss_delta_resnet.item():.6f}")
    
    x_sample_resnet_delta = rf_delta_resnet.sample(x_prev_l96)
    print(f"✓ ResNet1D (predict_delta) Sampling: {x_sample_resnet_delta.shape}")
    assert x_sample_resnet_delta.shape == (batch_size, state_dim_l96)
    
    # Verify that predict_delta flag is saved in hyperparameters
    assert rf_delta_mlp.predict_delta == True
    assert rf_mlp.predict_delta == False
    print("✓ predict_delta flag correctly stored")

    # Test Monte Carlo Guidance
    print("\n=== Testing Monte Carlo Guidance ===")
    
    # Define a simple observation function (e.g., observe first dimension)
    def observation_fn(x):
        return x[..., :1]
    
    obs_dim = 1
    
    rf_guidance = RFProposal(
        state_dim=state_dim,
        architecture='mlp',
        hidden_dim=64,
        depth=3,
        num_sampling_steps=10,
        mc_guidance=True,
        guidance_scale=0.5,
    )
    
    x_prev = torch.randn(batch_size, state_dim)
    # Create fake observations
    y_curr = torch.randn(batch_size, obs_dim)
    
    # Test sample with guidance
    x_sample_guided = rf_guidance.sample(x_prev, y_curr=y_curr, observation_fn=observation_fn)
    print(f"✓ Guided Sampling: {x_sample_guided.shape}")
    assert x_sample_guided.shape == (batch_size, state_dim)
    
    # Test sample_and_log_prob with guidance
    x_sample_guided_lp, log_prob_guided = rf_guidance.sample_and_log_prob(
        x_prev, y_curr=y_curr, observation_fn=observation_fn
    )
    print(f"✓ Guided Sampling with LogProb: {x_sample_guided_lp.shape}, {log_prob_guided.shape}")
    assert x_sample_guided_lp.shape == (batch_size, state_dim)
    assert log_prob_guided.shape == (batch_size,)
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_rectified_flow()
