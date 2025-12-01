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


class VelocityNetwork(nn.Module):
    """
    Neural network for velocity field v_θ(x, s | x_prev)
    
    Args:
        state_dim: Dimension of state space
        hidden_dim: Hidden layer dimension
        depth: Number of hidden layers
        time_embed_dim: Dimension of time embedding
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        depth: int = 4,
        time_embed_dim: int = 64,
        obs_dim: int = 0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.time_embed_dim = time_embed_dim
        self.obs_dim = obs_dim
        
        # Time embedding: map s ∈ [0,1] to higher dimensional space
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Main network: [x, time_embed, x_prev, y] -> velocity
        input_dim = state_dim + time_embed_dim + state_dim + obs_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.net = nn.Sequential(*layers)
        
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
        
        # Concatenate all inputs
        if self.obs_dim > 0 and y is not None:
            net_input = torch.cat([x, s_embed, x_prev, y], dim=-1)
        else:
            net_input = torch.cat([x, s_embed, x_prev], dim=-1)
        
        return self.net(net_input)


class RFProposal(pl.LightningModule):
    """
    Rectified Flow Proposal Distribution
    
    Learns p(x_t | x_{t-1}) using rectified flow from N(0,I) to the data distribution.
    Can be used as a proposal distribution in particle filters.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        depth: int = 4,
        learning_rate: float = 1e-3,
        num_sampling_steps: int = 10,  # small for testing
        num_likelihood_steps: int = 10,  # small for testing
        use_preprocessing: bool = False,
        obs_dim: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.learning_rate = learning_rate
        self.num_sampling_steps = num_sampling_steps
        self.num_likelihood_steps = num_likelihood_steps
        self.use_preprocessing = use_preprocessing
        
        # Velocity network
        self.velocity_net = VelocityNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            obs_dim=obs_dim,
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
        
        The loss is: E_{t~U(0,1), z~N(0,I)} ||v_θ(x(t), t | x_prev, y) - (x_curr - z)||^2
        where x(t) = (1-t)*z + t*x_curr is the straight-line interpolation
        
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
        
        # Linear interpolation: x(s) = (1-s)*z + s*x_curr
        x_s = (1 - s) * z + s * x_curr
        
        # Target velocity (constant along straight line)
        target_velocity = x_curr - z
        
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
            # verbose=True,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }
    
    @torch.no_grad()
    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_{t-1}, y_t) using Euler method
        
        Args:
            x_prev: Previous state, shape (state_dim,) or (batch, state_dim)
            y_curr: Current observation, shape (obs_dim,) or (batch, obs_dim) or None
            dt: Time step (unused, for interface compatibility)
            
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
            x = x + v * ds  # Euler step
        
        if was_1d:
            x = x.squeeze(0)
        
        return x

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
            use_exact_trace: If True, computes exact Jacobian trace (deterministic, O(D^2)).
                             If False, uses Hutchinson trace estimator (stochastic, O(D)).
        """
        was_1d = x_curr.dim() == 1
        if was_1d:
            x_curr = x_curr.unsqueeze(0)
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        
        batch_size = x_curr.shape[0]
        
        # Start from x_curr, integrate backward to s=0
        x = x_curr.clone()
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
                    # print("Computing exact trace...")
                    
                    # We need a function that takes x and returns v for jacobian computation
                    def v_func(x_in):
                        return self.velocity_net(x_in, s_tensor, x_prev, y_curr)

                    # Compute velocity for the step
                    v = v_func(x_for_grad)
                    
                    # Compute exact trace
                    divergence = torch.zeros(batch_size, device=self.device)
                    
                    # Iterate over state dimensions to compute diagonal of Jacobian
                    for k in range(self.state_dim):
                        # Create basis vector e_k
                        # We want d(v_k)/d(x_k)
                        grad_k = torch.autograd.grad(
                            v[:, k], 
                            x_for_grad, 
                            grad_outputs=torch.ones_like(v[:, k]),
                            create_graph=False,
                            retain_graph=True
                        )[0]
                        
                        # grad_k is [d(v_k)/dx_1, d(v_k)/dx_2, ...]
                        # We only want the k-th component: d(v_k)/dx_k
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
    
    # @torch.no_grad()
    # def sample(
    #     self,
    #     x_prev: torch.Tensor,
    #     y_curr: Optional[torch.Tensor] = None,
    #     dt: Optional[float] = None,
    # ) -> torch.Tensor:
    #     """
    #     Sample from the proposal distribution q(x_t | x_{t-1})
        
    #     Integrates the ODE: dx/ds = v_θ(x, s | x_prev) from s=0 to s=1
    #     Starting from z ~ N(0, I)
        
    #     Args:
    #         x_prev: Previous state, shape (state_dim,) or (batch, state_dim)
    #         y_curr: Current observation (unused in base RF, for interface compatibility)
    #         dt: Time step (unused, for interface compatibility)
            
    #     Returns:
    #         Sampled next state, shape same as x_prev
    #     """
    #     was_1d = x_prev.dim() == 1
    #     if was_1d:
    #         x_prev = x_prev.unsqueeze(0)
        
    #     batch_size = x_prev.shape[0]
        
    #     # Sample from source distribution z ~ N(0, I)
    #     z = torch.randn(batch_size, self.state_dim, device=self.device)
        
    #     # Define ODE function
    #     def ode_func(s, x):
    #         # s is scalar, x is (batch, state_dim)
    #         s_tensor = torch.full((batch_size, 1), s, device=self.device)
    #         return self.velocity_net(x, s_tensor, x_prev)
        
    #     # Integrate from s=0 to s=1
    #     # Use regular odeint (not adjoint) since we're in no_grad context
    #     s_span = torch.linspace(0, 1, self.num_sampling_steps, device=self.device)
    #     trajectory = odeint(
    #         ode_func,
    #         z,
    #         s_span,
    #         method='dopri5',
    #         rtol=1e-5,
    #         atol=1e-5,
    #     )
        
    #     # Return final state x(1)
    #     x_final = trajectory[-1]
        
    #     if was_1d:
    #         x_final = x_final.squeeze(0)
        
    #     return x_final
    
    # @torch.no_grad()
    # def log_prob(
    #     self,
    #     x_curr: torch.Tensor,
    #     x_prev: torch.Tensor,
    #     y_curr: Optional[torch.Tensor] = None,
    #     dt: Optional[float] = None,
    # ) -> torch.Tensor:
    #     """
    #     Compute log probability log q(x_curr | x_prev)
        
    #     Uses instantaneous change of variables formula for continuous normalizing flows:
    #     log p_1(x) = log p_0(z) - ∫_0^1 Tr(∂v/∂x) ds
        
    #     We integrate backward from x_curr to z, tracking the trace of Jacobian
    #     using Hutchinson's trace estimator.
        
    #     Args:
    #         x_curr: Current state, shape (state_dim,) or (batch, state_dim)
    #         x_prev: Previous state (conditioning), shape same as x_curr
    #         y_curr: Observation (unused, for interface compatibility)
    #         dt: Time step (unused, for interface compatibility)
            
    #     Returns:
    #         Log probability, shape () or (batch,)
    #     """
    #     was_1d = x_curr.dim() == 1
    #     if was_1d:
    #         x_curr = x_curr.unsqueeze(0)
    #         x_prev = x_prev.unsqueeze(0)
        
    #     batch_size = x_curr.shape[0]
        
    #     # Augmented state: [x, log_prob]
    #     # We'll track the log probability as we integrate backward
        
    #     def augmented_dynamics(s, state):
    #         """
    #         Augmented ODE system for computing likelihood
    #         state = [x, log_prob]
    #         """
    #         x = state[:, :self.state_dim]
            
    #         # Compute velocity
    #         s_tensor = torch.full((batch_size, 1), s, device=self.device)
    #         with torch.enable_grad():
    #             x_for_grad = x.detach().requires_grad_(True)
    #             v = self.velocity_net(x_for_grad, s_tensor, x_prev)
                
    #             # Hutchinson trace estimator with single random vector
    #             eps = torch.randn_like(x)
    #             vjp = torch.autograd.grad(
    #                 v, x_for_grad,
    #                 grad_outputs=eps,
    #                 create_graph=False,
    #                 retain_graph=False,
    #             )[0]
    #             trace_estimate = torch.sum(vjp * eps, dim=-1, keepdim=True)
            
    #         # dx/ds = -v(x, 1-s | x_prev)  [backward integration]
    #         # d(log_prob)/ds = Tr(∂v/∂x)
    #         dx_ds = -v.detach()
    #         dlogp_ds = trace_estimate.detach()
            
    #         return torch.cat([dx_ds, dlogp_ds], dim=-1)
        
    #     # Initial state: [x_curr, 0]
    #     initial_state = torch.cat([
    #         x_curr,
    #         torch.zeros(batch_size, 1, device=self.device)
    #     ], dim=-1)
        
    #     # Integrate backward from s=1 to s=0
    #     # Use regular odeint (not adjoint) since we compute gradients manually
    #     s_span = torch.linspace(1, 0, self.num_likelihood_steps, device=self.device)
        
    #     trajectory = odeint(
    #         augmented_dynamics,
    #         initial_state,
    #         s_span,
    #         method='dopri5',
    #         rtol=1e-5,
    #         atol=1e-5,
    #     )
        
    #     final_state = trajectory[-1]
    #     z = final_state[:, :self.state_dim]
    #     log_prob_correction = final_state[:, self.state_dim]
        
    #     # Base distribution log probability: log N(z; 0, I)
    #     log_prob_base = -0.5 * torch.sum(z ** 2, dim=-1) - 0.5 * self.state_dim * np.log(2 * np.pi)
        
    #     # Total log probability
    #     log_prob = log_prob_base - log_prob_correction
        
    #     if was_1d:
    #         log_prob = log_prob.squeeze(0)
        
    #     return log_prob
        
    
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


# Utility function for testing
def test_rectified_flow():
    """Test the rectified flow implementation"""
    print("Testing Rectified Flow Proposal...")
    
    state_dim = 3
    batch_size = 16
    
    # Create model
    rf = RFProposal(
        state_dim=state_dim,
        hidden_dim=64,
        depth=3,
        num_sampling_steps=20,
        num_likelihood_steps=20,
    )
    
    # Test forward pass
    x = torch.randn(batch_size, state_dim)
    s = torch.rand(batch_size, 1)
    x_prev = torch.randn(batch_size, state_dim)
    
    v = rf(x, s, x_prev)
    print(f"✓ Forward pass: {v.shape}")
    assert v.shape == (batch_size, state_dim)
    
    # Test loss computation
    x_curr = torch.randn(batch_size, state_dim)
    loss, metrics = rf.compute_rf_loss(x_prev, x_curr)
    print(f"✓ Loss computation: {loss.item():.6f}")
    
    # Test sampling (single)
    x_prev_single = torch.randn(state_dim)
    x_sample = rf.sample(x_prev_single)
    print(f"✓ Sampling (single): {x_sample.shape}")
    assert x_sample.shape == (state_dim,)
    
    # Test sampling (batch)
    x_prev_batch = torch.randn(batch_size, state_dim)
    x_sample_batch = rf.sample(x_prev_batch)
    print(f"✓ Sampling (batch): {x_sample_batch.shape}")
    assert x_sample_batch.shape == (batch_size, state_dim)
    
    # Test log probability (this is slow, so use small batch)
    x_prev_small = torch.randn(2, state_dim)
    x_curr_small = torch.randn(2, state_dim)
    log_prob = rf.log_prob(x_curr_small, x_prev_small)
    print(f"✓ Log probability: {log_prob.shape}, values: {log_prob}")
    assert log_prob.shape == (2,)
    
    # Test trajectory generation
    x_init = torch.randn(state_dim)
    trajectory = rf.generate_trajectory(x_init, n_steps=10, return_all=True)
    print(f"✓ Trajectory generation: {trajectory.shape}")
    assert trajectory.shape == (11, state_dim)
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_rectified_flow()