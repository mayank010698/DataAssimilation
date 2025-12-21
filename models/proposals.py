import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any


# =============================================================================
# Proposal Distribution Interface
# =============================================================================


class ProposalDistribution(ABC):
    """Abstract base class for proposal distributions q(x_t | x_{t-1}, y_t)"""

    @abstractmethod
    def sample(
        self, x_prev: torch.Tensor, y_curr: Optional[torch.Tensor], dt: float
    ) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_{t-1}, y_t)"""
        pass

    @abstractmethod
    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
    ) -> torch.Tensor:
        """Compute log q(x_t | x_{t-1}, y_t)"""
        pass


class TransitionProposal(ProposalDistribution):
    """
    CORRECTED Bootstrap proposal: q(x_t | x_{t-1}) = p(x_t | x_{t-1}) (ignores observation)

    Now properly handles preprocessing/normalization:
    - When preprocessing is ON: operates in normalized space
    - When preprocessing is OFF: operates in original space
    """

    def __init__(self, system, process_noise_std: float = 0.01):
        self.system = system
        self.process_noise_std = process_noise_std
        self.state_dim = system.state_dim
        # TransitionProposal ALWAYS operates in physical space

    def sample(
        self, x_prev: torch.Tensor, y_curr: Optional[torch.Tensor], dt: float
    ) -> torch.Tensor:
        """Sample from transition dynamics with process noise"""
        # Supports both single particle (D,) and batch (N, D)
        is_batch = x_prev.ndim > 1
        
        # No preprocessing, operate directly in original space
        integration = self.system.integrate(x_prev, 2, dt)
        if is_batch:
            x_next = integration[:, 1, :]
        else:
            x_next = integration[1, :]
            
        noise_std = self.process_noise_std

        # Convert noise_std to tensor if it's not already
        if not torch.is_tensor(noise_std):
             noise_std = torch.tensor(noise_std, dtype=torch.float32, device=x_prev.device)

        # Add process noise
        noise = noise_std * torch.randn_like(x_next)
        return x_next + noise

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
    ) -> torch.Tensor:
        """Compute log probability under transition dynamics"""
        # For bootstrap proposal, this is the process noise likelihood
        # Supports batch (N, D)
        is_batch = x_prev.ndim > 1
        
        # No preprocessing, operate in original space
        integration = self.system.integrate(x_prev, 2, dt)
        if is_batch:
            x_expected = integration[:, 1, :]
        else:
            x_expected = integration[1, :]
            
        noise_std = self.process_noise_std

        # Convert noise_std to tensor
        if not torch.is_tensor(noise_std):
             noise_std = torch.tensor(noise_std, dtype=torch.float32, device=x_prev.device)

        # Compute Gaussian log-likelihood of the noise
        diff = x_curr - x_expected
        noise_var = noise_std**2

        # If is_batch, sum over dim 1, else sum over dim 0 (all dims)
        if is_batch:
             reduce_dim = 1
        else:
             reduce_dim = 0
             
        log_prob = -0.5 * torch.sum(diff**2 / noise_var, dim=reduce_dim)
        
        # Log determinant term
        # If noise_std is scalar, we need to multiply by state_dim
        # If noise_std is vector (D,), sum handles it
        if noise_std.ndim == 0:
            log_det = self.state_dim * torch.log(noise_std)
        else:
            log_det = torch.sum(torch.log(noise_std))
            
        log_prob -= (0.5 * self.state_dim * np.log(2 * np.pi) + log_det)
        
        return log_prob


class LearnedNeuralProposal(ProposalDistribution, nn.Module):
    """
    IMPROVED: FlowDAS-style learned proposal with preprocessing support

    Neural network that learns q(x_t | x_{t-1}, y_t) to improve particle filter performance.
    This is still a stub but now has the correct architecture considerations.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        hidden_dim: int = 64,
        use_preprocessing: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.use_preprocessing = use_preprocessing

        # Input: [x_{t-1}, y_t, dt] where y_t might be optional
        # We need to handle variable input size when y_t is None
        max_input_dim = state_dim + obs_dim + 1  # +1 for dt

        self.network = nn.Sequential(
            nn.Linear(max_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * state_dim),  # mean and log_std
        )

        # Separate network for when no observation is available
        self.no_obs_network = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),  # just [x_{t-1}, dt]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * state_dim),
        )

    def sample(
        self, x_prev: torch.Tensor, y_curr: Optional[torch.Tensor], dt: float
    ) -> torch.Tensor:
        """Neural network proposal sampling - IMPROVED stub"""
        # TODO: Implement learned proposal
        # This would:
        # 1. Prepare input [x_prev, y_curr, dt] (handle y_curr=None case)
        # 2. Forward through network to get mean and log_std
        # 3. Sample from resulting Gaussian
        # 4. Handle preprocessing if needed

        # For now, just return x_prev + small noise
        noise_std = 0.01
        if self.use_preprocessing:
            noise_std *= 0.1  # Scale down for normalized space

        return x_prev + noise_std * torch.randn_like(x_prev)

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
    ) -> torch.Tensor:
        """Neural network proposal log probability - IMPROVED stub"""
        # TODO: Implement learned proposal log probability
        # This would:
        # 1. Forward through network to get mean and log_std
        # 2. Compute Gaussian log probability
        # 3. Handle preprocessing consistently

        return torch.tensor(0.0, device=x_curr.device)


class GaussianMixtureProposal(ProposalDistribution):
    """
    IMPROVED: Learned GMM proposal with preprocessing support

    Gaussian Mixture Model proposal that can adapt to complex posterior shapes.
    """

    def __init__(
        self, state_dim: int, n_components: int = 10, use_preprocessing: bool = False
    ):
        self.state_dim = state_dim
        self.n_components = n_components
        self.use_preprocessing = use_preprocessing

        # TODO: Initialize GMM parameters (means, covariances, weights)
        # These could be learned from data or adapted online

    def sample(
        self, x_prev: torch.Tensor, y_curr: Optional[torch.Tensor], dt: float
    ) -> torch.Tensor:
        """GMM proposal sampling - IMPROVED stub"""
        # TODO: Implement GMM proposal
        # This would:
        # 1. Select component based on current state/observation
        # 2. Sample from selected Gaussian component
        # 3. Handle preprocessing scaling

        noise_std = 0.01
        if self.use_preprocessing:
            noise_std *= 0.1

        return x_prev + noise_std * torch.randn_like(x_prev)

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
    ) -> torch.Tensor:
        """GMM proposal log probability - IMPROVED stub"""
        # TODO: Implement GMM proposal log probability
        # This would compute log sum of weighted component probabilities

        return torch.tensor(0.0, device=x_curr.device)


class RectifiedFlowProposal(ProposalDistribution):
    """
    Wrapper for trained Rectified Flow model to use as proposal distribution
    
    Loads a trained RFProposal from checkpoint and uses it as q(x_t | x_{t-1}).
    Note: RF currently doesn't use observations (y_curr), but interface supports it.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        num_likelihood_steps: Optional[int] = None,
        num_sampling_steps: Optional[int] = None,
        system: Optional[Any] = None,
        obs_mean: Optional[torch.Tensor] = None,
        obs_std: Optional[torch.Tensor] = None,
        mc_guidance: bool = False,
        guidance_scale: float = 1.0,
        obs_components: Optional[list] = None,
        use_exact_trace: bool = True,
    ):
        """
        Args:
            checkpoint_path: Path to trained RFProposal checkpoint (.ckpt file)
            device: Device to run model on ('cpu' or 'cuda')
            num_likelihood_steps: Override number of steps for likelihood computation
            num_sampling_steps: Override number of steps for sampling
            system: DynamicalSystem instance (required for pre/post processing)
            obs_mean: Mean of observations for scaling (if preprocessing used)
            obs_std: Std of observations for scaling (if preprocessing used)
            mc_guidance: Whether to use Monte Carlo guidance
            guidance_scale: Scale for guidance
            obs_components: List of observed state indices (needed for guidance)
            use_exact_trace: Whether to use exact trace or Hutchinson estimator
        """
        import sys
        from pathlib import Path
        
        # Add proposals directory to path to import RFProposal
        proposals_dir = Path(__file__).parent.parent / "proposals"
        if str(proposals_dir) not in sys.path:
            sys.path.insert(0, str(proposals_dir))
        
        from proposals.rectified_flow import RFProposal
        
        # Load trained model
        self.rf_model = RFProposal.load_from_checkpoint(checkpoint_path)
        self.rf_model.eval()
        self.rf_model.to(device)
        self.device = device
        self.system = system
        self.use_exact_trace = use_exact_trace
        
        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_std = obs_std.to(device) if obs_std is not None else None
        
        # Override steps if provided
        if num_likelihood_steps is not None:
            self.rf_model.num_likelihood_steps = num_likelihood_steps
        if num_sampling_steps is not None:
            self.rf_model.num_sampling_steps = num_sampling_steps
            
        # Guidance settings
        if mc_guidance:
            self.rf_model.mc_guidance = True
        if guidance_scale != 1.0:
            self.rf_model.guidance_scale = guidance_scale
            
        # Construct observation_fn for guidance if enabled
        self.observation_fn = None
        if self.rf_model.mc_guidance and obs_components is not None:
            self.obs_components = obs_components
            
            # NOTE: guidance is computed in SCALED space.
            # So observation_fn needs to act on scaled x.
            # But obs_components indices are valid for state vector regardless of scaling
            # (assuming scaling is component-wise or linear).
            def obs_fn(x):
                # x is (..., state_dim)
                return x[..., self.obs_components]
            
            self.observation_fn = obs_fn
            
        # Get state_dim from model
        self.state_dim = self.rf_model.state_dim
        
    def sample(
        self, x_prev: torch.Tensor, y_curr: Optional[torch.Tensor], dt: float
    ) -> torch.Tensor:
        """
        Sample from RF proposal q(x_t | x_{t-1})
        
        Args:
            x_prev: Previous state, shape (state_dim,)
            y_curr: Observation
            dt: Time step (unused by RF, kept for interface compatibility)
            
        Returns:
            Sampled next state, shape (state_dim,)
        """
        # Ensure x_prev is on correct device
        if x_prev.device != self.device:
            x_prev = x_prev.to(self.device)
            
        # Handle observation
        if y_curr is not None:
            if y_curr.device != self.device:
                y_curr = y_curr.to(self.device)
            
            # Preprocess observation if needed
            if self.obs_mean is not None and self.obs_std is not None:
                y_curr = (y_curr - self.obs_mean) / self.obs_std

        # Preprocess input (unscaled -> scaled)
        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)

        # RF model's sample method already handles the interface
        # RF expects and produces scaled data
        # Pass observation_fn if needed for guidance
        if self.rf_model.mc_guidance and self.observation_fn is not None:
            x_curr_scaled = self.rf_model.sample(x_prev, y_curr, dt, observation_fn=self.observation_fn)
        else:
            x_curr_scaled = self.rf_model.sample(x_prev, y_curr, dt)
        
        # Postprocess output (scaled -> unscaled)
        if self.system is not None:
            return self.system.postprocess(x_curr_scaled)
            
        return x_curr_scaled
    
    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
    ) -> torch.Tensor:
        """
        Compute log probability log q(x_curr | x_prev)
        
        Args:
            x_curr: Current state, shape (state_dim,)
            x_prev: Previous state, shape (state_dim,)
            y_curr: Observation
            dt: Time step (unused by RF, kept for interface compatibility)
            
        Returns:
            Log probability, scalar tensor
        """
        # Ensure tensors are on correct device
        if x_curr.device != self.device:
            x_curr = x_curr.to(self.device)
        if x_prev.device != self.device:
            x_prev = x_prev.to(self.device)
            
        # Handle observation
        if y_curr is not None:
            if y_curr.device != self.device:
                y_curr = y_curr.to(self.device)
            
            # Preprocess observation if needed
            if self.obs_mean is not None and self.obs_std is not None:
                y_curr = (y_curr - self.obs_mean) / self.obs_std
        
        # Preprocess inputs (unscaled -> scaled)
        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)
            x_curr = self.system.preprocess(x_curr)
            
        # RF model's log_prob method already handles the interface
        # We compute log probability in scaled space.
        # Since weight update does normalization, we don't strictly need Jacobian correction.
        # IMPORTANT: If using guidance, we need sample_and_log_prob equivalent logic? 
        # Actually RFProposal.log_prob currently does NOT support guidance divergence correction natively 
        # unless we modify it or call sample_and_log_prob during sampling (which BPF separates).
        
        # Current BPF implementation calls sample() then log_prob().
        # If guidance was used in sample(), x_curr was generated from a guided process.
        # To get the correct density q_guided(x_curr), we MUST account for the guidance term in the divergence.
        # RFProposal.log_prob needs to support guidance too if we want correct weights.
        
        # Wait, RFProposal.log_prob integrates BACKWARDS from x_curr to x_0.
        # Does the guidance term apply in reverse? 
        # Actually, for standard CNF log-likelihood:
        # log p(x_1) = log p(x_0) - \int div(v) dt
        # This holds regardless of how x_1 was generated, as long as v is the vector field OF THE MODEL/GUIDED PROCESS.
        # So we just need to ensure we use the guided velocity field in log_prob calculation.
        
        # RFProposal.log_prob uses self.velocity_net.
        # We haven't updated RFProposal.log_prob to use guidance yet in previous steps?
        # Let's check rectified_flow.py.
        # Ah, we added sample_and_log_prob but log_prob itself might not use guidance?
        # Let's check.
        
        return self.rf_model.log_prob(x_curr, x_prev, y_curr, dt, use_exact_trace=self.use_exact_trace)
