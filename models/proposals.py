import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


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
        self.use_preprocessing = getattr(system.config, "use_preprocessing", False)

        # Get normalization scaling factors if available
        if self.use_preprocessing and hasattr(system, "init_std"):
            self.init_std = system.init_std
        else:
            self.init_std = None

    def sample(
        self, x_prev: torch.Tensor, y_curr: Optional[torch.Tensor], dt: float
    ) -> torch.Tensor:
        """Sample from transition dynamics with process noise - CORRECTED for preprocessing"""
        # Convert to numpy for integration
        x_prev_np = x_prev.cpu().numpy()

        if self.use_preprocessing:
            # x_prev is in normalized space, need to convert to original space
            x_prev_orig = self.system.postprocess(x_prev_np)
            # Integrate in original space
            x_next_orig = self.system.integrate(x_prev_orig, 2, dt)[1]
            # Convert back to normalized space
            x_next = self.system.preprocess(x_next_orig.reshape(1, 1, -1))[0, 0]
            # Scale noise by normalization factor
            noise_std = (
                self.process_noise_std / self.init_std
                if self.init_std is not None
                else self.process_noise_std
            )
        else:
            # No preprocessing, operate directly in original space
            x_next = self.system.integrate(x_prev_np, 2, dt)[1]
            noise_std = self.process_noise_std

        # Convert everything to tensors for consistency
        noise_std = torch.tensor(noise_std, dtype=torch.float32, device=x_prev.device)
        x_next_tensor = torch.tensor(x_next, dtype=torch.float32, device=x_prev.device)

        # Add process noise
        noise = noise_std * torch.randn(self.state_dim, device=x_prev.device)
        return x_next_tensor + noise

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
    ) -> torch.Tensor:
        """Compute log probability under transition dynamics - CORRECTED for preprocessing"""
        # For bootstrap proposal, this is the process noise likelihood
        x_prev_np = x_prev.cpu().numpy()

        if self.use_preprocessing:
            # Convert to original space, integrate, convert back
            x_prev_orig = self.system.postprocess(x_prev_np)
            x_expected_orig = self.system.integrate(x_prev_orig, 2, dt)[1]
            x_expected = self.system.preprocess(x_expected_orig.reshape(1, 1, -1))[0, 0]
            # Scale noise by normalization factor
            noise_std = (
                self.process_noise_std / self.init_std
                if self.init_std is not None
                else self.process_noise_std
            )
        else:
            # No preprocessing, operate in original space
            x_expected = self.system.integrate(x_prev_np, 2, dt)[1]
            noise_std = self.process_noise_std

        x_expected_tensor = torch.tensor(
            x_expected, dtype=torch.float32, device=x_prev.device
        )

        # Convert everything to tensors for consistency
        noise_std = torch.tensor(noise_std, dtype=torch.float32, device=x_prev.device)

        # Compute Gaussian log-likelihood of the noise
        diff = x_curr - x_expected_tensor
        noise_var = noise_std**2

        log_prob = -0.5 * torch.sum(diff**2 / noise_var)
        log_prob -= 0.5 * torch.sum(torch.log(2 * np.pi * noise_var))

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
    
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Args:
            checkpoint_path: Path to trained RFProposal checkpoint (.ckpt file)
            device: Device to run model on ('cpu' or 'cuda')
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
        
        # Get state_dim from model
        self.state_dim = self.rf_model.state_dim
        
    def sample(
        self, x_prev: torch.Tensor, y_curr: Optional[torch.Tensor], dt: float
    ) -> torch.Tensor:
        """
        Sample from RF proposal q(x_t | x_{t-1})
        
        Args:
            x_prev: Previous state, shape (state_dim,)
            y_curr: Observation (unused by RF, kept for interface compatibility)
            dt: Time step (unused by RF, kept for interface compatibility)
            
        Returns:
            Sampled next state, shape (state_dim,)
        """
        # Ensure x_prev is on correct device
        if x_prev.device != self.device:
            x_prev = x_prev.to(self.device)
        
        # RF model's sample method already handles the interface
        return self.rf_model.sample(x_prev, y_curr, dt)
    
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
            y_curr: Observation (unused by RF, kept for interface compatibility)
            dt: Time step (unused by RF, kept for interface compatibility)
            
        Returns:
            Log probability, scalar tensor
        """
        # Ensure tensors are on correct device
        if x_curr.device != self.device:
            x_curr = x_curr.to(self.device)
        if x_prev.device != self.device:
            x_prev = x_prev.to(self.device)
        
        # RF model's log_prob method already handles the interface
        return self.rf_model.log_prob(x_curr, x_prev, y_curr, dt)


# Note: No top-level executable test code here; import these classes from models.proposals
