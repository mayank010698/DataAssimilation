import torch
import lightning.pytorch as pl
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


# =============================================================================
# Shared utility functions for filtering methods
# =============================================================================


def compute_unweighted_spread(ensemble: torch.Tensor) -> torch.Tensor:
    """
    Compute ensemble spread for unweighted ensembles.

    Args:
        ensemble: Tensor of shape (N, D) or (B, N, D), where N is ensemble size
                  and D is state dimension. If a 2D tensor is provided, a
                  batch dimension of size 1 is assumed.

    Returns:
        Tensor of shape (B,) with the RMS standard deviation over state
        dimensions for each batch element.
    """
    if ensemble.dim() == 2:
        ensemble = ensemble.unsqueeze(0)  # (1, N, D)
    elif ensemble.dim() != 3:
        raise ValueError(f"Expected ensemble with 2 or 3 dims, got {ensemble.shape}")

    # Variance over ensemble members, per state dimension
    var_per_dim = torch.var(ensemble, dim=1, unbiased=False)  # (B, D)
    # Mean variance over dimensions, then sqrt to get RMS std
    mean_var = torch.mean(var_per_dim, dim=1)  # (B,)
    spread = torch.sqrt(mean_var)
    return spread


def compute_weighted_spread(ensemble: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute ensemble spread for weighted ensembles (e.g. particle filters).

    Args:
        ensemble: Tensor of shape (N, D) or (B, N, D).
        weights:  Tensor of shape (N,) or (B, N). Weights are assumed to be
                  non-negative and will be renormalized internally to sum to 1
                  along the ensemble dimension.

    Returns:
        Tensor of shape (B,) with the RMS standard deviation over state
        dimensions for each batch element.
    """
    if ensemble.dim() == 2:
        ensemble = ensemble.unsqueeze(0)  # (1, N, D)
    if weights.dim() == 1:
        weights = weights.unsqueeze(0)  # (1, N)

    if ensemble.dim() != 3 or weights.dim() != 2:
        raise ValueError(
            f"Expected ensemble (B, N, D) and weights (B, N), "
            f"got ensemble {ensemble.shape}, weights {weights.shape}"
        )

    # Normalize weights along ensemble dimension
    w = torch.clamp(weights, min=0.0)
    w_sum = torch.sum(w, dim=1, keepdim=True)  # (B, 1)
    # Avoid division by zero: if all weights are zero, fall back to uniform
    zero_mask = (w_sum.squeeze(-1) == 0)
    if zero_mask.any():
        # Replace zero-sum rows with uniform weights
        n_particles = w.shape[1]
        w[zero_mask] = 1.0
        w_sum[zero_mask] = float(n_particles)

    w = w / w_sum  # (B, N)

    # Weighted mean over ensemble members
    w_expanded = w.unsqueeze(-1)  # (B, N, 1)
    mean = torch.sum(w_expanded * ensemble, dim=1, keepdim=True)  # (B, 1, D)

    # Weighted variance per dimension
    diff = ensemble - mean  # (B, N, D)
    var_per_dim = torch.sum(w_expanded * diff * diff, dim=1)  # (B, D)

    mean_var = torch.mean(var_per_dim, dim=1)  # (B,)
    spread = torch.sqrt(mean_var)
    return spread


# =============================================================================
# Single Base Class for All Filtering Methods
# =============================================================================


class FilteringMethod(pl.LightningModule, ABC):
    """Base class for all filtering approaches (particles/score/flow)"""

    def __init__(
        self, system, state_dim: int = 3, obs_dim: int = 1, device: str = "cpu"
    ):
        super().__init__()
        self.system = system
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # Register a dummy buffer to ensure self.device property works correctly
        # and allows moving the module to the correct device
        self.register_buffer("dummy_device_tracker", torch.zeros(1))
        self.to(device)

        # For tracking performance
        self.current_trajectory_idx = None
        self.current_time_idx = None

        # Save hyperparameters
        self.save_hyperparameters(ignore=["system"])

    @abstractmethod
    def initialize_filter(self, x0: torch.Tensor) -> None:
        """Initialize the filtering representation"""
        pass

    @abstractmethod
    def predict_step(self, dt: float, y_curr: Optional[torch.Tensor] = None) -> None:
        """Propagate filtering distribution forward"""
        pass

    @abstractmethod
    def update_step(self, observation: torch.Tensor) -> None:
        """Update filtering distribution with observation"""
        pass

    @abstractmethod
    def get_state_estimate(self) -> torch.Tensor:
        """Get E[x_t | y_{1:t}]"""
        pass

    @abstractmethod
    def get_state_covariance(self) -> torch.Tensor:
        """Get Cov[x_t | y_{1:t}]"""
        pass

    @abstractmethod
    def sample_posterior(self, n_samples: int) -> torch.Tensor:
        """Sample from p(x_t | y_{1:t})"""
        pass

    @abstractmethod
    def compute_log_likelihood(self, observation: torch.Tensor) -> float:
        """Compute log p(y_t | y_{1:t-1})"""
        pass

    def step(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        trajectory_idx: int,
        time_idx: int,
    ) -> Dict[str, Any]:
        """Single filtering step - common interface"""

        # Initialize filter if this is the start of a new trajectory
        if self.current_trajectory_idx != trajectory_idx:
            self.initialize_filter(x_prev)
            self.current_trajectory_idx = trajectory_idx
            self.current_time_idx = time_idx

        # Prediction step
        self.predict_step(dt, y_curr)

        # Update step (if observation available)
        log_likelihood = 0.0
        if y_curr is not None:
            log_likelihood = self.compute_log_likelihood(y_curr)
            self.update_step(y_curr)

        # Get estimates
        x_est = self.get_state_estimate()
        P_est = self.get_state_covariance()

        # Compute metrics
        # Ensure x_curr is on correct device
        x_curr = x_curr.to(self.device)
        
        error = x_est - x_curr
        rmse = torch.sqrt(torch.mean(error**2)).item()

        metrics = {
            "rmse": rmse,
            "log_likelihood": log_likelihood,
            "x_est": x_est,
            "P_est": P_est,
            "error": error,
            "trajectory_idx": trajectory_idx,
            "time_idx": time_idx,
        }

        return metrics

    def test_step(self, batch, batch_idx):
        """Lightning test step - common interface"""
        # Extract batch data
        x_prev = batch["x_prev"].squeeze(0)  # Remove batch dimension
        x_curr = batch["x_curr"].squeeze(0)
        y_curr = batch["y_curr"].squeeze(0) if batch["has_observation"].item() else None
        trajectory_idx = batch["trajectory_idx"].item()
        time_idx = batch["time_idx"].item()

        # Run filtering step
        dt = self.system.config.dt
        metrics = self.step(x_prev, x_curr, y_curr, dt, trajectory_idx, time_idx)

        # Log metrics
        self.log("test_rmse", metrics["rmse"], on_step=True, on_epoch=True)
        if y_curr is not None:
            self.log(
                "test_log_likelihood",
                metrics["log_likelihood"],
                on_step=True,
                on_epoch=True,
            )

        return metrics

    def training_step(self, batch, batch_idx):
        """Training step for learnable filtering methods"""
        # Extract batch data for training
        x_prev = batch["x_prev"].squeeze(0)
        x_curr = batch["x_curr"].squeeze(0)
        y_curr = batch["y_curr"].squeeze(0) if batch["has_observation"].item() else None
        trajectory_idx = batch["trajectory_idx"].item()
        time_idx = batch["time_idx"].item()

        # TODO: Implement training logic for learnable components
        # This could include:
        # 1. Training proposal distributions (FlowDAS-style)
        # 2. Training score networks (EnSF)
        # 3. Training flow networks (EnFF)
        # 4. Computing importance sampling losses
        # 5. Updating neural network parameters

        # For now, return dummy loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Log training metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizers for trainable components"""
        # TODO: Return actual optimizers when we have trainable parameters
        # For particle filters without learnable components, return None
        return None
