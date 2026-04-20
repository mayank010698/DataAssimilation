import logging
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
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_{t-1}, y_t). t is optional time step (e.g. for time-conditioned RF)."""
        pass

    @abstractmethod
    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Compute log q(x_t | x_{t-1}, y_t). t is optional time step (e.g. for time-conditioned RF)."""
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
        self._warned_offline_fallback = False

    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Sample from transition dynamics with process noise. t ignored."""
        # Supports both single particle (D,) and batch (N, D)
        is_batch = x_prev.ndim > 1
        if getattr(self.system, "requires_static_params", False):
            has_re = static_params is not None and (
                "reynolds" in static_params or "u" in static_params
            )
            if not has_re:
                raise ValueError(
                    "TransitionProposal requires static Reynolds params for this system "
                    "(expected static_params['reynolds'] or static_params['u'])."
                )
        
        # No preprocessing, operate directly in original space
        # Some systems (e.g., Kolmogorov) are offline-only and intentionally
        # do not implement online dynamics integration.
        try:
            integration = self.system.integrate(x_prev, 2, dt, static_params=static_params)
            if is_batch:
                x_next = integration[:, 1, :]
            else:
                x_next = integration[1, :]
        except NotImplementedError:
            if not getattr(self.system, "allow_persistence_fallback", True):
                raise
            # Optional fallback for legacy/placeholder systems.
            x_next = x_prev
            if not self._warned_offline_fallback:
                logging.warning(
                    "TransitionProposal: system.integrate() is unavailable. "
                    "Falling back to persistence proposal x_t = x_{t-1} + noise."
                )
                self._warned_offline_fallback = True
            
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
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """Compute log probability under transition dynamics. t ignored."""
        # For bootstrap proposal, this is the process noise likelihood
        # Supports batch (N, D)
        is_batch = x_prev.ndim > 1
        if getattr(self.system, "requires_static_params", False):
            has_re = static_params is not None and (
                "reynolds" in static_params or "u" in static_params
            )
            if not has_re:
                raise ValueError(
                    "TransitionProposal requires static Reynolds params for this system "
                    "(expected static_params['reynolds'] or static_params['u'])."
                )
        
        # No preprocessing, operate in original space
        try:
            integration = self.system.integrate(x_prev, 2, dt, static_params=static_params)
            if is_batch:
                x_expected = integration[:, 1, :]
            else:
                x_expected = integration[1, :]
        except NotImplementedError:
            if not getattr(self.system, "allow_persistence_fallback", True):
                raise
            # Must match sample() fallback for consistency.
            x_expected = x_prev
            
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
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
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
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
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
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
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
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
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
        trace_estimator: str = 'rademacher',
        num_trace_probes: int = 1,
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
            trace_estimator: 'gaussian' or 'rademacher' for Hutchinson estimator
            num_trace_probes: Number of probes for Hutchinson estimator
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
        
        # Freeze model parameters to ensure no gradients are tracked for weights
        for param in self.rf_model.parameters():
            param.requires_grad = False
            
        self.device = device
        self.system = system
        self.use_exact_trace = use_exact_trace
        
        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_std = obs_std.to(device) if obs_std is not None else None
        
        # Slice observation scalers if specific components requested
        if obs_components is not None:
            if self.obs_mean is not None:
                # obs_mean is from data_scaled.h5, which is DENSE
                # obs_components are indices into that dense vector
                self.obs_mean = self.obs_mean[obs_components]
            if self.obs_std is not None:
                self.obs_std = self.obs_std[obs_components]
        
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
            
        # Trace estimator settings
        self.trace_estimator = trace_estimator
        self.num_trace_probes = num_trace_probes
            
        # Construct observation_fn for guidance if enabled
        self.observation_fn = None
        
        if self.rf_model.mc_guidance:
            if self.system is not None:
                # Robust observation function that handles scaling and nonlinearity
                def obs_fn(x_scaled):
                    # x_scaled is in scaled space
                    # 1. Unscale x
                    x_unscaled = self.system.postprocess(x_scaled)
                    
                    # 2. Apply full observation operator (selection + nonlinearity)
                    y_pred = self.system.apply_observation_operator(x_unscaled)
                    
                    # 3. Scale y (to match the space where guidance is computed)
                    if self.obs_mean is not None and self.obs_std is not None:
                         y_pred_scaled = (y_pred - self.obs_mean) / self.obs_std
                         return y_pred_scaled
                    return y_pred
                
                self.observation_fn = obs_fn
                
            elif obs_components is not None:
                # Fallback to simple slicing if system is not available
                self.obs_components = obs_components
                
                def obs_fn_simple(x):
                    # x is (..., state_dim)
                    return x[..., self.obs_components]
                
                self.observation_fn = obs_fn_simple
            
        # Get state_dim from model
        self.state_dim = self.rf_model.state_dim
        
    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Sample from RF proposal q(x_t | x_{t-1})

        Args:
            x_prev: Previous state, shape (state_dim,) or (batch, state_dim)
            y_curr: Observation
            dt: Time step (unused by RF, kept for interface compatibility)
            t: Optional time step index; required when RF was trained with use_time_step=True.

        Returns:
            Sampled next state, shape (state_dim,) or (batch, state_dim)
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

        # When RF is time-conditioned, t must be provided by the caller (e.g. BPF passes time_idx).
        if getattr(self.rf_model, "use_time_step", False) and t is None:
            raise ValueError(
                "RF model has use_time_step=True but t was not provided to sample(). "
                "Callers must pass the trajectory time step (e.g. from the filter step)."
            )

        # RF model's sample method already handles the interface
        if self.rf_model.mc_guidance and self.observation_fn is not None:
            x_curr_scaled = self.rf_model.sample(
                x_prev, y_curr, dt, observation_fn=self.observation_fn, t=t
            )
        else:
            x_curr_scaled = self.rf_model.sample(x_prev, y_curr, dt, t=t)
        
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
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Compute log probability log q(x_curr | x_prev)

        t: Optional time step index; required when RF was trained with use_time_step=True.
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
            if self.obs_mean is not None and self.obs_std is not None:
                y_curr = (y_curr - self.obs_mean) / self.obs_std

        # Preprocess inputs (unscaled -> scaled)
        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)
            x_curr = self.system.preprocess(x_curr)

        # When RF is time-conditioned, t must be provided by the caller.
        if getattr(self.rf_model, "use_time_step", False) and t is None:
            raise ValueError(
                "RF model has use_time_step=True but t was not provided to log_prob(). "
                "Callers must pass the trajectory time step (e.g. from the filter step)."
            )

        return self.rf_model.log_prob(
            x_curr,
            x_prev,
            y_curr,
            dt,
            use_exact_trace=self.use_exact_trace,
            trace_estimator=self.trace_estimator,
            num_trace_probes=self.num_trace_probes,
            t=t,
        )


# =============================================================================
# Localized Rectified Flow Proposal (patch-based, per-dim log-density)
# =============================================================================


class LocalizedRFProposalWrapper(ProposalDistribution):
    """
    Wrapper for a trained LocalizedRFProposal usable as a ProposalDistribution.

    This sibling of RectifiedFlowProposal loads a patch-based local velocity
    network and exposes:

      - `sample(x_prev, y_curr, dt, ...)`             - standard PF interface
      - `log_prob(x_curr, x_prev, y_curr, dt, ...)`   - global scalar log-density
      - `sample_and_per_dim_log_prob(x_prev, y_curr, t=None)` - (x_t, ell) where
        ell has shape (..., N_x) such that `sum_j ell_j == log q(x_t | x_{t-1}, y_t)`

    The last method is the one exploited by `LocalizedParticleFilter` when
    `weight_type='full'`. All three methods handle preprocessing/postprocessing
    (via `system`) and observation scaling in the same way as
    `RectifiedFlowProposal`.

    Args:
        checkpoint_path: Path to the trained LocalizedRFProposal checkpoint.
        device: Torch device string.
        num_sampling_steps / num_likelihood_steps: Optional overrides.
        system: DynamicalSystem for pre/post scaling (optional).
        obs_mean / obs_std: Observation scalers (optional).
        obs_components: Indices of observed state sites (optional, for scaling).
        state_dim_override: Override the network's default state_dim at
            inference time (for zero-shot transfer e.g. L96-40 -> L96-400).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        num_sampling_steps: Optional[int] = None,
        num_likelihood_steps: Optional[int] = None,
        system: Optional[Any] = None,
        obs_mean: Optional[torch.Tensor] = None,
        obs_std: Optional[torch.Tensor] = None,
        obs_components: Optional[list] = None,
        state_dim_override: Optional[int] = None,
    ):
        import sys
        from pathlib import Path

        proposals_dir = Path(__file__).parent.parent / "proposals"
        if str(proposals_dir) not in sys.path:
            sys.path.insert(0, str(proposals_dir))

        from proposals.localized_rf import LocalizedRFProposal

        self.lrf_model = LocalizedRFProposal.load_from_checkpoint(
            checkpoint_path, map_location=device, strict=False
        )
        self.lrf_model.eval()
        self.lrf_model.to(device)
        for param in self.lrf_model.parameters():
            param.requires_grad = False

        self.device = device
        self.system = system

        if num_sampling_steps is not None:
            self.lrf_model.num_sampling_steps = num_sampling_steps
        if num_likelihood_steps is not None:
            self.lrf_model.num_likelihood_steps = num_likelihood_steps

        self.state_dim = (
            state_dim_override
            if state_dim_override is not None
            else self.lrf_model.state_dim_default
        )
        # Zero-shot override: expose a different state_dim without retraining.
        if state_dim_override is not None:
            self.lrf_model.state_dim_default = state_dim_override

        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_std = obs_std.to(device) if obs_std is not None else None
        if obs_components is not None:
            if self.obs_mean is not None:
                self.obs_mean = self.obs_mean[obs_components]
            if self.obs_std is not None:
                self.obs_std = self.obs_std[obs_components]
            # The model also needs obs_components to scatter sparse obs -> dense.
            # If we are doing a zero-shot state_dim override, the obs_components
            # baked into the checkpoint correspond to the *training* state_dim
            # and must be replaced by the target dataset's obs_components.
            if (
                self.lrf_model.obs_components is None
                or state_dim_override is not None
            ):
                self.lrf_model.obs_components = list(obs_components)

    # ---- helpers ----

    def _to_device(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.to(self.device) if x.device.type != self.device else x

    def _scale_obs(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is None:
            return None
        y = self._to_device(y)
        if self.obs_mean is not None and self.obs_std is not None:
            y = (y - self.obs_mean) / self.obs_std
        return y

    # ---- ProposalDistribution interface ----

    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        x_prev = self._to_device(x_prev)
        y_curr = self._scale_obs(y_curr)
        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)

        if getattr(self.lrf_model, "use_time_step", False) and t is None:
            raise ValueError(
                "LocalizedRFProposal has use_time_step=True but t was not provided."
            )

        x_curr_scaled = self.lrf_model.sample(x_prev, y_curr, dt, t=t)
        if self.system is not None:
            return self.system.postprocess(x_curr_scaled)
        return x_curr_scaled

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        x_curr = self._to_device(x_curr)
        x_prev = self._to_device(x_prev)
        y_curr = self._scale_obs(y_curr)
        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)
            x_curr = self.system.preprocess(x_curr)

        if getattr(self.lrf_model, "use_time_step", False) and t is None:
            raise ValueError(
                "LocalizedRFProposal has use_time_step=True but t was not provided."
            )

        return self.lrf_model.log_prob(x_curr, x_prev, y_curr, dt, t=t)

    # ---- Extended interface for LocalizedParticleFilter(weight_type='full') ----

    def sample_and_per_dim_log_prob(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Return (x_t, ell) where ell has shape (..., N_x) decomposes the
        log-density site-by-site. `sum(ell, dim=-1) == self.log_prob(x_t, x_prev, ...)`.

        Handles the same pre/post-processing as `sample`. The returned `ell`
        lives in the model's native (scaled) space — the same space used when
        computing per-dim transition log-probabilities for the full-weight
        correction — and the state `x_t` is post-processed back to physical space.
        """
        x_prev = self._to_device(x_prev)
        y_curr = self._scale_obs(y_curr)
        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)

        if getattr(self.lrf_model, "use_time_step", False) and t is None:
            raise ValueError(
                "LocalizedRFProposal has use_time_step=True but t was not provided."
            )

        x_curr_scaled, ell = self.lrf_model.sample_and_per_dim_log_prob(
            x_prev, y_curr, t=t
        )

        if self.system is not None:
            x_curr = self.system.postprocess(x_curr_scaled)
        else:
            x_curr = x_curr_scaled
        return x_curr, ell


# =============================================================================
# Shortcut + F2D2 Proposal (Stages 2 & 3 of the F2D2 pipeline)
# =============================================================================


class ShortcutF2D2Proposal(ProposalDistribution):
    """
    Fast proposal distribution backed by a trained ShortcutProposal (F2D2).

    Replaces RectifiedFlowProposal with a ~25–50× faster alternative:
    - sample()   : n_sampling_steps  forward Euler steps (default 1 NFE)
    - log_prob() : n_likelihood_steps backward Euler via divergence head (default 4 NFEs)

    Mirrors the constructor interface of RectifiedFlowProposal so it can be
    swapped in with minimal changes to eval.py / BPF scripts.

    Args:
        checkpoint_path:    Path to Stage 2 (shortcut) or Stage 3 (F2D2) checkpoint.
        device:             Torch device string.
        n_sampling_steps:   Override inference sampling steps (1 = single NFE).
        n_likelihood_steps: Override inference log-prob steps (4 NFEs default).
        system:             DynamicalSystem for pre/post processing (optional).
        obs_mean:           Observation mean tensor for normalisation.
        obs_std:            Observation std tensor for normalisation.
        obs_components:     List of observed state indices (to slice obs scalers).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        n_sampling_steps: Optional[int] = None,
        n_likelihood_steps: Optional[int] = None,
        system: Optional[Any] = None,
        obs_mean: Optional[torch.Tensor] = None,
        obs_std: Optional[torch.Tensor] = None,
        obs_components: Optional[list] = None,
    ):
        import sys
        from pathlib import Path

        proposals_dir = Path(__file__).parent.parent / "proposals"
        if str(proposals_dir) not in sys.path:
            sys.path.insert(0, str(proposals_dir))

        from proposals.shortcut_flow import ShortcutProposal

        self.sc_model = ShortcutProposal.load_from_checkpoint(
            checkpoint_path, map_location=device, strict=False
        )
        self.sc_model.eval()
        self.sc_model.to(device)
        for param in self.sc_model.parameters():
            param.requires_grad = False

        self.device = device
        self.system = system

        if n_sampling_steps is not None:
            self.sc_model.n_sampling_steps = n_sampling_steps
        if n_likelihood_steps is not None:
            self.sc_model.n_likelihood_steps = n_likelihood_steps

        self.state_dim = self.sc_model.state_dim

        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_std = obs_std.to(device) if obs_std is not None else None
        if obs_components is not None:
            if self.obs_mean is not None:
                self.obs_mean = self.obs_mean[obs_components]
            if self.obs_std is not None:
                self.obs_std = self.obs_std[obs_components]

    # ---- helpers ----

    def _to_device(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.to(self.device) if x.device.type != self.device else x

    def _scale_obs(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is None:
            return None
        y = self._to_device(y)
        if self.obs_mean is not None and self.obs_std is not None:
            y = (y - self.obs_mean) / self.obs_std
        return y

    # ---- ProposalDistribution interface ----

    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_{t-1}, y_t) in 1–few forward Euler steps.

        Args:
            x_prev: (state_dim,) or (B, state_dim), in physical space if system provided.
            y_curr: Observation or None.
            dt:     Unused (kept for interface compatibility).
            t:      Trajectory time step (required if use_time_step=True).

        Returns:
            Sampled state, same shape as x_prev, in physical space.
        """
        x_prev = self._to_device(x_prev)
        y_curr = self._scale_obs(y_curr)

        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)

        if getattr(self.sc_model, "use_time_step", False) and t is None:
            raise ValueError(
                "ShortcutProposal has use_time_step=True but t was not provided."
            )

        x_curr_scaled = self.sc_model.sample(x_prev, y_curr, t=t)

        if self.system is not None:
            return self.system.postprocess(x_curr_scaled)
        return x_curr_scaled

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Compute log q(x_curr | x_prev, y_curr) via divergence accumulation head.

        Args:
            x_curr: (state_dim,) or (B, state_dim), physical space.
            x_prev: same shape.
            y_curr: Observation or None.
            dt:     Unused.
            t:      Trajectory time step (required if use_time_step=True).

        Returns:
            log-probability, shape () or (B,).
        """
        x_curr = self._to_device(x_curr)
        x_prev = self._to_device(x_prev)
        y_curr = self._scale_obs(y_curr)

        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)
            x_curr = self.system.preprocess(x_curr)

        if getattr(self.sc_model, "use_time_step", False) and t is None:
            raise ValueError(
                "ShortcutProposal has use_time_step=True but t was not provided."
            )

        return self.sc_model.log_prob(x_curr, x_prev, y_curr, t=t)


# =============================================================================
# MeanFlow + F2D2 Proposal (MeanFlow Stages 2 & 3 of the F2D2 pipeline)
# =============================================================================


class MeanFlowF2D2Proposal(ProposalDistribution):
    """
    Fast proposal distribution backed by a trained MeanFlowProposal (F2D2).

    Same interface as ShortcutF2D2Proposal but loads a MeanFlowProposal
    checkpoint instead of a ShortcutProposal checkpoint:
    - sample()   : n_sampling_steps  forward Euler steps (default 1 NFE)
    - log_prob() : n_likelihood_steps backward Euler via divergence head (default 4 NFEs)

    Args:
        checkpoint_path:    Path to MeanFlow Stage 2 or Stage 3 (F2D2) checkpoint.
        device:             Torch device string.
        n_sampling_steps:   Override inference sampling steps (1 = single NFE).
        n_likelihood_steps: Override inference log-prob steps (4 NFEs default).
        system:             DynamicalSystem for pre/post processing (optional).
        obs_mean:           Observation mean tensor for normalisation.
        obs_std:            Observation std tensor for normalisation.
        obs_components:     List of observed state indices (to slice obs scalers).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        n_sampling_steps: Optional[int] = None,
        n_likelihood_steps: Optional[int] = None,
        system: Optional[Any] = None,
        obs_mean: Optional[torch.Tensor] = None,
        obs_std: Optional[torch.Tensor] = None,
        obs_components: Optional[list] = None,
    ):
        import sys
        from pathlib import Path

        proposals_dir = Path(__file__).parent.parent / "proposals"
        if str(proposals_dir) not in sys.path:
            sys.path.insert(0, str(proposals_dir))

        from proposals.meanflow_proposal import MeanFlowProposal

        self.mf_model = MeanFlowProposal.load_from_checkpoint(
            checkpoint_path, map_location=device, strict=False
        )
        self.mf_model.eval()
        self.mf_model.to(device)
        for param in self.mf_model.parameters():
            param.requires_grad = False

        self.device = device
        self.system = system

        if n_sampling_steps is not None:
            self.mf_model.n_sampling_steps = n_sampling_steps
        if n_likelihood_steps is not None:
            self.mf_model.n_likelihood_steps = n_likelihood_steps

        self.state_dim = self.mf_model.state_dim

        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_std = obs_std.to(device) if obs_std is not None else None
        if obs_components is not None:
            if self.obs_mean is not None:
                self.obs_mean = self.obs_mean[obs_components]
            if self.obs_std is not None:
                self.obs_std = self.obs_std[obs_components]

    # ---- helpers ----

    def _to_device(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.to(self.device) if x.device.type != self.device else x

    def _scale_obs(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is None:
            return None
        y = self._to_device(y)
        if self.obs_mean is not None and self.obs_std is not None:
            y = (y - self.obs_mean) / self.obs_std
        return y

    # ---- ProposalDistribution interface ----

    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        x_prev = self._to_device(x_prev)
        y_curr = self._scale_obs(y_curr)

        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)

        if getattr(self.mf_model, "use_time_step", False) and t is None:
            raise ValueError(
                "MeanFlowProposal has use_time_step=True but t was not provided."
            )

        x_curr_scaled = self.mf_model.sample(x_prev, y_curr, t=t)

        if self.system is not None:
            return self.system.postprocess(x_curr_scaled)
        return x_curr_scaled

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        x_curr = self._to_device(x_curr)
        x_prev = self._to_device(x_prev)
        y_curr = self._scale_obs(y_curr)

        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)
            x_curr = self.system.preprocess(x_curr)

        if getattr(self.mf_model, "use_time_step", False) and t is None:
            raise ValueError(
                "MeanFlowProposal has use_time_step=True but t was not provided."
            )

        return self.mf_model.log_prob(x_curr, x_prev, y_curr, t=t)


# =============================================================================
# NASMC Gaussian Proposal (Gu, Ghahramani & Turner, 2015)
# =============================================================================


class NASMCProposal(ProposalDistribution):
    """Wrapper for a trained NASMC :class:`GaussianProposal`.

    Mirrors the interface of :class:`RectifiedFlowProposal`:

    - Loads a :class:`GaussianProposal` checkpoint and freezes it.
    - Scales observations with ``obs_mean``/``obs_std`` when provided.
    - Pre/post-processes states with ``system.preprocess``/
      ``system.postprocess`` so the filter talks in *physical* space
      while the proposal talks in *scaled* space.
    - Returns ``log_prob`` in *scaled* space (matching the convention
      used elsewhere in the particle filter).

    The checkpoint does not carry state scalers, so the wrapper
    re-attaches them explicitly via ``attach_scalers`` on the loaded
    module so that the SMC log-density helpers stay consistent if the
    module is ever reused for downstream training.

    Args:
        checkpoint_path: Path to a :class:`GaussianProposal` checkpoint.
        device: Torch device string.
        system: DynamicalSystem for pre/post processing.
        obs_mean / obs_std: Observation scalers.
        obs_components: Indices of observed state sites (to slice the
            observation scalers).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        system: Optional[Any] = None,
        obs_mean: Optional[torch.Tensor] = None,
        obs_std: Optional[torch.Tensor] = None,
        obs_components: Optional[list] = None,
    ):
        import sys
        from pathlib import Path

        proposals_dir = Path(__file__).parent.parent / "proposals"
        if str(proposals_dir) not in sys.path:
            sys.path.insert(0, str(proposals_dir))

        from proposals.nasmc import GaussianProposal

        self.nasmc_model = GaussianProposal.load_from_checkpoint(
            checkpoint_path, map_location=device, strict=False
        )
        self.nasmc_model.eval()
        self.nasmc_model.to(device)
        for param in self.nasmc_model.parameters():
            param.requires_grad = False

        self.device = device
        self.system = system
        self.state_dim = self.nasmc_model.state_dim

        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_std = obs_std.to(device) if obs_std is not None else None
        if obs_components is not None:
            if self.obs_mean is not None:
                self.obs_mean = self.obs_mean[obs_components]
            if self.obs_std is not None:
                self.obs_std = self.obs_std[obs_components]

        # If the checkpoint was trained with the -f- variant
        # (use_dynamics_mean=True), we need the system attached to the
        # GaussianProposal to evaluate the deterministic dynamics mean.
        if getattr(self.nasmc_model, "use_dynamics_mean", False):
            if system is None:
                raise ValueError(
                    "NASMCProposal: checkpoint uses use_dynamics_mean=True, "
                    "which requires a DynamicalSystem to be passed via `system`."
                )
            dt = None
            if hasattr(system, "config"):
                dt = getattr(system.config, "dt", None)
            self.nasmc_model.attach_system(system, dt=dt)

        # Attach state scalers so log_prob helpers used by the -f- variant
        # operate in the correct space. The filter wrapper itself always
        # preprocesses state into scaled space before calling the module.
        state_mean = getattr(system, "init_mean", None) if system is not None else None
        state_std = getattr(system, "init_std", None) if system is not None else None
        if state_mean is not None and state_std is not None:
            self.nasmc_model.attach_scalers(
                state_scaler_mean=torch.as_tensor(state_mean, dtype=torch.float32),
                state_scaler_std=torch.as_tensor(state_std, dtype=torch.float32),
                obs_scaler_mean=self.obs_mean,
                obs_scaler_std=self.obs_std,
            )

    # ---- helpers ----

    def _to_device(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return x.to(self.device) if x.device.type != self.device else x

    def _scale_obs(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is None:
            return None
        y = self._to_device(y)
        if self.obs_mean is not None and self.obs_std is not None:
            y = (y - self.obs_mean) / self.obs_std
        return y

    # ---- ProposalDistribution interface ----

    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        x_prev = self._to_device(x_prev)
        y_curr = self._scale_obs(y_curr)

        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)

        if getattr(self.nasmc_model, "use_time_step", False) and t is None:
            raise ValueError(
                "NASMC model has use_time_step=True but t was not provided to sample()."
            )

        with torch.no_grad():
            x_curr_scaled = self.nasmc_model.sample(x_prev, y_curr, dt, t=t)

        if self.system is not None:
            return self.system.postprocess(x_curr_scaled)
        return x_curr_scaled

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        t: Optional[torch.Tensor] = None,
        static_params: Optional[dict] = None,
    ) -> torch.Tensor:
        x_curr = self._to_device(x_curr)
        x_prev = self._to_device(x_prev)
        y_curr = self._scale_obs(y_curr)

        if self.system is not None:
            x_prev = self.system.preprocess(x_prev)
            x_curr = self.system.preprocess(x_curr)

        if getattr(self.nasmc_model, "use_time_step", False) and t is None:
            raise ValueError(
                "NASMC model has use_time_step=True but t was not provided to log_prob()."
            )

        with torch.no_grad():
            return self.nasmc_model.log_prob(x_curr, x_prev, y_curr, dt, t=t)
