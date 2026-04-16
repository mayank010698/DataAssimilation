"""
Localized Particle Filter.

Keeps the existing global proposal sampling unchanged but replaces the global
importance-weight update with a localized approximate blockwise analysis step.

Shapes (batched):
  - particles:      (Batch, N, D)
  - local weights:  (Batch, num_blocks, N)
  - ancestors:      (Batch, num_blocks, N)
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union

from .base_pf import FilteringMethod, compute_unweighted_spread
from .proposals import ProposalDistribution, TransitionProposal
from .localization import (
    LocalizedPFConfig,
    LocalizationGeometry,
    compute_local_log_weights,
    local_systematic_resample,
    reconstruct_particles,
    apply_white_noise_regularization,
    apply_colored_noise_regularization,
    apply_weight_smoothing,
    compute_local_ess,
    compute_ancestor_repetition,
    compute_fixed_point_fraction,
    compute_mean_ancestor_displacement,
    compute_block_boundary_discontinuity,
)


class LocalizedParticleFilter(FilteringMethod):
    """
    Batched Localized Particle Filter.

    Uses the existing global proposal for prediction, then performs a localized
    approximate analysis: blockwise weights, blockwise resampling, patchwork
    particle reconstruction, with optional post-regularization and smoothing.

    Shapes:
      - particles: (Batch, N_particles, State_dim)
      - weights:   (Batch, N_particles)  -- always uniform after local resampling
    """

    def __init__(
        self,
        system,
        proposal_distribution: Optional[ProposalDistribution] = None,
        n_particles: int = 1000,
        state_dim: int = 40,
        obs_dim: int = 40,
        process_noise_std: float = 0.0,
        device: str = "cpu",
        localized_config: Optional[LocalizedPFConfig] = None,
    ):
        super().__init__(system, state_dim, obs_dim, device)
        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        self.loc_config = localized_config if localized_config is not None else LocalizedPFConfig()

        if proposal_distribution is None:
            self.proposal = TransitionProposal(system, process_noise_std)
        else:
            self.proposal = proposal_distribution

        self.transition_prior = TransitionProposal(system, process_noise_std)

        self.particles = None
        self.particles_prev = None
        self.weights = None

        # Geometry is built once during initialize_filter
        self.geometry: Optional[LocalizationGeometry] = None

        # Diagnostics from last update step
        self.last_local_ess = None
        self.last_ancestor_repetition = None
        self.last_fixed_point_fraction = None
        self.last_mean_ancestor_displacement = None
        self.last_boundary_discontinuity = None

        # Cache for per-dimension proposal log-density when weight_type="full".
        # Shape (Batch, N, D). Populated in predict_step, consumed in update_step.
        self._last_per_dim_ell: Optional[torch.Tensor] = None
        # Diagnostic: magnitude of the per-dim correction (mean abs residual)
        self.last_per_dim_correction_stats: Optional[Dict[str, float]] = None

        self.step_count = 0

        logging.info(
            f"Initialized LocalizedParticleFilter (Batched): "
            f"n_particles={n_particles}, block_size={self.loc_config.block_size}, "
            f"loc_radius={self.loc_config.localization_radius}, "
            f"post_reg={self.loc_config.post_regularization}, "
            f"smoothing={self.loc_config.weight_smoothing}"
        )

    # --------------------------------------------------------------------- #
    # Initialization
    # --------------------------------------------------------------------- #

    def initialize_filter(
        self, x0: torch.Tensor, init_std: Optional[Union[float, torch.Tensor]] = None
    ) -> None:
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        batch_size = x0.shape[0]
        x0 = x0.to(self.device)

        if init_std is None:
            init_std = self.system.config.obs_noise_std

        if isinstance(init_std, torch.Tensor):
            init_std = init_std.to(self.device)
            scale = init_std.reshape(1, 1, -1) if init_std.dim() > 0 else init_std
        else:
            scale = init_std

        noise = torch.randn(
            batch_size, self.n_particles, self.state_dim, device=self.device
        ) * scale
        self.particles = x0.unsqueeze(1) + noise
        self.particles_prev = self.particles.clone()

        self.weights = torch.ones(
            batch_size, self.n_particles, device=self.device
        ) / self.n_particles

        self.step_count = 0

        # Build geometry once
        if self.geometry is None:
            self.geometry = LocalizationGeometry(
                state_dim=self.state_dim,
                obs_components=self.system.config.obs_components,
                block_size=self.loc_config.block_size,
                localization_radius=self.loc_config.localization_radius,
                domain_length=float(self.state_dim),
                taper_type=self.loc_config.taper_type,
                device=self.device,
            )

        logging.info(
            f"LocalizedPF initialized: batch={batch_size}, "
            f"particles={self.n_particles}, blocks={self.geometry.num_blocks}"
        )

    # --------------------------------------------------------------------- #
    # Predict Step  (global proposal, same as batched BPF)
    # --------------------------------------------------------------------- #

    def predict_step(self, dt: float, y_curr: Optional[torch.Tensor] = None) -> None:
        self.particles_prev = self.particles.clone()

        batch_size = self.particles.shape[0]
        particles_flat = self.particles.reshape(-1, self.state_dim)

        y_curr_flat = None
        if y_curr is not None:
            y_curr_expanded = y_curr.unsqueeze(1).expand(
                batch_size, self.n_particles, -1
            )
            y_curr_flat = y_curr_expanded.reshape(
                batch_size * self.n_particles, y_curr.shape[-1]
            )

        proposal = self.proposal if y_curr is not None else self.transition_prior

        time_flat = None
        if getattr(self, "_current_time_idxs", None) is not None:
            time_flat = (
                self._current_time_idxs.unsqueeze(1)
                .expand(batch_size, self.n_particles)
                .reshape(-1)
                .float()
                .to(particles_flat.device)
            )

        # "full" weight path: need per-dim proposal log-density ell_j at each
        # particle; use the extended interface if the proposal supports it.
        want_full = (
            self.loc_config.weight_type == "full"
            and y_curr is not None
            and hasattr(proposal, "sample_and_per_dim_log_prob")
        )

        if want_full:
            x_new_flat, ell_flat = proposal.sample_and_per_dim_log_prob(
                particles_flat, y_curr_flat, dt, t=time_flat
            )
            self.particles = x_new_flat.reshape(
                batch_size, self.n_particles, self.state_dim
            )
            self._last_per_dim_ell = ell_flat.reshape(
                batch_size, self.n_particles, self.state_dim
            )
        else:
            x_new_flat = proposal.sample(particles_flat, y_curr_flat, dt, t=time_flat)
            self.particles = x_new_flat.reshape(
                batch_size, self.n_particles, self.state_dim
            )
            self._last_per_dim_ell = None

    # --------------------------------------------------------------------- #
    # Update Step  (localized analysis)
    # --------------------------------------------------------------------- #

    def update_step(self, observation: torch.Tensor) -> None:
        batch_size = self.particles.shape[0]
        observation = observation.to(self.device)

        # 1. Predicted observations
        particles_flat = self.particles.reshape(-1, self.state_dim)
        expected_obs_flat = self.system.apply_observation_operator(particles_flat)
        expected_obs = expected_obs_flat.reshape(batch_size, self.n_particles, self.obs_dim)

        # 2. Optional per-dim residual for "full" importance weights
        per_dim_residual = None
        self.last_per_dim_correction_stats = None
        if (
            self.loc_config.weight_type == "full"
            and self._last_per_dim_ell is not None
        ):
            # Compute per-dim transition log-prob for all particles.
            dt = getattr(self, "_current_dt", None)
            if dt is None:
                raise RuntimeError(
                    "LocalizedParticleFilter.update_step: weight_type='full' "
                    "requires `_current_dt` to be set by step(); this is done "
                    "automatically when calling step(), but direct callers "
                    "must set `self._current_dt` before update_step()."
                )
            log_p_per_dim = self._compute_transition_log_prob_per_dim(
                self.particles, self.particles_prev, dt
            )  # (Batch, N, D)
            per_dim_residual = log_p_per_dim - self._last_per_dim_ell
            # Diagnostic: typical magnitude of the correction.
            with torch.no_grad():
                self.last_per_dim_correction_stats = {
                    "mean_abs": per_dim_residual.abs().mean().item(),
                    "max_abs": per_dim_residual.abs().max().item(),
                }

        # 3. Local blockwise log weights (+ optional per-dim correction)
        obs_noise_var = self.system.config.obs_noise_std ** 2
        local_lw = compute_local_log_weights(
            expected_obs, observation, self.geometry, obs_noise_var,
            per_dim_residual=per_dim_residual,
        )  # (Batch, num_blocks, N)

        # 4. Diagnostics (before resampling)
        self.last_local_ess = compute_local_ess(local_lw)  # (Batch, num_blocks)

        # 5. Local resampling (with optional adjustment-minimizing reordering)
        ancestors = local_systematic_resample(
            local_lw, self.n_particles,
            adjustment_minimizing=self.loc_config.adjustment_minimizing,
        )

        # 6. Diagnostics (after resampling)
        self.last_ancestor_repetition = compute_ancestor_repetition(ancestors)
        self.last_fixed_point_fraction = compute_fixed_point_fraction(ancestors)
        self.last_mean_ancestor_displacement = compute_mean_ancestor_displacement(ancestors)

        # 7. Reconstruct patchwork particles
        particles_before_reg = self.particles.clone()
        self.particles = reconstruct_particles(self.particles, ancestors, self.geometry)

        # 8. Post-regularization
        if self.loc_config.post_regularization == "white":
            self.particles = apply_white_noise_regularization(
                self.particles, self.loc_config.post_jitter_std
            )
        elif self.loc_config.post_regularization == "colored":
            self.particles = apply_colored_noise_regularization(
                self.particles, self.loc_config.colored_jitter_scale
            )

        # 9. Weight smoothing
        if self.loc_config.weight_smoothing:
            self.particles = apply_weight_smoothing(
                self.particles,
                particles_before_reg,
                ancestors,
                self.geometry,
                self.loc_config.smoothing_radius,
                self.loc_config.smoothing_strength,
            )

        # 10. Boundary discontinuity diagnostic
        self.last_boundary_discontinuity = compute_block_boundary_discontinuity(
            self.particles, self.geometry
        )

        # Weights are uniform after local resampling
        self.weights = torch.ones(
            batch_size, self.n_particles, device=self.device
        ) / self.n_particles

    # --------------------------------------------------------------------- #
    # Per-dim transition log-prob  (used only when weight_type='full')
    # --------------------------------------------------------------------- #

    def _compute_transition_log_prob_per_dim(
        self, x_curr: torch.Tensor, x_prev: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Per-dimension Gaussian transition log-prob log p(x_{t,j} | x_{t-1})
        under an isotropic/diagonal process-noise model.

        Returns (Batch, N, D). Summing over D reproduces the scalar transition
        log-prob used by BPF.
        """
        batch_size, n_particles, D = x_curr.shape
        x_curr_flat = x_curr.reshape(-1, D)
        x_prev_flat = x_prev.reshape(-1, D)

        integration_result = self.system.integrate(x_prev_flat, 2, dt)
        x_expected_flat = integration_result[:, 1, :]

        noise_std = self.process_noise_std
        if not torch.is_tensor(noise_std):
            noise_std = torch.tensor(
                float(noise_std), device=x_curr.device, dtype=x_curr.dtype
            )
        else:
            noise_std = noise_std.to(device=x_curr.device, dtype=x_curr.dtype)

        diff = x_curr_flat - x_expected_flat
        noise_var = noise_std ** 2
        # -0.5 * diff^2 / sigma^2 - 0.5 log(2pi) - log(sigma)
        log_prob_per_dim_flat = (
            -0.5 * (diff ** 2) / noise_var
            - 0.5 * float(np.log(2 * np.pi))
            - torch.log(noise_std)
        )
        return log_prob_per_dim_flat.reshape(batch_size, n_particles, D)

    # --------------------------------------------------------------------- #
    # State Estimation
    # --------------------------------------------------------------------- #

    def get_state_estimate(self) -> torch.Tensor:
        """Ensemble mean with uniform weights. Returns (Batch, D)."""
        return self.particles.mean(dim=1)

    def get_state_covariance(self) -> torch.Tensor:
        """Ensemble covariance with uniform weights. Returns (Batch, D, D)."""
        mean = self.get_state_estimate().unsqueeze(1)  # (Batch, 1, D)
        centered = self.particles - mean  # (Batch, N, D)
        cov = torch.bmm(centered.transpose(1, 2), centered) / self.n_particles
        return cov

    # --------------------------------------------------------------------- #
    # Log-Likelihood (diagnostic only, not used for weight updates)
    # --------------------------------------------------------------------- #

    def compute_log_likelihood(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Approximate marginal log-likelihood using uniform weights.
        Returns (Batch,).
        """
        batch_size = self.particles.shape[0]
        observation = observation.to(self.device)

        particles_flat = self.particles.reshape(-1, self.state_dim)
        expected_obs_flat = self.system.apply_observation_operator(particles_flat)
        expected_obs = expected_obs_flat.reshape(
            batch_size, self.n_particles, self.obs_dim
        )

        obs_expanded = observation.unsqueeze(1)
        diff = obs_expanded - expected_obs
        obs_noise_var = self.system.config.obs_noise_std ** 2

        if self.obs_dim == 1:
            log_likelihoods = (
                -0.5 * (diff.squeeze(-1) ** 2) / obs_noise_var
                - 0.5 * np.log(2 * np.pi * obs_noise_var)
            )
        else:
            log_likelihoods = -0.5 * torch.sum(diff ** 2, dim=2) / obs_noise_var
            log_likelihoods -= 0.5 * self.obs_dim * np.log(2 * np.pi * obs_noise_var)

        # Uniform weights -> simple logsumexp - log(N)
        log_marginal = torch.logsumexp(log_likelihoods, dim=1) - np.log(
            self.n_particles
        )
        return log_marginal

    # --------------------------------------------------------------------- #
    # Full Step (predict + update + metrics)
    # --------------------------------------------------------------------- #

    def step(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        trajectory_idxs: torch.Tensor,
        time_idxs: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        start_time = time.perf_counter()
        self.step_count += 1
        self._current_time_idxs = time_idxs
        self._current_dt = dt

        # Predict
        self.predict_step(dt, y_curr)

        # Pre-update spread
        ensemble_spread_pre = torch.zeros(x_curr.shape[0], device=self.device)
        log_likelihoods = torch.zeros(x_curr.shape[0], device=self.device)

        if y_curr is not None:
            ensemble_spread_pre = compute_unweighted_spread(self.particles)
            log_likelihoods = self.compute_log_likelihood(y_curr)
            self.update_step(y_curr)

        # Estimates
        x_est = self.get_state_estimate()
        P_est = self.get_state_covariance()

        elapsed_time = (time.perf_counter() - start_time) / x_curr.shape[0]

        x_curr = x_curr.to(self.device)
        error = x_est - x_curr
        rmse = torch.sqrt(torch.mean(error ** 2, dim=1))  # (Batch,)

        # Local ESS stats
        if self.last_local_ess is not None:
            mean_local_ess = self.last_local_ess.mean(dim=1)  # (Batch,)
            min_local_ess = self.last_local_ess.min(dim=1)[0]  # (Batch,)
        else:
            mean_local_ess = torch.full(
                (x_curr.shape[0],), float(self.n_particles), device=self.device
            )
            min_local_ess = mean_local_ess.clone()

        # Ancestor uniqueness
        if self.last_ancestor_repetition is not None:
            mean_ancestor_uniq = self.last_ancestor_repetition.mean(dim=1)
        else:
            mean_ancestor_uniq = torch.ones(x_curr.shape[0], device=self.device)

        # Fixed-point fraction & displacement (adjustment-minimizing diagnostics)
        if self.last_fixed_point_fraction is not None:
            mean_fp_frac = self.last_fixed_point_fraction.mean(dim=1)
        else:
            mean_fp_frac = torch.zeros(x_curr.shape[0], device=self.device)

        if self.last_mean_ancestor_displacement is not None:
            mean_anc_disp = self.last_mean_ancestor_displacement.mean(dim=1)
        else:
            mean_anc_disp = torch.zeros(x_curr.shape[0], device=self.device)

        # Boundary discontinuity
        if self.last_boundary_discontinuity is not None:
            boundary_disc = self.last_boundary_discontinuity
        else:
            boundary_disc = torch.zeros(x_curr.shape[0], device=self.device)

        results = []
        for i in range(x_curr.shape[0]):
            metrics = {
                "rmse": rmse[i].item(),
                "log_likelihood": log_likelihoods[i].item(),
                "x_est": x_est[i].cpu().numpy(),
                "P_est": P_est[i].cpu().numpy(),
                "error": error[i].cpu().numpy(),
                "trajectory_idx": trajectory_idxs[i].item(),
                "time_idx": time_idxs[i].item(),
                "resampled": y_curr is not None,
                "ess_pre_resample": mean_local_ess[i].item(),
                "ess": mean_local_ess[i].item(),
                "min_local_ess": min_local_ess[i].item(),
                "mean_ancestor_uniqueness": mean_ancestor_uniq[i].item(),
                "mean_fixed_point_fraction": mean_fp_frac[i].item(),
                "mean_ancestor_displacement": mean_anc_disp[i].item(),
                "block_boundary_discontinuity": boundary_disc[i].item(),
                "ensemble_spread_pre_resample": (
                    ensemble_spread_pre[i].item() if y_curr is not None else 0.0
                ),
                "step_time": elapsed_time,
                "proposal_log_prob_mean": 0.0,
                "obs_log_prob_mean": 0.0,
                "obs_log_prob_std": 0.0,
                "weight_type": self.loc_config.weight_type,
                "per_dim_correction_mean_abs": (
                    self.last_per_dim_correction_stats["mean_abs"]
                    if self.last_per_dim_correction_stats is not None
                    else 0.0
                ),
                "per_dim_correction_max_abs": (
                    self.last_per_dim_correction_stats["max_abs"]
                    if self.last_per_dim_correction_stats is not None
                    else 0.0
                ),
            }
            results.append(metrics)

        return results

    # --------------------------------------------------------------------- #
    # Abstract method stubs
    # --------------------------------------------------------------------- #

    def sample_posterior(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError("Not implemented for batched localized filter")
