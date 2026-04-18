"""
Auxiliary Particle Filter (APF) baseline.

Implements the generic APF incremental-weight identity

    log w_t^i = log g(y_t | x_t^i)
              + log f(x_t^i | x_{t-1}^{k_i})
              - log m_t^{k_i}
              - log q(x_t^i | x_{t-1}^{k_i}, y_t)

with the classical Pitt-Shephard (1999) configuration as the default:
prior proposal q = f (TransitionProposal) and a point-surrogate
adjustment multiplier m_t(x_{t-1}) = g(y_t | mu_t) where
mu_t = E[x_t | x_{t-1}] is the deterministic transition mean.

When q = f, the f and q terms cancel exactly and the incremental
weight reduces to the Pitt-Shephard second-stage correction
log g(y_t | x_t) - log m_t^{k_i}. The generic formula is still
written in full so the code supports observation-aware proposals
(learned, EKF/UKF, flow-based, ...) as a drop-in.

Shapes follow BootstrapParticleFilter:
    particles:   (B, N, D)
    weights:     (B, N)
    log_weights: (B, N)

Integrates with eval.py's batched path.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base_pf import FilteringMethod, compute_weighted_spread
from .proposals import ProposalDistribution, TransitionProposal


# =============================================================================
# Helpers (kept local to avoid touching base_pf.py)
# =============================================================================


def _gaussian_obs_log_prob(
    y: torch.Tensor,
    y_pred: torch.Tensor,
    obs_noise_var: float,
    obs_dim: int,
) -> torch.Tensor:
    """
    Batched diagonal-Gaussian observation log-likelihood.

    Args:
        y:       (B, Obs)         observation per batch item
        y_pred:  (B, N, Obs)      predicted observation per particle
        obs_noise_var:             scalar observation-noise variance
        obs_dim:                   dimensionality of the observation

    Returns:
        (B, N) log p(y | y_pred).
    """
    diff = y.unsqueeze(1) - y_pred  # (B, N, Obs)
    if obs_dim == 1:
        log_prob = -0.5 * (diff.squeeze(-1) ** 2) / obs_noise_var \
            - 0.5 * float(np.log(2 * np.pi * obs_noise_var))
    else:
        log_prob = -0.5 * torch.sum(diff ** 2, dim=2) / obs_noise_var
        log_prob = log_prob - 0.5 * obs_dim * float(np.log(2 * np.pi * obs_noise_var))
    return log_prob


def _transition_log_prob_batched(
    x_curr: torch.Tensor,
    x_prev: torch.Tensor,
    system,
    process_noise_std: float,
    dt: float,
    state_dim: int,
) -> torch.Tensor:
    """
    log p(x_t | x_{t-1}) under the isotropic-Gaussian process-noise model, batched.

    Args:
        x_curr: (B, N, D)
        x_prev: (B, N, D)  (ancestor-selected previous particles)

    Returns:
        (B, N) log-prob.
    """
    batch_size, n_particles, _ = x_curr.shape
    x_curr_flat = x_curr.reshape(-1, state_dim)
    x_prev_flat = x_prev.reshape(-1, state_dim)

    integration_result = system.integrate(x_prev_flat, 2, dt)
    x_expected_flat = integration_result[:, 1, :]

    noise_std = process_noise_std
    if not torch.is_tensor(noise_std):
        noise_std = torch.tensor(noise_std, dtype=torch.float32, device=x_curr.device)

    diff = x_curr_flat - x_expected_flat
    noise_var = noise_std ** 2

    log_prob_flat = -0.5 * torch.sum(diff ** 2 / noise_var, dim=1)
    if noise_std.ndim == 0:
        log_det = state_dim * torch.log(noise_std)
    else:
        log_det = torch.sum(torch.log(noise_std))
    log_prob_flat = log_prob_flat - (0.5 * state_dim * float(np.log(2 * np.pi)) + log_det)

    return log_prob_flat.reshape(batch_size, n_particles)


def _systematic_resample_from_log_weights(
    log_w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched systematic resampling from (unnormalized) log weights.

    Args:
        log_w: (B, N)

    Returns:
        indices: (B, N) ancestor indices
        weights: (B, N) normalized probability weights used for sampling
    """
    batch_size, n = log_w.shape
    device = log_w.device

    max_log_w = torch.max(log_w, dim=1, keepdim=True).values
    w_unnorm = torch.exp(log_w - max_log_w)
    w_sum = torch.sum(w_unnorm, dim=1, keepdim=True)

    # Guard against all-zero rows.
    zero_rows = (w_sum.squeeze(-1) == 0)
    if zero_rows.any():
        w_unnorm[zero_rows] = 1.0
        w_sum[zero_rows] = float(n)

    weights = w_unnorm / w_sum  # (B, N)
    cumsum = torch.cumsum(weights, dim=1)  # (B, N)

    u = torch.rand(batch_size, 1, device=device) / n
    positions = u + torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0) / n
    indices = torch.searchsorted(cumsum, positions)
    indices = torch.clamp(indices, 0, n - 1)
    return indices, weights


def _gather_particles(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather particles along the ensemble axis. x: (B, N, D), idx: (B, N)."""
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, 1, idx_expanded)


# =============================================================================
# APF
# =============================================================================


class AuxiliaryParticleFilter(FilteringMethod):
    """
    Batched Auxiliary Particle Filter (Pitt & Shephard, 1999) with the
    generic APF incremental-weight identity and a one-resampling
    (Johansen-Doucet) main loop.

    Defaults reproduce classical Pitt-Shephard: prior proposal q = f
    and point-surrogate adjustment multiplier m_t = g(y_t | mu_t) with
    mu_t = deterministic transition mean.

    Args:
        adjustment_type: "point" (default) -> m_t = g(y_t | mu_t).
                        "predictive" -> Gaussian-inflated approximation of
                                         p(y_t | x_{t-1}) with
                                         total_var = obs_var + process_var.
    """

    def __init__(
        self,
        system,
        proposal_distribution: Optional[ProposalDistribution] = None,
        n_particles: int = 1000,
        state_dim: int = 3,
        obs_dim: int = 1,
        process_noise_std: float = 0.25,
        device: str = "cpu",
        adjustment_type: str = "point",
        resampling_threshold_ratio: float = 0.5,
    ):
        super().__init__(system, state_dim, obs_dim, device)

        if adjustment_type not in ("point", "predictive"):
            raise ValueError(
                f"adjustment_type must be 'point' or 'predictive', got {adjustment_type!r}"
            )

        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        self.adjustment_type = adjustment_type
        self.resampling_threshold_ratio = resampling_threshold_ratio

        if proposal_distribution is None:
            self.proposal = TransitionProposal(system, process_noise_std)
        else:
            self.proposal = proposal_distribution

        self.transition_prior = TransitionProposal(system, process_noise_std)
        self._proposal_is_transition = isinstance(self.proposal, TransitionProposal)

        self.particles = None
        self.particles_prev = None
        self.weights = None
        self.log_weights = None

        # Diagnostic buffers (populated on each update step).
        self.last_proposal_log_probs = None
        self.last_obs_log_likelihoods = None
        self.last_log_m = None
        self.last_ancestor_idx = None
        self.last_incremental_log_w = None

        self.step_count = 0
        self.resampling_history = []

        logging.info(
            "Initialized AuxiliaryParticleFilter "
            f"(n_particles={n_particles}, proposal={type(self.proposal).__name__}, "
            f"adjustment_type={adjustment_type}, obs_dim={obs_dim})"
        )

    # -------------------------------------------------------------------------
    # Filter-state management
    # -------------------------------------------------------------------------

    def initialize_filter(
        self,
        x0: torch.Tensor,
        init_std: Optional[Union[float, torch.Tensor]] = None,
    ) -> None:
        """Initialize particle cloud around x0. Mirrors BootstrapParticleFilter."""
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
        )
        noise = noise * scale

        self.particles = x0.unsqueeze(1) + noise
        self.particles_prev = self.particles.clone()

        self.weights = torch.ones(
            batch_size, self.n_particles, device=self.device
        ) / self.n_particles
        self.log_weights = torch.log(self.weights)

        self.step_count = 0
        logging.info(
            f"APF initialized: batch={batch_size}, n_particles={self.n_particles}"
        )

    # -------------------------------------------------------------------------
    # FilteringMethod abstract methods
    # -------------------------------------------------------------------------

    def predict_step(self, dt: float, y_curr: Optional[torch.Tensor] = None) -> None:
        """
        APF delegates prediction to `step()` when an observation is present
        (propagation happens AFTER ancestor resampling). When no observation
        is available, fall back to bootstrap-style prior propagation.
        """
        if y_curr is not None:
            # No-op: step() handles propagation after ancestor selection.
            return

        # Missing-observation branch: behave like BPF with transition prior.
        self.particles_prev = self.particles.clone()

        batch_size = self.particles.shape[0]
        particles_flat = self.particles.reshape(-1, self.state_dim)

        time_flat = None
        if getattr(self, "_current_time_idxs", None) is not None:
            time_flat = (
                self._current_time_idxs.unsqueeze(1)
                .expand(batch_size, self.n_particles)
                .reshape(-1)
                .float()
                .to(particles_flat.device)
            )

        x_new_flat = self.transition_prior.sample(particles_flat, None, dt, t=time_flat)
        self.particles = x_new_flat.reshape(batch_size, self.n_particles, self.state_dim)

    def update_step(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core APF update:
          1. Compute representative point mu_t and log_m.
          2. First-stage ancestor resample from lambda ∝ w_{t-1} * m_t.
          3. Propagate x_t ~ q(. | x_prev_sel, y_t).
          4. Second-stage log weights (generic APF formula).

        Returns:
            ancestor_idx: (B, N) ancestor indices selected in step 2.
            incremental_log_w: (B, N) the log-weight contribution
                from the current step (pre-normalization). Useful for
                diagnostics.
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        observation = observation.to(self.device)

        batch_size = self.particles.shape[0]
        dt = self.system.config.dt

        # x_{t-1}^i is currently in self.particles (step() stashes the
        # post-resample cloud BEFORE calling update_step).
        x_prev = self.particles  # (B, N, D)
        self.particles_prev = x_prev.clone()

        # ---------------------------------------------------------------------
        # Step A: representative point mu_t and adjustment multiplier m_t
        # ---------------------------------------------------------------------
        x_prev_flat = x_prev.reshape(-1, self.state_dim)
        integration_result = self.system.integrate(x_prev_flat, 2, dt)
        mu_t_flat = integration_result[:, 1, :]  # (B*N, D)

        if self.adjustment_type == "point":
            # m_t = g(y_t | mu_t): Gaussian with obs_noise_var only.
            y_pred_flat = self.system.apply_observation_operator(mu_t_flat)
            y_pred = y_pred_flat.reshape(batch_size, self.n_particles, self.obs_dim)
            obs_noise_var = self.system.config.obs_noise_std ** 2
            log_m = _gaussian_obs_log_prob(observation, y_pred, obs_noise_var, self.obs_dim)
        else:
            # "predictive": Gaussian-inflated approximation of p(y_t | x_{t-1})
            # using total_var = obs_var + process_var (same approximation as
            # BootstrapParticleFilter.compute_predictive_log_likelihood).
            y_pred_flat = self.system.apply_observation_operator(mu_t_flat)
            y_pred = y_pred_flat.reshape(batch_size, self.n_particles, self.obs_dim)
            obs_noise_var = self.system.config.obs_noise_std ** 2
            total_var = obs_noise_var + self.process_noise_std ** 2
            log_m = _gaussian_obs_log_prob(observation, y_pred, total_var, self.obs_dim)

        self.last_log_m = log_m

        # ---------------------------------------------------------------------
        # Step B: first-stage weights and ancestor resampling
        # ---------------------------------------------------------------------
        log_first_stage = self.log_weights + log_m
        ancestor_idx, _ = _systematic_resample_from_log_weights(log_first_stage)
        self.last_ancestor_idx = ancestor_idx

        x_prev_sel = _gather_particles(x_prev, ancestor_idx)  # (B, N, D)
        log_m_sel = torch.gather(log_m, 1, ancestor_idx)  # (B, N)

        # ---------------------------------------------------------------------
        # Step C: propagate from the selected ancestors
        # ---------------------------------------------------------------------
        x_prev_sel_flat = x_prev_sel.reshape(-1, self.state_dim)
        y_expanded_flat = (
            observation.unsqueeze(1)
            .expand(batch_size, self.n_particles, -1)
            .reshape(batch_size * self.n_particles, self.obs_dim)
        )

        time_flat = None
        if getattr(self, "_current_time_idxs", None) is not None:
            time_flat = (
                self._current_time_idxs.unsqueeze(1)
                .expand(batch_size, self.n_particles)
                .reshape(-1)
                .float()
                .to(x_prev_sel_flat.device)
            )

        x_new_flat = self.proposal.sample(
            x_prev_sel_flat, y_expanded_flat, dt, t=time_flat
        )
        self.particles = x_new_flat.reshape(batch_size, self.n_particles, self.state_dim)

        # Keep particles_prev aligned with the ancestor-selected cloud so
        # downstream diagnostics (and the ESS-triggered final resample) stay
        # consistent.
        self.particles_prev = x_prev_sel

        # ---------------------------------------------------------------------
        # Step D: second-stage incremental log weights (generic APF formula)
        # ---------------------------------------------------------------------
        x_t_flat = x_new_flat
        expected_obs_flat = self.system.apply_observation_operator(x_t_flat)
        expected_obs = expected_obs_flat.reshape(
            batch_size, self.n_particles, self.obs_dim
        )
        obs_noise_var = self.system.config.obs_noise_std ** 2
        log_g = _gaussian_obs_log_prob(observation, expected_obs, obs_noise_var, self.obs_dim)
        self.last_obs_log_likelihoods = log_g

        if self._proposal_is_transition:
            # Short-circuit: when q = f, log_f - log_q collapses to 0 exactly
            # (both are the same Gaussian kernel around system.integrate(x_prev)).
            incremental_log_w = log_g - log_m_sel
            log_f = None
            log_q = None
            proposal_log_probs_for_metric = None
        else:
            log_f = _transition_log_prob_batched(
                self.particles,
                x_prev_sel,
                self.system,
                self.process_noise_std,
                dt,
                self.state_dim,
            )
            log_q_flat = self.proposal.log_prob(
                x_t_flat,
                x_prev_sel_flat,
                y_expanded_flat,
                dt,
                t=time_flat,
            )
            log_q = log_q_flat.reshape(batch_size, self.n_particles)
            incremental_log_w = log_g + log_f - log_m_sel - log_q
            proposal_log_probs_for_metric = log_q

        self.last_proposal_log_probs = proposal_log_probs_for_metric
        self.last_incremental_log_w = incremental_log_w

        # One-resampling variant: weights are reset to the second-stage
        # incremental log weights (Johansen-Doucet). Ancestor resampling
        # already "spent" the old log_weights.
        self.log_weights = incremental_log_w

        return ancestor_idx, incremental_log_w

    def resample(self) -> Tuple[List[bool], torch.Tensor]:
        """
        ESS-triggered systematic resampling (identical policy to BPF).
        Used AFTER `update_step` to keep the effective sample size healthy.
        """
        batch_size = self.log_weights.shape[0]

        max_log_w = torch.max(self.log_weights, dim=1, keepdim=True).values
        w_unnorm = torch.exp(self.log_weights - max_log_w)
        w_sum = torch.sum(w_unnorm, dim=1, keepdim=True)

        zero_rows = (w_sum.squeeze(-1) == 0)
        if zero_rows.any():
            w_unnorm[zero_rows] = 1.0
            w_sum[zero_rows] = float(self.n_particles)

        self.weights = w_unnorm / w_sum
        ess = 1.0 / torch.sum(self.weights ** 2, dim=1)

        threshold = self.n_particles * self.resampling_threshold_ratio
        needs_resample = ess < threshold
        resampled_flags = needs_resample.cpu().tolist()

        if not needs_resample.any():
            self.log_weights = torch.log(self.weights)
            return resampled_flags, ess

        cumsum = torch.cumsum(self.weights, dim=1)
        u = torch.rand(batch_size, 1, device=self.device) / self.n_particles
        positions = u + torch.arange(
            self.n_particles, device=self.device, dtype=torch.float32
        ).unsqueeze(0) / self.n_particles
        indices = torch.searchsorted(cumsum, positions)
        indices = torch.clamp(indices, 0, self.n_particles - 1)

        idx_expanded = indices.unsqueeze(-1).expand(-1, -1, self.state_dim)
        particles_resampled = torch.gather(self.particles, 1, idx_expanded)
        particles_prev_resampled = torch.gather(self.particles_prev, 1, idx_expanded)

        mask = needs_resample.unsqueeze(-1).unsqueeze(-1)
        self.particles = torch.where(mask, particles_resampled, self.particles)
        self.particles_prev = torch.where(
            mask, particles_prev_resampled, self.particles_prev
        )

        new_weights = torch.where(
            needs_resample.unsqueeze(-1),
            torch.ones_like(self.weights) / self.n_particles,
            self.weights,
        )
        self.weights = new_weights
        self.log_weights = torch.log(self.weights)

        return resampled_flags, ess

    def get_state_estimate(self) -> torch.Tensor:
        """Weighted-mean state estimate: (B, D)."""
        return torch.sum(self.weights.unsqueeze(-1) * self.particles, dim=1)

    def get_state_covariance(self) -> torch.Tensor:
        """Weighted state covariance: (B, D, D)."""
        mean = self.get_state_estimate()
        centered = self.particles - mean.unsqueeze(1)
        outer = centered.unsqueeze(3) @ centered.unsqueeze(2)
        cov = torch.sum(self.weights.unsqueeze(-1).unsqueeze(-1) * outer, dim=1)
        return cov

    def sample_posterior(self, n_samples: int) -> torch.Tensor:
        """Weighted resample from the current particle cloud (batched)."""
        batch_size = self.particles.shape[0]
        samples = []
        for b in range(batch_size):
            idx = torch.multinomial(self.weights[b], n_samples, replacement=True)
            samples.append(self.particles[b, idx])
        return torch.stack(samples, dim=0)

    def compute_log_likelihood(self, observation: torch.Tensor) -> torch.Tensor:
        """
        log p(y_t | y_{1:t-1}) evaluated against the *current* particle
        cloud and weights (called before `update_step`).
        """
        batch_size = self.particles.shape[0]
        observation = observation.to(self.device)

        particles_flat = self.particles.reshape(-1, self.state_dim)
        expected_obs_flat = self.system.apply_observation_operator(particles_flat)
        expected_obs = expected_obs_flat.reshape(
            batch_size, self.n_particles, self.obs_dim
        )

        obs_noise_var = self.system.config.obs_noise_std ** 2
        log_likelihoods = _gaussian_obs_log_prob(
            observation, expected_obs, obs_noise_var, self.obs_dim
        )

        log_marginal = torch.logsumexp(self.log_weights + log_likelihoods, dim=1) \
            - torch.logsumexp(self.log_weights, dim=1)
        return log_marginal

    # -------------------------------------------------------------------------
    # Main entry point used by eval.py
    # -------------------------------------------------------------------------

    def step(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        trajectory_idxs: torch.Tensor,
        time_idxs: torch.Tensor,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """APF step for a batch of trajectories."""
        start_time = time.perf_counter()
        self.step_count += 1
        self._current_time_idxs = time_idxs

        batch_size = x_curr.shape[0]
        log_likelihoods = torch.zeros(batch_size, device=self.device)
        resampled_flags: List[bool] = [False] * batch_size
        ess_pre = 1.0 / torch.sum(self.weights ** 2, dim=1)
        ensemble_spread_pre = torch.zeros(batch_size, device=self.device)

        # Reset APF-specific diagnostics that are only populated when we
        # actually perform a full APF update.
        self.last_log_m = None
        self.last_ancestor_idx = None
        self.last_incremental_log_w = None
        self.last_proposal_log_probs = None
        self.last_obs_log_likelihoods = None

        if y_curr is None:
            # Propagate under transition prior (no ancestor selection, no
            # incremental weights). Matches BPF's missing-obs branch.
            self.predict_step(dt, None)
        else:
            # Predictive log-likelihood logged BEFORE the APF update (same
            # convention as BootstrapParticleFilter.step).
            log_likelihoods = self.compute_log_likelihood(y_curr)
            self.update_step(y_curr)
            # Stable weight normalization for spread diagnostic (log_weights
            # are unnormalized incremental log-weights at this point).
            _max = torch.max(self.log_weights, dim=1, keepdim=True).values
            _w = torch.exp(self.log_weights - _max)
            _w = _w / torch.sum(_w, dim=1, keepdim=True).clamp_min(1e-30)
            ensemble_spread_pre = compute_weighted_spread(self.particles, _w)
            resampled_flags, ess_pre = self.resample()

        x_est = self.get_state_estimate()
        P_est = self.get_state_covariance()

        x_curr = x_curr.to(self.device)
        error = x_est - x_curr
        rmse = torch.sqrt(torch.mean(error ** 2, dim=1))
        ess = 1.0 / torch.sum(self.weights ** 2, dim=1)

        elapsed = (time.perf_counter() - start_time) / batch_size

        if self.last_proposal_log_probs is not None:
            prop_means = self.last_proposal_log_probs.mean(dim=1)
        else:
            prop_means = torch.zeros(batch_size, device=self.device)

        if self.last_obs_log_likelihoods is not None:
            obs_means = self.last_obs_log_likelihoods.mean(dim=1)
            obs_stds = self.last_obs_log_likelihoods.std(dim=1)
        else:
            obs_means = torch.zeros(batch_size, device=self.device)
            obs_stds = torch.zeros(batch_size, device=self.device)

        if self.last_log_m is not None:
            log_m_mean = self.last_log_m.mean(dim=1)
            log_m_std = self.last_log_m.std(dim=1)
        else:
            log_m_mean = torch.zeros(batch_size, device=self.device)
            log_m_std = torch.zeros(batch_size, device=self.device)

        if self.last_incremental_log_w is not None:
            inc_w_std = self.last_incremental_log_w.std(dim=1)
        else:
            inc_w_std = torch.zeros(batch_size, device=self.device)

        if self.last_ancestor_idx is not None:
            unique_ancestors = torch.tensor(
                [
                    int(torch.unique(self.last_ancestor_idx[b]).numel())
                    for b in range(batch_size)
                ],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            unique_ancestors = torch.zeros(batch_size, device=self.device)

        results: List[Dict[str, Any]] = []
        for i in range(batch_size):
            metrics = {
                "rmse": rmse[i].item(),
                "log_likelihood": log_likelihoods[i].item(),
                "x_est": x_est[i].cpu().numpy(),
                "P_est": P_est[i].cpu().numpy(),
                "error": error[i].cpu().numpy(),
                "trajectory_idx": trajectory_idxs[i].item(),
                "time_idx": time_idxs[i].item(),
                "resampled": resampled_flags[i],
                "ess_pre_resample": ess_pre[i].item(),
                "ess": ess[i].item(),
                "ensemble_spread_pre_resample": (
                    ensemble_spread_pre[i].item() if y_curr is not None else 0.0
                ),
                "step_time": elapsed,
                "proposal_log_prob_mean": prop_means[i].item(),
                "obs_log_prob_mean": obs_means[i].item(),
                "obs_log_prob_std": obs_stds[i].item(),
                # APF-specific diagnostics:
                "apf_log_m_mean": log_m_mean[i].item(),
                "apf_log_m_std": log_m_std[i].item(),
                "apf_unique_ancestors": int(unique_ancestors[i].item()),
                "apf_inc_log_w_std": inc_w_std[i].item(),
            }
            results.append(metrics)

        return results
