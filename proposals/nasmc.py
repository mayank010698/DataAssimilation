"""Neural Adaptive Sequential Monte Carlo (NASMC) proposal.

This module implements the learned proposal from
Gu, Ghahramani & Turner (2015, arXiv:1506.03338) in a form compatible
with the rest of this codebase. The proposal is a diagonal Gaussian
whose mean and log-standard-deviation are produced by one of the
``architectures/gaussian_head.py`` backbones.

Phase 1 (``training_phase == 'pretrain'``): the module is trained by
standard maximum likelihood on ground-truth transition pairs
``(x_prev, x_curr, y_curr)`` coming from :class:`RFTransitionDataset`.
This is the "local MLE" baseline that also serves as a warm-start for
phase 2.

Phase 2 (``training_phase == 'refine'``): the module runs a forward
SMC sweep for each trajectory in the batch with the current proposal,
detaches the resulting particles/weights, and then takes a gradient
step on the weighted log-density of those particles under a fresh
forward pass of the proposal. This is the NASMC gradient estimator.

The module exposes ``sample`` / ``log_prob`` in scaled state space,
matching the :class:`RFProposal` convention so that the two proposals
can be swapped in the bootstrap particle filter.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .architectures import create_gaussian_head_network
except ImportError:
    from architectures import create_gaussian_head_network  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: sparse-obs log-likelihood in scaled space.
# ---------------------------------------------------------------------------


def _apply_obs_operator_scaled(
    x_scaled: torch.Tensor,
    state_scaler_mean: Optional[torch.Tensor],
    state_scaler_std: Optional[torch.Tensor],
    obs_scaler_mean: Optional[torch.Tensor],
    obs_scaler_std: Optional[torch.Tensor],
    obs_indices: Optional[List[int]],
    obs_nonlinearity: str,
) -> torch.Tensor:
    """Apply the observation operator to a state in *scaled* space.

    Exactly mirrors :meth:`RFProposal._apply_obs_operator_scaled`:
    unscale state -> slice observed components -> apply nonlinearity
    -> scale observation. Used inside the SMC inner loop to compute
    ``log p(y_t | x_t)`` when observations are in *scaled* space.
    """
    device = x_scaled.device

    if state_scaler_mean is not None and state_scaler_std is not None:
        mean = state_scaler_mean.to(device)
        std = state_scaler_std.to(device)
        x_phys = x_scaled * std + mean
    else:
        x_phys = x_scaled

    if obs_indices is not None:
        observed = x_phys[..., obs_indices]
    else:
        observed = x_phys

    nl = obs_nonlinearity or "arctan"
    if nl == "arctan":
        y_phys = torch.arctan(observed)
    elif nl == "square":
        y_phys = torch.square(observed)
    elif nl == "cube":
        y_phys = torch.pow(observed, 3)
    elif nl in ("linear_projection", "identity", "none"):
        y_phys = observed
    elif nl == "quad_capped_10":
        y_phys = torch.clamp(torch.pow(observed, 4), max=10.0) / 10.0
    elif hasattr(torch, nl):
        y_phys = getattr(torch, nl)(observed)
    else:
        raise ValueError(f"Unknown observation nonlinearity: {nl}")

    if obs_scaler_mean is not None and obs_scaler_std is not None:
        y_phys = (y_phys - obs_scaler_mean.to(device)) / obs_scaler_std.to(device)
    return y_phys


def _diagonal_gaussian_log_prob(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """Sum-of-coordinates diag-Gaussian log-density."""
    return -0.5 * (
        ((x - mu) / log_sigma.exp()) ** 2
        + 2.0 * log_sigma
        + math.log(2.0 * math.pi)
    ).sum(dim=-1)


# ---------------------------------------------------------------------------
# GaussianProposal: the standalone proposal model.
# ---------------------------------------------------------------------------


class GaussianProposal(pl.LightningModule):
    """Diagonal-Gaussian proposal ``q_phi(x_t | x_{t-1}, y_t)``.

    Matches the semantics of :class:`RFProposal` at the proposal
    level: all computations are in *scaled* state space (and scaled
    observation space for the auxiliary ``_apply_obs_operator_scaled``
    helper), so this module slots into the filter-side wrapper
    ``NASMCProposal`` which handles pre/post-processing.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int = 0,
        architecture: str = "mlp",
        hidden_dim: int = 128,
        depth: int = 4,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
        obs_indices: Optional[List[int]] = None,
        use_time_step: bool = False,
        trajectory_length: int = 1000,
        # Output parametrisation
        predict_delta: bool = True,
        use_dynamics_mean: bool = False,  # -f- variant (requires system)
        log_sigma_min: float = -7.0,
        log_sigma_max: float = 3.0,
        init_log_sigma: float = -1.0,
        zero_init_output: bool = True,
        # Optimisation (phase 1)
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        # Dataset-level metadata, used by the -f- variant and observation op.
        process_noise_std: float = 0.01,
        obs_noise_std: float = 0.1,
        obs_nonlinearity: str = "arctan",
        # Scalers (needed for the SMC loop and for the -f- variant; kept
        # off the hparams so the checkpoint stays portable).
        state_scaler_mean: Optional[torch.Tensor] = None,
        state_scaler_std: Optional[torch.Tensor] = None,
        obs_scaler_mean: Optional[torch.Tensor] = None,
        obs_scaler_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Only save JSON-serialisable hyperparameters; keep tensors separate
        # so that load_from_checkpoint reconstructs the module shape but
        # the caller re-attaches scalers/system for the SMC loop.
        self.save_hyperparameters(
            ignore=[
                "state_scaler_mean",
                "state_scaler_std",
                "obs_scaler_mean",
                "obs_scaler_std",
            ]
        )

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.architecture = architecture
        self.predict_delta = predict_delta
        self.use_dynamics_mean = use_dynamics_mean
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.use_time_step = use_time_step
        self.trajectory_length = trajectory_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.process_noise_std = float(process_noise_std)
        self.obs_noise_std = float(obs_noise_std)
        self.obs_nonlinearity = obs_nonlinearity or "arctan"
        self.obs_indices = list(obs_indices) if obs_indices is not None else None

        self.net = create_gaussian_head_network(
            architecture=architecture,
            state_dim=state_dim,
            obs_dim=obs_dim,
            use_time_step=use_time_step,
            hidden_dim=hidden_dim,
            depth=depth,
            channels=channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            obs_indices=obs_indices,
            zero_init_output=zero_init_output,
            init_log_sigma=init_log_sigma,
        )

        # Register scalers as non-persistent buffers so they travel with
        # the module but are not required to be present in the checkpoint.
        def _as_buffer(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, dtype=torch.float32)
            return t.float()

        sm = _as_buffer(state_scaler_mean)
        ss = _as_buffer(state_scaler_std)
        om = _as_buffer(obs_scaler_mean)
        os_ = _as_buffer(obs_scaler_std)
        self.register_buffer(
            "state_scaler_mean", sm if sm is not None else torch.zeros(state_dim),
            persistent=False,
        )
        self.register_buffer(
            "state_scaler_std", ss if ss is not None else torch.ones(state_dim),
            persistent=False,
        )
        if om is not None:
            self.register_buffer("obs_scaler_mean", om, persistent=False)
            self.register_buffer("obs_scaler_std", os_, persistent=False)
        else:
            self.obs_scaler_mean = None
            self.obs_scaler_std = None
        self._scalers_from_user = (sm is not None) and (ss is not None)

        # Optional DynamicalSystem handle for the -f- variant.
        # Set post-construction by callers that want use_dynamics_mean=True
        # (via ``attach_system``). Never part of the checkpoint.
        self._system = None
        self._dt: Optional[float] = None

        # Phase switch (may be overridden per-step by the trainer).
        # Either 'pretrain' (MLE on one-step pairs) or 'refine' (SMC).
        self.training_phase: str = "pretrain"

    # ------------------------------------------------------------------
    # External hooks used by train_nasmc and NASMCProposal.
    # ------------------------------------------------------------------

    def attach_system(self, system, dt: Optional[float] = None) -> None:
        """Attach a :class:`DynamicalSystem` for the -f- variant / SMC."""
        self._system = system
        if dt is None and system is not None and hasattr(system, "config"):
            dt = getattr(system.config, "dt", None)
        self._dt = dt

    def attach_scalers(
        self,
        state_scaler_mean: Optional[torch.Tensor] = None,
        state_scaler_std: Optional[torch.Tensor] = None,
        obs_scaler_mean: Optional[torch.Tensor] = None,
        obs_scaler_std: Optional[torch.Tensor] = None,
    ) -> None:
        """Override scalers after checkpoint load (for SMC / obs log-prob).

        Handles three cases for the obs scalers:
          1. Attribute was set to ``None`` (no obs scaler ever attached)
             -> register a fresh non-persistent buffer.
          2. A non-persistent buffer already exists -> overwrite the
             buffer's data in-place.
          3. A plain (non-buffer) attribute exists -> replace it with a
             new buffer (after detaching the plain attribute first).
        """
        device = self.device

        def _set_or_register(name: str, value: torch.Tensor) -> None:
            v = value.float().to(device)
            if name in self._buffers:
                self._buffers[name] = v
            else:
                # Drop any plain attribute of this name before registering.
                if name in self.__dict__:
                    del self.__dict__[name]
                self.register_buffer(name, v, persistent=False)

        if state_scaler_mean is not None:
            _set_or_register("state_scaler_mean", state_scaler_mean)
        if state_scaler_std is not None:
            _set_or_register("state_scaler_std", state_scaler_std)
        if obs_scaler_mean is not None:
            _set_or_register("obs_scaler_mean", obs_scaler_mean)
        if obs_scaler_std is not None:
            _set_or_register("obs_scaler_std", obs_scaler_std)
        self._scalers_from_user = True

    # ------------------------------------------------------------------
    # Time conditioning helper.
    # ------------------------------------------------------------------

    def _normalize_trajectory_time(
        self,
        t: Optional[torch.Tensor],
        batch_size: int,
        caller_name: str,
    ) -> Optional[torch.Tensor]:
        if not self.use_time_step:
            return None
        if t is None:
            raise ValueError(
                f"use_time_step=True but t was not provided to {caller_name}"
            )
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=self.device)
        else:
            t = t.to(self.device).float()
        if t.dim() == 0:
            t = t.reshape(1, 1).expand(batch_size, 1)
        elif t.dim() == 1:
            if t.shape[0] == 1 and batch_size != 1:
                t = t.expand(batch_size)
            t = t.unsqueeze(1)
        elif t.dim() == 2:
            if t.shape[1] != 1:
                raise ValueError(
                    f"{caller_name}: expected t with second dimension 1, got {tuple(t.shape)}"
                )
            if t.shape[0] == 1 and batch_size != 1:
                t = t.expand(batch_size, 1)
        else:
            raise ValueError(
                f"{caller_name}: expected scalar / 1D / 2D t tensor, got {t.dim()}D"
            )
        return t.float() / float(self.trajectory_length)

    # ------------------------------------------------------------------
    # Forward pass: produce (mu, log_sigma).
    # ------------------------------------------------------------------

    def _raw_forward(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        t_normalized: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.net(x_prev, y_curr, t_normalized)

    def mean_and_log_sigma(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian parameters ``(mu, log_sigma)`` in scaled state space."""
        batch_size = x_prev.shape[0]
        t_normalized = self._normalize_trajectory_time(
            t=t, batch_size=batch_size, caller_name="mean_and_log_sigma",
        )
        out = self._raw_forward(x_prev, y_curr, t_normalized)
        mu_raw, log_sigma = out.chunk(2, dim=-1)
        log_sigma = torch.clamp(log_sigma, min=self.log_sigma_min, max=self.log_sigma_max)

        if self.use_dynamics_mean:
            if self._system is None or self._dt is None:
                raise ValueError(
                    "use_dynamics_mean=True requires a DynamicalSystem to be "
                    "attached via attach_system(system, dt)."
                )
            # Deterministic one-step dynamics in *scaled* space:
            # unscale -> RK4/integrate -> scale again.
            with torch.no_grad():
                x_prev_phys = (
                    x_prev * self.state_scaler_std.to(x_prev.device)
                    + self.state_scaler_mean.to(x_prev.device)
                )
                integ = self._system.integrate(x_prev_phys, 2, self._dt)
                if x_prev.ndim == 2:
                    x_next_phys = integ[:, 1, :]
                else:
                    x_next_phys = integ[1, :]
                x_next_scaled = (
                    x_next_phys - self.state_scaler_mean.to(x_prev.device)
                ) / self.state_scaler_std.to(x_prev.device)
            mu = x_next_scaled + mu_raw
        elif self.predict_delta:
            mu = x_prev + mu_raw
        else:
            mu = mu_raw
        return mu, log_sigma

    # ------------------------------------------------------------------
    # ProposalDistribution-style API (scaled space).
    # ------------------------------------------------------------------

    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        t: Optional[torch.Tensor] = None,
        return_params: bool = False,
    ) -> torch.Tensor:
        """Draw a reparameterised Gaussian sample ``x_t`` in scaled space."""
        was_1d = x_prev.dim() == 1
        if was_1d:
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)

        mu, log_sigma = self.mean_and_log_sigma(x_prev, y_curr, t)
        eps = torch.randn_like(mu)
        x = mu + log_sigma.exp() * eps

        if was_1d:
            x = x.squeeze(0)
            mu = mu.squeeze(0)
            log_sigma = log_sigma.squeeze(0)
        if return_params:
            return x, mu, log_sigma
        return x

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Closed-form diagonal-Gaussian log-density in scaled space."""
        was_1d = x_curr.dim() == 1
        if was_1d:
            x_curr = x_curr.unsqueeze(0)
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)

        mu, log_sigma = self.mean_and_log_sigma(x_prev, y_curr, t)
        lp = _diagonal_gaussian_log_prob(x_curr, mu, log_sigma)

        if was_1d:
            lp = lp.squeeze(0)
        return lp

    # ------------------------------------------------------------------
    # Phase 1: local-MLE training on RFTransitionDataset batches.
    # ------------------------------------------------------------------

    def _mle_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_prev = batch["x_prev"]
        x_curr = batch["x_curr"]
        y_curr = batch.get("y_curr", None)
        t = batch.get("time_idx", None)

        log_prob = self.log_prob(x_curr, x_prev, y_curr, t=t)
        nll = -log_prob.mean()
        metrics = {
            "nll": float(nll.detach().item()),
            "log_prob_mean": float(log_prob.detach().mean().item()),
        }
        return nll, metrics

    # ------------------------------------------------------------------
    # Phase 2: SMC inner loop and weighted-log-density loss.
    # ------------------------------------------------------------------

    def _transition_log_prob_scaled(
        self,
        x_curr_scaled: torch.Tensor,
        x_prev_scaled: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian transition log-prob in *scaled* space.

        The ground-truth dynamics are ``x_t = RK4(x_{t-1}) + eta`` with
        ``eta ~ N(0, sigma_p^2 I)`` in *physical* space. In scaled space
        this becomes ``N(mu_scaled, (sigma_p / state_std)^2)`` per
        coordinate.
        """
        device = x_curr_scaled.device
        std = self.state_scaler_std.to(device)
        mean = self.state_scaler_mean.to(device)

        x_prev_phys = x_prev_scaled * std + mean
        integ = self._system.integrate(x_prev_phys, 2, self._dt)
        # integ is either (N, 2, D) when x_prev was 2D or (2, D) when 1D.
        if x_prev_scaled.ndim == 2:
            x_next_phys = integ[:, 1, :]
        else:
            x_next_phys = integ[1, :]
        x_next_scaled = (x_next_phys - mean) / std

        noise_std_scaled = self.process_noise_std / std  # per-dim std in scaled space
        diff = x_curr_scaled - x_next_scaled
        log_det = torch.sum(torch.log(noise_std_scaled))
        lp = (
            -0.5 * torch.sum((diff / noise_std_scaled) ** 2, dim=-1)
            - 0.5 * self.state_dim * math.log(2.0 * math.pi)
            - log_det
        )
        return lp

    def _observation_log_prob_scaled(
        self,
        x_scaled: torch.Tensor,
        y_scaled: torch.Tensor,
    ) -> torch.Tensor:
        """Gaussian obs log-prob in *scaled* space for state ``x_scaled``.

        Observations come from ``y = h(x) + nu`` with ``nu ~ N(0, obs_std^2 I)``
        in *physical* space. ``y_scaled = (y - obs_mean) / obs_std`` and
        ``h_scaled(x) = (h(phys(x)) - obs_mean) / obs_std``, so the
        Gaussian likelihood has per-dim std ``1`` in scaled space when
        the observations have been standardised. (If no obs scaler is
        available we use ``obs_std`` directly.)
        """
        h_scaled = _apply_obs_operator_scaled(
            x_scaled,
            state_scaler_mean=self.state_scaler_mean,
            state_scaler_std=self.state_scaler_std,
            obs_scaler_mean=self.obs_scaler_mean,
            obs_scaler_std=self.obs_scaler_std,
            obs_indices=self.obs_indices,
            obs_nonlinearity=self.obs_nonlinearity,
        )
        diff = y_scaled - h_scaled
        if self.obs_scaler_std is not None:
            # After scaling by obs_std, the noise std is obs_noise_std / obs_std.
            obs_std = self.obs_noise_std / self.obs_scaler_std.to(x_scaled.device)
            log_det = torch.sum(torch.log(obs_std))
            lp = (
                -0.5 * torch.sum((diff / obs_std) ** 2, dim=-1)
                - 0.5 * diff.shape[-1] * math.log(2.0 * math.pi)
                - log_det
            )
        else:
            obs_var = self.obs_noise_std ** 2
            lp = (
                -0.5 * torch.sum(diff ** 2, dim=-1) / obs_var
                - 0.5 * diff.shape[-1] * math.log(2.0 * math.pi * obs_var)
            )
        return lp

    @torch.no_grad()
    def _smc_forward(
        self,
        trajectories: torch.Tensor,   # (B, T, D), scaled
        observations: torch.Tensor,   # (B, T, O), scaled (if obs scaler present)
        obs_mask: torch.Tensor,       # (B, T), bool / {0,1}
        num_particles: int,
        resample_threshold: float = 0.5,
        use_bootstrap: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run an SMC sweep with the current proposal.

        Returns dict with detached tensors:
          - ``particles``: (B, T, N, D) scaled-space particles after
            sampling at each step (pre-resampling).
          - ``ancestors``: (B, T, N) long, ancestor index at step t.
          - ``weights``: (B, T, N) normalised (post-weight-update)
            weights per step.
          - ``log_weights_incr``: (B, T, N) unnormalised log importance
            weight increment per step (for diagnostics).
          - ``has_obs``: (B, T) bool; whether step t had an observation.
        """
        if self._system is None or self._dt is None:
            raise RuntimeError(
                "GaussianProposal requires a DynamicalSystem for SMC "
                "(call attach_system(system, dt) before phase-2 training)."
            )

        self.eval()
        device = self.device
        B, T, D = trajectories.shape
        N = num_particles
        dtype = trajectories.dtype

        particles_hist = torch.zeros(B, T, N, D, device=device, dtype=dtype)
        ancestors_hist = torch.zeros(B, T, N, dtype=torch.long, device=device)
        weights_hist = torch.zeros(B, T, N, device=device, dtype=dtype)
        log_incr_hist = torch.zeros(B, T, N, device=device, dtype=dtype)
        has_obs_hist = torch.zeros(B, T, dtype=torch.bool, device=device)

        # Initialise particles at the ground-truth x_0 in scaled space
        # (simple and standard for NASMC; §7.1 of the plan).
        x_prev = trajectories[:, 0].unsqueeze(1).expand(B, N, D).contiguous()
        log_weights = torch.zeros(B, N, device=device, dtype=dtype)

        # Store step 0 (no proposal sample yet; particles = x_0 copies).
        particles_hist[:, 0] = x_prev
        weights_hist[:, 0] = F.softmax(log_weights, dim=-1)
        ancestors_hist[:, 0] = torch.arange(N, device=device).unsqueeze(0).expand(B, N)

        resample_threshold_n = resample_threshold * N

        for t in range(1, T):
            y_t = observations[:, t]
            m_t = obs_mask[:, t]
            has_obs = bool(m_t.any().item())
            has_obs_hist[:, t] = m_t

            # Resample based on the previous step's weights if ESS is low.
            weights = F.softmax(log_weights, dim=-1)  # (B, N)
            ess = 1.0 / torch.sum(weights ** 2 + 1e-30, dim=-1)  # (B,)
            need_resample = ess < resample_threshold_n
            ancestors = torch.arange(N, device=device).unsqueeze(0).expand(B, N).clone()
            if need_resample.any():
                # Systematic resampling per batch item.
                for b in torch.nonzero(need_resample, as_tuple=False).squeeze(-1):
                    cumsum = torch.cumsum(weights[b], dim=-1)
                    u = torch.rand(1, device=device) / N
                    positions = u + torch.arange(N, device=device, dtype=dtype) / N
                    idx = torch.searchsorted(cumsum, positions)
                    idx = idx.clamp(0, N - 1)
                    ancestors[b] = idx
                    log_weights[b] = 0.0  # reset after resampling
                x_prev = torch.gather(
                    x_prev, 1, ancestors.unsqueeze(-1).expand(-1, -1, D)
                )
            ancestors_hist[:, t] = ancestors

            # Sample particles at step t.
            x_prev_flat = x_prev.reshape(B * N, D)
            y_flat = None
            if has_obs and self.obs_dim > 0:
                # Broadcast y_t to particles (masked-off rows just contribute
                # nothing to the loss because has_obs_hist is False for them).
                y_flat = y_t.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

            t_flat = None
            if self.use_time_step:
                t_flat = torch.full(
                    (B * N,), float(t), device=device, dtype=dtype
                )

            proposal_enabled = has_obs and not use_bootstrap
            if proposal_enabled:
                mu, log_sigma = self.mean_and_log_sigma(x_prev_flat, y_flat, t_flat)
                eps = torch.randn_like(mu)
                x_new_flat = mu + log_sigma.exp() * eps
                log_q = _diagonal_gaussian_log_prob(x_new_flat, mu, log_sigma)
            else:
                # Bootstrap (prior) proposal: sample from the transition.
                mu_trans = self._dynamics_next_scaled(x_prev_flat)
                std_scaled = self.process_noise_std / self.state_scaler_std.to(
                    x_prev_flat.device
                )
                eps = torch.randn_like(mu_trans)
                x_new_flat = mu_trans + std_scaled * eps
                log_q = _diagonal_gaussian_log_prob(
                    x_new_flat, mu_trans, torch.log(std_scaled).expand_as(mu_trans)
                )

            x_new = x_new_flat.reshape(B, N, D)
            particles_hist[:, t] = x_new

            log_p_trans = self._transition_log_prob_scaled(x_new_flat, x_prev_flat)
            if has_obs and self.obs_dim > 0:
                log_p_obs = self._observation_log_prob_scaled(x_new_flat, y_flat)
            else:
                log_p_obs = torch.zeros_like(log_p_trans)

            log_incr = (log_p_obs + log_p_trans - log_q).reshape(B, N)
            log_incr_hist[:, t] = log_incr

            # Update log-weights (SIS update). Mask out rows where has_obs is
            # False for that batch element so the weight increment collapses
            # to the transition-vs-prior ratio (which is 0 when we use the
            # bootstrap at unobserved steps).
            log_weights = log_weights + log_incr
            # Normalise-in-log so numbers stay sane.
            log_weights = log_weights - torch.logsumexp(
                log_weights, dim=-1, keepdim=True
            )
            weights_hist[:, t] = F.softmax(log_weights, dim=-1)

            x_prev = x_new

        return dict(
            particles=particles_hist,
            ancestors=ancestors_hist,
            weights=weights_hist,
            log_weights_incr=log_incr_hist,
            has_obs=has_obs_hist,
        )

    def _dynamics_next_scaled(self, x_scaled: torch.Tensor) -> torch.Tensor:
        device = x_scaled.device
        std = self.state_scaler_std.to(device)
        mean = self.state_scaler_mean.to(device)
        x_phys = x_scaled * std + mean
        integ = self._system.integrate(x_phys, 2, self._dt)
        if x_scaled.ndim == 2:
            x_next_phys = integ[:, 1, :]
        else:
            x_next_phys = integ[1, :]
        return (x_next_phys - mean) / std

    def _nasmc_loss(
        self,
        batch: Dict[str, torch.Tensor],
        num_particles: int,
        resample_threshold: float,
        use_bootstrap: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        trajectories = batch["trajectories"]     # (B, T, D), scaled
        observations = batch["observations"]     # (B, T, O), scaled
        obs_mask = batch["obs_mask"].bool()      # (B, T)
        B, T, D = trajectories.shape
        N = num_particles

        # Phase 2a: SMC forward (no gradients).
        smc = self._smc_forward(
            trajectories=trajectories,
            observations=observations,
            obs_mask=obs_mask,
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            use_bootstrap=use_bootstrap,
        )

        particles = smc["particles"]           # (B, T, N, D)
        ancestors = smc["ancestors"]           # (B, T, N)
        weights = smc["weights"]               # (B, T, N)
        has_obs = smc["has_obs"]               # (B, T)

        # Phase 2b: recompute log q_phi(x_t^n | x_{t-1}^{A_{t-1}^n}, y_t)
        # with gradients for every observed step and accumulate the
        # weighted negative log-density.
        self.train()
        total_loss = torch.zeros((), device=self.device)
        total_ess_sum = 0.0
        n_obs_steps = 0

        for t in range(1, T):
            if not bool(has_obs[:, t].any().item()):
                continue

            # Gather ancestors of this step's particles into x_prev.
            ancs = ancestors[:, t]  # (B, N)
            x_prev_step = torch.gather(
                particles[:, t - 1], 1, ancs.unsqueeze(-1).expand(-1, -1, D)
            )  # (B, N, D)
            x_curr_step = particles[:, t]  # (B, N, D)
            y_t = observations[:, t]       # (B, O)
            m_t = has_obs[:, t]            # (B,)

            x_prev_flat = x_prev_step.reshape(B * N, D)
            x_curr_flat = x_curr_step.reshape(B * N, D)
            y_flat = y_t.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1) if self.obs_dim > 0 else None
            t_flat = None
            if self.use_time_step:
                t_flat = torch.full(
                    (B * N,), float(t), device=self.device, dtype=x_prev_flat.dtype
                )

            mu, log_sigma = self.mean_and_log_sigma(x_prev_flat, y_flat, t_flat)
            log_q = _diagonal_gaussian_log_prob(x_curr_flat, mu, log_sigma)
            log_q = log_q.reshape(B, N)

            # Only count batch items that have an observation at step t.
            w = weights[:, t]                         # (B, N)
            mask_b = m_t.float().unsqueeze(-1)         # (B, 1)
            weighted = (w * log_q) * mask_b            # (B, N)
            # Mean over batch, sum over particles.
            denom = mask_b.sum().clamp_min(1.0)
            total_loss = total_loss - weighted.sum() / denom

            total_ess_sum += (
                (1.0 / torch.sum(w ** 2 + 1e-30, dim=-1)) * mask_b.squeeze(-1)
            ).sum().item() / max(denom.item(), 1.0)
            n_obs_steps += 1

        if n_obs_steps == 0:
            raise RuntimeError(
                "NASMC loss was requested but the batch has no observed steps."
            )
        total_loss = total_loss / n_obs_steps

        metrics = {
            "nasmc_loss": float(total_loss.detach().item()),
            "mean_ess": total_ess_sum / max(n_obs_steps, 1),
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Lightning hooks.
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if self.training_phase == "pretrain":
            loss, metrics = self._mle_loss(batch)
            self.log("train_nll", metrics["nll"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_log_prob", metrics["log_prob_mean"], on_epoch=True)
        elif self.training_phase == "refine":
            num_particles = int(self._refine_cfg.get("num_particles", 64))
            resample_threshold = float(self._refine_cfg.get("resample_threshold", 0.5))
            use_bootstrap = bool(self._refine_cfg.get("use_bootstrap", False))
            loss, metrics = self._nasmc_loss(
                batch,
                num_particles=num_particles,
                resample_threshold=resample_threshold,
                use_bootstrap=use_bootstrap,
            )
            self.log("train_nasmc", metrics["nasmc_loss"], prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_mean_ess", metrics["mean_ess"], on_epoch=True, prog_bar=True)
        else:
            raise ValueError(f"Unknown training_phase: {self.training_phase}")
        return loss

    def validation_step(self, batch, batch_idx):
        if self.training_phase == "pretrain":
            _, metrics = self._mle_loss(batch)
            self.log("val_nll", metrics["nll"], prog_bar=True, on_epoch=True)
            self.log("val_log_prob", metrics["log_prob_mean"], on_epoch=True)
        else:
            # Validation in refine mode is costly (another SMC pass).
            # We compute the phase-1 NLL on a dummy (x_prev, x_curr) view
            # of each trajectory so validation is always cheap and
            # comparable across phases.
            trajectories = batch["trajectories"]  # (B, T, D)
            observations = batch.get("observations", None)
            B, T, D = trajectories.shape
            x_prev = trajectories[:, :-1].reshape(B * (T - 1), D)
            x_curr = trajectories[:, 1:].reshape(B * (T - 1), D)
            if observations is not None and self.obs_dim > 0:
                y = observations[:, 1:].reshape(B * (T - 1), -1)
            else:
                y = None
            mle_batch = {"x_prev": x_prev, "x_curr": x_curr}
            if y is not None:
                mle_batch["y_curr"] = y
            _, metrics = self._mle_loss(mle_batch)
            self.log("val_nll", metrics["nll"], prog_bar=True, on_epoch=True)
            self.log("val_log_prob", metrics["log_prob_mean"], on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_nll",
            },
        }

    # ------------------------------------------------------------------
    # Configuration knobs set from the training script.
    # ------------------------------------------------------------------

    _refine_cfg: Dict[str, Any] = {}

    def set_phase(
        self,
        phase: str,
        *,
        num_particles: int = 64,
        resample_threshold: float = 0.5,
        use_bootstrap: bool = False,
    ) -> None:
        if phase not in ("pretrain", "refine"):
            raise ValueError(f"phase must be 'pretrain' or 'refine'; got {phase!r}")
        self.training_phase = phase
        self._refine_cfg = {
            "num_particles": num_particles,
            "resample_threshold": resample_threshold,
            "use_bootstrap": use_bootstrap,
        }
