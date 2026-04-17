"""
Localized Rectified Flow Proposal
---------------------------------

Patch-based rectified flow proposal distribution for high-dimensional, spatially
homogeneous systems (e.g. Lorenz-96). A single small velocity network is shared
across every grid site and consumes a window of radius `r` around the site.

Key properties enabled by the localized architecture:

1. Finite receptive field: site j's velocity only depends on state values at
   `[j-r, j+r]` (circularly indexed). The Jacobian of the full-state velocity
   field is banded with bandwidth r.

2. Spatial homogeneity: training on L96-40 transfers zero-shot to L96-400
   (one checkpoint, any state_dim).

3. Per-dimension log-density: the total log-density under the RF proposal
   decomposes along spatial dimensions and can be computed at O(N_x * r)
   cost rather than O(N_x^2). This enables principled local importance
   weighting in the localized particle filter.

Training: patch-level. Each item in the DataLoader is one site of one
transition; the RF loss is evaluated at that single centre. See
`proposals/patch_dataset.py`.

Inference: the same network is applied at every site in parallel via one
batched forward pass over (B * N_x) patches.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .architectures import create_local_velocity_network
    from .patch_utils import WindowSpec, extract_patches, dense_obs_from_components
except ImportError:  # direct script execution
    from architectures import create_local_velocity_network
    from patch_utils import WindowSpec, extract_patches, dense_obs_from_components

logger = logging.getLogger(__name__)


class LocalizedRFProposal(pl.LightningModule):
    """
    Localized Rectified Flow proposal q(x_t | x_{t-1}, y_t).

    Args:
        radius: Spatial radius r. Window size is 2*r+1.
        architecture: 'local_mlp' or 'local_resnet1d'.
        state_dim: Default full-state dimension at training time. May be
            overridden at inference time (zero-shot transfer to different
            grid sizes is an explicit goal).
        use_observations: Whether observations are fed into the local net.
        obs_components: Indices of observed state sites (for scatter at inference).
            Required whenever `use_observations=True` and the PF passes a sparse
            observation vector.
        predict_delta: If True, the net learns the increment x_t - x_{t-1}.
        num_sampling_steps: Number of Euler steps for sampling.
        num_likelihood_steps: Number of Euler steps for log-density integration.
        use_time_step: Whether the net receives a normalized trajectory-time scalar.
        trajectory_length: Used to normalize trajectory time.
        learning_rate: Optimizer LR.
        weight_decay: Optimizer weight decay.
        **arch_kwargs: Forwarded to `create_local_velocity_network`.
    """

    def __init__(
        self,
        radius: int,
        architecture: str = "local_mlp",
        state_dim: Optional[int] = None,
        use_observations: bool = True,
        obs_components: Optional[list] = None,
        predict_delta: bool = False,
        num_sampling_steps: int = 10,
        num_likelihood_steps: int = 10,
        use_time_step: bool = False,
        trajectory_length: int = 1000,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        obs_dropout: float = 0.0,
        **arch_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.radius = radius
        self.window_size = 2 * radius + 1
        self.architecture = architecture
        self.state_dim_default = state_dim
        self.use_observations = use_observations
        self.obs_components = list(obs_components) if obs_components is not None else None
        self.predict_delta = predict_delta
        self.num_sampling_steps = num_sampling_steps
        self.num_likelihood_steps = num_likelihood_steps
        self.use_time_step = use_time_step
        self.trajectory_length = trajectory_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.obs_dropout = obs_dropout

        self.local_net = create_local_velocity_network(
            architecture=architecture,
            radius=radius,
            use_obs=use_observations,
            use_time_step=use_time_step,
            **arch_kwargs,
        )

        self.training_step_outputs: list = []
        self.validation_step_outputs: list = []

    # ------------------------------------------------------------------
    # Trajectory-time normalization helper
    # ------------------------------------------------------------------
    def _normalize_t(
        self, t: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if not self.use_time_step:
            return None
        if t is None:
            raise ValueError(
                "LocalizedRFProposal configured with use_time_step=True but `t` is None."
            )
        if t.dim() == 0:
            t = t.expand(batch_size)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return (t.float() / float(self.trajectory_length)).to(self.device)

    # ------------------------------------------------------------------
    # Patch-level RF loss
    # ------------------------------------------------------------------
    def compute_rf_loss(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """
        RF loss evaluated on patches.

        Expects batch keys:
            x_prev_window   : (B, 2r+1)
            x_curr_window   : (B, 2r+1)
            target_j        : (B,) x_curr at centre
            obs_window      : (B, 2r+1), optional
            obs_mask_window : (B, 2r+1), optional
            time_idx        : (B,), optional
        """
        x_prev_w = batch["x_prev_window"]
        x_curr_w = batch["x_curr_window"]
        target_center = batch["target_j"].unsqueeze(-1)  # (B, 1)
        x_prev_center = batch["x_prev_center"].unsqueeze(-1)  # (B, 1)
        obs_w = batch.get("obs_window", None)
        mask_w = batch.get("obs_mask_window", None)
        t = batch.get("time_idx", None)

        B = x_prev_w.shape[0]
        device = x_prev_w.device

        # Determine target over the window (so that z interpolates toward
        # the true x_curr_window within the receptive field).
        if self.predict_delta:
            target_full_window = x_curr_w - x_prev_w
        else:
            target_full_window = x_curr_w

        # Sample s ~ U(0,1) and z ~ N(0, I) over the full window.
        s = torch.rand(B, 1, device=device)
        z_window = torch.randn_like(target_full_window)

        # Interpolated flow state z_window(s) = (1-s) z + s target over the window.
        z_s_window = (1.0 - s) * z_window + s * target_full_window

        # Target velocity at the CENTRE only:
        # if predict_delta: target_center_delta - z_center
        # else:             x_curr_center      - z_center
        z_center = z_window[:, self.radius].unsqueeze(-1)  # (B, 1)
        if self.predict_delta:
            target_center_delta = target_center - x_prev_center  # (B, 1)
            target_v_center = target_center_delta - z_center
        else:
            target_v_center = target_center - z_center

        t_norm = self._normalize_t(t, batch_size=B)

        # Observation dropout (training only): per-sample, with prob `obs_dropout`,
        # null the observation window (zero values + zero mask). This teaches the
        # model an unconditional branch analogous to classifier-free guidance.
        if (
            self.use_observations
            and self.training
            and self.obs_dropout > 0
            and obs_w is not None
        ):
            drop = (torch.rand(B, 1, device=device) < self.obs_dropout).to(obs_w.dtype)
            keep = 1.0 - drop
            obs_w = obs_w * keep
            if mask_w is not None:
                mask_w = mask_w * keep
            else:
                mask_w = keep.expand_as(obs_w)

        # Forward local net
        pred_v_center = self.local_net(
            z_s_window, x_prev_w,
            obs_w if self.use_observations else None,
            mask_w if self.use_observations else None,
            s, t_norm,
        )  # (B, 1)

        loss = torch.mean((pred_v_center - target_v_center) ** 2)

        metrics = {
            "loss": loss.item(),
            "velocity_norm": torch.mean(torch.abs(pred_v_center)).item(),
            "target_velocity_norm": torch.mean(torch.abs(target_v_center)).item(),
        }
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.compute_rf_loss(batch)
        self.log("train_loss", metrics["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_velocity_norm", metrics["velocity_norm"], on_step=False, on_epoch=True)
        self.training_step_outputs.append(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.compute_rf_loss(batch)
        self.log("val_loss", metrics["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_velocity_norm", metrics["velocity_norm"], on_step=False, on_epoch=True)
        self.validation_step_outputs.append(metrics)
        return loss

    def on_train_epoch_end(self):
        if len(self.training_step_outputs) > 0:
            avg = float(np.mean([m["loss"] for m in self.training_step_outputs]))
            logger.info(f"Epoch {self.current_epoch}: train_loss = {avg:.6f}")
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:
            avg = float(np.mean([m["loss"] for m in self.validation_step_outputs]))
            logger.info(f"Epoch {self.current_epoch}: val_loss = {avg:.6f}")
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=20
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }

    # ------------------------------------------------------------------
    # Helpers: build dense observation + mask from (possibly sparse) y
    # ------------------------------------------------------------------
    def _dense_obs(
        self, y: Optional[torch.Tensor], n_x: int, batch_size: int, device, dtype
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_observations or y is None:
            return None, None
        if y.shape[-1] == n_x and self.obs_components is None:
            # Already dense
            return y, torch.ones_like(y)
        if self.obs_components is None:
            raise ValueError(
                "LocalizedRFProposal received sparse observations but obs_components is None; "
                "cannot scatter to dense grid."
            )
        return dense_obs_from_components(y, self.obs_components, n_x)

    # ------------------------------------------------------------------
    # Patch-based velocity field at all sites in parallel
    # ------------------------------------------------------------------
    def _apply_local_net_all_sites(
        self,
        z: torch.Tensor,               # (B, N_x) current flow state
        s_scalar: torch.Tensor,        # scalar 0-d tensor
        x_prev: torch.Tensor,          # (B, N_x)
        obs_full: Optional[torch.Tensor],   # (B, N_x) dense obs or None
        obs_mask: Optional[torch.Tensor],   # (B, N_x) mask or None
        t_norm: Optional[torch.Tensor],     # (B, 1) or None
        window_spec: WindowSpec,
    ) -> torch.Tensor:
        """Return full-state velocity field (B, N_x) by applying local net at every site."""
        B, N_x = z.shape
        r = window_spec.radius
        W = window_spec.window_size

        z_win = extract_patches(z, window_spec)           # (B, N_x, W)
        xp_win = extract_patches(x_prev, window_spec)     # (B, N_x, W)
        if obs_full is not None:
            obs_win = extract_patches(obs_full, window_spec)
            mask_win = extract_patches(obs_mask, window_spec)
        else:
            obs_win = None
            mask_win = None

        # Flatten (B, N_x, W) -> (B*N_x, W) for a single batched forward pass.
        BNx = B * N_x
        z_flat = z_win.reshape(BNx, W)
        xp_flat = xp_win.reshape(BNx, W)
        obs_flat = obs_win.reshape(BNx, W) if obs_win is not None else None
        mask_flat = mask_win.reshape(BNx, W) if mask_win is not None else None

        s_flat = s_scalar.reshape(1, 1).expand(BNx, 1).to(z.dtype)
        if t_norm is not None:
            # Broadcast the per-batch trajectory time to every site
            t_flat = t_norm.unsqueeze(1).expand(B, N_x, 1).reshape(BNx, 1)
        else:
            t_flat = None

        v_flat = self.local_net(z_flat, xp_flat, obs_flat, mask_flat, s_flat, t_flat)
        # (B*N_x, 1) -> (B, N_x)
        return v_flat.reshape(B, N_x)

    def _default_window_spec(self, n_x: int) -> WindowSpec:
        return WindowSpec(radius=self.radius, stride=1, periodic=True)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        observation_fn=None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_{t-1}, y_t) with Euler integration.

        Args:
            x_prev: (N_x,) or (B, N_x).
            y_curr: Observation. Dense (B, N_x) or sparse (B, obs_dim) with
                obs_components from __init__.
            t: trajectory time step (unnormalized).
        """
        was_1d = x_prev.dim() == 1
        if was_1d:
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        B, N_x = x_prev.shape
        device = x_prev.device
        dtype = x_prev.dtype
        spec = self._default_window_spec(N_x)

        obs_full, obs_mask = self._dense_obs(y_curr, N_x, B, device, dtype)
        t_norm = self._normalize_t(t, batch_size=B)

        # Start at z ~ N(0, I)
        z = torch.randn(B, N_x, device=device, dtype=dtype)

        grid = torch.linspace(0.0, 1.0, self.num_sampling_steps + 1, device=device, dtype=dtype)
        for i in range(self.num_sampling_steps):
            s = grid[i]
            ds = grid[i + 1] - grid[i]
            v = self._apply_local_net_all_sites(
                z, s, x_prev, obs_full, obs_mask, t_norm, spec
            )
            z = z + v * ds

        if self.predict_delta:
            z = x_prev + z

        if was_1d:
            z = z.squeeze(0)
        return z

    # ------------------------------------------------------------------
    # Exact block diagonal of Jacobian via the "single-backward" trick
    # ------------------------------------------------------------------
    def _velocity_and_diag_jacobian(
        self,
        z: torch.Tensor,               # (B, N_x)
        s_scalar: torch.Tensor,        # scalar
        x_prev: torch.Tensor,
        obs_full: Optional[torch.Tensor],
        obs_mask: Optional[torch.Tensor],
        t_norm: Optional[torch.Tensor],
        spec: WindowSpec,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute v (B, N_x) and diag(d v / d z) (B, N_x) in ONE backward pass.

        Exploits: v_j = f(z_{j-r:j+r}, ...). Build z_window_leaf with
        requires_grad=True from the extracted patches (detached from z), forward
        through the net, then autograd.grad(sum(v)) over the LEAF gives per-patch
        gradient (B*N_x, 2r+1). The [:, radius] slice is d v_j / d z_j at the
        centre — exactly the diagonal entry we need.

        Correctness: independent patches → independent outputs → backprop over
        the stacked leaf does not mix sites.
        """
        B, N_x = z.shape
        r = spec.radius
        W = spec.window_size

        with torch.enable_grad():
            # Extract windows and make a new leaf tensor so the per-patch grads do
            # not smear back through the unfold (which would sum overlapping sites).
            z_win = extract_patches(z.detach(), spec)    # (B, N_x, W)
            z_leaf = z_win.reshape(B * N_x, W).clone().requires_grad_(True)

            xp_win = extract_patches(x_prev, spec).reshape(B * N_x, W)
            if obs_full is not None:
                obs_win = extract_patches(obs_full, spec).reshape(B * N_x, W)
                mask_win = extract_patches(obs_mask, spec).reshape(B * N_x, W)
            else:
                obs_win = None
                mask_win = None

            s_flat = s_scalar.reshape(1, 1).expand(B * N_x, 1).to(z.dtype)
            if t_norm is not None:
                t_flat = t_norm.unsqueeze(1).expand(B, N_x, 1).reshape(B * N_x, 1)
            else:
                t_flat = None

            v_flat = self.local_net(z_leaf, xp_win, obs_win, mask_win, s_flat, t_flat)  # (B*N_x, 1)

            # Single backward over the stacked leaf.
            grads = torch.autograd.grad(
                v_flat.sum(), z_leaf, create_graph=False, retain_graph=False
            )[0]  # (B*N_x, W)

        v = v_flat.detach().reshape(B, N_x)
        diag = grads.detach()[:, r].reshape(B, N_x)
        return v, diag

    # ------------------------------------------------------------------
    # Sample and per-dimension log-density
    # ------------------------------------------------------------------
    def sample_and_per_dim_log_prob(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t and return per-dimension log-density ell (B, N_x).

        The full-state log-density is `sum_j ell_j` (verified against
        `RFProposal.log_prob` in the unit tests). Each ell_j accumulates:
            ell_j(0)  = -0.5 * (z_j^2 + log(2 pi))
            ell_j    -= (d v_j / d z_j) * ds
        during forward Euler integration.
        """
        was_1d = x_prev.dim() == 1
        if was_1d:
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        B, N_x = x_prev.shape
        device = x_prev.device
        dtype = x_prev.dtype
        spec = self._default_window_spec(N_x)

        obs_full, obs_mask = self._dense_obs(y_curr, N_x, B, device, dtype)
        t_norm = self._normalize_t(t, batch_size=B)

        # Start at z ~ N(0, I)
        z = torch.randn(B, N_x, device=device, dtype=dtype)

        # Per-dim base log-density
        ell = -0.5 * z.pow(2) - 0.5 * math.log(2.0 * math.pi)  # (B, N_x)

        grid = torch.linspace(0.0, 1.0, self.num_sampling_steps + 1, device=device, dtype=dtype)
        for i in range(self.num_sampling_steps):
            s = grid[i]
            ds = grid[i + 1] - grid[i]
            v, diag = self._velocity_and_diag_jacobian(
                z, s, x_prev, obs_full, obs_mask, t_norm, spec
            )
            z = z + v * ds
            ell = ell - diag * ds

        if self.predict_delta:
            z = x_prev + z

        if was_1d:
            z = z.squeeze(0)
            ell = ell.squeeze(0)
        return z, ell

    # ------------------------------------------------------------------
    # Global log_prob (scalar) — for sum-consistency tests and compatibility
    # ------------------------------------------------------------------
    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        use_exact_trace: bool = True,
        trace_estimator: str = "gaussian",
        num_trace_probes: int = 1,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Global log-density via backward Euler + exact block diagonal trace.

        Because the Jacobian of the local velocity field is banded with
        bandwidth r and we only need the trace (diagonal sum), the single-
        backward diag trick over patches suffices and is O(N_x * r) per step.
        """
        was_1d = x_curr.dim() == 1
        if was_1d:
            x_curr = x_curr.unsqueeze(0)
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        B, N_x = x_curr.shape
        device = x_curr.device
        dtype = x_curr.dtype
        spec = self._default_window_spec(N_x)

        obs_full, obs_mask = self._dense_obs(y_curr, N_x, B, device, dtype)
        t_norm = self._normalize_t(t, batch_size=B)

        # Backward from s=1 to s=0 starting from x_curr (or delta)
        if self.predict_delta:
            x = (x_curr - x_prev).clone()
        else:
            x = x_curr.clone()

        log_prob_correction = torch.zeros(B, device=device, dtype=dtype)
        grid = torch.linspace(0.0, 1.0, self.num_likelihood_steps + 1, device=device, dtype=dtype)

        for i in range(self.num_likelihood_steps - 1, -1, -1):
            s = grid[i + 1]
            ds = grid[i + 1] - grid[i]
            if use_exact_trace:
                v, diag = self._velocity_and_diag_jacobian(
                    x, s, x_prev, obs_full, obs_mask, t_norm, spec
                )
                divergence = diag.sum(dim=-1)
            else:
                # Hutchinson fallback over the stacked patches — mostly for
                # sanity checks; exact-trace is cheap for local nets.
                v, divergence = self._velocity_and_hutchinson_trace(
                    x, s, x_prev, obs_full, obs_mask, t_norm, spec,
                    num_probes=num_trace_probes, estimator=trace_estimator,
                )
            x = x - v * ds
            log_prob_correction = log_prob_correction + divergence * ds

        log_prob_base = -0.5 * torch.sum(x ** 2, dim=-1) - 0.5 * N_x * math.log(2 * math.pi)
        lp = log_prob_base - log_prob_correction
        if was_1d:
            lp = lp.squeeze(0)
        return lp

    def _velocity_and_hutchinson_trace(
        self, z, s_scalar, x_prev, obs_full, obs_mask, t_norm, spec,
        num_probes: int = 1, estimator: str = "gaussian",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hutchinson trace of the full-state Jacobian via randomized probes over
        the stacked patches. Each probe accumulates eps^T J eps where J is the
        block-diagonal patch Jacobian; summed over patches it matches the global
        trace because the full Jacobian is banded / block diagonal in patches.
        """
        B, N_x = z.shape
        r = spec.radius
        W = spec.window_size

        with torch.enable_grad():
            z_win = extract_patches(z.detach(), spec)
            z_leaf = z_win.reshape(B * N_x, W).clone().requires_grad_(True)
            xp_win = extract_patches(x_prev, spec).reshape(B * N_x, W)
            if obs_full is not None:
                obs_win = extract_patches(obs_full, spec).reshape(B * N_x, W)
                mask_win = extract_patches(obs_mask, spec).reshape(B * N_x, W)
            else:
                obs_win, mask_win = None, None
            s_flat = s_scalar.reshape(1, 1).expand(B * N_x, 1).to(z.dtype)
            t_flat = (
                t_norm.unsqueeze(1).expand(B, N_x, 1).reshape(B * N_x, 1)
                if t_norm is not None else None
            )
            v_flat = self.local_net(z_leaf, xp_win, obs_win, mask_win, s_flat, t_flat)  # (B*N_x, 1)

            div_sum = torch.zeros(B * N_x, device=z.device, dtype=z.dtype)
            for p in range(num_probes):
                if estimator == "rademacher":
                    eps = torch.randint_like(z_leaf, low=0, high=2).float() * 2 - 1
                else:
                    eps = torch.randn_like(z_leaf)
                # Only the centre output exists, so we use grad_outputs on v_flat
                # and then use eps at the centre for the probe.
                eps_center = eps[:, r:r + 1]
                vjp = torch.autograd.grad(
                    v_flat, z_leaf, grad_outputs=eps_center,
                    create_graph=False, retain_graph=(p < num_probes - 1),
                )[0]
                div_sum = div_sum + (vjp * eps).sum(dim=-1)
            diag_per_patch = (div_sum / num_probes).reshape(B, N_x)

        v = v_flat.detach().reshape(B, N_x)
        divergence = diag_per_patch.sum(dim=-1)
        return v, divergence

    # ------------------------------------------------------------------
    # Compatibility: (x, log_prob) API similar to RFProposal
    # ------------------------------------------------------------------
    def sample_and_log_prob(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample and return global scalar log-prob (sum of per-dim ell)."""
        x, ell = self.sample_and_per_dim_log_prob(x_prev, y_curr, t=t)
        return x, ell.sum(dim=-1) if ell.dim() > 1 else ell
