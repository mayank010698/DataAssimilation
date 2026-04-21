"""Paige-Wood inference-network proposal (Lightning module).

This module implements the learned proposal described in
Paige & Wood, *Inference Networks for Sequential Monte Carlo in Graphical
Models*, ICML 2016, specialised to a state-space filtering setting.
Given samples ``(x_{t-1}, x_t, y_t)`` from the generative model's joint,
we train a conditional density estimator

    q_eta(x_t | x_{t-1}, y_t)

by plain maximum-likelihood (equivalently: forward-KL under the model's
joint). No particle filter in the inner loop; no REINFORCE; no ODE
integration at sampling/log-prob time. The design is intentionally the
cheapest-possible learned proposal: a shared feature backbone + a CDE
head whose density family we can sweep.

Relationship to other proposals in this codebase
-------------------------------------------------
* :class:`proposals.rectified_flow.RFProposal` — continuous normalising
  flow trained with flow matching. High expressivity; expensive sampling
  and likelihood.
* :class:`proposals.nasmc.GaussianProposal` — diagonal Gaussian trained
  in one of three modes: (a) supervised MLE ("gaussian_mle" preset,
  which is essentially the single-Gaussian special case of what this
  module trains), (b) NASMC weighted log-density, (c) MLE warmstart →
  NASMC refine. Our IN module subsumes (a) and extends it to richer
  density families (per-dim MoG, joint MoG, RNADE). It deliberately
  does NOT include the NASMC SMC-in-the-loop refinement — that lives
  in :class:`GaussianProposal` and is a different method.

See ``plan_in.md`` for the full design discussion; the CDE heads live in
:mod:`proposals.architectures.cde_heads` and the feature backbones in
:mod:`proposals.architectures.feature_backbones`.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn

try:
    from .architectures import create_feature_backbone, create_cde_head
    from .architectures.cde_heads import BaseCDEHead
except ImportError:  # pragma: no cover
    from architectures import create_feature_backbone, create_cde_head  # type: ignore
    from architectures.cde_heads import BaseCDEHead  # type: ignore

logger = logging.getLogger(__name__)


class InferenceNetworkProposal(pl.LightningModule):
    """Paige-Wood inference-network proposal as a Lightning module.

    Trains ``q_eta(x_t | x_{t-1}, y_t)`` by minimising the negative
    log-likelihood of ``x_t`` under the conditional density. The only
    thing the training loop needs is a stream of
    ``(x_prev, x_curr, y_curr, time_idx)`` triples — this is exactly
    what :class:`~proposals.rf_dataset.RFTransitionDataset` already
    provides.

    Everything is done in **scaled** state / observation space, matching
    :class:`~proposals.rectified_flow.RFProposal` and
    :class:`~proposals.nasmc.GaussianProposal`. The filter-side wrapper
    in :mod:`models.proposals` handles pre/post-processing to physical
    space so the particle filter's interface is unchanged.

    The predicted density can optionally be parameterised as an
    increment ``mu_raw = q_eta(x_t - x_{t-1} | ...)`` via
    ``predict_delta=True``. This helps when the one-step change is small
    relative to ``x_{t-1}`` (the same trick RF/NASMC use), because it
    gives the network a useful residual structure for free.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int = 0,
        obs_indices: Optional[List[int]] = None,
        architecture: str = "mlp",
        cde_head: str = "joint_mog",
        num_mixture_components: int = 8,
        hidden_dim: int = 128,
        depth: int = 4,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
        feature_dim: int = 128,
        use_time_step: bool = False,
        trajectory_length: int = 1000,
        predict_delta: bool = True,
        min_sigma: float = 1e-3,
        init_log_sigma: float = -1.0,
        zero_init_output: bool = True,
        rnade_hidden_size: int = 128,
        rnade_num_hidden_layers: int = 2,
        cond_dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        lr_scheduler_patience: int = 20,
    ) -> None:
        """
        Args:
            state_dim: Dimension ``D`` of the state.
            obs_dim: Dimension of ``y_t`` (0 = unconditional proposal,
                i.e. a learned one-step prior).
            obs_indices: Indices of observed state components. If ``None``
                and ``obs_dim > 0``, the backbone assumes a dense first-
                ``obs_dim``-coordinates layout (see the velocity nets).
            architecture: ``'mlp'`` or ``'resnet1d'``.
            cde_head: Density family. One of ``'gaussian'``, ``'mdn_k'``,
                ``'joint_mog'`` (default, recommended), ``'rnade'``.
            num_mixture_components: ``K`` for mixture heads. Ignored for
                the single-Gaussian head.
            hidden_dim, depth, channels, num_blocks, kernel_size,
            time_embed_dim, dropout, feature_dim: Backbone kwargs. MLP
                backbones use (hidden_dim, depth, time_embed_dim,
                dropout); ResNet1D backbones use (channels, num_blocks,
                kernel_size, time_embed_dim, dropout, feature_dim).
            use_time_step: Condition on normalised trajectory time.
            trajectory_length: Used to normalise ``time_idx`` into
                ``[0, 1]`` when ``use_time_step=True``.
            predict_delta: Predict ``x_t - x_{t-1}`` instead of ``x_t``.
                Turned on by default because the one-step change is
                typically small in scaled space.
            min_sigma, init_log_sigma, zero_init_output: Numerical knobs
                forwarded to the CDE head.
            rnade_hidden_size, rnade_num_hidden_layers: Only used when
                ``cde_head='rnade'``.
            cond_dropout: With this probability, replace ``y_t`` with zero
                and ``mask`` with zero during training. Gives us a single
                network that can handle both observed and unobserved
                steps (the BPF falls back to the transition prior at
                unobserved steps today; this trick lets IN cover them
                too without retraining).
            learning_rate, weight_decay, lr_scheduler_patience: Optimiser
                knobs. Matches the RF/NASMC defaults.
        """
        super().__init__()
        self.save_hyperparameters()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.obs_indices = list(obs_indices) if obs_indices is not None else None
        self.use_time_step = use_time_step
        self.trajectory_length = int(trajectory_length)
        self.predict_delta = bool(predict_delta)
        self.cond_dropout = float(cond_dropout)

        self.backbone = create_feature_backbone(
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
            feature_dim=feature_dim,
            obs_indices=obs_indices,
        )
        self.head: BaseCDEHead = create_cde_head(
            kind=cde_head,
            feature_dim=self.backbone.feature_dim,
            state_dim=state_dim,
            num_mixture_components=num_mixture_components,
            min_sigma=min_sigma,
            init_log_sigma=init_log_sigma,
            zero_init_output=zero_init_output,
            rnade_hidden_size=rnade_hidden_size,
            rnade_num_hidden_layers=rnade_num_hidden_layers,
        )

        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.lr_scheduler_patience = int(lr_scheduler_patience)
        self.cde_head_kind = str(cde_head)

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    def _normalize_trajectory_time(
        self,
        t: Optional[torch.Tensor],
        batch_size: int,
        caller_name: str,
    ) -> Optional[torch.Tensor]:
        """Map an integer/float trajectory time to a normalised ``[0,1]``
        tensor with shape ``(B, 1)``. Returns ``None`` when the backbone
        is not time-conditioned, or raises if the caller forgot ``t``.
        """
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
                    f"{caller_name}: expected t with second dimension 1, "
                    f"got {tuple(t.shape)}"
                )
        else:
            raise ValueError(
                f"{caller_name}: expected scalar / 1D / 2D t tensor, got {t.dim()}D"
            )
        return t.float() / float(self.trajectory_length)

    def _features(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        t: Optional[torch.Tensor],
        drop_obs: bool = False,
    ) -> torch.Tensor:
        """Run the backbone with optional conditioning dropout.

        ``drop_obs=True`` zeros out ``y_curr`` (so the backbone sees an
        all-zero observation with an all-zero mask). This is used by
        ``cond_dropout`` during training and by the filter wrapper when
        the current step has no observation.
        """
        y = None if drop_obs else y_curr
        t_normalized = self._normalize_trajectory_time(
            t=t, batch_size=x_prev.shape[0], caller_name="_features",
        )
        return self.backbone(x_prev, y, t_normalized)

    def _params(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        t: Optional[torch.Tensor],
        drop_obs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        h = self._features(x_prev, y_curr, t, drop_obs=drop_obs)
        return self.head(h)

    # ------------------------------------------------------------------
    # Public API: ``sample`` and ``log_prob`` in scaled space.
    # ------------------------------------------------------------------

    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Draw a single sample ``x_t ~ q_eta(x_t | x_{t-1}, y_t)`` in
        scaled state space. ``dt`` is accepted for ABC compatibility and
        ignored.
        """
        was_1d = x_prev.dim() == 1
        if was_1d:
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)

        params = self._params(x_prev, y_curr, t, drop_obs=False)
        delta = self.head.sample(params, n=1)
        x_curr = x_prev + delta if self.predict_delta else delta

        if was_1d:
            x_curr = x_curr.squeeze(0)
        return x_curr

    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate ``log q_eta(x_curr | x_prev, y_curr)`` in scaled
        state space. Returns a scalar or ``(B,)`` tensor.
        """
        was_1d = x_curr.dim() == 1
        if was_1d:
            x_curr = x_curr.unsqueeze(0)
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)

        params = self._params(x_prev, y_curr, t, drop_obs=False)
        target = x_curr - x_prev if self.predict_delta else x_curr
        lp = self.head.log_prob(target, params)

        if was_1d:
            lp = lp.squeeze(0)
        return lp

    def predictive_mean(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Closed-form conditional mean, useful for validation RMSE.

        For mixture heads this is the mixture-weighted mean; for the
        single Gaussian it is the predicted ``mu``. The value is returned
        in scaled state space, matching the ``sample`` / ``log_prob``
        convention.
        """
        was_1d = x_prev.dim() == 1
        if was_1d:
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)

        params = self._params(x_prev, y_curr, t, drop_obs=False)
        delta_mean = self.head.mean(params)
        x_mean = x_prev + delta_mean if self.predict_delta else delta_mean

        if was_1d:
            x_mean = x_mean.squeeze(0)
        return x_mean

    # ------------------------------------------------------------------
    # Training / validation: negative log-likelihood under the CDE.
    # ------------------------------------------------------------------

    def _nll(
        self,
        batch: Dict[str, torch.Tensor],
        apply_cond_dropout: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_prev = batch["x_prev"]
        x_curr = batch["x_curr"]
        y_curr = batch.get("y_curr", None)
        t = batch.get("time_idx", None)

        drop_obs = False
        if apply_cond_dropout and self.cond_dropout > 0.0 and self.obs_dim > 0:
            drop_obs = bool(torch.rand((), device=self.device) < self.cond_dropout)

        params = self._params(x_prev, y_curr, t, drop_obs=drop_obs)
        target = x_curr - x_prev if self.predict_delta else x_curr
        log_prob = self.head.log_prob(target, params)
        nll = -log_prob.mean()

        metrics: Dict[str, float] = {
            "nll": float(nll.detach().item()),
            "log_prob_mean": float(log_prob.detach().mean().item()),
        }
        if self.cde_head_kind in ("joint_mog", "mdn_k") and "log_w" in params:
            log_w = params["log_w"].detach()
            w = log_w.exp()
            mixture_entropy = -(w * log_w).sum(dim=-1).mean()
            metrics["mixture_entropy"] = float(mixture_entropy.item())
        if "sigma" in params:
            metrics["avg_sigma"] = float(params["sigma"].detach().mean().item())

        with torch.no_grad():
            mean_pred_delta = self.head.mean(params)
            x_mean = x_prev + mean_pred_delta if self.predict_delta else mean_pred_delta
            metrics["mean_l2"] = float(
                torch.linalg.vector_norm(x_curr - x_mean, dim=-1).mean().item()
            )
        return nll, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self._nll(batch, apply_cond_dropout=True)
        self.log("train_nll", metrics["nll"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_mean_l2", metrics["mean_l2"], on_epoch=True)
        if "mixture_entropy" in metrics:
            self.log("train_mixture_entropy", metrics["mixture_entropy"], on_epoch=True)
        if "avg_sigma" in metrics:
            self.log("train_avg_sigma", metrics["avg_sigma"], on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._nll(batch, apply_cond_dropout=False)
        self.log("val_loss", metrics["nll"], prog_bar=True, on_epoch=True)
        self.log("val_nll", metrics["nll"], prog_bar=True, on_epoch=True)
        self.log("val_mean_l2", metrics["mean_l2"], on_epoch=True, prog_bar=True)
        if "mixture_entropy" in metrics:
            self.log("val_mixture_entropy", metrics["mixture_entropy"], on_epoch=True)
        if "avg_sigma" in metrics:
            self.log("val_avg_sigma", metrics["avg_sigma"], on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
