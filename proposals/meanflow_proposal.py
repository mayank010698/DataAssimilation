"""
MeanFlow + F2D2 Proposal Distribution (Stages 2 & 3)

MeanFlowProposal is a PyTorch Lightning module that wraps a shortcut-style
velocity network and implements MeanFlow training plus optional F2D2
divergence distillation:

  Stage 2  (training_stage='mf'):
    - JVP-based MeanFlow self-consistency loss (no teacher needed)
    - (s, r) logit-normal time sampling with r > s
    - Adaptive per-sample loss reweighting

  Stage 3  (training_stage='mf_f2d2'):
    - Teacher velocity + divergence flow matching (same as Shortcut F2D2)
    - Optional MeanFlow JVP consistency regulariser (mf_consistency_weight)

Inference:
  sample()   — n_steps forward Euler steps (default 1; one NFE)
  log_prob() — n_steps backward Euler with divergence head (default 4 NFEs)

Uses RF convention throughout: s=0 is noise, s=1 is data.  The network
architecture is identical to the shortcut network (create_shortcut_network).
"""

import sys
import copy
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

try:
    from .architectures import create_shortcut_network
    from .architectures import ShortcutMLPVelocityNetwork, ShortcutResNet1DVelocityNetwork
    from .rectified_flow import RFProposal
except ImportError:
    from architectures import create_shortcut_network
    from architectures import ShortcutMLPVelocityNetwork, ShortcutResNet1DVelocityNetwork
    from rectified_flow import RFProposal


# ---------------------------------------------------------------------------
# EMA Callback (targets self.meanflow_net)
# ---------------------------------------------------------------------------

class MeanFlowEMACallback(Callback):
    """EMA over meanflow_net weights; swapped in for validation and saved at end."""

    def __init__(self, ema_beta: float = 0.9999):
        super().__init__()
        self.ema_beta = ema_beta
        self._ema: dict = {}
        self._backup: dict = {}

    def _net(self, pl_module):
        return pl_module.meanflow_net

    def on_fit_start(self, trainer, pl_module):
        net = self._net(pl_module)
        self._ema = {n: p.data.detach().clone() for n, p in net.named_parameters()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        net = self._net(pl_module)
        beta = self.ema_beta
        with torch.no_grad():
            for name, param in net.named_parameters():
                if name in self._ema:
                    self._ema[name].mul_(beta).add_(param.data, alpha=1.0 - beta)

    def on_validation_start(self, trainer, pl_module):
        if not self._ema:
            return
        net = self._net(pl_module)
        self._backup = {n: p.data.detach().clone() for n, p in net.named_parameters()}
        for name, param in net.named_parameters():
            if name in self._ema:
                param.data.copy_(self._ema[name])

    def on_validation_end(self, trainer, pl_module):
        if not self._backup:
            return
        net = self._net(pl_module)
        for name, param in net.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["meanflow_ema_params"] = {k: v.cpu() for k, v in self._ema.items()}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "meanflow_ema_params" in checkpoint:
            device = next(self._net(pl_module).parameters()).device
            self._ema = {
                k: v.to(device)
                for k, v in checkpoint["meanflow_ema_params"].items()
            }

    def on_train_end(self, trainer, pl_module):
        if not self._ema:
            return
        ckpt_dir = Path(trainer.checkpoint_callback.dirpath)
        ema_path = str(ckpt_dir / "meanflow_ema_weights.ckpt")
        net = self._net(pl_module)
        backup = {n: p.data.detach().clone() for n, p in net.named_parameters()}
        for name, param in net.named_parameters():
            if name in self._ema:
                param.data.copy_(self._ema[name])
        trainer.save_checkpoint(ema_path)
        for name, param in net.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
        logging.getLogger(__name__).info(f"MeanFlow EMA checkpoint: {ema_path}")


# ---------------------------------------------------------------------------
# Time sampling utilities (ported from CODEBASE_py-meanflow)
# ---------------------------------------------------------------------------

def _logit_normal_sample(
    P_mean: float, P_std: float, n: int, device: torch.device,
) -> torch.Tensor:
    rnd = torch.randn(n, device=device)
    return torch.sigmoid(rnd * P_std + P_mean).clamp(0.0, 1.0)


def _sample_sr_v0(
    P_mean_s: float, P_std_s: float,
    P_mean_r: float, P_std_r: float,
    ratio: float, n: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paper version: sort then collapse with prob 1-ratio."""
    s_raw = _logit_normal_sample(P_mean_s, P_std_s, n, device)
    r_raw = _logit_normal_sample(P_mean_r, P_std_r, n, device)
    s = torch.minimum(s_raw, r_raw)
    r = torch.maximum(s_raw, r_raw)
    mask = torch.rand(n, device=device) < (1 - ratio)
    r = torch.where(mask, s, r)
    return s, r


def _sample_sr_v1(
    P_mean_s: float, P_std_s: float,
    P_mean_r: float, P_std_r: float,
    ratio: float, n: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Improved version: collapse then clamp."""
    s_raw = _logit_normal_sample(P_mean_s, P_std_s, n, device)
    r_raw = _logit_normal_sample(P_mean_r, P_std_r, n, device)
    mask = torch.rand(n, device=device) < (1 - ratio)
    r_raw = torch.where(mask, s_raw, r_raw)
    r = torch.maximum(s_raw, r_raw)
    s = torch.minimum(s_raw, r_raw)
    return s, r


# ---------------------------------------------------------------------------
# MeanFlowProposal
# ---------------------------------------------------------------------------

class MeanFlowProposal(pl.LightningModule):
    """
    MeanFlow + F2D2 proposal distribution.

    Constructor arguments that are new compared to ShortcutProposal:
        tr_sampler:       'v0' or 'v1' for (s, r) sampling strategy.
        P_mean_s / P_std_s / P_mean_r / P_std_r: logit-normal params.
        ratio:            Probability that s != r (non-degenerate interval).
        norm_p / norm_eps: Adaptive loss reweighting exponent and epsilon.
        mf_consistency_weight: In Stage 3, weight for MeanFlow JVP regulariser.
    """

    def __init__(
        self,
        state_dim: int,
        teacher_ckpt_path: Optional[str] = None,
        training_stage: str = "mf",             # 'mf' | 'mf_f2d2'
        architecture: str = "mlp",
        hidden_dim: int = 128,
        depth: int = 4,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 3,
        time_embed_dim: int = 64,
        obs_dim: int = 0,
        obs_indices: Optional[list] = None,
        predict_delta: bool = False,
        use_time_step: bool = False,
        trajectory_length: int = 1000,
        cond_dropout: float = 0.0,
        debug_random_obs: bool = False,
        debug_random_prev_state: bool = False,
        # Prev-state corruption
        prev_state_corr_p0: float = 0.0,
        prev_state_corr_p_min: float = 0.05,
        prev_state_corr_total_steps: Optional[int] = None,
        prev_state_corr_sigma: float = 0.0,
        prev_state_corr_mask_ratio: float = 0.0,
        # Obs-consistency auxiliary loss
        obs_consistency_weight: float = 0.0,
        obs_nonlinearity: str = "arctan",
        state_scaler_mean: Optional[torch.Tensor] = None,
        state_scaler_std: Optional[torch.Tensor] = None,
        obs_scaler_mean: Optional[torch.Tensor] = None,
        obs_scaler_std: Optional[torch.Tensor] = None,
        # Training hyper-params
        learning_rate: float = 1e-4,
        lr_warmup_steps: int = 0,
        # MeanFlow-specific time sampling
        tr_sampler: str = "v1",
        P_mean_s: float = -0.6,
        P_std_s: float = 1.6,
        P_mean_r: float = -4.0,
        P_std_r: float = 1.6,
        ratio: float = 0.9,
        # MeanFlow adaptive loss weighting.
        # py-meanflow default is +0.75 (compresses outlier losses: L^0.25).
        # Negative values amplify outliers instead — opposite of the paper's intent.
        norm_p: float = 0.75,
        norm_eps: float = 1e-3,
        # F2D2 (Stage 3) params
        denoise_timesteps: int = 1024,
        teacher_div_estimator: str = "exact",
        div_scale: float = 1.0,
        mf_consistency_weight: float = 0.0,
        # When True (recommended for MeanFlow Stage 3): freeze backbone+velocity_head
        # and only train the divergence head.  This prevents Stage 3 from overwriting
        # the coarse-step structure (u(z,s,dt≈1)) learned in Stage 2.
        freeze_velocity: bool = False,
        # Inference
        n_sampling_steps: int = 1,
        n_likelihood_steps: int = 4,
        # Div head
        div_hidden_dim: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "state_scaler_mean", "state_scaler_std",
            "obs_scaler_mean", "obs_scaler_std",
        ])

        assert training_stage in ("mf", "mf_f2d2"), \
            f"training_stage must be 'mf' or 'mf_f2d2', got '{training_stage}'"
        assert tr_sampler in ("v0", "v1"), \
            f"tr_sampler must be 'v0' or 'v1', got '{tr_sampler}'"

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.obs_indices = obs_indices
        self.predict_delta = predict_delta
        self.use_time_step = use_time_step
        self.trajectory_length = trajectory_length
        self.training_stage = training_stage
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps
        self.cond_dropout = cond_dropout
        self.debug_random_obs = debug_random_obs
        self.debug_random_prev_state = debug_random_prev_state
        self.prev_state_corr_p0 = prev_state_corr_p0
        self.prev_state_corr_p_min = prev_state_corr_p_min
        self.prev_state_corr_total_steps = prev_state_corr_total_steps
        self.prev_state_corr_sigma = prev_state_corr_sigma
        self.prev_state_corr_mask_ratio = prev_state_corr_mask_ratio
        self.obs_consistency_weight = obs_consistency_weight
        self.obs_nonlinearity = obs_nonlinearity or "arctan"
        self._state_scaler_mean = state_scaler_mean
        self._state_scaler_std = state_scaler_std
        self._obs_scaler_mean = obs_scaler_mean
        self._obs_scaler_std = obs_scaler_std
        # MeanFlow time sampling
        self.tr_sampler = tr_sampler
        self.P_mean_s = P_mean_s
        self.P_std_s = P_std_s
        self.P_mean_r = P_mean_r
        self.P_std_r = P_std_r
        self.ratio = ratio
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        # F2D2
        self.denoise_timesteps = denoise_timesteps
        self.teacher_div_estimator = teacher_div_estimator
        self.div_scale = div_scale
        self.mf_consistency_weight = mf_consistency_weight
        self.freeze_velocity = freeze_velocity
        # Inference
        self.n_sampling_steps = n_sampling_steps
        self.n_likelihood_steps = n_likelihood_steps

        self.meanflow_net = create_shortcut_network(
            architecture=architecture,
            state_dim=state_dim,
            obs_dim=obs_dim,
            obs_indices=obs_indices,
            hidden_dim=hidden_dim,
            depth=depth,
            channels=channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            time_embed_dim=time_embed_dim,
            use_time_step=use_time_step,
            div_head_active=(training_stage == "mf_f2d2"),
            div_hidden_dim=div_hidden_dim,
        )

        self.teacher: Optional[RFProposal] = None
        self._train_outputs = []
        self._val_outputs = []

    # ---------- teacher loading (Stage 3 only) ----------

    def _load_teacher(self):
        ckpt_path = self.hparams.teacher_ckpt_path
        teacher = RFProposal.load_from_checkpoint(ckpt_path, map_location=self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.to(self.device)
        return teacher

    def on_fit_start(self):
        if self.training_stage == "mf_f2d2" and self.teacher is None:
            if self.hparams.teacher_ckpt_path is None:
                raise ValueError(
                    "teacher_ckpt_path is required for training_stage='mf_f2d2'"
                )
            self.teacher = self._load_teacher()
            logging.getLogger(__name__).info(
                f"Teacher loaded from {self.hparams.teacher_ckpt_path}"
            )

    # ---------- helpers (shared with ShortcutProposal / RFProposal) ----------

    def _normalize_trajectory_time(
        self, t: Optional[torch.Tensor], batch_size: int,
    ) -> Optional[torch.Tensor]:
        if not self.use_time_step:
            return None
        if t is None:
            raise ValueError("use_time_step=True but t was not provided.")
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=self.device, dtype=torch.float32)
        else:
            t = t.to(self.device)
        if t.dim() == 0:
            t = t.reshape(1, 1).expand(batch_size, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
            if t.shape[0] == 1 and batch_size != 1:
                t = t.expand(batch_size, 1)
        elif t.dim() == 2:
            if t.shape[0] == 1 and batch_size != 1:
                t = t.expand(batch_size, 1)
        return t.float() / self.trajectory_length

    def _get_corruption_prob(self) -> float:
        if self.prev_state_corr_total_steps is None or self.prev_state_corr_total_steps <= 0:
            return 0.0
        if self.prev_state_corr_p0 <= 0:
            return 0.0
        try:
            step = self.trainer.global_step
        except RuntimeError:
            step = 0
        progress = min(1.0, step / max(1, self.prev_state_corr_total_steps))
        return max(self.prev_state_corr_p_min, self.prev_state_corr_p0 * (1.0 - progress))

    def _corrupt_prev_state(self, x_prev: torch.Tensor):
        B = x_prev.shape[0]
        device = x_prev.device
        dtype = x_prev.dtype
        p = self._get_corruption_prob()
        if p <= 0 or (self.prev_state_corr_sigma <= 0 and self.prev_state_corr_mask_ratio <= 0):
            return x_prev, 0.0
        which = torch.rand(B, device=device) < p
        if not which.any():
            return x_prev, 0.0
        out = x_prev.clone()
        if self.prev_state_corr_sigma > 0:
            noise = self.prev_state_corr_sigma * torch.randn_like(x_prev)
            out = out + noise * which.unsqueeze(1).float()
        if self.prev_state_corr_mask_ratio > 0:
            d = self.state_dim
            n_mask = max(1, int(round(d * self.prev_state_corr_mask_ratio)))
            n_mask = min(d, n_mask)
            keys = torch.rand(B, d, device=device, dtype=dtype)
            idx = keys.argsort(dim=-1)[:, :n_mask]
            rows = torch.arange(B, device=device).unsqueeze(1).expand_as(idx)
            cur = out[rows, idx]
            out[rows, idx] = torch.where(
                which.unsqueeze(1).expand_as(idx),
                torch.zeros_like(cur),
                cur,
            )
        return out, which.float().mean().item()

    # ---------- (s, r) time sampling ----------

    def _sample_sr(self, B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (s, r) with s <= r, both in [0, 1]."""
        device = self.device
        if self.tr_sampler == "v0":
            return _sample_sr_v0(
                self.P_mean_s, self.P_std_s,
                self.P_mean_r, self.P_std_r,
                self.ratio, B, device,
            )
        else:
            return _sample_sr_v1(
                self.P_mean_s, self.P_std_s,
                self.P_mean_r, self.P_std_r,
                self.ratio, B, device,
            )

    # ---------- teacher divergence estimation (Stage 3) ----------

    def _teacher_divergence(
        self, v_teacher: torch.Tensor, x_t: torch.Tensor,
    ) -> torch.Tensor:
        assert x_t.requires_grad
        if self.teacher_div_estimator == "exact":
            B = x_t.shape[0]
            divergence = torch.zeros(B, device=self.device)
            for k in range(self.state_dim):
                grad_k = torch.autograd.grad(
                    v_teacher[:, k], x_t,
                    grad_outputs=torch.ones(B, device=self.device),
                    create_graph=False, retain_graph=True,
                )[0]
                divergence = divergence + grad_k[:, k]
            return divergence
        else:
            eps = torch.randint_like(x_t, low=0, high=2).float() * 2 - 1
            vjp = torch.autograd.grad(
                v_teacher, x_t, grad_outputs=eps,
                create_graph=False, retain_graph=False,
            )[0]
            return (vjp * eps).sum(dim=-1)

    # ---------- Stage 2: MeanFlow JVP loss ----------

    def compute_meanflow_loss(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        time_idx: Optional[torch.Tensor] = None,
        x_prev_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        MeanFlow JVP self-consistency loss.

        Uses RF convention: s=0 noise, s=1 data.
        z_s = (1-s)*noise + s*target, v = target - noise.
        """
        B = x_prev.shape[0]
        x_cond = x_prev if x_prev_cond is None else x_prev_cond
        t_normalized = self._normalize_trajectory_time(time_idx, B)

        target = (x_curr - x_prev) if self.predict_delta else x_curr

        noise = torch.randn_like(x_curr)
        s, r = self._sample_sr(B)  # s <= r, both (B,)

        s_col = s.unsqueeze(1)  # (B, 1)
        r_col = r.unsqueeze(1)

        z_s = (1 - s_col) * noise + s_col * target
        v = target - noise

        def u_func(z, s_scalar, r_scalar):
            h = r_scalar - s_scalar
            u, _ = self.meanflow_net(
                z,
                s_scalar.view(-1),
                h.view(-1),
                x_cond, y_curr, t_normalized,
            )
            return u

        ones_like_s = torch.ones_like(s_col)
        zeros_like_r = torch.zeros_like(r_col)

        with torch.amp.autocast("cuda", enabled=False):
            u_pred, du_ds = torch.func.jvp(
                u_func,
                (z_s.float(), s_col.float(), r_col.float()),
                (v.float(), ones_like_s.float(), zeros_like_r.float()),
            )

            u_tgt = (v.float() - (r_col.float() - s_col.float()) * du_ds).detach()

            loss_per_sample = (u_pred - u_tgt).pow(2).sum(dim=-1)  # (B,)

            adp_wt = (loss_per_sample.detach() + self.norm_eps) ** self.norm_p
            loss_weighted = (loss_per_sample / adp_wt).mean()

        metrics = {
            "loss": loss_weighted.item(),
            "mf_loss": loss_weighted.item(),
        }
        return loss_weighted, metrics

    # ---------- Stage 3: MeanFlow F2D2 loss ----------

    def compute_mf_f2d2_loss(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        time_idx: Optional[torch.Tensor] = None,
        x_prev_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Stage 3 loss: teacher velocity + divergence matching.
        Identical structure to ShortcutProposal.compute_f2d2_loss.
        """
        B = x_prev.shape[0]
        x_cond = x_prev if x_prev_cond is None else x_prev_cond
        t_normalized = self._normalize_trajectory_time(time_idx, B)
        dt_small_val = 1.0 / self.denoise_timesteps
        dt_small = torch.full((B,), dt_small_val, device=self.device)

        target = (x_curr - x_prev) if self.predict_delta else x_curr

        # ===== Flow branch (teacher matching) =====
        t_idx = torch.randint(0, self.denoise_timesteps, (B,), device=self.device)
        t_flow = (t_idx.float() / self.denoise_timesteps).clamp(1e-9, 1.0)
        z = torch.randn_like(x_curr)
        x_t = (1 - (1 - 1e-9) * t_flow.unsqueeze(1)) * z + t_flow.unsqueeze(1) * target
        x_t = x_t.detach().requires_grad_(True)

        with torch.enable_grad():
            v_teacher = self.teacher(x_t, t_flow.unsqueeze(1), x_cond, y_curr, t_normalized)
            div_teacher_raw = self._teacher_divergence(v_teacher, x_t)

        div_teacher = (div_teacher_raw * dt_small_val * self.div_scale).detach()

        v_hat, div_hat = self.meanflow_net(
            x_t.detach(), t_flow, dt_small, x_cond, y_curr, t_normalized,
        )
        v_teacher_det = v_teacher.detach()

        flow_loss_div = F.mse_loss(div_hat, div_teacher)

        if self.freeze_velocity:
            # Backbone is frozen: velocity matching is a no-op (no gradients reach
            # backbone/velocity_head).  Skip it to avoid misleading logging and the
            # redundant forward-pass through the frozen velocity head.
            flow_loss_v = torch.zeros(1, device=self.device)
            total_loss = flow_loss_div
        else:
            flow_loss_v = F.mse_loss(v_hat, v_teacher_det)
            total_loss = flow_loss_v + flow_loss_div

        metrics = {
            "flow_loss_v": flow_loss_v.item(),
            "flow_loss_div": flow_loss_div.item(),
        }

        # ===== Optional MeanFlow JVP consistency regulariser =====
        if self.mf_consistency_weight > 0:
            mf_loss, mf_metrics = self.compute_meanflow_loss(
                x_prev, x_curr, y_curr, time_idx, x_prev_cond,
            )
            total_loss = total_loss + self.mf_consistency_weight * mf_loss
            metrics["mf_loss"] = mf_metrics["mf_loss"]

        metrics["loss"] = total_loss.item()
        return total_loss, metrics

    # ---------- Lightning training / validation ----------

    def _shared_step(self, batch, stage: str):
        x_prev = batch["x_prev"]
        x_curr = batch["x_curr"]
        y_curr = batch.get("y_curr", None)
        time_idx = batch.get("time_idx", None)

        if self.debug_random_prev_state:
            x_prev = torch.randn_like(x_prev)
        if self.debug_random_obs and y_curr is not None:
            y_curr = torch.randn_like(y_curr)

        if stage == "train" and self.training and self.cond_dropout > 0:
            if torch.rand(1, device=self.device).item() < self.cond_dropout:
                y_curr = None

        x_prev_cond = None
        if stage == "train":
            corr_active = (
                self.prev_state_corr_p0 > 0
                and self.prev_state_corr_total_steps is not None
                and (self.prev_state_corr_sigma > 0 or self.prev_state_corr_mask_ratio > 0)
            )
            if corr_active:
                x_prev_cond, _ = self._corrupt_prev_state(x_prev)

        if self.training_stage == "mf_f2d2":
            loss, metrics = self.compute_mf_f2d2_loss(
                x_prev, x_curr, y_curr, time_idx, x_prev_cond,
            )
        else:
            loss, metrics = self.compute_meanflow_loss(
                x_prev, x_curr, y_curr, time_idx, x_prev_cond,
            )
        return loss, metrics

    def training_step(self, batch, batch_idx):
        if self.lr_warmup_steps > 0:
            step = self.trainer.global_step
            if step < self.lr_warmup_steps:
                scale = (step + 1) / max(1, self.lr_warmup_steps)
                for pg in self.optimizers().param_groups:
                    pg["lr"] = self.learning_rate * scale

        loss, metrics = self._shared_step(batch, "train")
        self.log("train_loss", metrics["loss"], on_step=True, on_epoch=True, prog_bar=True)
        if "mf_loss" in metrics:
            self.log("train_mf_loss", metrics["mf_loss"], on_step=True, on_epoch=True)
        if self.training_stage == "mf_f2d2":
            self.log("train_flow_v", metrics.get("flow_loss_v", 0), on_step=True, on_epoch=True)
            self.log("train_flow_div", metrics.get("flow_loss_div", 0), on_step=True, on_epoch=True)
        self._train_outputs.append(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch, "val")
        self.log("val_loss", metrics["loss"], on_step=False, on_epoch=True, prog_bar=True)
        if "mf_loss" in metrics:
            self.log("val_mf_loss", metrics["mf_loss"], on_step=False, on_epoch=True)
        if self.training_stage == "mf_f2d2":
            self.log("val_flow_v", metrics.get("flow_loss_v", 0), on_step=False, on_epoch=True)
            self.log("val_flow_div", metrics.get("flow_loss_div", 0), on_step=False, on_epoch=True)
        self._val_outputs.append(metrics)
        return loss

    def on_train_epoch_end(self):
        if self._train_outputs:
            avg = np.mean([m["loss"] for m in self._train_outputs])
            logging.getLogger(__name__).info(
                f"Epoch {self.current_epoch} train_loss={avg:.6f}"
            )
            self._train_outputs.clear()

    def on_validation_epoch_end(self):
        if self._val_outputs:
            avg = np.mean([m["loss"] for m in self._val_outputs])
            logging.getLogger(__name__).info(
                f"Epoch {self.current_epoch} val_loss={avg:.6f}"
            )
            self._val_outputs.clear()

    def configure_optimizers(self):
        if self.freeze_velocity:
            # Only optimise the divergence head; backbone + velocity_head are frozen.
            params = self.meanflow_net.div_head.parameters()
        else:
            params = self.parameters()
        optimizer = torch.optim.RAdam(
            params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    # ---------- Inference ----------

    @torch.no_grad()
    def sample(
        self,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        n_steps: Optional[int] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_{t-1}) using n_steps forward Euler steps.

        With n_steps=1 (default), this is a single network call: one NFE.
        Uses RF convention: s=0 noise, s=1 data.
        """
        if n_steps is None:
            n_steps = self.n_sampling_steps

        was_1d = x_prev.dim() == 1
        if was_1d:
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        B = x_prev.shape[0]

        t_normalized = self._normalize_trajectory_time(t, B)

        step_size = 1.0 / n_steps
        z = torch.randn(B, self.state_dim, device=self.device)

        for i in range(n_steps):
            t_cur = torch.full((B,), i * step_size, device=self.device)
            dt_cur = torch.full((B,), step_size, device=self.device)
            u, _ = self.meanflow_net(z, t_cur, dt_cur, x_prev, y_curr, t_normalized)
            z = z + step_size * u

        if self.predict_delta:
            z = x_prev + z

        if was_1d:
            z = z.squeeze(0)
        return z

    @torch.no_grad()
    def log_prob(
        self,
        x_curr: torch.Tensor,
        x_prev: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        n_steps: Optional[int] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute log q(x_curr | x_prev) using n_steps backward Euler with div head.
        """
        if n_steps is None:
            n_steps = self.n_likelihood_steps

        was_1d = x_curr.dim() == 1
        if was_1d:
            x_curr = x_curr.unsqueeze(0)
            x_prev = x_prev.unsqueeze(0)
            if y_curr is not None:
                y_curr = y_curr.unsqueeze(0)
        B = x_curr.shape[0]

        t_normalized = self._normalize_trajectory_time(t, B)

        if self.predict_delta:
            x = (x_curr - x_prev).clone()
        else:
            x = x_curr.clone()

        step_size = 1.0 / n_steps
        log_det = torch.zeros(B, device=self.device)

        for i in range(n_steps - 1, -1, -1):
            t_start_val = i * step_size
            t_cur = torch.full((B,), t_start_val, device=self.device)
            dt_cur = torch.full((B,), step_size, device=self.device)
            u, div = self.meanflow_net(x, t_cur, dt_cur, x_prev, y_curr, t_normalized)
            x = x - step_size * u
            log_det = log_det + div

        log_p0 = (
            -0.5 * torch.sum(x ** 2, dim=-1)
            - 0.5 * self.state_dim * np.log(2 * np.pi)
        )
        log_q = log_p0 - log_det

        if was_1d:
            log_q = log_q.squeeze(0)
        return log_q
