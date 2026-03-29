"""
Shortcut + F2D2 Proposal Distribution (Stages 2 & 3)

ShortcutProposal is a PyTorch Lightning module that wraps a shortcut velocity
network and implements the full F2D2 training pipeline:

  Stage 2  (training_stage='sc'):
    - Flow loss: match frozen teacher velocity at dt_small = 1/denoise_timesteps
    - Bootstrap loss: shortcut self-consistency via two-half-step composition
    - Time sampling: uniform discrete over denoise_timesteps grid

  Stage 3  (training_stage='f2d2'):
    - All of Stage 2, plus:
    - Divergence flow loss: match teacher divergence at dt_small
    - Divergence bootstrap loss: self-consistency of the divergence head
    - Teacher divergence estimated with exact trace (cheap for D≤40)
      or one-sample Hutchinson (configurable)

Inference:
  sample()   — n_steps forward Euler steps (default 1; effectively one NFE)
  log_prob() — n_steps backward Euler with divergence accumulation head (default 4 NFEs)

All conditioning features from the teacher (observations, obs_dropout,
prev_state_corruption, gated correction, trajectory-time conditioning) are
preserved and passed through to the shortcut network.
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

# ---------------------------------------------------------------------------
# Handle both `python shortcut_flow.py` and `import proposals.shortcut_flow`
# ---------------------------------------------------------------------------
try:
    from .architectures import create_shortcut_network
    from .architectures import ShortcutMLPVelocityNetwork, ShortcutResNet1DVelocityNetwork
    from .rectified_flow import RFProposal
except ImportError:
    from architectures import create_shortcut_network
    from architectures import ShortcutMLPVelocityNetwork, ShortcutResNet1DVelocityNetwork
    from rectified_flow import RFProposal


# ---------------------------------------------------------------------------
# EMA Callback (same pattern as in train_fm.py)
# ---------------------------------------------------------------------------

class ShortcutEMACallback(Callback):
    """EMA over shortcut_net weights; swapped in for validation and saved at end."""

    def __init__(self, ema_beta: float = 0.9999):
        super().__init__()
        self.ema_beta = ema_beta
        self._ema: dict = {}
        self._backup: dict = {}

    def _net(self, pl_module):
        return pl_module.shortcut_net

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
        checkpoint["shortcut_ema_params"] = {k: v.cpu() for k, v in self._ema.items()}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "shortcut_ema_params" in checkpoint:
            device = next(self._net(pl_module).parameters()).device
            self._ema = {
                k: v.to(device)
                for k, v in checkpoint["shortcut_ema_params"].items()
            }

    def on_train_end(self, trainer, pl_module):
        if not self._ema:
            return
        ckpt_dir = Path(trainer.checkpoint_callback.dirpath)
        ema_path = str(ckpt_dir / "shortcut_ema_weights.ckpt")
        net = self._net(pl_module)
        backup = {n: p.data.detach().clone() for n, p in net.named_parameters()}
        for name, param in net.named_parameters():
            if name in self._ema:
                param.data.copy_(self._ema[name])
        trainer.save_checkpoint(ema_path)
        for name, param in net.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])
        logging.getLogger(__name__).info(f"Shortcut EMA checkpoint: {ema_path}")


# ---------------------------------------------------------------------------
# ShortcutProposal
# ---------------------------------------------------------------------------

class ShortcutProposal(pl.LightningModule):
    """
    Shortcut + F2D2 proposal distribution.

    Constructor arguments that are new compared to RFProposal:
        teacher_ckpt_path: Path to the frozen teacher RFProposal checkpoint.
        training_stage:    'sc' (Stage 2) or 'f2d2' (Stage 3).
        denoise_timesteps: Grid size for uniform t-sampling (default 1024).
        lr_warmup_steps:   Linear LR warmup steps (0 = disabled).
        grad_clip_val:     Gradient clipping norm (default 0.01, matching F2D2).
        teacher_div_estimator: 'exact' or 'hutchinson' for Stage 3 divergence targets.
        div_scale:         Scale applied to divergence targets before MSE
                           (F2D2 uses 1/20000 for images; use 1.0 for our small D).
        n_sampling_steps:  Default number of steps for sample() at inference.
        n_likelihood_steps: Default number of steps for log_prob() at inference.

    All architecture and conditioning args from RFProposal are preserved:
        state_dim, obs_dim, architecture, hidden_dim, depth, channels, num_blocks,
        kernel_size, time_embed_dim, obs_indices, predict_delta, cond_dropout,
        use_time_step, trajectory_length, use_gated_obs_correction, prev_state_corr_*,
        obs_consistency_weight, etc.
    """

    # ---------- constructor ----------

    def __init__(
        self,
        state_dim: int,
        teacher_ckpt_path: str,
        training_stage: str = "sc",            # 'sc' | 'f2d2'
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
        denoise_timesteps: int = 1024,
        lr_warmup_steps: int = 0,
        teacher_div_estimator: str = "exact",  # 'exact' | 'hutchinson'
        div_scale: float = 1.0,
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

        assert training_stage in ("sc", "f2d2"), \
            f"training_stage must be 'sc' or 'f2d2', got '{training_stage}'"
        assert teacher_div_estimator in ("exact", "hutchinson"), \
            f"teacher_div_estimator must be 'exact' or 'hutchinson'"

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.obs_indices = obs_indices
        self.predict_delta = predict_delta
        self.use_time_step = use_time_step
        self.trajectory_length = trajectory_length
        self.training_stage = training_stage
        self.learning_rate = learning_rate
        self.denoise_timesteps = denoise_timesteps
        self.lr_warmup_steps = lr_warmup_steps
        self.teacher_div_estimator = teacher_div_estimator
        self.div_scale = div_scale
        self.n_sampling_steps = n_sampling_steps
        self.n_likelihood_steps = n_likelihood_steps
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

        # Shortcut network (div head inactive until Stage 3)
        self.shortcut_net = create_shortcut_network(
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
            div_head_active=(training_stage == "f2d2"),
            div_hidden_dim=div_hidden_dim,
        )

        # Teacher loaded lazily in on_fit_start (not saved in Lightning checkpoint)
        self.teacher: Optional[RFProposal] = None

        # Per-epoch accumulators for logging
        self._train_outputs = []
        self._val_outputs = []

    # ---------- teacher loading ----------

    def _load_teacher(self):
        """Load and freeze the teacher model onto the current device."""
        ckpt_path = self.hparams.teacher_ckpt_path
        teacher = RFProposal.load_from_checkpoint(ckpt_path, map_location=self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.to(self.device)
        return teacher

    def on_fit_start(self):
        if self.teacher is None:
            self.teacher = self._load_teacher()
            logging.getLogger(__name__).info(
                f"Teacher loaded from {self.hparams.teacher_ckpt_path}"
            )

    # ---------- helpers shared with RFProposal ----------

    def _normalize_trajectory_time(
        self, t: Optional[torch.Tensor], batch_size: int
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
        p = self._get_corruption_prob()
        if p <= 0 or (self.prev_state_corr_sigma <= 0 and self.prev_state_corr_mask_ratio <= 0):
            return x_prev, 0.0
        out = x_prev.clone()
        which = torch.rand(B, device=x_prev.device) < p
        n_corr = which.sum().item()
        if n_corr == 0:
            return out, 0.0
        if self.prev_state_corr_sigma > 0:
            noise = self.prev_state_corr_sigma * torch.randn_like(x_prev)
            out = out + noise * which.unsqueeze(1).float()
        if self.prev_state_corr_mask_ratio > 0:
            n_mask = max(1, int(round(self.state_dim * self.prev_state_corr_mask_ratio)))
            for i in range(B):
                if which[i]:
                    idx = torch.randperm(self.state_dim, device=x_prev.device)[:n_mask]
                    out[i, idx] = 0.0
        return out, n_corr / B

    # ---------- bootstrap timestep sampling ----------

    def _sample_bootstrap_timesteps(self, B: int):
        """
        Sample (dt, t_start, t_mid) for the bootstrap self-consistency loss.
        dt ∈ {1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64}
        t_start is a valid start time for the chosen dt.
        """
        device = self.device
        dt_choices = torch.tensor(
            [1.0 / (2 ** i) for i in range(7)], device=device
        )
        dt_idx = torch.randint(0, len(dt_choices), (B,), device=device)
        dt = dt_choices[dt_idx]
        max_steps = ((1.0 - dt) / dt).floor().long() + 1
        step = (torch.rand(B, device=device) * max_steps.float()).floor()
        t_start = step * dt
        t_mid = t_start + dt / 2
        return dt, t_start, t_mid   # each (B,)

    # ---------- teacher divergence estimation ----------

    def _teacher_divergence(
        self, v_teacher: torch.Tensor, x_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate ∫_t^{t+dt_small} div v_teacher ds ≈ div(v_teacher) * dt_small.
        Returns shape (B,).  dt_small factor NOT applied here — caller multiplies.
        """
        assert x_t.requires_grad
        if self.teacher_div_estimator == "exact":
            B = x_t.shape[0]
            divergence = torch.zeros(B, device=self.device)
            for k in range(self.state_dim):
                grad_k = torch.autograd.grad(
                    v_teacher[:, k],
                    x_t,
                    grad_outputs=torch.ones(B, device=self.device),
                    create_graph=False,
                    retain_graph=True,
                )[0]
                divergence = divergence + grad_k[:, k]
            return divergence
        else:
            # One-sample Rademacher Hutchinson
            eps = torch.randint_like(x_t, low=0, high=2).float() * 2 - 1
            vjp = torch.autograd.grad(
                v_teacher, x_t, grad_outputs=eps, create_graph=False, retain_graph=False
            )[0]
            return (vjp * eps).sum(dim=-1)

    # ---------- Stage 2: shortcut loss ----------

    def compute_shortcut_loss(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        time_idx: Optional[torch.Tensor] = None,
        x_prev_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Stage 2 loss: velocity flow-matching + bootstrap self-consistency.
        No divergence supervision.
        """
        B = x_prev.shape[0]
        x_cond = x_prev if x_prev_cond is None else x_prev_cond
        t_normalized = self._normalize_trajectory_time(time_idx, B)
        dt_small = torch.full((B,), 1.0 / self.denoise_timesteps, device=self.device)

        target = (x_curr - x_prev) if self.predict_delta else x_curr

        # ===== Flow loss =====
        t_idx = torch.randint(0, self.denoise_timesteps, (B,), device=self.device)
        t_flow = (t_idx.float() / self.denoise_timesteps).clamp(1e-9, 1.0)
        z = torch.randn_like(x_curr)
        x_t = (1 - (1 - 1e-9) * t_flow.unsqueeze(1)) * z + t_flow.unsqueeze(1) * target

        with torch.no_grad():
            v_teacher = self.teacher(x_t, t_flow.unsqueeze(1), x_cond, y_curr, t_normalized)

        v_hat, _ = self.shortcut_net(x_t, t_flow, dt_small, x_cond, y_curr, t_normalized)
        flow_loss_v = F.mse_loss(v_hat, v_teacher)

        # ===== Bootstrap loss =====
        dt, t_start, t_mid = self._sample_bootstrap_timesteps(B)
        t_start = t_start.clamp(1e-9, 1.0)
        z2 = torch.randn_like(x_curr)
        x_start = (1 - (1 - 1e-9) * t_start.unsqueeze(1)) * z2 + t_start.unsqueeze(1) * target

        with torch.no_grad():
            v1, _ = self.shortcut_net(x_start, t_start, dt / 2, x_cond, y_curr, t_normalized)
            x_mid = x_start + (dt / 2).unsqueeze(1) * v1
            v2, _ = self.shortcut_net(x_mid, t_mid, dt / 2, x_cond, y_curr, t_normalized)
            v_target = (v1 + v2) / 2

        v_boot, _ = self.shortcut_net(x_start, t_start, dt, x_cond, y_curr, t_normalized)
        boot_loss_v = F.mse_loss(v_boot, v_target)

        total_loss = flow_loss_v + boot_loss_v
        metrics = {
            "loss": total_loss.item(),
            "flow_loss_v": flow_loss_v.item(),
            "boot_loss_v": boot_loss_v.item(),
            "flow_loss_div": 0.0,
            "boot_loss_div": 0.0,
        }
        return total_loss, metrics

    # ---------- Stage 3: F2D2 / likelihood loss ----------

    def compute_f2d2_loss(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        time_idx: Optional[torch.Tensor] = None,
        x_prev_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Stage 3 loss: velocity + divergence flow-matching + bootstrap.
        """
        B = x_prev.shape[0]
        x_cond = x_prev if x_prev_cond is None else x_prev_cond
        t_normalized = self._normalize_trajectory_time(time_idx, B)
        dt_small_val = 1.0 / self.denoise_timesteps
        dt_small = torch.full((B,), dt_small_val, device=self.device)

        target = (x_curr - x_prev) if self.predict_delta else x_curr

        # ===== Flow branch =====
        t_idx = torch.randint(0, self.denoise_timesteps, (B,), device=self.device)
        t_flow = (t_idx.float() / self.denoise_timesteps).clamp(1e-9, 1.0)
        z = torch.randn_like(x_curr)
        x_t = (1 - (1 - 1e-9) * t_flow.unsqueeze(1)) * z + t_flow.unsqueeze(1) * target
        x_t = x_t.detach().requires_grad_(True)

        # Teacher velocity (need graph for divergence)
        with torch.enable_grad():
            v_teacher = self.teacher(x_t, t_flow.unsqueeze(1), x_cond, y_curr, t_normalized)
            div_teacher_raw = self._teacher_divergence(v_teacher, x_t)

        div_teacher = (div_teacher_raw * dt_small_val * self.div_scale).detach()

        v_hat, div_hat = self.shortcut_net(
            x_t.detach(), t_flow, dt_small, x_cond, y_curr, t_normalized
        )
        v_teacher_det = v_teacher.detach()

        flow_loss_v = F.mse_loss(v_hat, v_teacher_det)
        flow_loss_div = F.mse_loss(div_hat, div_teacher)

        # ===== Bootstrap branch =====
        dt, t_start, t_mid = self._sample_bootstrap_timesteps(B)
        t_start = t_start.clamp(1e-9, 1.0)
        z2 = torch.randn_like(x_curr)
        x_start = (1 - (1 - 1e-9) * t_start.unsqueeze(1)) * z2 + t_start.unsqueeze(1) * target

        with torch.no_grad():
            v1, div1 = self.shortcut_net(x_start, t_start, dt / 2, x_cond, y_curr, t_normalized)
            x_mid = x_start + (dt / 2).unsqueeze(1) * v1
            v2, div2 = self.shortcut_net(x_mid, t_mid, dt / 2, x_cond, y_curr, t_normalized)
            v_target = (v1 + v2) / 2
            div_target = (div1 + div2) / 2

        v_boot, div_boot = self.shortcut_net(
            x_start, t_start, dt, x_cond, y_curr, t_normalized
        )
        boot_loss_v = F.mse_loss(v_boot, v_target)
        boot_loss_div = F.mse_loss(div_boot, div_target)

        total_loss = flow_loss_v + flow_loss_div + boot_loss_v + boot_loss_div
        metrics = {
            "loss": total_loss.item(),
            "flow_loss_v": flow_loss_v.item(),
            "flow_loss_div": flow_loss_div.item(),
            "boot_loss_v": boot_loss_v.item(),
            "boot_loss_div": boot_loss_div.item(),
        }
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

        # Conditioning dropout (training only)
        if stage == "train" and self.training and self.cond_dropout > 0:
            if torch.rand(1, device=self.device).item() < self.cond_dropout:
                y_curr = None

        # Prev-state corruption (training only)
        x_prev_cond = None
        if stage == "train":
            corr_active = (
                self.prev_state_corr_p0 > 0
                and self.prev_state_corr_total_steps is not None
                and (self.prev_state_corr_sigma > 0 or self.prev_state_corr_mask_ratio > 0)
            )
            if corr_active:
                x_prev_cond, _ = self._corrupt_prev_state(x_prev)

        compute_fn = self.compute_f2d2_loss if self.training_stage == "f2d2" else self.compute_shortcut_loss
        loss, metrics = compute_fn(x_prev, x_curr, y_curr, time_idx, x_prev_cond)
        return loss, metrics

    def training_step(self, batch, batch_idx):
        # Manual linear LR warmup
        if self.lr_warmup_steps > 0:
            step = self.trainer.global_step
            if step < self.lr_warmup_steps:
                scale = (step + 1) / max(1, self.lr_warmup_steps)
                for pg in self.optimizers().param_groups:
                    pg["lr"] = self.learning_rate * scale

        loss, metrics = self._shared_step(batch, "train")
        self.log("train_loss", metrics["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_flow_v", metrics["flow_loss_v"], on_step=True, on_epoch=True)
        self.log("train_boot_v", metrics["boot_loss_v"], on_step=True, on_epoch=True)
        if self.training_stage == "f2d2":
            self.log("train_flow_div", metrics["flow_loss_div"], on_step=True, on_epoch=True)
            self.log("train_boot_div", metrics["boot_loss_div"], on_step=True, on_epoch=True)
        self._train_outputs.append(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch, "val")
        self.log("val_loss", metrics["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_flow_v", metrics["flow_loss_v"], on_step=False, on_epoch=True)
        self.log("val_boot_v", metrics["boot_loss_v"], on_step=False, on_epoch=True)
        if self.training_stage == "f2d2":
            self.log("val_flow_div", metrics["flow_loss_div"], on_step=False, on_epoch=True)
            self.log("val_boot_div", metrics["boot_loss_div"], on_step=False, on_epoch=True)
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
        optimizer = torch.optim.RAdam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20
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
        With n_steps>1 the shortcut model refines the sample.

        Args:
            x_prev:  (state_dim,) or (B, state_dim)
            y_curr:  (obs_dim,) or (B, obs_dim) or None
            dt:      Unused (interface compatibility)
            n_steps: Number of Euler steps (None → self.n_sampling_steps)
            t:       Trajectory time step (unnormalized); required when use_time_step=True

        Returns:
            Sampled next state, same shape as x_prev
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
            u, _ = self.shortcut_net(z, t_cur, dt_cur, x_prev, y_curr, t_normalized)
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

        Each backward step calls (u, div) = shortcut_net(...).
        log_det accumulates the forward divergence; base density evaluated at z_0.

        Args:
            x_curr:  (state_dim,) or (B, state_dim)
            x_prev:  same shape
            y_curr:  (obs_dim,) or (B, obs_dim) or None
            dt:      Unused (interface compatibility)
            n_steps: Number of backward steps (None → self.n_likelihood_steps)
            t:       Trajectory time step (unnormalized); required when use_time_step=True

        Returns:
            log_prob, shape () or (B,)
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

        # Backward integration from t=1 toward t=0
        for i in range(n_steps - 1, -1, -1):
            t_start_val = i * step_size
            t_cur = torch.full((B,), t_start_val, device=self.device)
            dt_cur = torch.full((B,), step_size, device=self.device)
            u, div = self.shortcut_net(x, t_cur, dt_cur, x_prev, y_curr, t_normalized)
            x = x - step_size * u   # backward step
            log_det = log_det + div  # div head gives ∫_{t}^{t+dt} div v ds (forward direction)

        # Base Gaussian log-prob at z_0
        log_p0 = (
            -0.5 * torch.sum(x ** 2, dim=-1)
            - 0.5 * self.state_dim * np.log(2 * np.pi)
        )
        log_q = log_p0 - log_det

        if was_1d:
            log_q = log_q.squeeze(0)
        return log_q
