"""
Modern Flow Matching Teacher Training (Stage 1b)

Trains a flow matching teacher using:
  - EDM-style log-normal time sampling (concentrated on hard intermediate steps)
  - Exponential Moving Average (EMA) of weights for evaluation/checkpointing
  - RAdam optimizer with linear learning-rate warmup

The checkpoint produced is a standard RFProposal and can be loaded directly as
`RFProposal.load_from_checkpoint(ckpt_path)` for Stage 2 shortcut distillation.

Leaves proposals/train_rf.py completely untouched.
"""

import copy
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)
from lightning.pytorch.loggers import WandbLogger
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import sys

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))
    sys.path.append(str(Path(__file__).parent.parent))

from rectified_flow import RFProposal
from rf_dataset import RFDataModule

try:
    from eval_proposal import run_proposal_eval
except ImportError:
    try:
        from proposals.eval_proposal import run_proposal_eval
    except ImportError:
        run_proposal_eval = None


# ---------------------------------------------------------------------------
# EMA Callback
# ---------------------------------------------------------------------------

class EMACallback(Callback):
    """
    Exponential Moving Average of velocity_net weights.

    - Updates shadow EMA params after every training step.
    - Swaps EMA weights in during validation (so val_loss reflects EMA quality).
    - Swaps back to main weights after validation.
    - Saves EMA state into Lightning checkpoints.
    - At training end, also saves a dedicated `ema_weights.ckpt` with the EMA
      weights installed as the main model weights (loadable as plain RFProposal).
    """

    def __init__(self, ema_beta: float = 0.9999, net_attr: str = "velocity_net"):
        super().__init__()
        self.ema_beta = ema_beta
        self.net_attr = net_attr
        self._ema: dict = {}
        self._backup: dict = {}

    def _get_net(self, pl_module):
        return getattr(pl_module, self.net_attr)

    # ---- Initialise / update EMA ----

    def on_train_start(self, trainer, pl_module):
        net = self._get_net(pl_module)
        self._ema = {n: p.data.detach().clone() for n, p in net.named_parameters()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        net = self._get_net(pl_module)
        beta = self.ema_beta
        with torch.no_grad():
            for name, param in net.named_parameters():
                if name in self._ema:
                    self._ema[name].mul_(beta).add_(param.data, alpha=1.0 - beta)

    # ---- Swap EMA in/out around validation ----

    def on_validation_start(self, trainer, pl_module):
        if not self._ema:
            return
        net = self._get_net(pl_module)
        self._backup = {n: p.data.detach().clone() for n, p in net.named_parameters()}
        for name, param in net.named_parameters():
            if name in self._ema:
                param.data.copy_(self._ema[name])

    def on_validation_end(self, trainer, pl_module):
        if not self._backup:
            return
        net = self._get_net(pl_module)
        for name, param in net.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    # ---- Persist EMA inside Lightning checkpoints ----

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["ema_params"] = {k: v.cpu() for k, v in self._ema.items()}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_params" in checkpoint:
            device = next(self._get_net(pl_module).parameters()).device
            self._ema = {k: v.to(device) for k, v in checkpoint["ema_params"].items()}

    # ---- Save a standalone EMA checkpoint at the very end ----

    def on_train_end(self, trainer, pl_module):
        if not self._ema:
            return
        ckpt_dir = Path(trainer.checkpoint_callback.dirpath)
        ema_ckpt_path = str(ckpt_dir / "ema_weights.ckpt")

        # Temporarily install EMA weights
        net = self._get_net(pl_module)
        backup = {n: p.data.detach().clone() for n, p in net.named_parameters()}
        for name, param in net.named_parameters():
            if name in self._ema:
                param.data.copy_(self._ema[name])

        trainer.save_checkpoint(ema_ckpt_path)

        # Restore
        for name, param in net.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

        logging.getLogger(__name__).info(
            f"EMA checkpoint saved to: {ema_ckpt_path}"
        )


# ---------------------------------------------------------------------------
# FMProposal — modern teacher (subclass of RFProposal)
# ---------------------------------------------------------------------------

class FMProposal(RFProposal):
    """
    Flow Matching teacher with EDM log-normal time sampling, EMA, and RAdam.

    Inherits everything from RFProposal (architecture, conditioning, all kwargs)
    and overrides only the training-time differences so that the saved checkpoint
    is fully compatible with RFProposal.load_from_checkpoint().

    New constructor arguments (added to hparams):
        p_mean: Log-normal mean for noise-level σ (default -1.2, EDM default)
        p_std:  Log-normal std  for noise-level σ (default  1.2, EDM default)
        lr_warmup_steps: Linear warmup over this many gradient steps
    """

    def __init__(
        self,
        # Pass-through to RFProposal
        state_dim: int,
        architecture: str = "mlp",
        hidden_dim: int = 128,
        depth: int = 4,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 3,
        learning_rate: float = 1e-3,
        num_sampling_steps: int = 50,
        num_likelihood_steps: int = 50,
        use_preprocessing: bool = False,
        obs_dim: int = 0,
        train_cond_method: str = "concat",
        cond_embed_dim: int = 128,
        num_attn_heads: int = 4,
        predict_delta: bool = False,
        time_embed_dim: int = 64,
        mc_guidance: bool = False,
        guidance_scale: float = 1.0,
        obs_indices: Optional[list] = None,
        cond_dropout: float = 0.0,
        debug_random_obs: bool = False,
        debug_random_prev_state: bool = False,
        use_time_step: bool = False,
        trajectory_length: int = 1000,
        use_gated_obs_correction: bool = False,
        gate_type: str = "scalar",
        gate_hidden_dim: int = 64,
        gate_init_bias: float = 0.0,
        prior_zero_init: bool = True,
        obs_zero_init: bool = False,
        prev_state_corr_p0: float = 0.0,
        prev_state_corr_p_min: float = 0.05,
        prev_state_corr_total_steps: Optional[int] = None,
        prev_state_corr_sigma: float = 0.0,
        prev_state_corr_mask_ratio: float = 0.0,
        obs_consistency_weight: float = 0.0,
        obs_nonlinearity: str = "arctan",
        state_scaler_mean: Optional[torch.Tensor] = None,
        state_scaler_std: Optional[torch.Tensor] = None,
        obs_scaler_mean: Optional[torch.Tensor] = None,
        obs_scaler_std: Optional[torch.Tensor] = None,
        # New args
        p_mean: float = -1.2,
        p_std: float = 1.2,
        lr_warmup_steps: int = 0,
    ):
        super().__init__(
            state_dim=state_dim,
            architecture=architecture,
            hidden_dim=hidden_dim,
            depth=depth,
            channels=channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            learning_rate=learning_rate,
            num_sampling_steps=num_sampling_steps,
            num_likelihood_steps=num_likelihood_steps,
            use_preprocessing=use_preprocessing,
            obs_dim=obs_dim,
            train_cond_method=train_cond_method,
            cond_embed_dim=cond_embed_dim,
            num_attn_heads=num_attn_heads,
            predict_delta=predict_delta,
            time_embed_dim=time_embed_dim,
            mc_guidance=mc_guidance,
            guidance_scale=guidance_scale,
            obs_indices=obs_indices,
            cond_dropout=cond_dropout,
            debug_random_obs=debug_random_obs,
            debug_random_prev_state=debug_random_prev_state,
            use_time_step=use_time_step,
            trajectory_length=trajectory_length,
            use_gated_obs_correction=use_gated_obs_correction,
            gate_type=gate_type,
            gate_hidden_dim=gate_hidden_dim,
            gate_init_bias=gate_init_bias,
            prior_zero_init=prior_zero_init,
            obs_zero_init=obs_zero_init,
            prev_state_corr_p0=prev_state_corr_p0,
            prev_state_corr_p_min=prev_state_corr_p_min,
            prev_state_corr_total_steps=prev_state_corr_total_steps,
            prev_state_corr_sigma=prev_state_corr_sigma,
            prev_state_corr_mask_ratio=prev_state_corr_mask_ratio,
            obs_consistency_weight=obs_consistency_weight,
            obs_nonlinearity=obs_nonlinearity,
            state_scaler_mean=state_scaler_mean,
            state_scaler_std=state_scaler_std,
            obs_scaler_mean=obs_scaler_mean,
            obs_scaler_std=obs_scaler_std,
        )
        self.p_mean = p_mean
        self.p_std = p_std
        self.lr_warmup_steps = lr_warmup_steps
        self.save_hyperparameters()

    # ---- EDM log-normal time sampling ----

    def _sample_t_lognormal(self, batch_size: int) -> torch.Tensor:
        """
        Sample flow times using the EDM log-normal schedule.
        σ ~ LogNormal(p_mean, p_std), then t = 1/(1+σ) ∈ (0,1].
        Concentrates training effort on intermediate (harder) timesteps.
        """
        rnd = torch.randn(batch_size, device=self.device)
        sigma = (rnd * self.p_std + self.p_mean).exp()
        t = 1.0 / (1.0 + sigma)
        return t.clamp(1e-4, 1.0).unsqueeze(1)  # (B, 1)

    # ---- Override compute_rf_loss to use log-normal t ----

    def compute_rf_loss(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        x_prev_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size = x_prev.shape[0]
        x_cond = x_prev if x_prev_cond is None else x_prev_cond

        # Log-normal time sampling (EDM-style) — key difference from parent
        s = self._sample_t_lognormal(batch_size)  # (B, 1)

        z = torch.randn_like(x_curr)

        if self.predict_delta:
            target = x_curr - x_prev
        else:
            target = x_curr

        x_s = (1 - s) * z + s * target
        target_velocity = target - z

        t_normalized = self._normalize_trajectory_time(
            t=t, batch_size=batch_size, caller_name="compute_rf_loss"
        )

        pred_velocity = self.velocity_net(x_s, s, x_cond, y_curr, t_normalized)

        loss_rf = torch.mean((pred_velocity - target_velocity) ** 2)
        loss_obs = torch.tensor(0.0, device=loss_rf.device, dtype=loss_rf.dtype)
        if (
            self.obs_consistency_weight > 0
            and y_curr is not None
            and self._state_scaler_mean is not None
            and self._obs_scaler_mean is not None
        ):
            _, y_hat_scaled = self._predict_endpoint_and_obs(
                x_s, s, x_cond, y_curr, t_normalized
            )
            loss_obs = torch.mean((y_hat_scaled - y_curr) ** 2)

        total_loss = loss_rf + self.obs_consistency_weight * loss_obs
        metrics = {
            "loss": total_loss.item(),
            "loss_rf": loss_rf.item(),
            "loss_obs": loss_obs.item(),
            "loss_total": total_loss.item(),
            "velocity_norm": torch.mean(torch.norm(pred_velocity, dim=-1)).item(),
            "target_velocity_norm": torch.mean(torch.norm(target_velocity, dim=-1)).item(),
        }
        return total_loss, metrics

    # ---- RAdam + linear warmup ----

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(
            self.parameters(),
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
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    def training_step(self, batch, batch_idx):
        # Linear warmup: override LR for the first lr_warmup_steps steps
        if self.lr_warmup_steps > 0:
            step = self.trainer.global_step
            if step < self.lr_warmup_steps:
                scale = (step + 1) / max(1, self.lr_warmup_steps)
                opt = self.optimizers()
                for pg in opt.param_groups:
                    pg["lr"] = self.learning_rate * scale

        # Delegate to parent (which handles corruption, dropout, logging, etc.)
        return super().training_step(batch, batch_idx)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path):
    log_file = log_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def train_modern_teacher(
    data_dir: str,
    output_dir: str,
    state_dim: int = 3,
    obs_dim: int = 0,
    architecture: str = "mlp",
    hidden_dim: int = 128,
    depth: int = 4,
    channels: int = 64,
    num_blocks: int = 6,
    kernel_size: int = 3,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_epochs: int = 500,
    num_workers: int = 4,
    use_observations: bool = False,
    train_cond_method: str = "concat",
    cond_embed_dim: int = 128,
    num_attn_heads: int = 4,
    gpus: int = 1,
    predict_delta: bool = False,
    time_embed_dim: int = 64,
    obs_indices: Optional[list] = None,
    cond_dropout: float = 0.0,
    wandb_project: str = "rf-train-fm",
    save_every_n_epochs: Optional[int] = None,
    debug_random_obs: bool = False,
    debug_random_prev_state: bool = False,
    use_time_step: bool = False,
    trajectory_length: int = 1000,
    use_gated_obs_correction: bool = False,
    gate_type: str = "scalar",
    gate_hidden_dim: int = 64,
    gate_init_bias: float = 0.0,
    prior_zero_init: bool = True,
    obs_zero_init: bool = False,
    prev_state_corr_p0: float = 0.0,
    prev_state_corr_p_min: float = 0.05,
    prev_state_corr_sigma: float = 0.0,
    prev_state_corr_mask_ratio: float = 0.0,
    obs_consistency_weight: float = 0.0,
    obs_nonlinearity: str = "arctan",
    state_scaler_mean=None,
    state_scaler_std=None,
    obs_scaler_mean=None,
    obs_scaler_std=None,
    # FM-specific
    p_mean: float = -1.2,
    p_std: float = 1.2,
    ema_beta: float = 0.9999,
    lr_warmup_steps: int = 0,
    grad_clip_val: float = 1.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not use_observations:
        obs_dim = 0

    logger_obj = setup_logging(output_dir)
    logger_obj.info("=" * 80)
    logger_obj.info("Modern Flow Matching Teacher Training (Stage 1b)")
    logger_obj.info("=" * 80)
    logger_obj.info(f"  Log-normal t-sampling: p_mean={p_mean}, p_std={p_std}")
    logger_obj.info(f"  EMA beta: {ema_beta}")
    logger_obj.info(f"  LR warmup steps: {lr_warmup_steps}")
    logger_obj.info(f"  Architecture: {architecture}, state_dim={state_dim}, obs_dim={obs_dim}")

    data_module = RFDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        window=1,
        use_observations=use_observations,
        obs_components=obs_indices,
    )
    data_module.setup("fit")
    num_batches = len(data_module.train_dataloader())
    total_steps = num_batches * max_epochs
    logger_obj.info(f"  Batches/epoch: {num_batches}, total steps: {total_steps}")

    model = FMProposal(
        state_dim=state_dim,
        obs_dim=obs_dim,
        architecture=architecture,
        hidden_dim=hidden_dim,
        depth=depth,
        channels=channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        learning_rate=learning_rate,
        num_sampling_steps=50,
        num_likelihood_steps=50,
        use_preprocessing=True,
        train_cond_method=train_cond_method,
        cond_embed_dim=cond_embed_dim,
        num_attn_heads=num_attn_heads,
        predict_delta=predict_delta,
        time_embed_dim=time_embed_dim,
        obs_indices=obs_indices,
        cond_dropout=cond_dropout,
        debug_random_obs=debug_random_obs,
        debug_random_prev_state=debug_random_prev_state,
        use_time_step=use_time_step,
        trajectory_length=trajectory_length,
        use_gated_obs_correction=use_gated_obs_correction,
        gate_type=gate_type,
        gate_hidden_dim=gate_hidden_dim,
        gate_init_bias=gate_init_bias,
        prior_zero_init=prior_zero_init,
        obs_zero_init=obs_zero_init,
        prev_state_corr_p0=prev_state_corr_p0,
        prev_state_corr_p_min=prev_state_corr_p_min,
        prev_state_corr_total_steps=total_steps,
        prev_state_corr_sigma=prev_state_corr_sigma,
        prev_state_corr_mask_ratio=prev_state_corr_mask_ratio,
        obs_consistency_weight=obs_consistency_weight,
        obs_nonlinearity=obs_nonlinearity,
        state_scaler_mean=state_scaler_mean,
        state_scaler_std=state_scaler_std,
        obs_scaler_mean=obs_scaler_mean,
        obs_scaler_std=obs_scaler_std,
        p_mean=p_mean,
        p_std=p_std,
        lr_warmup_steps=lr_warmup_steps,
    )

    logger_obj.info(
        f"  Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    ema_callback = EMACallback(ema_beta=ema_beta, net_attr="velocity_net")

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="fm-{epoch:03d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=50, mode="min", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [ema_callback, checkpoint_callback, early_stop, lr_monitor]

    if save_every_n_epochs is not None and save_every_n_epochs > 0:
        periodic_ckpt = ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="fm-periodic-{epoch:03d}",
            every_n_epochs=save_every_n_epochs,
            save_top_k=-1,
            save_on_train_epoch_end=True,
        )
        callbacks.append(periodic_ckpt)

    wandb_logger = WandbLogger(
        entity="ml-climate",
        project=wandb_project,
        name=output_dir.name,
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 and torch.cuda.is_available() else "cpu",
        devices=gpus if gpus > 0 and torch.cuda.is_available() else 1,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
        gradient_clip_val=grad_clip_val,
        precision=32,
    )

    logger_obj.info("Starting training …")
    trainer.fit(model, data_module)
    logger_obj.info("Training completed.")
    logger_obj.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    final_path = output_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_path)

    return model, checkpoint_callback.best_model_path, wandb_logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train modern flow matching teacher (Stage 1b) with log-normal t-sampling"
    )

    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    # Architecture
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--obs_dim", type=int, default=0)
    parser.add_argument("--architecture", type=str, default="mlp", choices=["mlp", "resnet1d"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--train_cond_method", type=str, default="concat",
                        choices=["concat", "film", "adaln", "cross_attn"])
    parser.add_argument("--cond_embed_dim", type=int, default=128)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--time_embed_dim", type=int, default=64)
    parser.add_argument("--predict_delta", action="store_true")
    parser.add_argument("--use_time_step", action="store_true")
    parser.add_argument("--trajectory_length", type=int, default=None)

    # Gated obs correction
    parser.add_argument("--use_gated_obs_correction", action="store_true")
    parser.add_argument("--gate_type", type=str, default="scalar", choices=["scalar", "spatial"])
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument("--gate_init_bias", type=float, default=0.0)
    parser.add_argument("--prior_zero_init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obs_zero_init", action=argparse.BooleanOptionalAction, default=False)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--grad_clip_val", type=float, default=1.0)

    # FM-specific (new)
    parser.add_argument("--p_mean", type=float, default=-1.2,
                        help="Log-normal mean for EDM time sampling")
    parser.add_argument("--p_std", type=float, default=1.2,
                        help="Log-normal std for EDM time sampling")
    parser.add_argument("--ema_beta", type=float, default=0.9999,
                        help="EMA decay rate for shadow weights")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Linear LR warmup over this many gradient steps (0 = disabled)")

    # Conditioning / regularization
    parser.add_argument("--use_observations", action="store_true")
    parser.add_argument("--cond_dropout", type=float, default=0.1)
    parser.add_argument("--obs_components", type=str, default=None)
    parser.add_argument("--prev_state_corr_p0", type=float, default=0.0)
    parser.add_argument("--prev_state_corr_p_min", type=float, default=0.05)
    parser.add_argument("--prev_state_corr_sigma", type=float, default=0.0)
    parser.add_argument("--prev_state_corr_mask_ratio", type=float, default=0.0)
    parser.add_argument("--obs_consistency_weight", type=float, default=0.0)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="rf-train-fm")
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--debug_random_obs", action="store_true")
    parser.add_argument("--debug_random_prev_state", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()

    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/data/da_outputs/fm_runs/run_{ts}"

    # Load config
    config = None
    config_path = Path(args.data_dir) / "config.yaml"
    if config_path.exists():
        try:
            from data import load_config_yaml
            config = load_config_yaml(config_path)
        except Exception:
            pass

    # Parse obs_components
    obs_components = None
    if args.obs_components is not None:
        obs_components = [int(i) for i in args.obs_components.split(",") if i.strip()]
    elif config is not None:
        obs_components = getattr(config, "obs_components", None)
    if obs_components is not None:
        args.obs_dim = len(obs_components)

    trajectory_length = args.trajectory_length
    if trajectory_length is None and config is not None:
        trajectory_length = getattr(config, "len_trajectory", 1000)
    if trajectory_length is None:
        trajectory_length = 1000

    obs_nonlinearity = "arctan"
    if config is not None:
        obs_nonlinearity = getattr(config, "obs_nonlinearity", "arctan")

    # Scalers for obs-consistency
    state_scaler_mean = state_scaler_std = obs_scaler_mean = obs_scaler_std = None
    if args.obs_consistency_weight > 0 and args.use_observations:
        data_dir = Path(args.data_dir)
        for fname in ("data_scaled.h5", "data.h5"):
            data_file = data_dir / fname
            if data_file.exists():
                try:
                    import h5py
                    with h5py.File(data_file, "r") as f:
                        if "scaler_mean" in f and "scaler_std" in f:
                            state_scaler_mean = torch.from_numpy(f["scaler_mean"][:]).float()
                            state_scaler_std = torch.from_numpy(f["scaler_std"][:]).float()
                        if "obs_scaler_mean" in f and "obs_scaler_std" in f:
                            om = f["obs_scaler_mean"][:]
                            os_ = f["obs_scaler_std"][:]
                            if obs_components is not None and len(om) > len(obs_components):
                                om = om[obs_components]
                                os_ = os_[obs_components]
                            obs_scaler_mean = torch.from_numpy(om).float()
                            obs_scaler_std = torch.from_numpy(os_).float()
                except Exception:
                    pass
                break

    if not args.use_observations or args.obs_dim == 0:
        args.obs_consistency_weight = 0.0

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    model, best_ckpt, wandb_logger = train_modern_teacher(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        state_dim=args.state_dim,
        obs_dim=args.obs_dim,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        channels=args.channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        use_observations=args.use_observations,
        train_cond_method=args.train_cond_method,
        cond_embed_dim=args.cond_embed_dim,
        num_attn_heads=args.num_attn_heads,
        gpus=args.gpus,
        predict_delta=args.predict_delta,
        time_embed_dim=args.time_embed_dim,
        obs_indices=obs_components,
        cond_dropout=args.cond_dropout,
        wandb_project=args.wandb_project,
        save_every_n_epochs=args.save_every_n_epochs,
        debug_random_obs=args.debug_random_obs,
        debug_random_prev_state=args.debug_random_prev_state,
        use_time_step=args.use_time_step,
        trajectory_length=trajectory_length,
        use_gated_obs_correction=args.use_gated_obs_correction,
        gate_type=args.gate_type,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_init_bias=args.gate_init_bias,
        prior_zero_init=args.prior_zero_init,
        obs_zero_init=args.obs_zero_init,
        prev_state_corr_p0=args.prev_state_corr_p0,
        prev_state_corr_p_min=args.prev_state_corr_p_min,
        prev_state_corr_sigma=args.prev_state_corr_sigma,
        prev_state_corr_mask_ratio=args.prev_state_corr_mask_ratio,
        obs_consistency_weight=args.obs_consistency_weight,
        obs_nonlinearity=obs_nonlinearity,
        state_scaler_mean=state_scaler_mean,
        state_scaler_std=state_scaler_std,
        obs_scaler_mean=obs_scaler_mean,
        obs_scaler_std=obs_scaler_std,
        p_mean=args.p_mean,
        p_std=args.p_std,
        ema_beta=args.ema_beta,
        lr_warmup_steps=args.lr_warmup_steps,
        grad_clip_val=args.grad_clip_val,
    )

    if args.evaluate and run_proposal_eval is not None:
        wandb_run = None
        if wandb_logger and hasattr(wandb_logger, "experiment"):
            wandb_run = wandb_logger.experiment
        run_proposal_eval(
            checkpoint_path=best_ckpt,
            data_dir=args.data_dir,
            n_trajectories=None,
            n_vis_trajectories=10,
            batch_size=args.batch_size,
            device="cuda" if (args.gpus > 0 and torch.cuda.is_available()) else "cpu",
            wandb_run=wandb_run,
        )


if __name__ == "__main__":
    main()
