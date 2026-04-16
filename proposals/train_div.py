"""
Stage 3: Divergence Head Training (F2D2 Likelihood)

Resumes from a Stage 2 shortcut or MeanFlow checkpoint, activates the
divergence head, and trains the full network (velocity + divergence) using
the F2D2 likelihood loss.

Supports both Shortcut and MeanFlow Stage 2 checkpoints.  The checkpoint
type is auto-detected from hyperparameters ('tr_sampler' present => MeanFlow).

After Stage 3 the checkpoint can be loaded as ShortcutProposal or
MeanFlowProposal and used for fast log_prob() evaluation.

Usage:
    python proposals/train_div.py \
        --shortcut_ckpt /path/to/sc_best.ckpt \
        --teacher_ckpt  /path/to/teacher.ckpt \
        --data_dir      /path/to/data \
        --output_dir    /path/to/output \
        --state_dim 40 --obs_dim 20 --architecture resnet1d \
        --teacher_div_estimator exact \
        --max_epochs 100
"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))
    sys.path.append(str(Path(__file__).parent.parent))

from shortcut_flow import ShortcutProposal, ShortcutEMACallback
from meanflow_proposal import MeanFlowProposal, MeanFlowEMACallback
from rf_dataset import RFDataModule

try:
    from eval_proposal import run_proposal_eval, _derive_eval_run_name
except ImportError:
    from proposals.eval_proposal import run_proposal_eval, _derive_eval_run_name


def _is_meanflow_checkpoint(ckpt_path: str) -> bool:
    """Detect whether a Stage 2 checkpoint is MeanFlow (vs Shortcut)."""
    load_kw = {"map_location": "cpu"}
    try:
        raw = torch.load(ckpt_path, weights_only=False, **load_kw)
    except TypeError:
        raw = torch.load(ckpt_path, **load_kw)
    hparams = raw.get("hyper_parameters", {})
    return isinstance(hparams, dict) and "tr_sampler" in hparams


def setup_logging(log_dir: Path):
    log_file = log_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def train_divergence(
    shortcut_ckpt: str,
    teacher_ckpt: str,
    data_dir: str,
    output_dir: str,
    state_dim: int = 3,
    obs_dim: int = 0,
    architecture: str = "mlp",
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    max_epochs: int = 100,
    num_workers: int = 4,
    use_observations: bool = False,
    obs_indices: Optional[list] = None,
    cond_dropout: float = 0.0,
    gpus: int = 1,
    denoise_timesteps: int = 1024,
    ema_beta: float = 0.9999,
    lr_warmup_steps: int = 0,
    grad_clip_val: float = 0.01,
    teacher_div_estimator: str = "exact",
    div_scale: float = 1.0,
    wandb_project: str = "rf-f2d2",
    save_every_n_epochs: Optional[int] = None,
    debug_random_obs: bool = False,
    debug_random_prev_state: bool = False,
    # MeanFlow-specific: freeze backbone+velocity_head so Stage 3 only trains
    # the divergence head, preserving the coarse-step structure from Stage 2.
    freeze_velocity: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not use_observations:
        obs_dim = 0

    logger_obj = setup_logging(output_dir)
    logger_obj.info("=" * 80)
    logger_obj.info("Stage 3: Divergence Head (F2D2 Likelihood)")
    logger_obj.info("=" * 80)
    logger_obj.info(f"  Shortcut ckpt:    {shortcut_ckpt}")
    logger_obj.info(f"  Teacher ckpt:     {teacher_ckpt}")
    logger_obj.info(f"  Div estimator:    {teacher_div_estimator}, scale={div_scale}")

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

    # Detect checkpoint type and load the correct class
    is_meanflow = _is_meanflow_checkpoint(shortcut_ckpt)
    logger_obj.info(f"  Checkpoint type:  {'MeanFlow' if is_meanflow else 'Shortcut'}")

    if is_meanflow:
        model = MeanFlowProposal.load_from_checkpoint(
            shortcut_ckpt,
            strict=False,
            training_stage="mf_f2d2",
            teacher_ckpt_path=teacher_ckpt,
            teacher_div_estimator=teacher_div_estimator,
            div_scale=div_scale,
            learning_rate=learning_rate,
            denoise_timesteps=denoise_timesteps,
            lr_warmup_steps=lr_warmup_steps,
            debug_random_obs=debug_random_obs,
            debug_random_prev_state=debug_random_prev_state,
            freeze_velocity=freeze_velocity,
        )
        model.meanflow_net.div_head_active = True
        model.training_stage = "mf_f2d2"
        if freeze_velocity:
            # Hard-freeze backbone + velocity_head so Stage 3 cannot overwrite
            # the coarse-step mean-field structure trained in Stage 2.
            for name, param in model.meanflow_net.named_parameters():
                if "div_head" not in name:
                    param.requires_grad_(False)
            logger_obj.info(
                "  freeze_velocity=True: backbone and velocity_head are frozen. "
                "Only div_head will be trained."
            )
        ema_callback = MeanFlowEMACallback(ema_beta=ema_beta)
    else:
        model = ShortcutProposal.load_from_checkpoint(
            shortcut_ckpt,
            strict=False,
            training_stage="f2d2",
            teacher_ckpt_path=teacher_ckpt,
            teacher_div_estimator=teacher_div_estimator,
            div_scale=div_scale,
            learning_rate=learning_rate,
            denoise_timesteps=denoise_timesteps,
            lr_warmup_steps=lr_warmup_steps,
            debug_random_obs=debug_random_obs,
            debug_random_prev_state=debug_random_prev_state,
        )
        model.shortcut_net.div_head_active = True
        model.training_stage = "f2d2"
        ema_callback = ShortcutEMACallback(ema_beta=ema_beta)

    logger_obj.info(
        f"  Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="f2d2-{epoch:03d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=30, mode="min", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [ema_callback, checkpoint_callback, early_stop, lr_monitor]

    if save_every_n_epochs is not None and save_every_n_epochs > 0:
        periodic_ckpt = ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="f2d2-periodic-{epoch:03d}",
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

    logger_obj.info("Starting Stage 3 (divergence) training …")
    trainer.fit(model, data_module)
    logger_obj.info("Stage 3 training completed.")
    logger_obj.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    final_path = output_dir / "f2d2_final.ckpt"
    trainer.save_checkpoint(final_path)

    return model, checkpoint_callback.best_model_path, wandb_logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Train divergence head (F2D2 likelihood) from Stage 2 shortcut or MeanFlow ckpt"
    )

    # Required
    parser.add_argument("--shortcut_ckpt", type=str, required=True,
                        help="Path to Stage 2 ShortcutProposal or MeanFlowProposal checkpoint")
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="Path to frozen teacher RFProposal checkpoint (for divergence targets)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    # Architecture (should match Stage 2)
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--obs_dim", type=int, default=0)
    parser.add_argument("--architecture", type=str, default="mlp", choices=["mlp", "resnet1d"])
    parser.add_argument("--use_observations", action="store_true")
    parser.add_argument("--obs_components", type=str, default=None)
    parser.add_argument("--cond_dropout", type=float, default=0.0)

    # Stage 3 specific
    parser.add_argument("--teacher_div_estimator", type=str, default="exact",
                        choices=["exact", "hutchinson"],
                        help="Method for estimating teacher divergence: 'exact' (sum of diagonal Jacobian, "
                             "cheap for D≤40) or 'hutchinson' (stochastic, scalable to larger D)")
    parser.add_argument("--div_scale", type=float, default=1.0,
                        help="Scale applied to divergence targets before MSE. Use 1.0 for low-D data; "
                             "F2D2 uses 1/20000 for high-D image data.")
    parser.add_argument("--denoise_timesteps", type=int, default=1024)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--ema_beta", type=float, default=0.9999)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--grad_clip_val", type=float, default=0.01)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="rf-f2d2")
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--debug_random_obs", action="store_true")
    parser.add_argument("--debug_random_prev_state", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--freeze_velocity",
        action="store_true",
        default=True,
        help="(MeanFlow only) Freeze backbone+velocity_head; only train div_head. "
             "Recommended — prevents Stage 3 from erasing Stage 2 coarse-step structure.",
    )
    parser.add_argument(
        "--no_freeze_velocity",
        dest="freeze_velocity",
        action="store_false",
        help="Disable velocity freezing (trains full network, as in Shortcut F2D2).",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="After training, run autoregressive eval (eval_proposal); logs to the same W&B run",
    )
    parser.add_argument(
        "--no_wandb_eval",
        action="store_true",
        help="With --evaluate only when no training W&B run exists, skip starting eval W&B (unused after training)",
    )
    parser.add_argument(
        "--eval_run_name",
        type=str,
        default=None,
        help="W&B run name for eval if a new run must be started (default: derived from checkpoint path)",
    )
    parser.add_argument(
        "--eval_num_sampling_steps",
        type=int,
        default=None,
        help="Override Euler steps for autoregressive sampling in --evaluate (Shortcut: default from checkpoint)",
    )
    parser.add_argument(
        "--eval_num_likelihood_steps",
        type=int,
        default=None,
        help="Override likelihood integration steps in --evaluate (default: checkpoint)",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/data/da_outputs/f2d2_runs/run_{ts}"

    obs_components = None
    if args.obs_components is not None:
        obs_components = [int(i) for i in args.obs_components.split(",") if i.strip()]
    if obs_components is not None:
        args.obs_dim = len(obs_components)

    config_path = Path(args.data_dir) / "config.yaml"
    if config_path.exists():
        try:
            from data import load_config_yaml
            config = load_config_yaml(config_path)
            if obs_components is None:
                obs_components = getattr(config, "obs_components", None)
                if obs_components is not None:
                    args.obs_dim = len(obs_components)
        except Exception:
            pass

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    _, best_checkpoint, wandb_logger = train_divergence(
        shortcut_ckpt=args.shortcut_ckpt,
        teacher_ckpt=args.teacher_ckpt,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        state_dim=args.state_dim,
        obs_dim=args.obs_dim,
        architecture=args.architecture,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        use_observations=args.use_observations,
        obs_indices=obs_components,
        cond_dropout=args.cond_dropout,
        gpus=args.gpus,
        denoise_timesteps=args.denoise_timesteps,
        ema_beta=args.ema_beta,
        lr_warmup_steps=args.lr_warmup_steps,
        grad_clip_val=args.grad_clip_val,
        teacher_div_estimator=args.teacher_div_estimator,
        div_scale=args.div_scale,
        wandb_project=args.wandb_project,
        save_every_n_epochs=args.save_every_n_epochs,
        debug_random_obs=args.debug_random_obs,
        debug_random_prev_state=args.debug_random_prev_state,
        freeze_velocity=args.freeze_velocity,
    )

    if args.evaluate:
        import wandb

        out = Path(args.output_dir)
        checkpoint_to_eval = best_checkpoint
        if not checkpoint_to_eval:
            last_ckpt = out / "checkpoints" / "last.ckpt"
            if last_ckpt.is_file():
                checkpoint_to_eval = str(last_ckpt)
            else:
                final_ckpt = out / "f2d2_final.ckpt"
                checkpoint_to_eval = str(final_ckpt) if final_ckpt.is_file() else None

        if not checkpoint_to_eval:
            raise RuntimeError("--evaluate set but no checkpoint path (best/last/final missing)")

        wandb_run = None
        wandb_started_for_eval_only = False
        if wandb_logger is not None and hasattr(wandb_logger, "experiment"):
            wandb_run = wandb_logger.experiment
        elif not args.no_wandb_eval:
            ckpt = Path(checkpoint_to_eval).resolve()
            try:
                wandb_dir = ckpt.parents[1] / "wandb"
            except IndexError:
                wandb_dir = Path(".") / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            run_name = args.eval_run_name or _derive_eval_run_name(str(ckpt))
            wandb_run = wandb.init(
                entity="ml-climate",
                project=args.wandb_project,
                name=run_name,
                dir=str(wandb_dir),
            )
            wandb_started_for_eval_only = True

        try:
            run_proposal_eval(
                checkpoint_path=checkpoint_to_eval,
                data_dir=args.data_dir,
                n_trajectories=None,
                n_vis_trajectories=10,
                batch_size=args.batch_size,
                device="cuda" if (args.gpus > 0 and torch.cuda.is_available()) else "cpu",
                wandb_run=wandb_run,
                num_sampling_steps=args.eval_num_sampling_steps,
                num_likelihood_steps=args.eval_num_likelihood_steps,
            )
        finally:
            if wandb_started_for_eval_only:
                wandb.finish()


if __name__ == "__main__":
    main()
