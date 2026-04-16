"""
Stage 2: Shortcut Distillation Training

Distils a frozen teacher RFProposal (from train_rf.py or train_fm.py) into a
ShortcutProposal using the shortcut self-consistency loss.

After training you can proceed to Stage 3 (train_div.py) using the best checkpoint.

Usage:
    python proposals/train_sc.py \
        --teacher_ckpt /path/to/teacher.ckpt \
        --data_dir /path/to/data \
        --output_dir /path/to/output \
        --state_dim 40 --obs_dim 20 --architecture resnet1d \
        --denoise_timesteps 1024 --max_epochs 200 \
        --evaluate

    Eval-only (same as train_rf.py --checkpoint pattern):
    python proposals/train_sc.py --data_dir /path/to/data \
        --checkpoint /path/to/sc.ckpt --evaluate \
        --eval_num_sampling_steps 4
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
from rf_dataset import RFDataModule

try:
    from eval_proposal import run_proposal_eval, _derive_eval_run_name
except ImportError:
    from proposals.eval_proposal import run_proposal_eval, _derive_eval_run_name


def setup_logging(log_dir: Path):
    log_file = log_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def train_shortcut(
    teacher_ckpt: str,
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
    time_embed_dim: int = 64,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    max_epochs: int = 300,
    num_workers: int = 4,
    use_observations: bool = False,
    obs_indices: Optional[list] = None,
    cond_dropout: float = 0.0,
    gpus: int = 1,
    predict_delta: bool = False,
    use_time_step: bool = False,
    trajectory_length: int = 1000,
    denoise_timesteps: int = 1024,
    ema_beta: float = 0.9999,
    lr_warmup_steps: int = 0,
    grad_clip_val: float = 0.01,
    wandb_project: str = "rf-shortcut",
    save_every_n_epochs: Optional[int] = None,
    debug_random_obs: bool = False,
    debug_random_prev_state: bool = False,
    prev_state_corr_p0: float = 0.0,
    prev_state_corr_p_min: float = 0.05,
    prev_state_corr_sigma: float = 0.0,
    prev_state_corr_mask_ratio: float = 0.0,
    div_hidden_dim: int = 64,
    n_sampling_steps: int = 1,
    n_likelihood_steps: int = 4,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not use_observations:
        obs_dim = 0

    logger_obj = setup_logging(output_dir)
    logger_obj.info("=" * 80)
    logger_obj.info("Stage 2: Shortcut Distillation")
    logger_obj.info("=" * 80)
    logger_obj.info(f"  Teacher:         {teacher_ckpt}")
    logger_obj.info(f"  Architecture:    {architecture}, state_dim={state_dim}, obs_dim={obs_dim}")
    logger_obj.info(f"  denoise_timesteps: {denoise_timesteps}")
    logger_obj.info(f"  EMA beta:        {ema_beta}")
    logger_obj.info(f"  LR warmup steps: {lr_warmup_steps}")

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

    model = ShortcutProposal(
        state_dim=state_dim,
        teacher_ckpt_path=teacher_ckpt,
        training_stage="sc",
        architecture=architecture,
        hidden_dim=hidden_dim,
        depth=depth,
        channels=channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        time_embed_dim=time_embed_dim,
        obs_dim=obs_dim,
        obs_indices=obs_indices,
        predict_delta=predict_delta,
        use_time_step=use_time_step,
        trajectory_length=trajectory_length,
        cond_dropout=cond_dropout,
        debug_random_obs=debug_random_obs,
        debug_random_prev_state=debug_random_prev_state,
        prev_state_corr_p0=prev_state_corr_p0,
        prev_state_corr_p_min=prev_state_corr_p_min,
        prev_state_corr_total_steps=total_steps,
        prev_state_corr_sigma=prev_state_corr_sigma,
        prev_state_corr_mask_ratio=prev_state_corr_mask_ratio,
        learning_rate=learning_rate,
        denoise_timesteps=denoise_timesteps,
        lr_warmup_steps=lr_warmup_steps,
        n_sampling_steps=n_sampling_steps,
        n_likelihood_steps=n_likelihood_steps,
        div_hidden_dim=div_hidden_dim,
    )

    logger_obj.info(
        f"  Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    ema_callback = ShortcutEMACallback(ema_beta=ema_beta)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="sc-{epoch:03d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=40, mode="min", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [ema_callback, checkpoint_callback, early_stop, lr_monitor]

    if save_every_n_epochs is not None and save_every_n_epochs > 0:
        periodic_ckpt = ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="sc-periodic-{epoch:03d}",
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

    logger_obj.info("Starting Stage 2 (shortcut) training …")
    trainer.fit(model, data_module)
    logger_obj.info("Stage 2 training completed.")
    logger_obj.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    final_path = output_dir / "sc_final.ckpt"
    trainer.save_checkpoint(final_path)

    return model, checkpoint_callback.best_model_path, wandb_logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Shortcut distillation from a flow-matching teacher"
    )

    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=None,
        help="Path to frozen teacher RFProposal checkpoint (required for training; omit if only --checkpoint + --evaluate)",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    # Architecture (must match teacher)
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--obs_dim", type=int, default=0)
    parser.add_argument("--architecture", type=str, default="mlp", choices=["mlp", "resnet1d"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--time_embed_dim", type=int, default=64)
    parser.add_argument("--predict_delta", action="store_true")
    parser.add_argument("--use_time_step", action="store_true")
    parser.add_argument("--trajectory_length", type=int, default=1000)
    parser.add_argument("--div_hidden_dim", type=int, default=64)

    # Conditioning
    parser.add_argument("--use_observations", action="store_true")
    parser.add_argument("--obs_components", type=str, default=None,
                        help="Comma-separated observed state indices, e.g. '0,2,4'")
    parser.add_argument("--cond_dropout", type=float, default=0.0)

    # Shortcut-specific
    parser.add_argument("--denoise_timesteps", type=int, default=1024,
                        help="Number of discrete time steps for uniform t-sampling")
    parser.add_argument("--ema_beta", type=float, default=0.9999)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--grad_clip_val", type=float, default=0.01)

    # Inference defaults saved into checkpoint
    parser.add_argument("--n_sampling_steps", type=int, default=1)
    parser.add_argument("--n_likelihood_steps", type=int, default=4)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)

    # Prev-state corruption
    parser.add_argument("--prev_state_corr_p0", type=float, default=0.0)
    parser.add_argument("--prev_state_corr_p_min", type=float, default=0.05)
    parser.add_argument("--prev_state_corr_sigma", type=float, default=0.0)
    parser.add_argument("--prev_state_corr_mask_ratio", type=float, default=0.0)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="rf-shortcut")
    parser.add_argument(
        "--no_wandb_eval",
        action="store_true",
        help="With --checkpoint --evaluate, skip starting a new W&B run for eval-only (no effect after training)",
    )
    parser.add_argument(
        "--eval_run_name",
        type=str,
        default=None,
        help="W&B run name for eval-only mode; default derives from checkpoint path",
    )
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--debug_random_obs", action="store_true")
    parser.add_argument("--debug_random_prev_state", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run autoregressive eval (eval_proposal): after training, logs to the same W&B run; with --checkpoint only, starts a new W&B run unless --no_wandb_eval",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Skip training and only run --evaluate on this shortcut checkpoint (same pattern as train_rf.py)",
    )
    parser.add_argument(
        "--eval_num_sampling_steps",
        type=int,
        default=None,
        help="Override Euler steps for autoregressive sampling in --evaluate (Shortcut: matches oracle n_steps; default: checkpoint)",
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
        args.output_dir = f"/data/da_outputs/sc_runs/run_{ts}"

    # Parse obs_components
    obs_components = None
    if args.obs_components is not None:
        obs_components = [int(i) for i in args.obs_components.split(",") if i.strip()]
    if obs_components is not None:
        args.obs_dim = len(obs_components)

    # Try loading trajectory_length from config
    config_path = Path(args.data_dir) / "config.yaml"
    if config_path.exists():
        try:
            from data import load_config_yaml
            config = load_config_yaml(config_path)
            if args.trajectory_length == 1000:
                args.trajectory_length = getattr(config, "len_trajectory", 1000)
            if obs_components is None:
                obs_components = getattr(config, "obs_components", None)
                if obs_components is not None:
                    args.obs_dim = len(obs_components)
        except Exception:
            pass

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    if args.checkpoint is None and not args.teacher_ckpt:
        parser.error("--teacher_ckpt is required when training (omit only for eval-only: --checkpoint with --evaluate)")

    wandb_logger = None
    if args.checkpoint is None:
        _, best_checkpoint, wandb_logger = train_shortcut(
            teacher_ckpt=args.teacher_ckpt,
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
            time_embed_dim=args.time_embed_dim,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            num_workers=args.num_workers,
            use_observations=args.use_observations,
            obs_indices=obs_components,
            cond_dropout=args.cond_dropout,
            gpus=args.gpus,
            predict_delta=args.predict_delta,
            use_time_step=args.use_time_step,
            trajectory_length=args.trajectory_length,
            denoise_timesteps=args.denoise_timesteps,
            ema_beta=args.ema_beta,
            lr_warmup_steps=args.lr_warmup_steps,
            grad_clip_val=args.grad_clip_val,
            wandb_project=args.wandb_project,
            save_every_n_epochs=args.save_every_n_epochs,
            debug_random_obs=args.debug_random_obs,
            debug_random_prev_state=args.debug_random_prev_state,
            prev_state_corr_p0=args.prev_state_corr_p0,
            prev_state_corr_p_min=args.prev_state_corr_p_min,
            prev_state_corr_sigma=args.prev_state_corr_sigma,
            prev_state_corr_mask_ratio=args.prev_state_corr_mask_ratio,
            div_hidden_dim=args.div_hidden_dim,
            n_sampling_steps=args.n_sampling_steps,
            n_likelihood_steps=args.n_likelihood_steps,
        )
        checkpoint_to_eval = best_checkpoint
    else:
        checkpoint_to_eval = args.checkpoint

    if args.evaluate:
        import wandb

        wandb_run = None
        wandb_started_for_eval_only = False
        if wandb_logger is not None and hasattr(wandb_logger, "experiment"):
            wandb_run = wandb_logger.experiment
        elif args.checkpoint is not None and not args.no_wandb_eval:
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
