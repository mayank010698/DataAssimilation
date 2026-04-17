"""
Training driver for the Localized Rectified Flow Proposal.

Patches are assembled on the fly by `PatchDataModule` (Option A): each dataset
item corresponds to one site of one transition. The localized velocity network
is shared across all spatial locations, so training on L96-40 transfers
zero-shot to larger grids (e.g. L96-400).
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

# Support both package and script execution
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))
    sys.path.append(str(Path(__file__).parent.parent))

try:
    from .localized_rf import LocalizedRFProposal
    from .patch_dataset import PatchDataModule
    from .patch_utils import WindowSpec
    from .eval_proposal import run_proposal_eval
except ImportError:
    from localized_rf import LocalizedRFProposal
    from patch_dataset import PatchDataModule
    from patch_utils import WindowSpec
    try:
        from eval_proposal import run_proposal_eval
    except ImportError:
        from proposals.eval_proposal import run_proposal_eval


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def train_localized_rf(
    data_dir: str,
    output_dir: str,
    radius: int,
    state_dim: int,
    architecture: str = "local_mlp",
    hidden_dim: int = 128,
    depth: int = 4,
    channels: int = 32,
    num_blocks: int = 2,
    kernel_size: int = 3,
    time_embed_dim: int = 64,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 500,
    num_workers: int = 4,
    use_observations: bool = True,
    obs_components: list = None,
    predict_delta: bool = False,
    use_time_step: bool = False,
    trajectory_length: int = 1000,
    num_sampling_steps: int = 25,
    num_likelihood_steps: int = 25,
    gpus: int = 1,
    wandb_project: str = "lrf-train-96",
    wandb_run_name: str = None,
    save_every_n_epochs: int = None,
    train_fraction: float = 1.0,
    early_stopping_patience: int = 50,
    obs_dropout: float = 0.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(output_dir)

    log.info("=" * 80)
    log.info("Training Localized Rectified Flow Proposal")
    log.info("=" * 80)
    log.info(f"data_dir: {data_dir}")
    log.info(f"output_dir: {output_dir}")
    log.info(f"radius: {radius}  (window_size = {2*radius+1})")
    log.info(f"state_dim: {state_dim}")
    log.info(f"architecture: {architecture}")
    log.info(f"use_observations: {use_observations}, obs_components: {obs_components}")
    log.info(f"predict_delta: {predict_delta}, use_time_step: {use_time_step}")

    window_spec = WindowSpec(radius=radius, stride=1, periodic=True)

    data_module = PatchDataModule(
        data_dir=data_dir,
        radius=radius,
        batch_size=batch_size,
        num_workers=num_workers,
        use_observations=use_observations,
        obs_components=obs_components,
        window_spec=window_spec,
        train_fraction=train_fraction,
    )
    data_module.setup("fit")

    arch_kwargs = dict(
        time_embed_dim=time_embed_dim,
    )
    if architecture == "local_mlp":
        arch_kwargs.update(hidden_dim=hidden_dim, depth=depth)
    elif architecture == "local_resnet1d":
        arch_kwargs.update(
            channels=channels, num_blocks=num_blocks, kernel_size=kernel_size
        )

    model = LocalizedRFProposal(
        radius=radius,
        architecture=architecture,
        state_dim=state_dim,
        use_observations=use_observations,
        obs_components=obs_components,
        predict_delta=predict_delta,
        num_sampling_steps=num_sampling_steps,
        num_likelihood_steps=num_likelihood_steps,
        use_time_step=use_time_step,
        trajectory_length=trajectory_length,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        obs_dropout=obs_dropout,
        **arch_kwargs,
    )
    log.info(
        f"Model params: {sum(p.numel() for p in model.parameters()):,} total, "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="lrf-{epoch:03d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        mode="min",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, early_stop, lr_monitor]
    if save_every_n_epochs:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename="lrf-periodic-{epoch:03d}",
                every_n_epochs=save_every_n_epochs,
                save_top_k=-1,
                save_last=False,
                save_on_train_epoch_end=True,
            )
        )

    wandb_logger = WandbLogger(
        entity="ml-climate",
        project=wandb_project,
        name=wandb_run_name if wandb_run_name else output_dir.name,
        save_dir=str(output_dir),
    )
    wandb_logger.log_hyperparams(
        dict(
            radius=radius,
            state_dim=state_dim,
            architecture=architecture,
            hidden_dim=hidden_dim,
            depth=depth,
            channels=channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            time_embed_dim=time_embed_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            use_observations=use_observations,
            obs_components=obs_components,
            predict_delta=predict_delta,
            use_time_step=use_time_step,
            trajectory_length=trajectory_length,
            num_sampling_steps=num_sampling_steps,
            num_likelihood_steps=num_likelihood_steps,
            data_dir=str(data_dir),
            train_fraction=train_fraction,
            obs_dropout=obs_dropout,
        )
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 and torch.cuda.is_available() else "cpu",
        devices=gpus if gpus > 0 and torch.cuda.is_available() else 1,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
        precision=32,
    )

    log.info("Starting training...")
    trainer.fit(model, data_module)
    log.info(f"Best model: {checkpoint_callback.best_model_path}")
    log.info(f"Best val_loss: {checkpoint_callback.best_model_score}")

    final_path = output_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_path)
    log.info(f"Final checkpoint: {final_path}")
    return model, checkpoint_callback.best_model_path, wandb_logger


def main():
    parser = argparse.ArgumentParser("Train Localized Rectified Flow Proposal")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--state_dim", type=int, required=True)
    parser.add_argument(
        "--architecture", type=str, default="local_mlp",
        choices=["local_mlp", "local_resnet1d"],
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--time_embed_dim", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--use_observations", action="store_true")
    parser.add_argument(
        "--obs_components", type=str, default=None,
        help="Comma-separated indices of observed variables",
    )
    parser.add_argument("--predict_delta", action="store_true")
    parser.add_argument("--use_time_step", action="store_true")
    parser.add_argument("--trajectory_length", type=int, default=None)

    parser.add_argument("--num_sampling_steps", type=int, default=25)
    parser.add_argument("--num_likelihood_steps", type=int, default=25)

    parser.add_argument("--wandb_project", type=str, default="lrf-train-96")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Custom wandb run name. Defaults to the output_dir basename.",
    )
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=50,
        help="Number of epochs with no val_loss improvement before stopping.",
    )
    parser.add_argument(
        "--obs_dropout",
        type=float,
        default=0.0,
        help=(
            "Per-sample probability of dropping the observation window during "
            "training (zero values + zero mask). Analogous to classifier-free "
            "guidance dropout in train_rf.py."
        ),
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of training trajectories to use, in (0, 1]. "
            "E.g. 0.5 trains on the first half of the train split. "
            "Validation and test splits are unaffected."
        ),
    )

    # Post-training autoregressive evaluation (logged to the same wandb run)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "After training, run autoregressive proposal evaluation "
            "(proposals.eval_proposal.run_proposal_eval) on the best checkpoint "
            "and log eval/* metrics + trajectory plots to the same wandb run."
        ),
    )
    parser.add_argument(
        "--eval_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional explicit checkpoint to evaluate. If unset and --evaluate is "
            "passed, uses the best checkpoint from training."
        ),
    )
    parser.add_argument("--eval_n_trajectories", type=int, default=None,
                        help="Number of test trajectories for eval (None = all).")
    parser.add_argument("--eval_n_vis_trajectories", type=int, default=10,
                        help="How many trajectories to visualize in wandb.")
    parser.add_argument("--eval_n_samples_per_traj", type=int, default=20,
                        help="Ensemble size per trajectory for CRPS / spread.")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for autoregressive eval.")
    parser.add_argument("--eval_num_sampling_steps", type=int, default=None,
                        help="Override Euler steps for sampling at eval time.")
    parser.add_argument("--eval_num_likelihood_steps", type=int, default=None,
                        help="Override Euler steps for likelihood at eval time.")
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/data/da_outputs/lrf_runs/run_{ts}"

    # Parse obs_components
    obs_components = None
    if args.obs_components is not None:
        obs_components = [int(i) for i in args.obs_components.split(",") if i.strip()]
    else:
        # Try loading from config.yaml (matches train_rf behaviour)
        config_path = Path(args.data_dir) / "config.yaml"
        if config_path.exists():
            try:
                from data import load_config_yaml
                config = load_config_yaml(config_path)
                obs_components = config.obs_components
                if args.trajectory_length is None:
                    args.trajectory_length = config.len_trajectory
            except Exception:
                pass

    if args.trajectory_length is None:
        args.trajectory_length = 1000

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    model, best_checkpoint, wandb_logger = train_localized_rf(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        radius=args.radius,
        state_dim=args.state_dim,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        channels=args.channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        time_embed_dim=args.time_embed_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        use_observations=args.use_observations,
        obs_components=obs_components,
        predict_delta=args.predict_delta,
        use_time_step=args.use_time_step,
        trajectory_length=args.trajectory_length,
        num_sampling_steps=args.num_sampling_steps,
        num_likelihood_steps=args.num_likelihood_steps,
        gpus=args.gpus,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_every_n_epochs=args.save_every_n_epochs,
        train_fraction=args.train_fraction,
        early_stopping_patience=args.early_stopping_patience,
        obs_dropout=args.obs_dropout,
    )

    if args.evaluate:
        checkpoint_to_eval = args.eval_checkpoint or best_checkpoint
        if not checkpoint_to_eval:
            logging.getLogger(__name__).warning(
                "--evaluate set but no checkpoint available (training produced no "
                "best_model_path and --eval_checkpoint is unset). Skipping eval."
            )
        else:
            wandb_run = None
            if wandb_logger is not None and hasattr(wandb_logger, "experiment"):
                wandb_run = wandb_logger.experiment

            logging.getLogger(__name__).info(
                f"Running post-training autoregressive eval on {checkpoint_to_eval}"
            )
            run_proposal_eval(
                checkpoint_path=checkpoint_to_eval,
                data_dir=args.data_dir,
                n_trajectories=args.eval_n_trajectories,
                n_vis_trajectories=args.eval_n_vis_trajectories,
                batch_size=args.eval_batch_size,
                n_samples_per_traj=args.eval_n_samples_per_traj,
                device="cuda" if (args.gpus > 0 and torch.cuda.is_available()) else "cpu",
                wandb_run=wandb_run,
                num_sampling_steps=args.eval_num_sampling_steps,
                num_likelihood_steps=args.eval_num_likelihood_steps,
            )


if __name__ == "__main__":
    main()
