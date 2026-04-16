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
except ImportError:
    from localized_rf import LocalizedRFProposal
    from patch_dataset import PatchDataModule
    from patch_utils import WindowSpec


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
    save_every_n_epochs: int = None,
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
    early_stop = EarlyStopping(monitor="val_loss", patience=50, mode="min", verbose=True)
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
        name=output_dir.name,
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
    return model, checkpoint_callback.best_model_path


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
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
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

    train_localized_rf(
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
        save_every_n_epochs=args.save_every_n_epochs,
    )


if __name__ == "__main__":
    main()
