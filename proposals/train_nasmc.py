"""CLI entrypoint for training the NASMC :class:`GaussianProposal`.

The script supports the two-phase training recipe from
``nasmc_plan.md``: phase 1 runs local MLE on
:class:`RFTransitionDataset` pairs (this is the warm-start that
stabilises NASMC); phase 2 runs the NASMC weighted log-density loss on
subtrajectories from :class:`NASMCTrajectoryDataset`.

Example:

    python -m proposals.train_nasmc \\
        --data_dir /path/to/dw_dataset \\
        --output_dir /tmp/nasmc_run \\
        --state_dim 1 --obs_dim 1 \\
        --architecture mlp --hidden_dim 128 --depth 4 \\
        --use_observations \\
        --max_epochs_pretrain 30 --max_epochs_refine 30 \\
        --num_particles 32 --segment_length 64

Only phase 1 is run if ``--max_epochs_refine == 0``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# Add local + parent dirs to path so we can run as a script too.
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))
    sys.path.append(str(Path(__file__).parent.parent))

try:
    from .nasmc import GaussianProposal
    from .nasmc_dataset import NASMCDataModule
except ImportError:  # pragma: no cover
    from nasmc import GaussianProposal  # type: ignore
    from nasmc_dataset import NASMCDataModule  # type: ignore


logger = logging.getLogger(__name__)


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _load_scalers_and_system(data_dir: Path, obs_components: Optional[List[int]] = None):
    """Load state/obs scalers, system, and dt from the data directory."""
    import h5py
    import numpy as np

    data_file = data_dir / "data_scaled.h5"
    if not data_file.exists():
        data_file = data_dir / "data.h5"

    with h5py.File(data_file, "r") as f:
        state_mean = torch.from_numpy(f["scaler_mean"][:]).float()
        state_std = torch.from_numpy(f["scaler_std"][:]).float()
        obs_mean = None
        obs_std = None
        if "obs_scaler_mean" in f:
            obs_mean_full = f["obs_scaler_mean"][:]
            obs_std_full = f["obs_scaler_std"][:]
            if obs_components is not None and len(obs_mean_full) >= len(obs_components):
                obs_mean = torch.from_numpy(obs_mean_full[obs_components]).float()
                obs_std = torch.from_numpy(obs_std_full[obs_components]).float()
            else:
                obs_mean = torch.from_numpy(np.asarray(obs_mean_full)).float()
                obs_std = torch.from_numpy(np.asarray(obs_std_full)).float()

    system = None
    dt = None
    config_path = data_dir / "config.yaml"
    if config_path.exists():
        try:
            from data import (
                DoubleWell,
                Lorenz63,
                Lorenz96,
                KuramotoSivashinsky,
                load_config_yaml,
            )

            cfg = load_config_yaml(config_path)
            if "J" in cfg.system_params:
                system_class = KuramotoSivashinsky
            elif "dim" in cfg.system_params or "F" in cfg.system_params:
                system_class = Lorenz96
            elif cfg.system_params.get("system_name") == "double_well":
                system_class = DoubleWell
            else:
                system_class = Lorenz63
            system = system_class(cfg)
            dt = cfg.dt
            # Ensure the system has scalers attached for preprocess/postprocess.
            system.init_mean = state_mean
            system.init_std = state_std
        except Exception as exc:
            logger.warning(
                "Could not construct DynamicalSystem from %s: %s", config_path, exc,
            )

    return state_mean, state_std, obs_mean, obs_std, system, dt


def _make_callbacks(output_dir: Path, monitor: str, phase_tag: str, patience: int = 50):
    ckpt = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=f"nasmc-{phase_tag}-{{epoch:03d}}-{{{monitor}:.6f}}",
        monitor=monitor,
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(monitor=monitor, patience=patience, mode="min", verbose=False)
    lrm = LearningRateMonitor(logging_interval="epoch")
    return [ckpt, early, lrm], ckpt


def train_nasmc(
    data_dir: str,
    output_dir: str,
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
    predict_delta: bool = True,
    use_dynamics_mean: bool = False,
    use_time_step: bool = False,
    trajectory_length: int = 1000,
    init_log_sigma: float = -1.0,
    log_sigma_min: float = -7.0,
    log_sigma_max: float = 3.0,
    use_observations: bool = True,
    obs_components: Optional[List[int]] = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs_pretrain: int = 30,
    max_epochs_refine: int = 30,
    num_particles: int = 32,
    segment_length: int = 64,
    resample_threshold: float = 0.5,
    use_bootstrap_proposal: bool = False,
    num_workers: int = 4,
    gpus: int = 0,
    seed: Optional[int] = None,
    process_noise_std: float = 0.01,
    obs_noise_std: float = 0.1,
    obs_nonlinearity: str = "arctan",
    pretrain_checkpoint: Optional[str] = None,
):
    """Train the NASMC proposal end-to-end (phase 1 → phase 2)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(output_dir)

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    log.info("=" * 80)
    log.info("NASMC training")
    log.info("=" * 80)
    log.info("data_dir        = %s", data_dir)
    log.info("output_dir      = %s", output_dir)
    log.info("state_dim       = %d, obs_dim = %d", state_dim, obs_dim)
    log.info("architecture    = %s", architecture)
    log.info("predict_delta   = %s, use_dynamics_mean = %s", predict_delta, use_dynamics_mean)
    log.info("use_time_step   = %s, trajectory_length = %d", use_time_step, trajectory_length)
    log.info("num_particles   = %d, segment_length = %d", num_particles, segment_length)
    log.info("phase 1 epochs  = %d, phase 2 epochs = %d", max_epochs_pretrain, max_epochs_refine)

    data_dir_path = Path(data_dir)
    state_mean, state_std, obs_mean, obs_std, system, dt = _load_scalers_and_system(
        data_dir_path, obs_components=obs_components,
    )

    # Build the proposal model.
    model = GaussianProposal(
        state_dim=state_dim,
        obs_dim=obs_dim if use_observations else 0,
        architecture=architecture,
        hidden_dim=hidden_dim,
        depth=depth,
        channels=channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        time_embed_dim=time_embed_dim,
        dropout=dropout,
        obs_indices=obs_components,
        use_time_step=use_time_step,
        trajectory_length=trajectory_length,
        predict_delta=predict_delta,
        use_dynamics_mean=use_dynamics_mean,
        log_sigma_min=log_sigma_min,
        log_sigma_max=log_sigma_max,
        init_log_sigma=init_log_sigma,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        process_noise_std=process_noise_std,
        obs_noise_std=obs_noise_std,
        obs_nonlinearity=obs_nonlinearity,
        state_scaler_mean=state_mean,
        state_scaler_std=state_std,
        obs_scaler_mean=obs_mean,
        obs_scaler_std=obs_std,
    )
    log.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    if pretrain_checkpoint is not None:
        log.info("Loading pretrain checkpoint from %s", pretrain_checkpoint)
        ck = torch.load(pretrain_checkpoint, map_location="cpu")
        # Accept either a full Lightning checkpoint or just a state_dict.
        state = ck.get("state_dict", ck)
        model.load_state_dict(state, strict=False)

    if use_dynamics_mean or max_epochs_refine > 0:
        if system is None:
            raise RuntimeError(
                "Phase 2 (and the -f- variant) require a DynamicalSystem, but "
                "none could be constructed from the data directory's config.yaml."
            )
        model.attach_system(system, dt=dt)

    accelerator = "gpu" if gpus > 0 and torch.cuda.is_available() else "cpu"
    devices = gpus if accelerator == "gpu" else 1

    best_pretrain_ckpt = None

    # -------------------- Phase 1: local-MLE pretrain ----------------------
    if max_epochs_pretrain > 0 and pretrain_checkpoint is None:
        log.info("Phase 1: local-MLE pretraining (%d epochs)", max_epochs_pretrain)
        model.set_phase("pretrain")
        data_module = NASMCDataModule(
            data_dir=str(data_dir_path),
            phase="pretrain",
            segment_length=segment_length,
            batch_size=batch_size,
            num_workers=num_workers,
            use_observations=use_observations,
            obs_components=obs_components,
        )
        callbacks, ckpt = _make_callbacks(
            output_dir / "pretrain", monitor="val_nll", phase_tag="pretrain"
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs_pretrain,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            precision=32,
            default_root_dir=str(output_dir / "pretrain"),
            enable_progress_bar=True,
        )
        trainer.fit(model, data_module)
        best_pretrain_ckpt = ckpt.best_model_path or None
        log.info("Phase 1 complete. Best checkpoint: %s", best_pretrain_ckpt)

    # -------------------- Phase 2: NASMC weighted refine --------------------
    best_refine_ckpt = None
    if max_epochs_refine > 0:
        log.info("Phase 2: NASMC weighted log-density refinement (%d epochs)", max_epochs_refine)
        model.set_phase(
            "refine",
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            use_bootstrap=use_bootstrap_proposal,
        )
        data_module = NASMCDataModule(
            data_dir=str(data_dir_path),
            phase="refine",
            segment_length=segment_length,
            batch_size=batch_size,
            num_workers=num_workers,
            use_observations=use_observations,
            obs_components=obs_components,
        )
        callbacks, ckpt = _make_callbacks(
            output_dir / "refine", monitor="val_nll", phase_tag="refine"
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs_refine,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            log_every_n_steps=10,
            gradient_clip_val=5.0,  # NASMC gradients are noisier; tighter clip.
            precision=32,
            default_root_dir=str(output_dir / "refine"),
            enable_progress_bar=True,
        )
        trainer.fit(model, data_module)
        best_refine_ckpt = ckpt.best_model_path or None
        log.info("Phase 2 complete. Best checkpoint: %s", best_refine_ckpt)

    # Save a final canonical checkpoint for evaluation. Use the most-recent
    # trainer if available (it has hparams wired up); otherwise fall back to
    # a fresh trainer bound to the model.
    final_ckpt = output_dir / "final_model.ckpt"
    trainer_for_save = trainer if "trainer" in locals() else pl.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
    )
    try:
        trainer_for_save.save_checkpoint(str(final_ckpt))
    except Exception as exc:
        log.warning("Falling back to torch.save for final checkpoint: %s", exc)
        torch.save({"state_dict": model.state_dict()}, final_ckpt)
    log.info("Final model saved to %s", final_ckpt)

    return {
        "model": model,
        "pretrain_checkpoint": best_pretrain_ckpt,
        "refine_checkpoint": best_refine_ckpt,
        "final_checkpoint": str(final_ckpt),
    }


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train NASMC Gaussian proposal")

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--obs_dim", type=int, default=1)
    parser.add_argument("--architecture", choices=["mlp", "resnet1d"], default="mlp")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--time_embed_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--predict_delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_dynamics_mean", action="store_true")
    parser.add_argument("--use_time_step", action="store_true")
    parser.add_argument("--trajectory_length", type=int, default=None)
    parser.add_argument("--init_log_sigma", type=float, default=-1.0)
    parser.add_argument("--log_sigma_min", type=float, default=-7.0)
    parser.add_argument("--log_sigma_max", type=float, default=3.0)

    parser.add_argument("--use_observations", action="store_true")
    parser.add_argument("--obs_components", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--max_epochs_pretrain", type=int, default=30)
    parser.add_argument("--max_epochs_refine", type=int, default=30)

    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--segment_length", type=int, default=64)
    parser.add_argument("--resample_threshold", type=float, default=0.5)
    parser.add_argument("--use_bootstrap_proposal", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--pretrain_checkpoint", type=str, default=None,
                        help="Skip phase 1 and load this checkpoint for phase 2.")

    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"/tmp/nasmc_runs/run_{ts}"

    obs_components = None
    if args.obs_components is not None:
        obs_components = _parse_int_list(args.obs_components)

    # Load a subset of config.yaml values for convenience defaults.
    trajectory_length = args.trajectory_length
    process_noise_std = 0.01
    obs_noise_std = 0.1
    obs_nonlinearity = "arctan"

    config_path = Path(args.data_dir) / "config.yaml"
    if config_path.exists():
        try:
            from data import load_config_yaml

            cfg = load_config_yaml(config_path)
            if obs_components is None:
                obs_components = getattr(cfg, "obs_components", None)
            if trajectory_length is None:
                trajectory_length = getattr(cfg, "len_trajectory", None)
            process_noise_std = float(getattr(cfg, "process_noise_std", process_noise_std))
            obs_noise_std = float(getattr(cfg, "obs_noise_std", obs_noise_std))
            obs_nonlinearity = str(getattr(cfg, "obs_nonlinearity", obs_nonlinearity))
        except Exception as exc:
            logger.warning("Failed to read config.yaml: %s", exc)

    if trajectory_length is None:
        trajectory_length = 1000

    obs_dim = args.obs_dim
    if obs_components is not None:
        obs_dim = len(obs_components)

    train_nasmc(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        state_dim=args.state_dim,
        obs_dim=obs_dim,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        channels=args.channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        time_embed_dim=args.time_embed_dim,
        dropout=args.dropout,
        predict_delta=args.predict_delta,
        use_dynamics_mean=args.use_dynamics_mean,
        use_time_step=args.use_time_step,
        trajectory_length=trajectory_length,
        init_log_sigma=args.init_log_sigma,
        log_sigma_min=args.log_sigma_min,
        log_sigma_max=args.log_sigma_max,
        use_observations=args.use_observations,
        obs_components=obs_components,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs_pretrain=args.max_epochs_pretrain,
        max_epochs_refine=args.max_epochs_refine,
        num_particles=args.num_particles,
        segment_length=args.segment_length,
        resample_threshold=args.resample_threshold,
        use_bootstrap_proposal=args.use_bootstrap_proposal,
        num_workers=args.num_workers,
        gpus=args.gpus,
        seed=args.seed,
        process_noise_std=process_noise_std,
        obs_noise_std=obs_noise_std,
        obs_nonlinearity=obs_nonlinearity,
        pretrain_checkpoint=args.pretrain_checkpoint,
    )


if __name__ == "__main__":
    main()
