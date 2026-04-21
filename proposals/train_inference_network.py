"""CLI entrypoint for training the Paige-Wood inference-network proposal.

This trains an :class:`InferenceNetworkProposal` by plain conditional
NLL on (x_prev, x_curr, y_curr) triples drawn from ``data_scaled.h5``.
No particle filter inside the training loop, no ODE integration: the
whole point of IN relative to :mod:`proposals.rectified_flow` is that
training is a boring supervised density-estimation problem on a
finite-sample joint.

The CLI deliberately mirrors :mod:`proposals.train_nasmc` and
:mod:`proposals.train_rf` in flag naming so that sbatch scripts can be
copied/adapted with minimal friction. The one unique knob is
``--cde_head``, which selects the conditional-density family:

  * ``gaussian``  — single diagonal Gaussian (MDN-1).
  * ``mdn_k``     — per-coordinate mixture of K Gaussians.
  * ``joint_mog`` — joint mixture of K diagonal Gaussians (default).
  * ``rnade``     — conditional autoregressive MoG (faithful Paige-Wood).

Four named presets cover the main configurations used in the paper
comparison:

  * ``--preset inn_gaussian``       single diagonal Gaussian head.
  * ``--preset inn_mog``            joint mixture of Gaussians (K=8).
  * ``--preset inn_rnade``          autoregressive MoG (K=8, 2x128 MLP).
  * ``--preset inn_mdn_k``          per-dim MoG (K=8). Mostly for ablations.

Example:

    python -m proposals.train_inference_network \\
        --data_dir /path/to/dw_dataset \\
        --output_dir /tmp/inn_run \\
        --state_dim 1 --obs_dim 1 \\
        --architecture mlp --hidden_dim 128 --depth 4 \\
        --use_observations \\
        --preset inn_mog --num_mixture_components 8 \\
        --max_epochs 60
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

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent))
    sys.path.append(str(Path(__file__).parent.parent))

try:
    from .inference_network import InferenceNetworkProposal
    from .rf_dataset import RFDataModule
except ImportError:  # pragma: no cover
    from inference_network import InferenceNetworkProposal  # type: ignore
    from rf_dataset import RFDataModule  # type: ignore

logger = logging.getLogger(__name__)


PRESETS = {
    "inn_gaussian": {
        "tag": "inn_gaussian",
        "description": "IN with a single diagonal Gaussian head (MDN-1).",
        "cde_head": "gaussian",
        "num_mixture_components": 1,
    },
    "inn_mog": {
        "tag": "inn_mog",
        "description": "IN with a joint mixture of K=8 diagonal Gaussians (default).",
        "cde_head": "joint_mog",
        "num_mixture_components": 8,
    },
    "inn_mdn_k": {
        "tag": "inn_mdn_k",
        "description": "IN with a per-coordinate MoG (K=8).",
        "cde_head": "mdn_k",
        "num_mixture_components": 8,
    },
    "inn_rnade": {
        "tag": "inn_rnade",
        "description": "IN with a conditional RNADE head (K=8, 2x128 MLP). Paper-faithful Paige-Wood.",
        "cde_head": "rnade",
        "num_mixture_components": 8,
    },
}


def _apply_preset(args: argparse.Namespace) -> Optional[dict]:
    name = getattr(args, "preset", "none")
    if name in (None, "none"):
        return None
    if name not in PRESETS:
        raise ValueError(f"Unknown --preset '{name}'. Choices: {list(PRESETS.keys())}.")
    preset = PRESETS[name]
    defaults = args._parser_defaults  # type: ignore[attr-defined]
    for key in ("cde_head", "num_mixture_components"):
        if key not in preset:
            continue
        current = getattr(args, key, None)
        if current == defaults.get(key, None):
            setattr(args, key, preset[key])
    return preset


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _load_scalers_and_trajectory_length(
    data_dir: Path, obs_components: Optional[List[int]] = None,
):
    """Load state/obs scalers and trajectory length from config.yaml.

    Matches the logic in :mod:`proposals.train_nasmc` but omits the
    ``DynamicalSystem`` construction (IN training doesn't need it; the
    filter wrapper reconstructs the system from the run config).
    """
    import h5py
    import numpy as np

    data_file = data_dir / "data_scaled.h5"
    if not data_file.exists():
        data_file = data_dir / "data.h5"

    state_mean = None
    state_std = None
    obs_mean = None
    obs_std = None
    with h5py.File(data_file, "r") as f:
        if "scaler_mean" in f:
            state_mean = torch.from_numpy(f["scaler_mean"][:]).float()
            state_std = torch.from_numpy(f["scaler_std"][:]).float()
        if "obs_scaler_mean" in f:
            obs_mean_full = f["obs_scaler_mean"][:]
            obs_std_full = f["obs_scaler_std"][:]
            if obs_components is not None and len(obs_mean_full) >= len(obs_components):
                obs_mean = torch.from_numpy(obs_mean_full[obs_components]).float()
                obs_std = torch.from_numpy(obs_std_full[obs_components]).float()
            else:
                obs_mean = torch.from_numpy(np.asarray(obs_mean_full)).float()
                obs_std = torch.from_numpy(np.asarray(obs_std_full)).float()

    trajectory_length = None
    config_path = data_dir / "config.yaml"
    if config_path.exists():
        try:
            from data import load_config_yaml

            cfg = load_config_yaml(config_path)
            trajectory_length = int(getattr(cfg, "len_trajectory", 1000))
        except Exception as exc:
            logger.warning("Failed to load config.yaml: %s", exc)

    return state_mean, state_std, obs_mean, obs_std, trajectory_length


def _make_callbacks(output_dir: Path, patience: int = 40):
    ckpt = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="inn-{epoch:03d}-{val_nll:.4f}",
        monitor="val_nll",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(monitor="val_nll", patience=patience, mode="min")
    lrm = LearningRateMonitor(logging_interval="epoch")
    return [ckpt, early, lrm], ckpt


def _build_wandb_logger(
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
    wandb_entity: Optional[str],
    disable_wandb: bool,
    output_dir: Path,
):
    if disable_wandb or not wandb_project:
        return None
    try:
        from lightning.pytorch.loggers import WandbLogger
    except Exception as exc:  # pragma: no cover
        logger.warning("Wandb not available (%s); continuing without logging.", exc)
        return None
    return WandbLogger(
        entity=wandb_entity,
        project=wandb_project,
        name=wandb_run_name or output_dir.name,
        save_dir=str(output_dir),
    )


def train_inference_network(
    data_dir: str,
    output_dir: str,
    state_dim: int,
    obs_dim: int = 0,
    obs_components: Optional[List[int]] = None,
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
    rnade_hidden_size: int = 128,
    rnade_num_hidden_layers: int = 2,
    use_time_step: bool = False,
    trajectory_length: int = 1000,
    predict_delta: bool = True,
    min_sigma: float = 1e-3,
    init_log_sigma: float = -1.0,
    cond_dropout: float = 0.0,
    use_observations: bool = True,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 60,
    patience: int = 40,
    num_workers: int = 4,
    gpus: int = 0,
    seed: Optional[int] = None,
    preset_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    disable_wandb: bool = True,
    gradient_clip_val: float = 1.0,
):
    """Train an :class:`InferenceNetworkProposal` via supervised NLL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(output_dir)

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    log.info("=" * 80)
    log.info("Inference-network proposal training (Paige-Wood)")
    log.info("=" * 80)
    log.info("preset          = %s", preset_name or "(none)")
    log.info("data_dir        = %s", data_dir)
    log.info("output_dir      = %s", output_dir)
    log.info("state_dim       = %d, obs_dim = %d", state_dim, obs_dim)
    log.info("architecture    = %s, cde_head = %s (K=%d)",
             architecture, cde_head, num_mixture_components)
    log.info("predict_delta   = %s, use_time_step = %s", predict_delta, use_time_step)
    log.info("trajectory_length = %d", trajectory_length)
    log.info("max_epochs      = %d, batch_size = %d", max_epochs, batch_size)

    data_dir_path = Path(data_dir)
    _, _, _, _, cfg_traj_len = _load_scalers_and_trajectory_length(
        data_dir_path, obs_components=obs_components,
    )
    if cfg_traj_len is not None and trajectory_length != cfg_traj_len:
        log.info(
            "Using trajectory_length=%d from config.yaml (CLI gave %d).",
            cfg_traj_len, trajectory_length,
        )
        trajectory_length = cfg_traj_len

    model = InferenceNetworkProposal(
        state_dim=state_dim,
        obs_dim=obs_dim if use_observations else 0,
        obs_indices=obs_components,
        architecture=architecture,
        cde_head=cde_head,
        num_mixture_components=num_mixture_components,
        hidden_dim=hidden_dim,
        depth=depth,
        channels=channels,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        time_embed_dim=time_embed_dim,
        dropout=dropout,
        feature_dim=feature_dim,
        rnade_hidden_size=rnade_hidden_size,
        rnade_num_hidden_layers=rnade_num_hidden_layers,
        use_time_step=use_time_step,
        trajectory_length=trajectory_length,
        predict_delta=predict_delta,
        min_sigma=min_sigma,
        init_log_sigma=init_log_sigma,
        cond_dropout=cond_dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    log.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    data_module = RFDataModule(
        data_dir=str(data_dir_path),
        batch_size=batch_size,
        num_workers=num_workers,
        use_observations=use_observations,
        obs_components=obs_components,
    )

    accelerator = "gpu" if gpus > 0 and torch.cuda.is_available() else "cpu"
    devices = gpus if accelerator == "gpu" else 1

    callbacks, ckpt_cb = _make_callbacks(output_dir, patience=patience)
    wandb_logger = _build_wandb_logger(
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
        disable_wandb=disable_wandb,
        output_dir=output_dir,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=wandb_logger if wandb_logger is not None else True,
        log_every_n_steps=10,
        gradient_clip_val=gradient_clip_val,
        precision=32,
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
    )
    trainer.fit(model, data_module)

    best_ckpt = ckpt_cb.best_model_path or None
    log.info("Best checkpoint: %s", best_ckpt)

    final_ckpt = output_dir / "final_model.ckpt"
    try:
        trainer.save_checkpoint(str(final_ckpt))
    except Exception as exc:
        log.warning("Falling back to torch.save for final checkpoint: %s", exc)
        torch.save({"state_dict": model.state_dict(),
                    "hyper_parameters": dict(model.hparams)}, final_ckpt)
    log.info("Final model saved to %s", final_ckpt)

    return {
        "model": model,
        "best_checkpoint": best_ckpt,
        "final_checkpoint": str(final_ckpt),
    }


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the Paige-Wood inference-network proposal. See module "
            "docstring for the four named presets (inn_gaussian, inn_mog, "
            "inn_mdn_k, inn_rnade) and flag semantics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument(
        "--preset",
        choices=["none", *PRESETS.keys()],
        default="none",
        help=(
            "Preset configuration. Sets --cde_head / --num_mixture_components. "
            "CLI values override preset values."
        ),
    )

    parser.add_argument("--state_dim", type=int, required=True)
    parser.add_argument("--obs_dim", type=int, default=0)
    parser.add_argument("--obs_components", type=str, default=None)
    parser.add_argument("--use_observations", action="store_true")

    parser.add_argument("--architecture", choices=["mlp", "resnet1d"], default="mlp")
    parser.add_argument(
        "--cde_head",
        choices=["gaussian", "mdn_k", "joint_mog", "rnade"],
        default="joint_mog",
        help="Conditional density family. joint_mog is the recommended default.",
    )
    parser.add_argument("--num_mixture_components", type=int, default=8)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--time_embed_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--feature_dim", type=int, default=128,
                        help="ResNet1D feature-projection width. Ignored for MLP.")

    parser.add_argument("--rnade_hidden_size", type=int, default=128)
    parser.add_argument("--rnade_num_hidden_layers", type=int, default=2)

    parser.add_argument("--use_time_step", action="store_true")
    parser.add_argument("--trajectory_length", type=int, default=1000)
    parser.add_argument("--predict_delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_sigma", type=float, default=1e-3)
    parser.add_argument("--init_log_sigma", type=float, default=-1.0)
    parser.add_argument(
        "--cond_dropout",
        type=float,
        default=0.0,
        help=(
            "Probability of dropping y_t (and its mask) during training. Gives "
            "one network that handles both observed and unobserved steps."
        ),
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default="ml-climate")
    parser.add_argument("--disable_wandb", action="store_true")

    args = parser.parse_args()
    args._parser_defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    preset = _apply_preset(args)
    preset_tag = preset["tag"] if preset else None

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{ts}_{preset_tag}" if preset_tag else f"run_{ts}"
        args.output_dir = f"/tmp/inn_runs/{name}"
    elif preset_tag and preset_tag not in Path(args.output_dir).name:
        args.output_dir = str(Path(args.output_dir).with_name(
            f"{Path(args.output_dir).name}_{preset_tag}"
        ))

    if args.wandb_run_name is None and preset_tag:
        args.wandb_run_name = Path(args.output_dir).name

    obs_components = None
    if args.obs_components is not None:
        obs_components = _parse_int_list(args.obs_components)

    obs_dim = args.obs_dim
    if obs_components is not None:
        obs_dim = len(obs_components)

    train_inference_network(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        state_dim=args.state_dim,
        obs_dim=obs_dim,
        obs_components=obs_components,
        architecture=args.architecture,
        cde_head=args.cde_head,
        num_mixture_components=args.num_mixture_components,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        channels=args.channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        time_embed_dim=args.time_embed_dim,
        dropout=args.dropout,
        feature_dim=args.feature_dim,
        rnade_hidden_size=args.rnade_hidden_size,
        rnade_num_hidden_layers=args.rnade_num_hidden_layers,
        use_time_step=args.use_time_step,
        trajectory_length=args.trajectory_length,
        predict_delta=args.predict_delta,
        min_sigma=args.min_sigma,
        init_log_sigma=args.init_log_sigma,
        cond_dropout=args.cond_dropout,
        use_observations=args.use_observations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        gpus=args.gpus,
        seed=args.seed,
        preset_name=preset_tag,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        disable_wandb=args.disable_wandb,
        gradient_clip_val=args.gradient_clip_val,
    )


if __name__ == "__main__":
    main()
