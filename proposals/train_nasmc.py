"""CLI entrypoint for training the NASMC :class:`GaussianProposal`.

The script supports the two-phase training recipe from
``nasmc_plan.md``: phase 1 runs local MLE on
:class:`RFTransitionDataset` pairs (a warm-start we add to stabilise
NASMC); phase 2 runs the NASMC weighted log-density loss on
subtrajectories from :class:`NASMCTrajectoryDataset`.

Paper-faithful vs. our additions
--------------------------------
Paper-faithful (Gu, Ghahramani & Turner, 2015 -- arXiv:1506.03338):
  * Phase 2 weighted log-density objective (Eq. 12-13 of the paper).
  * Detached (stop-gradient) SMC particles/weights in the gradient pass.
  * SMC with systematic resampling and an ESS threshold.
  * Bootstrap-filter fallback for particle generation (the paper notes
    particles may be drawn from the bootstrap proposal, especially
    early in training). Exposed here as ``--bootstrap_warmup_epochs``.

Our additions / simplifications:
  * Phase 1 local-MLE pretrain on ground-truth transitions. This is
    NOT in Gu et al. 2015; it is a supervised warm-start that makes
    the phase-2 importance weights less degenerate at iteration zero.
  * Single-component diagonal Gaussian head (the paper uses an MDN).
  * Markovian conditioning ``q_phi(x_t | x_{t-1}, y_t)`` (the paper
    uses an LSTM over history).
  * ``predict_delta=True`` parametrisation (``mu = x_{t-1} + mu_raw``).
    Related to, but not identical to, the paper's "-f-" variant.

Three named presets capture common configurations cleanly:

  * ``--preset gaussian_mle``
      Phase 1 only; a standalone baseline: "Gaussian proposal trained
      by MLE on ground-truth transitions". Not NASMC -- this is our
      supervised-MLE baseline.

  * ``--preset pure_nasmc``
      Phase 2 only (no MLE pretrain), with a bootstrap-proposal
      warmup for the first ``--bootstrap_warmup_epochs`` epochs. This
      is the literal Gu et al. 2015 method (modulo MDN/LSTM, which we
      deliberately omit; see block above).

  * ``--preset nasmc_warmstart``
      Phase 1 then phase 2. The "steelman" version of NASMC. This is
      what you get by default when both ``max_epochs_pretrain`` and
      ``max_epochs_refine`` are > 0 without a preset.

Example:

    python -m proposals.train_nasmc \\
        --data_dir /path/to/dw_dataset \\
        --output_dir /tmp/nasmc_run \\
        --state_dim 1 --obs_dim 1 \\
        --architecture mlp --hidden_dim 128 --depth 4 \\
        --use_observations \\
        --preset nasmc_warmstart \\
        --num_particles 32 --segment_length 64

Only phase 1 is run if ``--max_epochs_refine == 0``. Only phase 2 is
run if ``--max_epochs_pretrain == 0`` (or if ``--pretrain_checkpoint``
is supplied).
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
    Callback,
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


# ---------------------------------------------------------------------------
# Preset definitions.
#
# Each preset sets a small number of NASMC-specific training knobs. They are
# applied BEFORE the user's own overrides so that any flag the user passes
# explicitly wins. The ``tag`` is used in the output directory name and wandb
# run name so runs are easy to tell apart in dashboards.
# ---------------------------------------------------------------------------
PRESETS = {
    "gaussian_mle": {
        "tag": "gaussian_mle",
        "description": (
            "Phase 1 only (supervised MLE on ground-truth transitions). "
            "NOT NASMC; a standalone Gaussian-proposal baseline."
        ),
        "max_epochs_pretrain": 30,
        "max_epochs_refine": 0,
        "bootstrap_warmup_epochs": 0,
        "use_bootstrap_proposal": False,
    },
    "pure_nasmc": {
        "tag": "pure_nasmc",
        "description": (
            "Phase 2 only (Gu et al. 2015 NASMC, no MLE pretrain). "
            "Bootstrap warmup for the first K epochs, then learned proposal."
        ),
        "max_epochs_pretrain": 0,
        "max_epochs_refine": 60,
        "bootstrap_warmup_epochs": 5,
        # use_bootstrap_proposal acts as the initial state; the warmup callback
        # flips it off after K epochs. Keep this True so epoch 0 already uses
        # the bootstrap proposal.
        "use_bootstrap_proposal": True,
    },
    "nasmc_warmstart": {
        "tag": "nasmc_warmstart",
        "description": (
            "Phase 1 (MLE pretrain) then Phase 2 (NASMC refinement). "
            "The 'steelman' version of NASMC."
        ),
        "max_epochs_pretrain": 30,
        "max_epochs_refine": 30,
        "bootstrap_warmup_epochs": 0,
        "use_bootstrap_proposal": False,
    },
}


def _apply_preset(args: argparse.Namespace) -> Optional[dict]:
    """If ``--preset`` was given, fill in unspecified flags from the preset.

    We only overwrite a flag if the user did not pass it on the CLI (i.e.
    the value still matches the parser default). This keeps preset defaults
    opt-in without blocking explicit overrides from the sbatch / CLI.

    Returns the preset dict (including its ``tag``) or ``None`` if the user
    did not select a preset.
    """
    name = getattr(args, "preset", "none")
    if name in (None, "none"):
        return None
    if name not in PRESETS:
        raise ValueError(
            f"Unknown --preset '{name}'. Choices: {list(PRESETS.keys())}."
        )

    preset = PRESETS[name]
    # Only fill in a value if the user didn't explicitly set one. We detect
    # "unset" by checking against the parser defaults captured below.
    defaults = args._parser_defaults  # type: ignore[attr-defined]
    for key in (
        "max_epochs_pretrain",
        "max_epochs_refine",
        "bootstrap_warmup_epochs",
        "use_bootstrap_proposal",
    ):
        if key not in preset:
            continue
        current = getattr(args, key, None)
        if current == defaults.get(key, None):
            setattr(args, key, preset[key])
    return preset


class BootstrapWarmupCallback(Callback):
    """Flip the phase-2 proposal from bootstrap -> learned at a given epoch.

    During the first ``warmup_epochs`` epochs of phase 2, particles are
    drawn from the transition (bootstrap) proposal, matching Gu et al. 2015's
    note that early iterations can use the bootstrap filter. Once epoch
    >= ``warmup_epochs`` the callback flips the model's internal refine
    config so subsequent epochs use the learned ``q_phi``.

    The gradient objective is unchanged in either regime: we always train
    ``q_phi`` on the particles produced by the current sweep (whether the
    sweep was driven by the bootstrap or by ``q_phi`` itself).
    """

    def __init__(self, warmup_epochs: int):
        super().__init__()
        self.warmup_epochs = int(warmup_epochs)
        self._flipped = False

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module) -> None:
        if self.warmup_epochs <= 0 or self._flipped:
            return
        if trainer.current_epoch >= self.warmup_epochs:
            cfg = getattr(pl_module, "_refine_cfg", None)
            if isinstance(cfg, dict):
                cfg["use_bootstrap"] = False
                self._flipped = True
                logging.info(
                    "BootstrapWarmupCallback: epoch %d reached warmup_epochs=%d; "
                    "switching phase-2 sweep from bootstrap to learned q_phi.",
                    trainer.current_epoch,
                    self.warmup_epochs,
                )


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


def _build_wandb_logger(
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
    wandb_entity: Optional[str],
    disable_wandb: bool,
    output_dir: Path,
):
    """Create a Lightning ``WandbLogger`` or return ``None`` if disabled.

    The import is deferred so that environments without wandb still work.
    """
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
    batch_size_refine: Optional[int] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs_pretrain: int = 30,
    max_epochs_refine: int = 30,
    num_particles: int = 32,
    segment_length: int = 64,
    resample_threshold: float = 0.5,
    use_bootstrap_proposal: bool = False,
    bootstrap_warmup_epochs: int = 0,
    num_workers: int = 4,
    gpus: int = 0,
    seed: Optional[int] = None,
    process_noise_std: float = 0.01,
    obs_noise_std: float = 0.1,
    obs_nonlinearity: str = "arctan",
    pretrain_checkpoint: Optional[str] = None,
    # Logging / experiment-identification knobs.
    preset_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    disable_wandb: bool = True,
):
    """Train the NASMC proposal end-to-end (phase 1 → phase 2).

    See module docstring for a description of the three named presets
    (``gaussian_mle``, ``pure_nasmc``, ``nasmc_warmstart``) and which
    aspects are paper-faithful vs. our additions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(output_dir)

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    log.info("=" * 80)
    log.info("NASMC training")
    log.info("=" * 80)
    log.info("preset          = %s", preset_name or "(none)")
    log.info("data_dir        = %s", data_dir)
    log.info("output_dir      = %s", output_dir)
    log.info("state_dim       = %d, obs_dim = %d", state_dim, obs_dim)
    log.info("architecture    = %s", architecture)
    log.info("predict_delta   = %s, use_dynamics_mean = %s", predict_delta, use_dynamics_mean)
    log.info("use_time_step   = %s, trajectory_length = %d", use_time_step, trajectory_length)
    log.info("num_particles   = %d, segment_length = %d", num_particles, segment_length)
    log.info(
        "phase 1 epochs  = %d, phase 2 epochs = %d, bootstrap_warmup_epochs = %d",
        max_epochs_pretrain, max_epochs_refine, bootstrap_warmup_epochs,
    )

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
        pretrain_wandb = _build_wandb_logger(
            wandb_project=wandb_project,
            wandb_run_name=(f"{wandb_run_name}-pretrain" if wandb_run_name else None),
            wandb_entity=wandb_entity,
            disable_wandb=disable_wandb,
            output_dir=output_dir / "pretrain",
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs_pretrain,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=pretrain_wandb if pretrain_wandb is not None else True,
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
        # If the user asked for a bootstrap warmup, start phase 2 in bootstrap
        # mode regardless of ``use_bootstrap_proposal``; the callback flips it
        # off at epoch ``bootstrap_warmup_epochs``.
        initial_use_bootstrap = use_bootstrap_proposal or (bootstrap_warmup_epochs > 0)
        model.set_phase(
            "refine",
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            use_bootstrap=initial_use_bootstrap,
        )
        refine_bs = int(batch_size_refine) if batch_size_refine is not None else batch_size
        if refine_bs != batch_size:
            log.info(
                "Phase 2 using batch_size_refine=%d (phase 1 used batch_size=%d)",
                refine_bs, batch_size,
            )
        data_module = NASMCDataModule(
            data_dir=str(data_dir_path),
            phase="refine",
            segment_length=segment_length,
            batch_size=refine_bs,
            num_workers=num_workers,
            use_observations=use_observations,
            obs_components=obs_components,
        )
        callbacks, ckpt = _make_callbacks(
            output_dir / "refine", monitor="val_nll", phase_tag="refine"
        )
        if bootstrap_warmup_epochs > 0:
            callbacks.append(BootstrapWarmupCallback(bootstrap_warmup_epochs))
            log.info(
                "Phase 2 will use the bootstrap proposal for the first %d "
                "epoch(s), then switch to the learned q_phi.",
                bootstrap_warmup_epochs,
            )
        refine_wandb = _build_wandb_logger(
            wandb_project=wandb_project,
            wandb_run_name=(f"{wandb_run_name}-refine" if wandb_run_name else None),
            wandb_entity=wandb_entity,
            disable_wandb=disable_wandb,
            output_dir=output_dir / "refine",
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs_refine,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=refine_wandb if refine_wandb is not None else True,
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
    parser = argparse.ArgumentParser(
        description=(
            "Train NASMC Gaussian proposal. See the module docstring for "
            "the three named presets (gaussian_mle / pure_nasmc / "
            "nasmc_warmstart) and which knobs are paper-faithful vs. "
            "our additions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=None)

    # Named presets for the paper's comparison table. See PRESETS dict above.
    parser.add_argument(
        "--preset",
        choices=["none", *PRESETS.keys()],
        default="none",
        help=(
            "Preset training configuration. Options: "
            "'gaussian_mle' (phase 1 only; supervised Gaussian-MLE baseline, "
            "NOT NASMC); "
            "'pure_nasmc' (phase 2 only, with --bootstrap_warmup_epochs>0; "
            "the literal Gu et al. 2015 method modulo MDN/LSTM); "
            "'nasmc_warmstart' (phase 1 + phase 2; our default 'steelman' "
            "version). Preset values only apply if the user did not also "
            "pass the corresponding flag explicitly."
        ),
    )

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
    parser.add_argument(
        "--batch_size_refine",
        type=int,
        default=None,
        help=(
            "Optional override for the phase-2 (NASMC refine) batch size. "
            "Defaults to --batch_size. Phase 2 has ~segment_length * num_particles "
            "more activation memory per sample than phase 1, so it usually needs "
            "a much smaller batch (e.g. 128 when phase 1 runs at 512)."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--max_epochs_pretrain", type=int, default=30)
    parser.add_argument("--max_epochs_refine", type=int, default=30)

    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--segment_length", type=int, default=64)
    parser.add_argument("--resample_threshold", type=float, default=0.5)
    parser.add_argument("--use_bootstrap_proposal", action="store_true")
    parser.add_argument(
        "--bootstrap_warmup_epochs",
        type=int,
        default=0,
        help=(
            "If > 0, phase 2 uses the bootstrap proposal for the first K "
            "epochs, then flips to the learned q_phi. Paper-faithful: Gu "
            "et al. 2015 explicitly allow bootstrap-filter particles early "
            "in training."
        ),
    )

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--pretrain_checkpoint", type=str, default=None,
                        help="Skip phase 1 and load this checkpoint for phase 2.")

    # Wandb / experiment identification.
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name. Omit (or --disable_wandb) to disable.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name. Defaults to the output directory name.")
    parser.add_argument("--wandb_entity", type=str, default="ml-climate",
                        help="Wandb entity / team name. Defaults to 'ml-climate' "
                             "to match train_rf.py; pass '' or your username to "
                             "log under a different entity.")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable wandb logging even if --wandb_project is set.")

    args = parser.parse_args()
    args._parser_defaults = {
        a.dest: a.default for a in parser._actions if hasattr(a, "dest")
    }
    preset = _apply_preset(args)

    preset_tag = preset["tag"] if preset else None
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{ts}"
        if preset_tag:
            name = f"run_{ts}_{preset_tag}"
        args.output_dir = f"/tmp/nasmc_runs/{name}"
    elif preset_tag and preset_tag not in Path(args.output_dir).name:
        # Make the preset visible in the directory name for easy filtering.
        args.output_dir = str(Path(args.output_dir).with_name(
            f"{Path(args.output_dir).name}_{preset_tag}"
        ))

    if args.wandb_run_name is None and preset_tag:
        args.wandb_run_name = f"{Path(args.output_dir).name}"

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
        batch_size_refine=args.batch_size_refine,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs_pretrain=args.max_epochs_pretrain,
        max_epochs_refine=args.max_epochs_refine,
        num_particles=args.num_particles,
        segment_length=args.segment_length,
        resample_threshold=args.resample_threshold,
        use_bootstrap_proposal=args.use_bootstrap_proposal,
        bootstrap_warmup_epochs=args.bootstrap_warmup_epochs,
        num_workers=args.num_workers,
        gpus=args.gpus,
        seed=args.seed,
        process_noise_std=process_noise_std,
        obs_noise_std=obs_noise_std,
        obs_nonlinearity=obs_nonlinearity,
        pretrain_checkpoint=args.pretrain_checkpoint,
        preset_name=preset_tag,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        disable_wandb=args.disable_wandb,
    )


if __name__ == "__main__":
    main()
