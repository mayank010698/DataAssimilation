#!/usr/bin/env python3
"""
Oracle one-step RMSE in scaled state space: ground-truth x_prev every step (no
autoregressive drift). Compares persistence, FM teacher, and shortcut over a
sweep of shortcut Euler steps.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    DataAssimilationDataModule,
    Lorenz63,
    Lorenz96,
    KuramotoSivashinsky,
    TimeAlignedBatchSampler,
    load_config_yaml,
)
from proposals.eval_proposal import (  # noqa: E402
    _get_inference_sampling_steps,
    _set_inference_sampling_steps,
    load_proposal_from_checkpoint,
)
from proposals.shortcut_flow import ShortcutProposal  # noqa: E402


def detect_system_class(config_path: Path, data_dir: str, config) -> type:
    """Match run_proposal_eval in proposals/eval_proposal.py."""
    config_lower = str(config_path).lower()
    data_dir_lower = str(data_dir).lower()
    if (
        "ks" in data_dir_lower
        or "kuramoto" in data_dir_lower
        or "ks" in config_lower
    ):
        return KuramotoSivashinsky
    if (
        "lorenz96" in data_dir_lower
        or "96" in config_lower
        or "lorenz96" in config_lower
        or ("dim" in config.system_params and config.system_params["dim"] > 3)
    ):
        return Lorenz96
    return Lorenz63


def default_teacher_sampling_steps(teacher_ckpt: Path) -> int:
    load_kw = {"map_location": "cpu"}
    try:
        raw = torch.load(teacher_ckpt, weights_only=False, **load_kw)
    except TypeError:
        raw = torch.load(teacher_ckpt, **load_kw)
    h = raw.get("hyper_parameters")
    if isinstance(h, dict) and h.get("num_sampling_steps") is not None:
        return int(h["num_sampling_steps"])
    return 50


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def batch_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(((pred - target) ** 2).mean(dim=-1)).mean().item()


def prepare_batch_tensors(batch, device: torch.device):
    """Return x_prev_scaled, x_curr_scaled, y_curr_scaled or None, obs mask (B,), time (B,) float."""
    x_prev = batch["x_prev_scaled"].to(device)
    x_curr = batch["x_curr_scaled"].to(device)
    time_idx = batch["time_idx"].squeeze(-1).to(device=device, dtype=torch.float32)
    has_obs = batch["has_observation"].squeeze(-1).to(device)
    if has_obs.any():
        y = batch["y_curr_scaled"].to(device)
    else:
        y = None
    return x_prev, x_curr, y, has_obs, time_idx


def apply_obs_mask(x_prev, x_curr, y, has_obs, time_idx):
    """Keep only rows with observations (oracle uses y)."""
    if not has_obs.any():
        return None
    idx = has_obs.nonzero(as_tuple=True)[0]
    x_prev_s = x_prev.index_select(0, idx)
    x_curr_s = x_curr.index_select(0, idx)
    t_s = time_idx.index_select(0, idx)
    if y is not None:
        y_s = y.index_select(0, idx)
    else:
        y_s = None
    return x_prev_s, x_curr_s, y_s, t_s


def main():
    parser = argparse.ArgumentParser(
        description="Oracle (GT x_prev) one-step RMSE: teacher vs shortcut + NFE sweep"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--shortcut_ckpt", type=str, required=True)
    parser.add_argument(
        "--teacher_sampling_steps",
        type=int,
        default=None,
        help="FM Euler steps (default: from teacher ckpt hyper_parameters, else 50)",
    )
    parser.add_argument(
        "--shortcut_sampling_steps",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated NFE counts for ShortcutProposal.sample(n_steps=...)",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Stop after this many time-batches (default: full test loader)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("oracle_debug")

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    data_dir = Path(args.data_dir)
    config_path = data_dir / "config.yaml"
    config = load_config_yaml(config_path)
    system_class = detect_system_class(config_path, str(data_dir), config)

    teacher_ckpt = Path(args.teacher_ckpt)
    shortcut_ckpt = Path(args.shortcut_ckpt)

    teacher_n = (
        args.teacher_sampling_steps
        if args.teacher_sampling_steps is not None
        else default_teacher_sampling_steps(teacher_ckpt)
    )
    shortcut_ns = parse_int_list(args.shortcut_sampling_steps)

    device = torch.device(args.device)

    dm = DataAssimilationDataModule(
        config=config,
        system_class=system_class,
        data_dir=str(data_dir),
        batch_size=args.batch_size,
    )
    dm.setup("test")
    test_ds = dm.test_dataset
    n_traj = test_ds.n_trajectories
    traj_len = config.len_trajectory - 1

    batch_sampler = TimeAlignedBatchSampler(
        data_source_len=len(test_ds),
        num_trajectories=n_traj,
        traj_len=traj_len,
        batch_size=args.batch_size,
        shuffle=False,
    )
    loader = DataLoader(test_ds, batch_sampler=batch_sampler, num_workers=0)

    teacher = load_proposal_from_checkpoint(str(teacher_ckpt))
    if not hasattr(teacher, "mc_guidance"):
        teacher.mc_guidance = False
    teacher.to(device)
    teacher.eval()
    _set_inference_sampling_steps(teacher, teacher_n)

    shortcut = load_proposal_from_checkpoint(str(shortcut_ckpt))
    if not hasattr(shortcut, "mc_guidance"):
        shortcut.mc_guidance = False
    shortcut.to(device)
    shortcut.eval()
    if not isinstance(shortcut, ShortcutProposal):
        log.error("--shortcut_ckpt must be a ShortcutProposal (Stage 2/3) checkpoint.")
        sys.exit(1)

    sum_persist = 0.0
    sum_teacher = 0.0
    sum_shortcut = {k: 0.0 for k in shortcut_ns}
    n_batches = 0
    n_rows = 0

    log.info(f"system_class={system_class.__name__}")
    log.info(
        f"teacher num_sampling_steps={_get_inference_sampling_steps(teacher)} "
        f"(set from CLI or ckpt default)"
    )
    log.info(f"shortcut n_steps sweep: {shortcut_ns}")
    log.info(f"test trajectories={n_traj}, time-batches per epoch={len(loader)}")

    for b_idx, batch in enumerate(tqdm(loader, desc="oracle")):
        if args.max_batches is not None and b_idx >= args.max_batches:
            break

        x_prev, x_curr, y, has_obs, time_idx = prepare_batch_tensors(batch, device)
        sel = apply_obs_mask(x_prev, x_curr, y, has_obs, time_idx)
        if sel is None:
            continue
        x_prev_s, x_curr_s, y_s, t_s = sel
        B = x_prev_s.shape[0]
        n_rows += B

        sum_persist += batch_rmse(x_prev_s, x_curr_s) * B

        t_teacher = t_s if getattr(teacher, "use_time_step", False) else None
        with torch.no_grad():
            pred_t = teacher.sample(x_prev_s, y_s, t=t_teacher)
        sum_teacher += batch_rmse(pred_t, x_curr_s) * B

        t_sc = t_s if getattr(shortcut, "use_time_step", False) else None
        with torch.no_grad():
            for k in shortcut_ns:
                pred_s = shortcut.sample(x_prev_s, y_s, n_steps=k, t=t_sc)
                sum_shortcut[k] += batch_rmse(pred_s, x_curr_s) * B

        n_batches += 1

    if n_rows == 0:
        log.error("No batches with observations; check data_dir and obs_mask.")
        sys.exit(1)

    mean_p = sum_persist / n_rows
    mean_t = sum_teacher / n_rows
    mean_s = {k: sum_shortcut[k] / n_rows for k in shortcut_ns}

    log.info("--- Oracle mean RMSE (scaled state) weighted by batch size ---")
    log.info(f"batches_used={n_batches}  rows_used={n_rows}")
    log.info(f"persistence        : {mean_p:.6f}")
    log.info(f"teacher (N={teacher_n}): {mean_t:.6f}")
    for k in shortcut_ns:
        log.info(f"shortcut (n_steps={k}): {mean_s[k]:.6f}")


if __name__ == "__main__":
    main()
