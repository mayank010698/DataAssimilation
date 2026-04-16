"""
Patch-level dataset for training localized RF proposals (Option A).

Wraps a trajectory dataset and exposes `N_transitions * N_x` training items;
each item corresponds to one grid site (center) of one transition.

The item contains enough information for the training loop to form the RF
interpolation on a patch:
    - x_prev_window     : (2r+1,) previous-state window
    - x_curr_window     : (2r+1,) current-state window (NOT x_curr_center only,
                          because z is sampled over the window and the RF
                          target at the centre depends on x_curr at the centre)
    - obs_window        : (2r+1,) dense observation window (zeros where no obs)
    - obs_mask_window   : (2r+1,) binary mask of observation presence
    - target_j          : scalar x_curr at the centre
    - x_prev_center     : scalar x_prev at the centre (needed for predict_delta)
    - j                 : centre index
    - trajectory_idx    : original trajectory index
    - time_idx          : within-trajectory time index

The optional `WindowSpec` makes the extraction policy pluggable — today we
ship the default stride-1 circular uniform windows, but overlapping /
multi-scale / strided variants can drop in without changing this dataset's
interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging

try:
    from .patch_utils import WindowSpec
except ImportError:
    from patch_utils import WindowSpec


def _extract_window_np(
    signal: np.ndarray, center: int, radius: int, periodic: bool = True
) -> np.ndarray:
    """Extract a circular window around `center` from a 1-D array."""
    n = signal.shape[-1]
    if periodic:
        idx = (np.arange(-radius, radius + 1) + center) % n
    else:
        raise NotImplementedError("Non-periodic windows not supported yet.")
    return signal[..., idx]


class PatchTransitionDataset(Dataset):
    """
    Dataset exposing `N_transitions * N_x` patch items from a trajectory HDF5.

    Args:
        trajectories: Array of shape (n_trajectories, n_steps, state_dim).
        observations: Optional array of shape (n_trajectories, n_steps, obs_dim).
        obs_components: Indices into the state grid for each observation column.
            Required if `observations` is provided; used to scatter obs back
            onto a dense state-sized vector.
        window_spec: WindowSpec controlling extraction. If `None`, defaults to
            uniform stride-1 windows of size `2*radius+1` at every site.
        radius: Only used when `window_spec is None`.

    Notes:
        - `__len__ = n_transitions * n_windows_per_transition`.
        - Items are keyed `(transition_idx, window_idx)` via integer division.
        - Observations are pre-scattered to dense N_x columns on __init__
          (lazy per-access would be wasteful for repeat gets).
    """

    def __init__(
        self,
        trajectories: np.ndarray,
        observations: Optional[np.ndarray] = None,
        obs_components: Optional[Sequence[int]] = None,
        window_spec: Optional[WindowSpec] = None,
        radius: int = 3,
    ):
        self.trajectories = trajectories
        self.n_trajectories, self.n_steps, self.state_dim = trajectories.shape

        if window_spec is None:
            window_spec = WindowSpec(radius=radius, stride=1, centers=None, periodic=True)
        self.window_spec = window_spec
        self.radius = window_spec.radius
        self.window_size = window_spec.window_size

        # Pre-compute the list of window centres once
        centers = window_spec.enumerate_centers(self.state_dim).numpy()
        self.centers = centers
        self.n_windows = centers.shape[0]
        self.n_transitions = self.n_trajectories * (self.n_steps - 1)

        # Scatter sparse observations to dense (n_traj, n_steps, N_x) + mask
        self.obs_full = None
        self.obs_mask_full = None
        if observations is not None:
            if obs_components is None:
                # Fallback: observations are already dense
                if observations.shape[-1] != self.state_dim:
                    raise ValueError(
                        "observations has non-state_dim last axis and no obs_components "
                        "were provided; cannot determine scatter indices."
                    )
                self.obs_full = observations.astype(np.float32)
                self.obs_mask_full = np.ones_like(self.obs_full, dtype=np.float32)
            else:
                dense = np.zeros(
                    (self.n_trajectories, self.n_steps, self.state_dim), dtype=np.float32
                )
                mask = np.zeros_like(dense)
                idx = np.asarray(list(obs_components), dtype=np.int64)
                dense[..., idx] = observations
                mask[..., idx] = 1.0
                self.obs_full = dense
                self.obs_mask_full = mask

    def __len__(self) -> int:
        return self.n_transitions * self.n_windows

    def _decode_index(self, k: int) -> tuple[int, int, int, int]:
        """Map a flat index k to (traj_idx, t, window_idx, center)."""
        transition_idx = k // self.n_windows
        window_idx = k % self.n_windows
        traj_idx = transition_idx // (self.n_steps - 1)
        within_traj = transition_idx % (self.n_steps - 1)
        t = within_traj + 1  # because we need t-1 to be valid
        center = int(self.centers[window_idx])
        return traj_idx, t, window_idx, center

    def __getitem__(self, k: int):
        traj_idx, t, window_idx, j = self._decode_index(k)

        x_prev_full = self.trajectories[traj_idx, t - 1]
        x_curr_full = self.trajectories[traj_idx, t]

        x_prev_window = _extract_window_np(x_prev_full, j, self.radius)
        x_curr_window = _extract_window_np(x_curr_full, j, self.radius)

        item = {
            "x_prev_window": torch.from_numpy(np.ascontiguousarray(x_prev_window)).float(),
            "x_curr_window": torch.from_numpy(np.ascontiguousarray(x_curr_window)).float(),
            "target_j": torch.tensor(float(x_curr_full[j])),
            "x_prev_center": torch.tensor(float(x_prev_full[j])),
            "j": torch.tensor(j, dtype=torch.long),
            "trajectory_idx": torch.tensor(traj_idx, dtype=torch.long),
            "time_idx": torch.tensor(t, dtype=torch.long),
        }

        if self.obs_full is not None:
            obs_full_t = self.obs_full[traj_idx, t]
            obs_mask_t = self.obs_mask_full[traj_idx, t]
            item["obs_window"] = torch.from_numpy(
                np.ascontiguousarray(_extract_window_np(obs_full_t, j, self.radius))
            ).float()
            item["obs_mask_window"] = torch.from_numpy(
                np.ascontiguousarray(_extract_window_np(obs_mask_t, j, self.radius))
            ).float()

        return item


class PatchDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for patch-based RF training. Mirrors the interface of
    `RFDataModule` but yields per-patch items.
    """

    def __init__(
        self,
        data_dir: str,
        radius: int,
        batch_size: int = 256,
        num_workers: int = 4,
        use_observations: bool = False,
        obs_components: Optional[Sequence[int]] = None,
        window_spec: Optional[WindowSpec] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.radius = radius
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_observations = use_observations
        self.obs_components = obs_components
        self.window_spec = window_spec or WindowSpec(radius=radius, stride=1, periodic=True)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        data_file = self.data_dir / "data_scaled.h5"
        if not data_file.exists():
            data_file = self.data_dir / "data.h5"
            if not data_file.exists():
                raise FileNotFoundError(
                    f"Data file not found in {self.data_dir}. "
                    "Run data generation first."
                )
            logging.warning(
                "data_scaled.h5 not found; using data.h5 (unscaled). "
                "This is usually wrong for RF training."
            )

        with h5py.File(data_file, "r") as f:
            if stage == "fit" or stage is None:
                train_traj = f["train/trajectories"][:]
                val_traj = f["val/trajectories"][:]
                train_obs = None
                val_obs = None
                if self.use_observations:
                    train_obs = f["train/observations"][:]
                    val_obs = f["val/observations"][:]
                    if self.obs_components is not None:
                        train_obs = train_obs[..., list(self.obs_components)]
                        val_obs = val_obs[..., list(self.obs_components)]

                self.train_dataset = PatchTransitionDataset(
                    train_traj,
                    observations=train_obs,
                    obs_components=self.obs_components,
                    window_spec=self.window_spec,
                    radius=self.radius,
                )
                self.val_dataset = PatchTransitionDataset(
                    val_traj,
                    observations=val_obs,
                    obs_components=self.obs_components,
                    window_spec=self.window_spec,
                    radius=self.radius,
                )
                logging.info(
                    f"Loaded patch training data: {len(self.train_dataset)} items "
                    f"({self.train_dataset.n_transitions} transitions × "
                    f"{self.train_dataset.n_windows} windows)"
                )

            if stage == "test" or stage is None:
                test_traj = f["test/trajectories"][:]
                test_obs = None
                if self.use_observations:
                    test_obs = f["test/observations"][:]
                    if self.obs_components is not None:
                        test_obs = test_obs[..., list(self.obs_components)]
                self.test_dataset = PatchTransitionDataset(
                    test_traj,
                    observations=test_obs,
                    obs_components=self.obs_components,
                    window_spec=self.window_spec,
                    radius=self.radius,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
