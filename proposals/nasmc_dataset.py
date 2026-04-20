"""Datasets for the NASMC training pipeline.

Two datasets live in this file:

- :class:`NASMCTrajectoryDataset`: returns subtrajectories of shape
  ``(T, D)`` together with dense per-step observations (shape ``(T, O)``,
  zero-padded at unobserved steps) and an ``obs_mask`` of shape ``(T,)``.
  This is the batch format consumed by phase 2 of :class:`GaussianProposal`.

- :class:`NASMCDataModule`: Lightning data module. In phase 1 it
  constructs an :class:`RFTransitionDataset` (one-step pairs). In
  phase 2 it constructs an :class:`NASMCTrajectoryDataset` (full
  subtrajectories).

Both read the scaled data file ``data_scaled.h5`` produced by
:func:`save_generated_data` so that all tensors are already in the
z-scored state and observation space that :class:`GaussianProposal`
operates in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import h5py
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .rf_dataset import RFTransitionDataset
except ImportError:  # pragma: no cover - script-mode fallback
    from rf_dataset import RFTransitionDataset  # type: ignore


class NASMCTrajectoryDataset(Dataset):
    """Subtrajectory dataset with dense per-step observations.

    Given the raw scaled trajectories, observations (stored only at
    observed time steps), and the trajectory-level ``obs_mask``, each
    sample is a contiguous window of length ``segment_length`` starting
    at a random (or deterministic) position and is returned as a dict
    with keys:

    - ``trajectories``: float tensor ``(T, D)``, in scaled state space.
    - ``observations``: float tensor ``(T, O)``, in scaled observation
      space. Rows where ``obs_mask`` is False are filled with zeros.
    - ``obs_mask``: bool tensor ``(T,)``; True where an observation is
      available at that step.

    The segment-level ``obs_mask`` is computed from the trajectory-level
    ``obs_mask`` so that ``obs_frequency > 1`` datasets are handled
    correctly.
    """

    def __init__(
        self,
        trajectories: np.ndarray,
        observations: np.ndarray,
        obs_mask: np.ndarray,
        segment_length: int,
        stride: Optional[int] = None,
        deterministic: bool = False,
        obs_components: Optional[List[int]] = None,
    ):
        """
        Args:
            trajectories: Shape ``(n_trajectories, n_steps, state_dim)``, scaled.
            observations: Shape ``(n_trajectories, n_obs_steps, obs_dim)``, scaled.
                May be ``None`` if there are no observations; the observation
                channel of each sample will then be a zero tensor.
            obs_mask: Shape ``(n_steps,)`` bool — trajectory-level mask.
            segment_length: ``T`` in the output shape.
            stride: If not None, enumerate fixed-stride windows
                ``[0, stride, 2*stride, ...]`` per trajectory. If None,
                draw a random starting position in ``__getitem__``.
            deterministic: If True and ``stride`` is None, always start
                segments at the beginning of each trajectory (useful
                for validation).
            obs_components: Optional list of indices to slice the last
                dimension of the observations to (for e.g. sparse L96).
        """
        if trajectories.ndim != 3:
            raise ValueError(
                f"trajectories must be 3D, got shape {trajectories.shape}"
            )
        self.trajectories = trajectories
        self.obs_mask = np.asarray(obs_mask, dtype=bool)
        self.n_trajectories, self.n_steps, self.state_dim = trajectories.shape

        if observations is not None and obs_components is not None:
            observations = observations[..., obs_components]
        self.observations = observations
        if observations is not None:
            self.obs_dim = observations.shape[-1]
        else:
            self.obs_dim = 0
        self.obs_time_indices = np.where(self.obs_mask)[0]
        # state-time-index -> position in observations[trajectory, :, :]
        # or -1 if unobserved.
        self._state_to_obs_idx = -np.ones(self.n_steps, dtype=np.int64)
        for pos, t in enumerate(self.obs_time_indices):
            self._state_to_obs_idx[t] = pos

        if segment_length < 2:
            raise ValueError("segment_length must be >= 2")
        if segment_length > self.n_steps:
            raise ValueError(
                f"segment_length={segment_length} > trajectory length {self.n_steps}"
            )
        self.segment_length = int(segment_length)
        self.deterministic = bool(deterministic)

        if stride is not None:
            stride = int(stride)
            if stride < 1:
                raise ValueError("stride must be >= 1")
            self.stride = stride
            max_start = self.n_steps - self.segment_length
            starts_per_traj = list(range(0, max_start + 1, stride))
            # Ensure we always cover the end of the trajectory.
            if starts_per_traj[-1] != max_start:
                starts_per_traj.append(max_start)
            self._items = [
                (traj_idx, start)
                for traj_idx in range(self.n_trajectories)
                for start in starts_per_traj
            ]
        else:
            self.stride = None
            self._items = [(traj_idx, None) for traj_idx in range(self.n_trajectories)]

    def __len__(self) -> int:
        return len(self._items)

    def _pick_start(self, traj_idx: int, provided_start: Optional[int]) -> int:
        if provided_start is not None:
            return int(provided_start)
        if self.deterministic:
            return 0
        max_start = self.n_steps - self.segment_length
        if max_start == 0:
            return 0
        return int(np.random.randint(0, max_start + 1))

    def __getitem__(self, idx: int):
        traj_idx, provided_start = self._items[idx]
        start = self._pick_start(traj_idx, provided_start)
        end = start + self.segment_length

        traj_segment = self.trajectories[traj_idx, start:end]  # (T, D)
        mask_segment = self.obs_mask[start:end]                # (T,)

        if self.obs_dim > 0 and self.observations is not None:
            obs_segment = np.zeros(
                (self.segment_length, self.obs_dim),
                dtype=self.observations.dtype,
            )
            for local_t in range(self.segment_length):
                global_t = start + local_t
                obs_pos = self._state_to_obs_idx[global_t]
                if obs_pos >= 0:
                    obs_segment[local_t] = self.observations[traj_idx, obs_pos]
        else:
            obs_segment = np.zeros((self.segment_length, 0), dtype=np.float32)

        return {
            "trajectories": torch.as_tensor(traj_segment, dtype=torch.float32),
            "observations": torch.as_tensor(obs_segment, dtype=torch.float32),
            "obs_mask": torch.as_tensor(mask_segment, dtype=torch.bool),
            "trajectory_idx": traj_idx,
            "start_idx": start,
        }


class NASMCDataModule(pl.LightningDataModule):
    """Lightning data module that switches between phase-1 and phase-2 batches.

    When ``phase == 'pretrain'`` the module returns one-step
    ``(x_prev, x_curr, y_curr)`` batches using :class:`RFTransitionDataset`
    (the same one the RF baseline uses). When ``phase == 'refine'`` it
    returns subtrajectory batches from :class:`NASMCTrajectoryDataset`.
    """

    def __init__(
        self,
        data_dir: str,
        phase: str = "pretrain",
        segment_length: int = 64,
        batch_size: int = 64,
        num_workers: int = 4,
        use_observations: bool = True,
        obs_components: Optional[List[int]] = None,
        val_segment_stride: Optional[int] = None,
    ):
        super().__init__()
        if phase not in ("pretrain", "refine"):
            raise ValueError(f"Unknown phase: {phase!r}")
        self.data_dir = Path(data_dir)
        self.phase = phase
        self.segment_length = int(segment_length)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.use_observations = bool(use_observations)
        self.obs_components = list(obs_components) if obs_components is not None else None
        self.val_segment_stride = val_segment_stride

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def _load_split(self, data_file: Path, split_name: str):
        with h5py.File(data_file, "r") as f:
            traj = f[f"{split_name}/trajectories"][:]
            obs = None
            if self.use_observations and f"{split_name}/observations" in f:
                obs = f[f"{split_name}/observations"][:]
                if self.obs_components is not None and obs.shape[-1] > len(self.obs_components):
                    obs = obs[..., self.obs_components]
            obs_mask = f["obs_mask"][:]
        return traj, obs, obs_mask

    def setup(self, stage: Optional[str] = None):
        data_file = self.data_dir / "data_scaled.h5"
        if not data_file.exists():
            data_file = self.data_dir / "data.h5"
            if not data_file.exists():
                raise FileNotFoundError(
                    f"No data file found at {self.data_dir}; expected data_scaled.h5."
                )
            logging.warning(
                "data_scaled.h5 not found; falling back to data.h5. "
                "This is almost certainly NOT what you want for NASMC training."
            )

        if stage in (None, "fit"):
            train_traj, train_obs, obs_mask = self._load_split(data_file, "train")
            val_traj, val_obs, _ = self._load_split(data_file, "val")

            if self.phase == "pretrain":
                # Flatten observations to (N, T, O) at *state* indices by
                # broadcasting across obs_time_indices. RFTransitionDataset
                # expects (N, T_state, O); many of our datasets already
                # satisfy obs_frequency == 1 so we pass obs through when
                # shapes match, otherwise we densify.
                train_obs_dense = _densify_obs(train_obs, obs_mask, train_traj.shape[1])
                val_obs_dense = _densify_obs(val_obs, obs_mask, val_traj.shape[1])
                self.train_dataset = RFTransitionDataset(
                    train_traj,
                    observations=train_obs_dense,
                    window=1,
                )
                self.val_dataset = RFTransitionDataset(
                    val_traj,
                    observations=val_obs_dense,
                    window=1,
                )
            else:
                self.train_dataset = NASMCTrajectoryDataset(
                    train_traj,
                    train_obs,
                    obs_mask,
                    segment_length=self.segment_length,
                    stride=None,
                    deterministic=False,
                )
                self.val_dataset = NASMCTrajectoryDataset(
                    val_traj,
                    val_obs,
                    obs_mask,
                    segment_length=self.segment_length,
                    stride=self.val_segment_stride,
                    deterministic=True,
                )
            logging.info(
                "NASMCDataModule (phase=%s): train=%d, val=%d",
                self.phase,
                len(self.train_dataset),
                len(self.val_dataset),
            )

        if stage in (None, "test"):
            test_traj, test_obs, obs_mask = self._load_split(data_file, "test")
            if self.phase == "pretrain":
                test_obs_dense = _densify_obs(test_obs, obs_mask, test_traj.shape[1])
                self.test_dataset = RFTransitionDataset(
                    test_traj, observations=test_obs_dense, window=1,
                )
            else:
                self.test_dataset = NASMCTrajectoryDataset(
                    test_traj,
                    test_obs,
                    obs_mask,
                    segment_length=self.segment_length,
                    stride=self.val_segment_stride,
                    deterministic=True,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=self.phase == "refine",
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


def _densify_obs(
    obs: Optional[np.ndarray],
    obs_mask: np.ndarray,
    n_steps: int,
) -> Optional[np.ndarray]:
    """Turn ``(N, n_obs_steps, O)`` observations into ``(N, n_steps, O)``.

    Indices where ``obs_mask`` is False are filled with zeros. When
    ``obs`` is already dense (i.e. the second axis has length
    ``n_steps``), we return it unchanged.
    """
    if obs is None:
        return None
    obs_mask = np.asarray(obs_mask, dtype=bool)
    if obs.shape[1] == n_steps:
        return obs
    obs_time_indices = np.where(obs_mask)[0]
    if obs.shape[1] != obs_time_indices.size:
        raise ValueError(
            "Observations shape is inconsistent with obs_mask: "
            f"obs has {obs.shape[1]} observed steps but mask selects "
            f"{obs_time_indices.size}."
        )
    dense = np.zeros(
        (obs.shape[0], n_steps, obs.shape[2]), dtype=obs.dtype
    )
    dense[:, obs_time_indices] = obs
    return dense
