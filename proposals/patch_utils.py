"""
Patch extraction utilities for localized proposals on periodic 1D domains.

The central abstraction is `WindowSpec`, which encapsulates a window-
extraction policy. The default is a uniform stride-1 window of size 2r+1,
but the abstraction is designed to support overlapping/non-uniform/strided
windows in future without touching downstream code.

All routines assume state tensors with shape `(..., N_x)` on a periodic
1D grid and use `F.pad(mode='circular')` + `unfold` for efficient batched
patch extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


# =============================================================================
# Window specification (extensible)
# =============================================================================


@dataclass
class WindowSpec:
    """
    Declarative specification for how to extract spatial windows.

    Default: uniform stride-1 windows of size `2*radius+1` centered at every
    grid point; this matches the LDM/Gottwald 2025 formulation.

    Attributes:
        radius: Spatial radius r. Window size is 2*r+1.
        stride: Spacing between window centres (1 = every site, 2 = every other, ...).
        centers: Optional explicit list of centre indices. If provided, overrides
            stride-based enumeration (useful for non-uniform/learned placements later).
        periodic: Whether the domain is periodic (circular indexing).
    """

    radius: int
    stride: int = 1
    centers: Optional[Sequence[int]] = None
    periodic: bool = True

    @property
    def window_size(self) -> int:
        return 2 * self.radius + 1

    def enumerate_centers(self, n_x: int) -> torch.Tensor:
        """Return the centre indices for a domain of size n_x."""
        if self.centers is not None:
            return torch.tensor(list(self.centers), dtype=torch.long)
        return torch.arange(0, n_x, self.stride, dtype=torch.long)

    def num_windows(self, n_x: int) -> int:
        return int(self.enumerate_centers(n_x).numel())


# =============================================================================
# Patch extraction
# =============================================================================


def _circular_unfold(x: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Extract stride-1 circular windows of size 2*r+1 centered at every index.

    Args:
        x: Tensor of shape `(..., N_x)`.
        radius: Spatial radius r.

    Returns:
        Tensor of shape `(..., N_x, 2*r+1)` where `[..., j, :]` is the window
        centred at j with circular indexing.
    """
    # F.pad pads the last dim; works for any leading dims.
    pad = (radius, radius)
    x_padded = F.pad(x, pad, mode="circular")
    # unfold on the last dim: (..., N_x, 2r+1)
    windows = x_padded.unfold(dimension=-1, size=2 * radius + 1, step=1)
    return windows


def extract_patches(
    x: torch.Tensor,
    spec: WindowSpec,
) -> torch.Tensor:
    """
    Extract windows from a 1D periodic signal according to `spec`.

    Args:
        x: Shape `(..., N_x)`.
        spec: WindowSpec instance.

    Returns:
        Patches of shape `(..., num_windows, 2*r+1)`.

    Notes:
        - Only the default case (periodic, stride-1, no explicit centres) uses
          `F.pad` + `unfold` for maximum efficiency.
        - Stride > 1 or explicit centres fall back to a gather-based path
          that is still vectorized but slightly more expensive.
    """
    n_x = x.shape[-1]

    if not spec.periodic:
        raise NotImplementedError("Non-periodic windows are not supported yet.")

    # Fast path: stride 1, all sites, periodic
    if spec.stride == 1 and spec.centers is None:
        return _circular_unfold(x, spec.radius)

    # General path: gather windows at the requested centres
    centers = spec.enumerate_centers(n_x).to(x.device)  # (W,)
    offsets = torch.arange(-spec.radius, spec.radius + 1, device=x.device)  # (2r+1,)
    # Absolute indices with circular wrap: (W, 2r+1)
    idx = (centers.unsqueeze(1) + offsets.unsqueeze(0)) % n_x
    idx_flat = idx.reshape(-1)  # (W * (2r+1),)
    gathered = x.index_select(dim=-1, index=idx_flat)
    # Reshape to (..., W, 2r+1)
    return gathered.reshape(*x.shape[:-1], idx.shape[0], idx.shape[1])


def extract_obs_windows(
    obs_full: torch.Tensor,
    obs_mask_full: torch.Tensor,
    spec: WindowSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract observation windows (dense) with an accompanying mask.

    Args:
        obs_full: Shape `(..., N_x)` dense observations (zeros at unobserved sites).
        obs_mask_full: Shape `(..., N_x)` binary mask (1 where an observation is present).
        spec: WindowSpec.

    Returns:
        (obs_windows, mask_windows), each `(..., num_windows, 2*r+1)`.
    """
    obs_win = extract_patches(obs_full, spec)
    mask_win = extract_patches(obs_mask_full, spec)
    return obs_win, mask_win


def dense_obs_from_components(
    obs: torch.Tensor,
    obs_components: Sequence[int],
    n_x: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter an observation vector (values at observed components only) onto
    a dense state-sized vector, and return an accompanying mask.

    Args:
        obs: Shape `(..., obs_dim)` observation values.
        obs_components: Indices into the state grid (length obs_dim).
        n_x: Full state dimension.

    Returns:
        (obs_full, obs_mask_full), each of shape `(..., n_x)`.
    """
    device = obs.device
    dtype = obs.dtype
    batch_shape = obs.shape[:-1]
    obs_full = torch.zeros(*batch_shape, n_x, device=device, dtype=dtype)
    mask_full = torch.zeros(*batch_shape, n_x, device=device, dtype=dtype)

    idx = torch.as_tensor(list(obs_components), dtype=torch.long, device=device)
    obs_full.index_copy_(dim=-1, index=idx, source=obs)
    mask_full.index_fill_(dim=-1, index=idx, value=1.0)
    return obs_full, mask_full


# =============================================================================
# Assembly (inverse of patch extraction for centre-only outputs)
# =============================================================================


def assemble_centers(
    patch_outputs: torch.Tensor,
    spec: WindowSpec,
    n_x: int,
) -> torch.Tensor:
    """
    Assemble a per-site prediction from centre-only patch outputs.

    Args:
        patch_outputs: Shape `(..., num_windows, 1)` or `(..., num_windows)`.
            Each entry is the prediction at the centre of the corresponding window.
        spec: WindowSpec used to produce the windows.
        n_x: Full state dimension.

    Returns:
        Tensor of shape `(..., n_x)` where the centre predictions are placed
        at their corresponding grid indices. Sites not covered by any centre
        are left at zero (the typical use is stride=1, which covers every site).
    """
    if patch_outputs.dim() >= 1 and patch_outputs.shape[-1] == 1:
        patch_outputs = patch_outputs.squeeze(-1)

    centers = spec.enumerate_centers(n_x).to(patch_outputs.device)
    if spec.stride == 1 and spec.centers is None and centers.numel() == n_x:
        # Fast path: centres are 0..N_x-1 in order, trivial reshape
        return patch_outputs.reshape(*patch_outputs.shape)

    out = torch.zeros(
        *patch_outputs.shape[:-1], n_x,
        device=patch_outputs.device,
        dtype=patch_outputs.dtype,
    )
    out.index_copy_(dim=-1, index=centers, source=patch_outputs)
    return out
