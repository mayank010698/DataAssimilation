"""
Localization utilities for the Localized Particle Filter.

Provides geometry precomputation, local weight computation, local resampling,
particle reconstruction, post-regularization, weight smoothing, and diagnostics.

All core functions operate on batched tensors:
  - particles:    (Batch, N, D)
  - local weights: (Batch, num_blocks, N)
  - ancestors:     (Batch, num_blocks, N)

The LocalizationGeometry is batch-independent (spatial structure only).
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .enkf import dist2coeff, pairwise_distances


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class LocalizedPFConfig:
    block_size: int = 4
    localization_radius: float = 8.0
    taper_type: str = "gaspari_cohn"        # "gaspari_cohn" | "gaussian" | "none"
    resampler_type: str = "systematic"      # "systematic" | "multinomial"
    adjustment_minimizing: bool = True       # reorder ancestors to maximize identity preservation
    post_regularization: str = "none"       # "none" | "white" | "colored"
    post_jitter_std: float = 0.01
    colored_jitter_scale: float = 0.1
    weight_smoothing: bool = False
    smoothing_radius: int = 1               # in units of blocks
    smoothing_strength: float = 0.5
    # Importance weight formulation:
    #   "likelihood_only" (default): Farchi & Bocquet Eq. 29 — only the tapered
    #       observation log-likelihood contributes (proposal and transition
    #       terms cancel under the bootstrap choice q=p).
    #   "full": adds the per-block per-dim correction
    #       sum_{j in b} [log p(x_j^i | x_{t-1}^i) - ell_j^i]
    #       where ell_j is the proposal's per-dim log-density (requires a
    #       proposal exposing `sample_and_per_dim_log_prob`).
    weight_type: str = "likelihood_only"


# =============================================================================
# Localization Geometry
# =============================================================================


class LocalizationGeometry:
    """
    Precomputed block structure and observation neighborhoods for localized PF.
    Assumes a 1D periodic domain (e.g. Lorenz-96).
    """

    def __init__(
        self,
        state_dim: int,
        obs_components: List[int],
        block_size: int,
        localization_radius: float,
        domain_length: Optional[float] = None,
        taper_type: str = "gaspari_cohn",
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.block_size = block_size
        self.localization_radius = localization_radius
        self.domain_length = domain_length if domain_length is not None else float(state_dim)
        self.device = device

        state_coords = torch.arange(state_dim, dtype=torch.float32, device=device)
        obs_coords = torch.tensor(obs_components, dtype=torch.float32, device=device)

        # Build blocks: partition state indices into contiguous chunks
        self.block_indices: List[torch.LongTensor] = []
        for start in range(0, state_dim, block_size):
            end = min(start + block_size, state_dim)
            self.block_indices.append(
                torch.arange(start, end, dtype=torch.long, device=device)
            )
        self._num_blocks = len(self.block_indices)

        # Block center coordinates (mean of state coords in each block)
        self.block_centers = torch.stack([
            state_coords[idx].float().mean() for idx in self.block_indices
        ])  # (num_blocks,)

        # Precompute taper matrix: (num_blocks, obs_dim)
        # For each block, compute distance from block center to each obs coord,
        # then apply Gaspari-Cohn (or no taper).
        dist_matrix = pairwise_distances(
            self.block_centers, obs_coords, domain_length=self.domain_length
        )  # (num_blocks, obs_dim)

        if taper_type == "gaspari_cohn":
            self.taper_matrix = dist2coeff(dist_matrix, localization_radius)
        elif taper_type == "gaussian":
            # G(d/r) = exp(-(d/r)^2 / 2), matching Farchi & Bocquet Eq. (29).
            # At d=r the taper is exp(-0.5)≈0.607; at d=2r it is exp(-2)≈0.135.
            self.taper_matrix = torch.exp(-0.5 * (dist_matrix / localization_radius) ** 2)
        elif taper_type == "none":
            # Binary: 1 if within radius, else 0
            self.taper_matrix = (dist_matrix <= localization_radius).float()
        else:
            raise ValueError(f"Unknown taper_type: {taper_type}")

        # Precompute dim-to-block mapping for weight smoothing
        self.dim_to_block = torch.zeros(state_dim, dtype=torch.long, device=device)
        for b, idx in enumerate(self.block_indices):
            self.dim_to_block[idx] = b

        # Precompute block-to-dim mask: (num_blocks, state_dim) with 1 where
        # state dim is owned by that block. Used by the vectorized "full"
        # weight path to scatter per-dim residuals (log p - ell) into block sums.
        self.block_dim_mask = torch.zeros(
            self._num_blocks, state_dim, dtype=torch.float32, device=device
        )
        for b, idx in enumerate(self.block_indices):
            self.block_dim_mask[b, idx] = 1.0

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def get_block_indices(self, b: int) -> torch.LongTensor:
        return self.block_indices[b]

    def get_neighbor_blocks(self, b: int, radius: int) -> List[int]:
        """Return block indices within `radius` blocks of block `b` (periodic)."""
        neighbors = []
        for offset in range(-radius, radius + 1):
            nb = (b + offset) % self._num_blocks
            neighbors.append(nb)
        return neighbors


# =============================================================================
# Local Weight Computation
# =============================================================================


def compute_local_log_weights(
    expected_obs: torch.Tensor,
    observation: torch.Tensor,
    geometry: LocalizationGeometry,
    obs_noise_var: float,
    per_dim_residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute local blockwise log importance weights.

    Args:
        expected_obs: (Batch, N, obs_dim) -- h(x) for each particle.
        observation:  (Batch, obs_dim).
        geometry:     LocalizationGeometry with precomputed taper_matrix.
        obs_noise_var: Observation noise variance (scalar).
        per_dim_residual: Optional (Batch, N, state_dim) tensor holding the
            per-dim "full"-weight residual
                r_{i,j} = log p(x_{t,j}^i | x_{t-1}^i) - ell_j^i
            where ell is the proposal's per-dim log-density. When provided,
            `block_dim_mask` scatters these residuals into an additive
            per-block correction -> (Batch, num_blocks, N) via
            einsum("bnd,gd->bgn"). This implements the "full" weight of the
            spec's Eq. for `weight_type='full'`.

    Returns:
        (Batch, num_blocks, N) local log weights.
    """
    diff = observation.unsqueeze(1) - expected_obs
    scaled_sq = diff ** 2 / obs_noise_var
    local_log_weights = -0.5 * torch.einsum(
        "bno,go->bgn", scaled_sq, geometry.taper_matrix
    )

    if per_dim_residual is not None:
        # Scatter residual into block-sum correction: (B, num_blocks, N).
        correction = torch.einsum(
            "bnd,gd->bgn", per_dim_residual, geometry.block_dim_mask
        )
        local_log_weights = local_log_weights + correction

    return local_log_weights


# =============================================================================
# Local Systematic Resampling
# =============================================================================


def _adjustment_minimizing_reorder(indices: torch.Tensor) -> torch.Tensor:
    """
    Reorder resampled ancestor indices to maximize fixed points (ancestor[i] == i).

    Given the multiset of ancestors from systematic resampling, find the
    permutation that preserves particle identity as much as possible.  This is
    the "adjustment-minimizing" refinement from Chopin (2004) / Farchi &
    Bocquet (2018, Section 3.2).

    Algorithm per row:
      1. Count multiplicities m_k for each ancestor value k.
      2. Assign position k -> ancestor k wherever m_k >= 1  (fixed points).
      3. Fill remaining empty positions with surplus copies, ordered by index
         to minimize total displacement.

    Args:
        indices: (M, N) raw ancestor indices from searchsorted (long).

    Returns:
        (M, N) reordered ancestor indices (long).
    """
    M, N = indices.shape
    device = indices.device
    identity = torch.arange(N, device=device)

    counts = torch.zeros(M, N, dtype=torch.long, device=device)
    counts.scatter_add_(1, indices, torch.ones_like(indices))

    result = torch.full((M, N), -1, dtype=torch.long, device=device)

    can_fix = counts >= 1
    result[can_fix] = identity.unsqueeze(0).expand(M, -1)[can_fix]
    counts[can_fix] -= 1

    for m in range(M):
        empty = (result[m] == -1).nonzero(as_tuple=False).squeeze(-1)
        if empty.numel() == 0:
            continue
        surplus = torch.repeat_interleave(identity, counts[m])
        result[m, empty] = surplus

    return result


def local_systematic_resample(
    local_log_weights: torch.Tensor,
    n_particles: int,
    adjustment_minimizing: bool = True,
) -> torch.Tensor:
    """
    Systematic resampling applied independently per (batch, block),
    optionally with adjustment-minimizing reordering.

    Args:
        local_log_weights: (Batch, num_blocks, N).
        n_particles: Number of particles N.
        adjustment_minimizing: If True, reorder ancestors to maximize
            identity preservation (ancestor[i] == i).

    Returns:
        (Batch, num_blocks, N) ancestor indices (long).
    """
    batch_size, num_blocks, N = local_log_weights.shape
    device = local_log_weights.device

    # Flatten (Batch, num_blocks) -> (Batch * num_blocks,) as independent problems
    flat_log_w = local_log_weights.reshape(-1, N)  # (B*G, N)
    M = flat_log_w.shape[0]

    # Normalize: log-sum-exp then softmax
    max_lw = flat_log_w.max(dim=1, keepdim=True)[0]
    shifted = flat_log_w - max_lw
    weights = torch.exp(shifted)
    weight_sum = weights.sum(dim=1, keepdim=True)

    # Handle zero-sum rows (degenerate weights -> uniform)
    zero_mask = (weight_sum.squeeze(1) == 0)
    if zero_mask.any():
        weights[zero_mask] = 1.0
        weight_sum[zero_mask] = float(N)

    weights = weights / weight_sum  # (M, N)

    # Systematic resampling
    cumsum = torch.cumsum(weights, dim=1)  # (M, N)
    u = torch.rand(M, 1, device=device) / N
    positions = u + torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0) / N
    indices = torch.searchsorted(cumsum, positions)
    indices = torch.clamp(indices, 0, N - 1)

    if adjustment_minimizing:
        indices = _adjustment_minimizing_reorder(indices)

    return indices.reshape(batch_size, num_blocks, N)


# =============================================================================
# Particle Reconstruction
# =============================================================================


def reconstruct_particles(
    particles: torch.Tensor,
    ancestors: torch.Tensor,
    geometry: LocalizationGeometry,
) -> torch.Tensor:
    """
    Reconstruct patchwork particles from blockwise ancestor indices.

    Args:
        particles: (Batch, N, D) pre-resampling particles.
        ancestors: (Batch, num_blocks, N) ancestor indices per block.
        geometry:  LocalizationGeometry.

    Returns:
        (Batch, N, D) reconstructed particles.
    """
    batch_size, N, D = particles.shape
    particles_new = torch.empty_like(particles)

    for b in range(geometry.num_blocks):
        idx_b = geometry.get_block_indices(b)  # (block_size,)
        block_dim = idx_b.shape[0]
        # ancestors[:, b, :] is (Batch, N) -- which source particle for each target
        anc = ancestors[:, b, :]  # (Batch, N)
        # Gather source values for this block's state dimensions
        # particles[:, :, idx_b] is (Batch, N, block_dim)
        src = particles[:, :, idx_b]  # (Batch, N, block_dim)
        # Expand ancestor indices for gather: (Batch, N, block_dim)
        anc_expanded = anc.unsqueeze(-1).expand(-1, -1, block_dim)
        particles_new[:, :, idx_b] = torch.gather(src, 1, anc_expanded)

    return particles_new


# =============================================================================
# Post-Regularization
# =============================================================================


def apply_white_noise_regularization(
    particles: torch.Tensor,
    jitter_std: float,
) -> torch.Tensor:
    """Add iid Gaussian noise to all particles after local resampling."""
    return particles + jitter_std * torch.randn_like(particles)


def apply_colored_noise_regularization(
    particles: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Add noise scaled by per-dimension ensemble std.
    Estimates diagonal std from the ensemble per batch element.
    """
    # ensemble std per batch, per dimension: (Batch, D)
    ensemble_std = torch.std(particles, dim=1, keepdim=True)  # (Batch, 1, D)
    noise = torch.randn_like(particles) * ensemble_std * scale
    return particles + noise


# =============================================================================
# Weight Smoothing
# =============================================================================


def apply_weight_smoothing(
    particles_raw: torch.Tensor,
    particles_prior: torch.Tensor,
    ancestors: torch.Tensor,
    geometry: LocalizationGeometry,
    smoothing_radius: int,
    smoothing_strength: float,
) -> torch.Tensor:
    """
    Smooth block-boundary artifacts by averaging ancestor-selected values
    across neighboring blocks for each state dimension.

    Args:
        particles_raw:   (Batch, N, D) after local resampling.
        particles_prior: (Batch, N, D) before resampling (source values).
        ancestors:       (Batch, num_blocks, N) ancestor indices.
        geometry:        LocalizationGeometry.
        smoothing_radius: Number of neighboring blocks to average over.
        smoothing_strength: Blending coefficient in [0, 1].

    Returns:
        (Batch, N, D) smoothed particles.
    """
    if smoothing_strength <= 0.0:
        return particles_raw

    batch_size, N, D = particles_raw.shape
    smoothed = torch.zeros_like(particles_raw)

    for n in range(D):
        owning_block = geometry.dim_to_block[n].item()
        neighbor_blocks = geometry.get_neighbor_blocks(owning_block, smoothing_radius)

        # Compute distance-based taper weights for each neighbor block
        dim_coord = float(n)
        taper_weights = []
        for nb in neighbor_blocks:
            center = geometry.block_centers[nb].item()
            # Periodic distance
            raw_dist = abs(dim_coord - center)
            dist = min(raw_dist, geometry.domain_length - raw_dist)
            w = dist2coeff(
                torch.tensor(dist, device=geometry.device),
                geometry.localization_radius,
            ).item()
            taper_weights.append(w)

        total_weight = sum(taper_weights)
        if total_weight == 0:
            smoothed[:, :, n] = particles_raw[:, :, n]
            continue

        # Weighted average of ancestor-selected values from neighboring blocks
        accum = torch.zeros(batch_size, N, device=particles_raw.device)
        for nb, tw in zip(neighbor_blocks, taper_weights):
            if tw == 0:
                continue
            anc_nb = ancestors[:, nb, :]  # (Batch, N)
            # Gather the n-th dimension from prior using this block's ancestors
            vals = torch.gather(particles_prior[:, :, n], 1, anc_nb)  # (Batch, N)
            accum += tw * vals
        accum /= total_weight

        smoothed[:, :, n] = (
            smoothing_strength * accum
            + (1.0 - smoothing_strength) * particles_raw[:, :, n]
        )

    return smoothed


# =============================================================================
# Diagnostics
# =============================================================================


def compute_local_ess(local_log_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute ESS for each (batch, block).

    Args:
        local_log_weights: (Batch, num_blocks, N).

    Returns:
        (Batch, num_blocks) ESS values.
    """
    # Normalize to proper weights per block
    max_lw = local_log_weights.max(dim=2, keepdim=True)[0]
    shifted = local_log_weights - max_lw
    weights = torch.exp(shifted)
    weight_sum = weights.sum(dim=2, keepdim=True)
    weight_sum = torch.clamp(weight_sum, min=1e-30)
    weights = weights / weight_sum
    ess = 1.0 / (weights ** 2).sum(dim=2)  # (Batch, num_blocks)
    return ess


def compute_ancestor_repetition(ancestors: torch.Tensor) -> torch.Tensor:
    """
    Fraction of unique ancestors per (batch, block).

    Args:
        ancestors: (Batch, num_blocks, N) long tensor.

    Returns:
        (Batch, num_blocks) uniqueness fraction in [0, 1].
    """
    batch_size, num_blocks, N = ancestors.shape
    result = torch.zeros(batch_size, num_blocks, device=ancestors.device)
    for bi in range(batch_size):
        for b in range(num_blocks):
            n_unique = torch.unique(ancestors[bi, b, :]).numel()
            result[bi, b] = n_unique / N
    return result


def compute_fixed_point_fraction(ancestors: torch.Tensor) -> torch.Tensor:
    """
    Fraction of positions where ancestor[i] == i per (batch, block).

    This is the primary diagnostic for adjustment-minimizing resampling:
    higher means more particles kept their identity.

    Args:
        ancestors: (Batch, num_blocks, N) long tensor.

    Returns:
        (Batch, num_blocks) fixed-point fraction in [0, 1].
    """
    N = ancestors.shape[2]
    identity = torch.arange(N, device=ancestors.device).view(1, 1, N)
    return (ancestors == identity).float().mean(dim=2)


def compute_mean_ancestor_displacement(ancestors: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute displacement |ancestor[i] - i| per (batch, block).

    Lower values indicate better identity preservation.

    Args:
        ancestors: (Batch, num_blocks, N) long tensor.

    Returns:
        (Batch, num_blocks) mean absolute displacement.
    """
    N = ancestors.shape[2]
    identity = torch.arange(N, device=ancestors.device).view(1, 1, N)
    return (ancestors - identity).abs().float().mean(dim=2)


def compute_block_boundary_discontinuity(
    particles: torch.Tensor,
    geometry: LocalizationGeometry,
) -> torch.Tensor:
    """
    Average absolute jump across neighboring block boundaries.

    Args:
        particles: (Batch, N, D).
        geometry:  LocalizationGeometry.

    Returns:
        (Batch,) mean discontinuity.
    """
    batch_size = particles.shape[0]
    jumps = []
    for b in range(geometry.num_blocks):
        idx_b = geometry.get_block_indices(b)
        next_b = (b + 1) % geometry.num_blocks
        idx_next = geometry.get_block_indices(next_b)
        # Last dim of block b vs first dim of block (b+1)
        last_dim = idx_b[-1]
        first_dim = idx_next[0]
        jump = (particles[:, :, last_dim] - particles[:, :, first_dim]).abs()  # (Batch, N)
        jumps.append(jump.mean(dim=1))  # (Batch,)
    return torch.stack(jumps, dim=1).mean(dim=1)  # (Batch,)
