"""
Unit tests for the Localized FPPF implementation.

Covers tests 1, 2, 3, 4, 5, 6, 7 from the Localized FPPF implementation spec:

  1. Patch extraction correctness (circular indexing + reassembly).
  2. Velocity field assembly (output shape; v_j depends only on z[j-r : j+r+1]).
  3. Per-dimension log-density consistency (sum_j ell_j == global log_prob).
  4. Jacobian sparsity (|j - k| > r -> dv_j / dz_k == 0).
  5. Gaspari-Cohn taper sanity (GC(0)=1, GC(2r)=0, monotone non-increasing).
  6. Local resampling correctness (uniform -> identity; single block -> global).
  7. Recovery of global FPPF-like behavior with r = N_x/2 and no localization.

Runs on CPU (no CUDA required). Designed for `pytest` or direct `python`.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

# Make the project root importable when run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.enkf import dist2coeff  # Gaspari-Cohn implementation
from models.localization import (
    LocalizationGeometry,
    LocalizedPFConfig,
    compute_local_log_weights,
    local_systematic_resample,
)
from proposals.localized_rf import LocalizedRFProposal
from proposals.patch_utils import (
    WindowSpec,
    assemble_centers,
    dense_obs_from_components,
    extract_patches,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def _make_model(
    radius: int = 3,
    state_dim: int = 20,
    architecture: str = "local_mlp",
    num_steps: int = 5,
    predict_delta: bool = False,
) -> LocalizedRFProposal:
    model = LocalizedRFProposal(
        radius=radius,
        architecture=architecture,
        state_dim=state_dim,
        use_observations=True,
        obs_components=list(range(state_dim)),
        predict_delta=predict_delta,
        num_sampling_steps=num_steps,
        num_likelihood_steps=num_steps,
        hidden_dim=32,
        depth=3,
    )
    model.eval()
    return model


# =============================================================================
# Test 1: Patch extraction correctness
# =============================================================================


def test_patch_extraction_circular_indexing():
    """extract_patches must use circular indexing and preserve values exactly."""
    x = torch.arange(10, dtype=torch.float32).unsqueeze(0)  # (1, 10)
    spec = WindowSpec(radius=2, stride=1, periodic=True)
    patches = extract_patches(x, spec)  # (1, 10, 5)

    assert patches.shape == (1, 10, 5)

    # Window centred at j=0 should wrap around: [8, 9, 0, 1, 2]
    expected_0 = torch.tensor([8.0, 9.0, 0.0, 1.0, 2.0])
    assert torch.allclose(patches[0, 0], expected_0)

    # Window centred at j=9 should wrap: [7, 8, 9, 0, 1]
    expected_9 = torch.tensor([7.0, 8.0, 9.0, 0.0, 1.0])
    assert torch.allclose(patches[0, 9], expected_9)

    # Interior window should be contiguous.
    for j in range(2, 8):
        expected = torch.arange(j - 2, j + 3, dtype=torch.float32)
        assert torch.allclose(patches[0, j], expected)


def test_patch_extraction_reassembly():
    """Centre-only output assembly composes with extraction as identity."""
    x = torch.randn(3, 15)  # batch=3, N_x=15
    spec = WindowSpec(radius=3, stride=1, periodic=True)
    patches = extract_patches(x, spec)  # (3, 15, 7)

    # Take the centre element of each window -> that must be x itself.
    centres = patches[..., spec.radius]  # (3, 15)
    assert torch.allclose(centres, x)

    # Round-trip via assemble_centers.
    assembled = assemble_centers(centres.unsqueeze(-1), spec, n_x=15)
    assert torch.allclose(assembled, x)


# =============================================================================
# Test 2: Velocity field assembly
# =============================================================================


def test_velocity_field_shape_and_local_dependence():
    """Output[j] must depend only on z[(j-r):(j+r+1)] (circular)."""
    state_dim = 16
    radius = 3
    model = _make_model(radius=radius, state_dim=state_dim)

    z = torch.randn(2, state_dim)
    x_prev = torch.randn(2, state_dim)
    obs_full = torch.zeros(2, state_dim)
    obs_mask = torch.zeros(2, state_dim)
    spec = model._default_window_spec(state_dim)
    s = torch.tensor(0.5)

    with torch.no_grad():
        v = model._apply_local_net_all_sites(z, s, x_prev, obs_full, obs_mask, None, spec)
    assert v.shape == (2, state_dim)

    # Perturb z only at sites outside the receptive field of centre j0,
    # i.e. sites with circular distance > r. v[:, j0] must not change.
    j0 = 5
    inside = set(((j0 + o) % state_dim) for o in range(-radius, radius + 1))

    z_perturbed = z.clone()
    for k in range(state_dim):
        if k not in inside:
            z_perturbed[:, k] += 100.0  # huge perturbation outside receptive field

    with torch.no_grad():
        v_perturbed = model._apply_local_net_all_sites(
            z_perturbed, s, x_prev, obs_full, obs_mask, None, spec
        )

    # v at sites inside `inside` will move because we perturbed z at only
    # sites outside j0's window, which is still inside the windows of other
    # sites. But v[:, j0] should be untouched.
    assert torch.allclose(v[:, j0], v_perturbed[:, j0], atol=1e-6)


# =============================================================================
# Test 3: Per-dim log-density sum consistency
# =============================================================================


@pytest.mark.parametrize("arch", ["local_mlp", "local_resnet1d"])
@pytest.mark.parametrize("predict_delta", [False, True])
def test_sum_consistency_of_per_dim_log_prob(arch, predict_delta):
    """sum_j ell_j must match the global backward-Euler log_prob (rel tol 1e-4)."""
    state_dim = 20
    model = _make_model(
        radius=3, state_dim=state_dim, architecture=arch,
        num_steps=5, predict_delta=predict_delta,
    )

    x_prev = torch.randn(4, state_dim)
    y = torch.randn(4, state_dim)

    x_sampled, ell = model.sample_and_per_dim_log_prob(x_prev, y)
    assert x_sampled.shape == (4, state_dim)
    assert ell.shape == (4, state_dim)

    lp_global = model.log_prob(x_sampled, x_prev, y, use_exact_trace=True)

    diff = (ell.sum(dim=-1) - lp_global).abs().max().item()
    scale = max(lp_global.abs().max().item(), 1.0)
    rel = diff / scale
    assert rel < 1e-4, f"sum_j ell_j != log_prob (rel error {rel:.2e})"


# =============================================================================
# Test 4: Jacobian sparsity
# =============================================================================


def test_jacobian_sparsity():
    """dv_j / dz_k must be zero whenever circular distance |j - k| > r."""
    state_dim = 12
    radius = 2
    model = _make_model(radius=radius, state_dim=state_dim)

    z = torch.randn(1, state_dim, requires_grad=True)
    x_prev = torch.randn(1, state_dim)
    obs_full = torch.zeros(1, state_dim)
    obs_mask = torch.zeros(1, state_dim)
    spec = model._default_window_spec(state_dim)
    s = torch.tensor(0.5)

    v = model._apply_local_net_all_sites(z, s, x_prev, obs_full, obs_mask, None, spec)
    # (1, N_x) -> build N_x x N_x Jacobian (one row per output).
    J = torch.zeros(state_dim, state_dim)
    for j in range(state_dim):
        grad = torch.autograd.grad(v[0, j], z, retain_graph=True)[0]
        J[j] = grad[0]

    # Any entry with circular distance > r must be essentially zero.
    for j in range(state_dim):
        for k in range(state_dim):
            raw = abs(j - k)
            dist = min(raw, state_dim - raw)
            if dist > radius:
                assert J[j, k].abs().item() < 1e-6, (
                    f"non-zero Jacobian at j={j},k={k},dist={dist}: {J[j,k].item()}"
                )


# =============================================================================
# Test 5: Gaspari-Cohn taper function sanity
# =============================================================================


def test_gaspari_cohn_sanity():
    """GC(0)=1, GC(2r)=0, and non-increasing on [0, 2r]."""
    r = 4.0
    ds = torch.linspace(0.0, 2.5 * r, 200)
    vals = dist2coeff(ds, r)

    # Endpoint sanity.
    assert abs(vals[0].item() - 1.0) < 1e-6
    assert vals[-1].item() < 1e-6  # beyond 2r

    # Locate index closest to 2r.
    idx_2r = int(torch.argmin((ds - 2 * r).abs()).item())
    assert vals[idx_2r].item() < 1e-4

    # Non-increasing up to 2r (allow tiny numerical wiggle).
    diffs = vals[1 : idx_2r + 1] - vals[: idx_2r]
    assert diffs.max().item() <= 1e-8


# =============================================================================
# Test 6: Local resampling correctness
# =============================================================================


def test_local_resampling_uniform_is_identity():
    """Uniform local log-weights + adjustment_minimizing => identity assignment."""
    B, num_blocks, N = 2, 5, 64
    log_w = torch.zeros(B, num_blocks, N)
    anc = local_systematic_resample(log_w, n_particles=N, adjustment_minimizing=True)
    identity = torch.arange(N).view(1, 1, N).expand(B, num_blocks, N)
    assert torch.equal(anc, identity)


def test_local_resampling_single_block_matches_global():
    """With one big block (block_size == state_dim) and no taper, the local
    softmax should match the classical global multinomial softmax distribution
    (i.e. the per-particle marginal weights agree)."""
    state_dim = 8
    n_particles = 1000
    config = LocalizedPFConfig(block_size=state_dim, taper_type="none")
    geom = LocalizationGeometry(
        state_dim=state_dim,
        obs_components=list(range(state_dim)),
        block_size=config.block_size,
        localization_radius=1e9,  # covers everything
        taper_type="none",
    )
    assert geom.num_blocks == 1

    # Synthetic log-likelihood per particle (batched B=1).
    obs_noise_var = 1.0
    expected_obs = torch.randn(1, n_particles, state_dim)
    obs = torch.randn(1, state_dim)

    local_lw = compute_local_log_weights(
        expected_obs, obs, geom, obs_noise_var,
    )  # (1, 1, N)
    # Compare to the manual global log-likelihood.
    diff = obs.unsqueeze(1) - expected_obs
    global_lw = -0.5 * (diff ** 2 / obs_noise_var).sum(dim=-1)  # (1, N)

    assert torch.allclose(local_lw[:, 0, :], global_lw, atol=1e-5)


def test_local_resampling_per_dim_correction_vectorized():
    """compute_local_log_weights with per_dim_residual must match a direct
    block-sum of the residual term."""
    state_dim = 12
    n_particles = 16
    block_size = 4
    geom = LocalizationGeometry(
        state_dim=state_dim,
        obs_components=list(range(state_dim)),
        block_size=block_size,
        localization_radius=1e9,
        taper_type="none",
    )
    expected_obs = torch.randn(2, n_particles, state_dim)
    obs = torch.randn(2, state_dim)
    per_dim_residual = torch.randn(2, n_particles, state_dim)

    actual = compute_local_log_weights(
        expected_obs, obs, geom, obs_noise_var=1.0, per_dim_residual=per_dim_residual,
    )  # (2, num_blocks, N)

    # Manual reference: base likelihood + per-block sum of residual.
    diff = obs.unsqueeze(1) - expected_obs
    base = -0.5 * torch.einsum(
        "bno,go->bgn", diff ** 2, geom.taper_matrix
    )
    expected_correction = torch.einsum(
        "bnd,gd->bgn", per_dim_residual, geom.block_dim_mask
    )
    reference = base + expected_correction

    assert torch.allclose(actual, reference, atol=1e-6)


# =============================================================================
# Test 7: Recovery of "global" behavior when r ~ N_x/2 and no PF localization
# =============================================================================


def test_large_radius_matches_dense_window():
    """With r = (N_x - 1)/2 each window sees the full state; the per-dim
    log-density still sums to the scalar log_prob. This verifies the model
    degenerates gracefully to a fully-connected velocity field."""
    state_dim = 9
    model = _make_model(
        radius=(state_dim - 1) // 2, state_dim=state_dim, num_steps=4,
    )
    x_prev = torch.randn(3, state_dim)
    y = torch.randn(3, state_dim)

    x_sampled, ell = model.sample_and_per_dim_log_prob(x_prev, y)
    lp = model.log_prob(x_sampled, x_prev, y, use_exact_trace=True)
    err = (ell.sum(dim=-1) - lp).abs().max().item()
    assert err < 1e-3, f"large-radius sum-consistency failed: err={err}"


def test_single_block_no_taper_recovers_global_weights():
    """With block_size == state_dim, localization_radius huge, taper='none',
    the local PF weights are exactly the global BPF log-likelihood per particle."""
    state_dim = 6
    n_particles = 32
    geom = LocalizationGeometry(
        state_dim=state_dim,
        obs_components=list(range(state_dim)),
        block_size=state_dim,
        localization_radius=1e9,
        taper_type="none",
    )
    expected_obs = torch.randn(1, n_particles, state_dim)
    obs = torch.randn(1, state_dim)
    obs_noise_var = 0.5

    local_lw = compute_local_log_weights(expected_obs, obs, geom, obs_noise_var)
    global_lw = -0.5 * ((obs.unsqueeze(1) - expected_obs) ** 2 / obs_noise_var).sum(dim=-1)
    assert torch.allclose(local_lw.squeeze(1), global_lw, atol=1e-5)


# -----------------------------------------------------------------------------
# Convenience entry point.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
