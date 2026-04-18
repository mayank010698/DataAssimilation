"""
Unit tests for the Auxiliary Particle Filter (AuxiliaryParticleFilter).

Covers:
  1. Shape / numerical-sanity smoke test over several steps on a small L63 setup.
  2. Equivalence of the generic APF weight formula and the Pitt-Shephard
     short-circuit (log_g - log_m_sel) when q == f (TransitionProposal).
  3. Ancestor-gather correctness: when log_m concentrates on a single
     index per batch row, every selected ancestor equals that index.

Runs on CPU (no CUDA required). Designed for `pytest` or direct `python`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import DataAssimilationConfig, Lorenz63
from models.apf import AuxiliaryParticleFilter, _systematic_resample_from_log_weights
from models.proposals import TransitionProposal


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def _make_l63_system(obs_components=(0,), obs_nonlinearity="identity", dt=0.05):
    config = DataAssimilationConfig(
        num_trajectories=1,
        len_trajectory=20,
        warmup_steps=0,
        dt=dt,
        obs_noise_std=0.3,
        obs_frequency=1,
        obs_components=list(obs_components),
        obs_nonlinearity=obs_nonlinearity,
        system_params={},
    )
    return Lorenz63(config)


def _make_apf(system, n_particles=64, adjustment_type="point", proposal=None):
    return AuxiliaryParticleFilter(
        system=system,
        proposal_distribution=proposal,
        n_particles=n_particles,
        state_dim=system.state_dim,
        obs_dim=system.obs_dim,
        process_noise_std=0.2,
        device="cpu",
        adjustment_type=adjustment_type,
        resampling_threshold_ratio=0.5,
    )


# -----------------------------------------------------------------------------
# 1. Smoke test: run a few steps, no NaNs, shapes stay consistent.
# -----------------------------------------------------------------------------


def test_apf_smoke_runs_multiple_steps():
    system = _make_l63_system()
    apf = _make_apf(system, n_particles=64)

    batch_size = 2
    n_steps = 5
    dt = system.config.dt

    # Initialize with a ground-truth batch.
    x_true = system.sample_initial_state(n_samples=batch_size)  # (B, 3)
    apf.initialize_filter(x_true, init_std=0.3)

    for t in range(1, n_steps + 1):
        # Advance truth one step for the x_curr argument.
        x_true = system.integrate(x_true, 2, dt)[:, 1, :]
        # Synthesize an observation from the new truth.
        y = system.observe(x_true, add_noise=True)  # (B, obs_dim)

        traj_idxs = torch.arange(batch_size, dtype=torch.long)
        time_idxs = torch.full((batch_size,), t, dtype=torch.long)

        results = apf.step(x_true, x_true, y, dt, traj_idxs, time_idxs)

        assert len(results) == batch_size
        for r in results:
            assert np.isfinite(r["rmse"])
            assert np.isfinite(r["ess"])
            assert 0.0 < r["ess"] <= apf.n_particles + 1e-6
            assert np.all(np.isfinite(r["x_est"]))
            assert np.all(np.isfinite(r["P_est"]))
            assert "apf_log_m_mean" in r
            assert "apf_unique_ancestors" in r
            assert 1 <= r["apf_unique_ancestors"] <= apf.n_particles

        assert torch.isfinite(apf.particles).all()
        assert torch.isfinite(apf.log_weights).all()


def test_apf_handles_missing_observation():
    """When y_curr is None, APF should propagate the cloud without crashing
    and produce finite metrics (bootstrap-prior branch)."""
    system = _make_l63_system()
    apf = _make_apf(system, n_particles=32)

    x_true = system.sample_initial_state(n_samples=1).unsqueeze(0)  # (1, 3)
    apf.initialize_filter(x_true, init_std=0.3)

    dt = system.config.dt
    x_true = system.integrate(x_true, 2, dt)[:, 1, :]

    traj_idxs = torch.tensor([0], dtype=torch.long)
    time_idxs = torch.tensor([1], dtype=torch.long)

    results = apf.step(x_true, x_true, None, dt, traj_idxs, time_idxs)
    assert len(results) == 1
    assert np.isfinite(results[0]["rmse"])
    assert np.isfinite(results[0]["ess"])
    assert torch.isfinite(apf.particles).all()


# -----------------------------------------------------------------------------
# 2. Equivalence test: short-circuit vs generic formula when q = f.
# -----------------------------------------------------------------------------


def test_generic_and_shortcircuit_agree_when_proposal_is_transition():
    """With q == TransitionProposal, the generic APF incremental weight
    (log_g + log_f - log_m_sel - log_q) must agree with the Pitt-Shephard
    short-circuit (log_g - log_m_sel) up to small numerical noise, because
    log_f and log_q come from the same Gaussian kernel around the
    deterministic transition mean.

    We verify this by running two APF instances with identical inputs and
    identical torch RNG state, forcing one to take the generic path.
    """
    system = _make_l63_system()

    apf_short = _make_apf(system, n_particles=32)
    apf_generic = _make_apf(system, n_particles=32)
    # Force generic path: pretend the proposal is not a TransitionProposal.
    apf_generic._proposal_is_transition = False

    x0 = system.sample_initial_state(n_samples=1).unsqueeze(0)  # (1, 3)

    # Seed both filters identically so they produce identical particle clouds.
    torch.manual_seed(42)
    apf_short.initialize_filter(x0, init_std=0.3)
    torch.manual_seed(42)
    apf_generic.initialize_filter(x0, init_std=0.3)

    # Observation drawn from a propagated truth.
    dt = system.config.dt
    x_true = system.integrate(x0, 2, dt)[:, 1, :]
    torch.manual_seed(7)
    y = system.observe(x_true, add_noise=True)

    # Run update_step under the SAME RNG sequence for both. update_step
    # draws from:
    #   (a) systematic-resample uniform u (1 tensor),
    #   (b) the proposal's process-noise draw (1 tensor of shape [B*N, D]).
    # Both instances consume these in the same order, so identical seeds
    # give identical draws.
    torch.manual_seed(123)
    _, inc_w_short = apf_short.update_step(y)
    torch.manual_seed(123)
    _, inc_w_generic = apf_generic.update_step(y)

    # The two paths must produce identical incremental log weights.
    assert torch.allclose(inc_w_short, inc_w_generic, atol=1e-5, rtol=1e-4), (
        f"Short-circuit vs generic path disagree: "
        f"max_abs_diff={torch.max(torch.abs(inc_w_short - inc_w_generic)).item():.3e}"
    )


# -----------------------------------------------------------------------------
# 3. Ancestor-gather correctness under a peaked log_m.
# -----------------------------------------------------------------------------


def test_ancestor_resample_concentrates_on_peaked_log_m():
    """If log_m has a single dominant entry per batch, systematic
    resampling must pick that index for (essentially) all ancestors."""
    B, N = 3, 128
    log_w = torch.full((B, N), -1e6)
    peaks = torch.tensor([7, 42, 100])
    for b in range(B):
        log_w[b, peaks[b]] = 0.0  # exp -> dominates

    idx, weights = _systematic_resample_from_log_weights(log_w)

    assert idx.shape == (B, N)
    # The dominant weight should be effectively 1.0 after normalization.
    for b in range(B):
        assert weights[b, peaks[b]].item() > 1.0 - 1e-6
        # Every single systematic-resample draw should pick the peak.
        assert (idx[b] == peaks[b]).all(), (
            f"batch {b}: expected all ancestors = {peaks[b].item()}, "
            f"got unique {torch.unique(idx[b]).tolist()}"
        )


def test_ancestor_resample_with_uniform_log_weights_covers_range():
    """With uniform log-weights, systematic resampling should produce
    indices spanning the full particle range (not trivially degenerate)."""
    B, N = 2, 64
    log_w = torch.zeros(B, N)  # uniform
    idx, weights = _systematic_resample_from_log_weights(log_w)
    assert idx.shape == (B, N)
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / N))
    for b in range(B):
        n_unique = int(torch.unique(idx[b]).numel())
        # Systematic resampling with uniform weights gives exactly N unique
        # indices (each bucket gets picked once).
        assert n_unique == N, f"batch {b}: expected {N} unique, got {n_unique}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
