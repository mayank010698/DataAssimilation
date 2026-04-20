"""Unit tests for the NASMC implementation.

Covers:
  - Gaussian head shape / parameter correctness.
  - GaussianProposal sample / log_prob shapes and consistency with the
    closed-form diagonal-Gaussian density.
  - NASMCTrajectoryDataset returns the right shapes and observation
    indexing (including obs_frequency > 1).
  - GaussianProposal phase-1 MLE training step runs on a toy batch.
  - GaussianProposal SMC inner loop runs end-to-end on a toy trajectory.

Run with:
    pytest tests/test_nasmc.py
or directly:
    python tests/test_nasmc.py
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proposals.architectures import (
    MLPGaussianHead,
    ResNet1DGaussianHead,
    create_gaussian_head_network,
)
from proposals.nasmc import GaussianProposal
from proposals.nasmc_dataset import (
    NASMCTrajectoryDataset,
    NASMCDataModule,
    _densify_obs,
)


torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# Backbone shape tests.
# ---------------------------------------------------------------------------


def test_mlp_gaussian_head_shapes():
    state_dim = 4
    obs_dim = 2
    net = MLPGaussianHead(
        state_dim=state_dim,
        obs_dim=obs_dim,
        obs_indices=[0, 2],
        hidden_dim=32,
        depth=2,
        use_time_step=True,
    )
    B = 5
    x_prev = torch.randn(B, state_dim)
    y = torch.randn(B, obs_dim)
    t = torch.randint(0, 100, (B,)).float() / 100.0
    out = net(x_prev, y, t)
    assert out.shape == (B, 2 * state_dim)


def test_resnet1d_gaussian_head_shapes():
    state_dim = 8
    obs_dim = 4
    net = ResNet1DGaussianHead(
        state_dim=state_dim,
        obs_dim=obs_dim,
        obs_indices=[0, 2, 4, 6],
        channels=16,
        num_blocks=2,
    )
    B = 3
    x_prev = torch.randn(B, state_dim)
    y = torch.randn(B, obs_dim)
    out = net(x_prev, y)
    assert out.shape == (B, 2 * state_dim)


def test_factory_constructs_heads():
    mlp = create_gaussian_head_network(
        architecture="mlp", state_dim=3, obs_dim=1, hidden_dim=16, depth=2,
    )
    assert isinstance(mlp, MLPGaussianHead)
    rn = create_gaussian_head_network(
        architecture="resnet1d", state_dim=6, obs_dim=2, channels=8, num_blocks=1,
    )
    assert isinstance(rn, ResNet1DGaussianHead)
    with pytest.raises(ValueError):
        create_gaussian_head_network(architecture="foo", state_dim=3)


# ---------------------------------------------------------------------------
# GaussianProposal API tests.
# ---------------------------------------------------------------------------


def test_gaussian_proposal_sample_log_prob_shapes():
    state_dim = 4
    obs_dim = 2
    model = GaussianProposal(
        state_dim=state_dim,
        obs_dim=obs_dim,
        architecture="mlp",
        hidden_dim=32,
        depth=2,
        obs_indices=[0, 2],
        predict_delta=True,
    )
    model.eval()
    B = 6
    x_prev = torch.randn(B, state_dim)
    y = torch.randn(B, obs_dim)

    x_next = model.sample(x_prev, y, dt=0.01)
    assert x_next.shape == (B, state_dim)

    lp = model.log_prob(x_next, x_prev, y, dt=0.01)
    assert lp.shape == (B,)

    # Single-particle (unbatched) interface also works.
    x_next_single = model.sample(x_prev[0], y[0], dt=0.01)
    assert x_next_single.shape == (state_dim,)
    lp_single = model.log_prob(x_next_single, x_prev[0], y[0], dt=0.01)
    assert lp_single.shape == ()


def test_gaussian_proposal_log_prob_matches_closed_form():
    state_dim = 3
    model = GaussianProposal(
        state_dim=state_dim,
        obs_dim=0,
        architecture="mlp",
        hidden_dim=16,
        depth=2,
        predict_delta=True,
    )
    model.eval()
    B = 4
    x_prev = torch.randn(B, state_dim)
    mu, log_sigma = model.mean_and_log_sigma(x_prev)
    x_curr = mu + log_sigma.exp() * torch.randn_like(mu)
    lp = model.log_prob(x_curr, x_prev, y_curr=None, dt=0.01)
    expected = -0.5 * (
        ((x_curr - mu) / log_sigma.exp()) ** 2
        + 2 * log_sigma
        + math.log(2 * math.pi)
    ).sum(dim=-1)
    assert torch.allclose(lp, expected, atol=1e-5)


def test_gaussian_proposal_zero_init_recovers_delta_zero():
    """With zero-init the network should output mu_raw = 0 and log_sigma = init."""
    state_dim = 3
    init_log_sigma = -2.0
    model = GaussianProposal(
        state_dim=state_dim, obs_dim=0, hidden_dim=16, depth=2,
        predict_delta=True, init_log_sigma=init_log_sigma, zero_init_output=True,
    )
    model.eval()
    x_prev = torch.randn(5, state_dim)
    mu, log_sigma = model.mean_and_log_sigma(x_prev)
    assert torch.allclose(mu, x_prev, atol=1e-5)
    assert torch.allclose(log_sigma, torch.full_like(log_sigma, init_log_sigma), atol=1e-5)


# ---------------------------------------------------------------------------
# Trajectory dataset tests.
# ---------------------------------------------------------------------------


def test_trajectory_dataset_sparse_obs_indexing():
    n_traj = 2
    n_steps = 20
    state_dim = 3
    obs_freq = 5
    obs_dim = 2

    traj = np.random.randn(n_traj, n_steps, state_dim).astype(np.float32)
    obs_mask = np.zeros(n_steps, dtype=bool)
    obs_mask[::obs_freq] = True
    obs_time_indices = np.where(obs_mask)[0]
    obs = np.random.randn(n_traj, obs_time_indices.size, obs_dim).astype(np.float32)

    ds = NASMCTrajectoryDataset(
        traj, obs, obs_mask, segment_length=10, stride=5, deterministic=False,
    )
    assert len(ds) >= n_traj
    item = ds[0]
    assert item["trajectories"].shape == (10, state_dim)
    assert item["observations"].shape == (10, obs_dim)
    assert item["obs_mask"].shape == (10,)
    assert item["obs_mask"].dtype == torch.bool
    # Unobserved rows should be exactly zero.
    local_mask = item["obs_mask"].numpy()
    assert np.all(item["observations"].numpy()[~local_mask] == 0.0)
    # Observed rows should equal the stored observations.
    start = item["start_idx"]
    for local_t in range(10):
        global_t = start + local_t
        if obs_mask[global_t]:
            pos = int(np.where(obs_time_indices == global_t)[0])
            np.testing.assert_allclose(
                item["observations"].numpy()[local_t], obs[item["trajectory_idx"], pos]
            )


def test_densify_obs_preserves_dense_inputs():
    n_steps = 10
    obs_mask = np.ones(n_steps, dtype=bool)
    obs = np.random.randn(3, n_steps, 2).astype(np.float32)
    out = _densify_obs(obs, obs_mask, n_steps)
    assert out is obs or np.array_equal(out, obs)


def test_densify_obs_expands_sparse_inputs():
    n_steps = 8
    obs_mask = np.zeros(n_steps, dtype=bool)
    obs_mask[::2] = True
    obs = np.random.randn(2, 4, 3).astype(np.float32)
    out = _densify_obs(obs, obs_mask, n_steps)
    assert out.shape == (2, n_steps, 3)
    np.testing.assert_allclose(out[:, ::2], obs)
    assert np.all(out[:, 1::2] == 0.0)


# ---------------------------------------------------------------------------
# Phase-1 training step smoke test.
# ---------------------------------------------------------------------------


def test_gaussian_proposal_mle_training_step_runs():
    state_dim = 3
    obs_dim = 1
    model = GaussianProposal(
        state_dim=state_dim, obs_dim=obs_dim, hidden_dim=16, depth=2,
        obs_indices=[0], predict_delta=True,
    )
    model.set_phase("pretrain")
    B = 8
    batch = {
        "x_prev": torch.randn(B, state_dim),
        "x_curr": torch.randn(B, state_dim),
        "y_curr": torch.randn(B, obs_dim),
    }
    loss = model.training_step(batch, 0)
    assert loss.dim() == 0
    loss.backward()


# ---------------------------------------------------------------------------
# SMC inner loop smoke test.
# ---------------------------------------------------------------------------


class _ToySystem:
    """Minimal DynamicalSystem stand-in: identity dynamics, Euler integration.

    Matches the (state,) / (N, state) conventions used elsewhere and exposes
    `integrate(x, n_steps, dt)` returning a `(n_steps, D)` / `(N, n_steps, D)`
    tensor where each step is the previous state (persistence).
    """

    state_dim = 3

    def integrate(self, x, n_steps, dt, process_noise_std=0.0, static_params=None):
        if x.ndim == 1:
            return x.unsqueeze(0).expand(n_steps, -1).contiguous()
        return x.unsqueeze(1).expand(-1, n_steps, -1).contiguous()


def test_gaussian_proposal_smc_runs_end_to_end():
    state_dim = 3
    obs_dim = 1
    model = GaussianProposal(
        state_dim=state_dim, obs_dim=obs_dim, hidden_dim=16, depth=2,
        obs_indices=[0], predict_delta=True,
    )
    model.attach_system(_ToySystem(), dt=0.01)
    # Override scalers with identity so computations are in the same space.
    model.attach_scalers(
        state_scaler_mean=torch.zeros(state_dim),
        state_scaler_std=torch.ones(state_dim),
    )

    B, T = 2, 6
    trajectories = torch.randn(B, T, state_dim)
    obs_mask = torch.zeros(B, T, dtype=torch.bool)
    obs_mask[:, 2:] = True
    observations = torch.randn(B, T, obs_dim)

    model.set_phase("refine", num_particles=4)
    batch = {
        "trajectories": trajectories,
        "observations": observations,
        "obs_mask": obs_mask,
    }
    loss = model.training_step(batch, 0)
    assert loss.dim() == 0
    loss.backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
