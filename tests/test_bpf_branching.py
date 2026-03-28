"""Tests for auxiliary branching PF allocation and BPF equivalence (uniform K)."""

import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bpf import BootstrapParticleFilter
from models.bpf_branching import (
    RFAuxiliaryBranchingParticleFilter,
    allocate_offspring_counts_batched,
    build_branching_parent_indices,
)
from models.proposals import TransitionProposal


class _MockConfig:
    dt = 0.05
    obs_noise_std = 1.0


class MockSystem:
    state_dim = 3
    obs_dim = 1
    config = _MockConfig()

    def integrate(
        self,
        x0,
        n_steps,
        dt=None,
        process_noise_std=0.0,
        step_start=0,
        static_params=None,
    ):
        if dt is None:
            dt = self.config.dt
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32)
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        traj = [x0]
        x = x0
        for _ in range(n_steps - 1):
            x = x.clone()
            traj.append(x)
        return torch.stack(traj, dim=1)

    def apply_observation_operator(self, x):
        return x[..., :1]


def _fake_allocate_all_ones(log_w_prev, predictive_log_lik, n_budget, **kwargs):
    b, n = log_w_prev.shape
    return torch.ones(b, n, dtype=torch.long, device=log_w_prev.device)


def _snapshot_filter(pf):
    return {
        "particles": pf.particles.clone(),
        "particles_prev": pf.particles_prev.clone(),
        "log_weights": pf.log_weights.clone(),
        "weights": pf.weights.clone(),
        "step_count": pf.step_count,
    }


def _restore_filter(pf, st):
    pf.particles = st["particles"].clone()
    pf.particles_prev = st["particles_prev"].clone()
    pf.log_weights = st["log_weights"].clone()
    pf.weights = st["weights"].clone()
    pf.step_count = st["step_count"]


class TestAllocateOffspring(unittest.TestCase):
    def test_sum_equals_budget(self):
        torch.manual_seed(0)
        b, n, budget = 4, 17, 17
        log_w = torch.randn(b, n)
        pred = torch.randn(b, n)
        k = allocate_offspring_counts_batched(log_w, pred, budget)
        self.assertEqual(k.shape, (b, n))
        self.assertTrue((k >= 0).all())
        self.assertTrue(torch.all(k.sum(dim=1) == budget))

    def test_uniform_weights_and_scores_gives_all_ones(self):
        b, n, budget = 2, 10, 10
        log_w = torch.zeros(b, n)
        pred = torch.zeros(b, n)
        k = allocate_offspring_counts_batched(log_w, pred, budget)
        self.assertTrue(torch.all(k == 1))

    def test_build_parent_indices(self):
        k = torch.tensor([[2, 0, 1], [1, 1, 1]], dtype=torch.long)
        p = build_branching_parent_indices(k)
        self.assertEqual(p.tolist(), [[0, 0, 2], [0, 1, 2]])


class TestBranchingMatchesBPF(unittest.TestCase):
    def test_uniform_k_matches_standard_bpf_one_step(self):
        torch.manual_seed(42)
        system = MockSystem()
        proposal = TransitionProposal(system, process_noise_std=0.25)
        n_particles = 8
        common_kw = dict(
            system=system,
            proposal_distribution=proposal,
            n_particles=n_particles,
            state_dim=3,
            obs_dim=1,
            process_noise_std=0.25,
            device="cpu",
            resampling_threshold_ratio=0.0,
        )
        bpf = BootstrapParticleFilter(**common_kw)
        abpf = RFAuxiliaryBranchingParticleFilter(
            **common_kw, use_apf_first_stage_correction=False
        )

        x0 = torch.tensor([[1.0, 0.5, -0.2]])
        bpf.initialize_filter(x0)
        abpf.initialize_filter(x0)

        st = _snapshot_filter(bpf)

        y_curr = torch.tensor([[0.3]])
        dt = 0.05
        trajectory_idxs = torch.zeros(1, dtype=torch.long)
        time_idxs = torch.ones(1, dtype=torch.long)

        torch.manual_seed(100)
        _restore_filter(bpf, st)
        bpf.step(x0, x0, y_curr, dt, trajectory_idxs, time_idxs)
        particles_b = bpf.particles.clone()
        logw_b = bpf.log_weights.clone()

        torch.manual_seed(100)
        _restore_filter(abpf, st)
        with patch(
            "models.bpf_branching.allocate_offspring_counts_batched",
            side_effect=_fake_allocate_all_ones,
        ):
            abpf.step(x0, x0, y_curr, dt, trajectory_idxs, time_idxs)

        torch.testing.assert_close(particles_b, abpf.particles, rtol=0, atol=0)
        torch.testing.assert_close(logw_b, abpf.log_weights, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
