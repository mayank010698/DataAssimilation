"""Auxiliary branching PF helpers and RFAuxiliaryBranchingParticleFilter (batched)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base_pf import compute_weighted_spread
from .bpf import BootstrapParticleFilter


def allocate_offspring_counts_batched(
    log_w_prev: torch.Tensor,
    predictive_log_lik: torch.Tensor,
    n_budget: int,
    min_per_parent: int = 0,
    max_per_parent: Optional[int] = None,
) -> torch.Tensor:
    """
    Integer offspring counts K_i per parent, shape (B, N), sum_i K_i = n_budget per row.
    Largest-remainder on targets n_budget * softmax(log w + predictive log score).
    """
    if min_per_parent != 0 or max_per_parent is not None:
        raise NotImplementedError(
            "min_per_parent and max_per_parent are reserved (v1 uses neither)."
        )
    if log_w_prev.shape != predictive_log_lik.shape:
        raise ValueError("log_w_prev and predictive_log_lik must have the same shape")
    batch_size, n_parents = log_w_prev.shape
    log_a = log_w_prev + predictive_log_lik
    log_a = log_a - torch.max(log_a, dim=1, keepdim=True).values
    p = torch.softmax(log_a, dim=1)
    targets = p * float(n_budget)
    k = torch.floor(targets).to(torch.long)
    need = n_budget - k.sum(dim=1)
    frac = targets - k.float()
    for b in range(batch_size):
        nb = int(need[b].item())
        if nb > 0:
            _, inds = torch.topk(frac[b], nb)
            k[b, inds] += 1
    return k


def build_branching_parent_indices(k: torch.Tensor) -> torch.Tensor:
    """Parent index per descendant, shape (B, n_budget); parent i repeated K[b,i] times."""
    batch_size, n_parents = k.shape
    device = k.device
    ar = torch.arange(n_parents, device=device, dtype=torch.long)
    rows = []
    n_budget = int(k[0].sum().item())
    for b in range(batch_size):
        if int(k[b].sum().item()) != n_budget:
            raise ValueError("Each batch row of k must sum to the same n_budget")
        rows.append(torch.repeat_interleave(ar, k[b]))
    return torch.stack(rows, dim=0)


class RFAuxiliaryBranchingParticleFilter(BootstrapParticleFilter):
    """Batched auxiliary branching particle filter (fixed budget N proposals per obs step)."""

    def __init__(
        self,
        *args,
        use_apf_first_stage_correction: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_apf_first_stage_correction = use_apf_first_stage_correction

    def step(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        trajectory_idxs: torch.Tensor,
        time_idxs: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        if y_curr is None:
            return super().step(
                x_prev, x_curr, y_curr, dt, trajectory_idxs, time_idxs
            )

        start_time = time.perf_counter()
        self.step_count += 1
        self._current_time_idxs = time_idxs

        batch_size = x_curr.shape[0]
        n_budget = self.n_particles
        dt_model = self.system.config.dt
        observation = y_curr.to(self.device)

        x_anc = self.particles
        pred_ll = self.compute_predictive_log_likelihood(
            x_anc, observation, dt_model
        )
        k = allocate_offspring_counts_batched(
            self.log_weights, pred_ll, n_budget
        )
        parent_idx = build_branching_parent_indices(k)

        idx_exp = parent_idx.unsqueeze(-1).expand(
            -1, -1, self.state_dim
        )
        x_prev_rep = torch.gather(x_anc, 1, idx_exp)

        particles_flat = x_prev_rep.reshape(-1, self.state_dim)
        obs_expanded = observation.unsqueeze(1).expand(
            batch_size, n_budget, -1
        )
        obs_flat = obs_expanded.reshape(batch_size * n_budget, -1)

        time_flat = None
        if getattr(self, "_current_time_idxs", None) is not None:
            time_flat = (
                self._current_time_idxs.unsqueeze(1)
                .expand(batch_size, n_budget)
                .reshape(-1)
                .float()
                .to(particles_flat.device)
            )

        x_desc_flat = self.proposal.sample(
            particles_flat, obs_flat, dt_model, t=time_flat
        )
        x_desc = x_desc_flat.reshape(
            batch_size, n_budget, self.state_dim
        )

        log_k = torch.log(k.float())
        log_k_desc = log_k.gather(1, parent_idx)
        log_w_split = self.log_weights.gather(1, parent_idx) - log_k_desc

        self.particles = x_desc
        self.particles_prev = x_prev_rep
        self.log_weights = log_w_split

        log_likelihoods = self.compute_log_likelihood(observation)

        expected_obs_flat = self.system.apply_observation_operator(
            x_desc_flat
        )
        expected_obs = expected_obs_flat.reshape(
            batch_size, n_budget, self.obs_dim
        )
        obs_exp = observation.unsqueeze(1)
        diff = obs_exp - expected_obs
        obs_noise_var = self.system.config.obs_noise_std**2
        if self.obs_dim == 1:
            obs_log_likelihoods = (
                -0.5 * (diff.squeeze(-1) ** 2) / obs_noise_var
                - 0.5 * np.log(2 * np.pi * obs_noise_var)
            )
        else:
            obs_log_likelihoods = (
                -0.5 * torch.sum(diff**2, dim=2) / obs_noise_var
            )
            obs_log_likelihoods -= 0.5 * (
                self.obs_dim * np.log(2 * np.pi * obs_noise_var)
            )

        self.last_obs_log_likelihoods = obs_log_likelihoods

        transition_log_probs = self.compute_transition_log_prob(
            x_desc, x_prev_rep, dt_model
        )

        particles_prev_flat = x_prev_rep.reshape(-1, self.state_dim)
        proposal_log_probs_flat = self.proposal.log_prob(
            x_desc_flat,
            particles_prev_flat,
            obs_flat,
            dt_model,
            t=time_flat,
        )
        proposal_log_probs = proposal_log_probs_flat.reshape(
            batch_size, n_budget
        )
        self.last_proposal_log_probs = proposal_log_probs

        importance = (
            obs_log_likelihoods
            + transition_log_probs
            - proposal_log_probs
        )
        if self.use_apf_first_stage_correction:
            mu_term = pred_ll.gather(1, parent_idx)
            importance = importance - mu_term

        self.log_weights = log_w_split + importance

        lw = self.log_weights
        w_spread = torch.exp(lw - torch.max(lw, dim=1, keepdim=True).values)
        w_spread = w_spread / torch.sum(w_spread, dim=1, keepdim=True)
        ensemble_spread_pre = compute_weighted_spread(
            self.particles, w_spread
        )
        resampled_flags, ess_pre = self.resample()

        abpf_max_offspring = k.max(dim=1).values.float()
        abpf_frac_zero_parents = (k == 0).float().mean(dim=1)
        abpf_n_unique = torch.tensor(
            [
                int(parent_idx[b].unique().numel())
                for b in range(batch_size)
            ],
            device=self.device,
            dtype=torch.float32,
        )

        x_est = self.get_state_estimate()
        P_est = self.get_state_covariance()
        elapsed_time = (time.perf_counter() - start_time) / batch_size

        x_curr = x_curr.to(self.device)
        error = x_est - x_curr
        rmse = torch.sqrt(torch.mean(error**2, dim=1))
        ess = 1.0 / torch.sum(self.weights**2, dim=1)

        prop_means = (
            self.last_proposal_log_probs.mean(dim=1)
            if self.last_proposal_log_probs is not None
            else torch.zeros(batch_size, device=self.device)
        )
        obs_means = (
            self.last_obs_log_likelihoods.mean(dim=1)
            if self.last_obs_log_likelihoods is not None
            else torch.zeros(batch_size, device=self.device)
        )
        obs_stds = (
            self.last_obs_log_likelihoods.std(dim=1)
            if self.last_obs_log_likelihoods is not None
            else torch.zeros(batch_size, device=self.device)
        )

        results: List[Dict[str, Any]] = []
        for i in range(batch_size):
            results.append(
                {
                    "rmse": rmse[i].item(),
                    "log_likelihood": log_likelihoods[i].item(),
                    "x_est": x_est[i].cpu().numpy(),
                    "P_est": P_est[i].cpu().numpy(),
                    "error": error[i].cpu().numpy(),
                    "trajectory_idx": trajectory_idxs[i].item(),
                    "time_idx": time_idxs[i].item(),
                    "resampled": resampled_flags[i],
                    "ess_pre_resample": ess_pre[i].item(),
                    "ess": ess[i].item(),
                    "ensemble_spread_pre_resample": ensemble_spread_pre[
                        i
                    ].item(),
                    "step_time": elapsed_time,
                    "proposal_log_prob_mean": prop_means[i].item(),
                    "obs_log_prob_mean": obs_means[i].item(),
                    "obs_log_prob_std": obs_stds[i].item(),
                    "abpf_n_unique_ancestors": abpf_n_unique[i].item(),
                    "abpf_max_offspring": abpf_max_offspring[i].item(),
                    "abpf_frac_zero_parents": abpf_frac_zero_parents[
                        i
                    ].item(),
                }
            )
        return results
