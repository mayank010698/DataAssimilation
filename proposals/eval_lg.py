"""
Oracle-comparison evaluation for LinearGaussian Benchmark B.

Computes diagnostic metrics in five areas:
  1. Proposal accuracy  — mean/covariance error, Gaussian 2-Wasserstein vs oracle
  2. Log-density accuracy — MAE/RMSE/bias/95th-pct of log q_phi vs log q*
  3. Weight-level accuracy — incremental log-weight errors, ESS comparison
  4. Kalman-filter comparison — particle filter vs exact Kalman posterior
  5. Forward KL KL(q*||q_phi) — both CNF and discrete-Euler density estimates

Usage:
    python proposals/eval_lg.py \\
        --checkpoint path/to/mf.ckpt \\
        --data_dir   path/to/lg_dataset \\
        --n_samples  256 \\
        --n_pairs    1000 \\
        --device     cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm  # type: ignore[reportMissingModuleSource]
except ImportError:
    # Fallback no-op iterator if tqdm isn't available.
    def tqdm(iterable, **kwargs):
        return iterable

# Ensure project root is importable
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from data import (
    DataAssimilationDataModule,
    LinearGaussian,
    load_config_yaml,
)
from proposals.eval_proposal import (
    load_proposal_from_checkpoint,
    _get_inference_sampling_steps,
    _get_inference_likelihood_steps,
    _set_inference_sampling_steps,
    _set_inference_likelihood_steps,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

def kalman_filter(
    system: LinearGaussian,
    observations: torch.Tensor,
    x0_mean: Optional[torch.Tensor] = None,
    P0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward Kalman filter for the LinearGaussian system.

    Args:
        system:       LinearGaussian instance (with precomputed matrices).
        observations: (T, m) tensor of observations y_1, …, y_T.
        x0_mean:      Prior mean (d,). Defaults to system._init_mean.
        P0:           Prior covariance (d, d). Defaults to system._init_cov.

    Returns:
        means:  (T, d)  — posterior means E[x_t | y_{1:t}]
        covs:   (T, d, d) — posterior covariances
    """
    device = observations.device
    system._to_device(device)

    A = system._A.to(device)
    H = system._H.to(device)
    Q = system._Q.to(device)
    R = system._R.to(device)

    T = observations.shape[0]
    d = system._d
    m = system._m

    if x0_mean is None:
        x0_mean = system._init_mean.to(device)
    if P0 is None:
        P0 = system._init_cov.to(device)

    # Predict from x0 prior to t=1
    x_pred = (A @ x0_mean.unsqueeze(-1)).squeeze(-1)   # (d,)
    P_pred = A @ P0 @ A.T + Q                          # (d, d)

    means = []
    covs  = []

    for t in range(T):
        y = observations[t]   # (m,)

        # Innovation
        S = H @ P_pred @ H.T + R                           # (m, m)
        K = P_pred @ H.T @ torch.linalg.inv(S)             # (d, m)  Kalman gain

        # Update
        innov  = y - (H @ x_pred.unsqueeze(-1)).squeeze(-1)
        x_filt = x_pred + (K @ innov.unsqueeze(-1)).squeeze(-1)
        P_filt = (torch.eye(d, device=device) - K @ H) @ P_pred

        means.append(x_filt)
        covs.append(P_filt)

        # Predict for t+1
        if t < T - 1:
            x_pred = (A @ x_filt.unsqueeze(-1)).squeeze(-1)
            P_pred = A @ P_filt @ A.T + Q

    return torch.stack(means), torch.stack(covs)   # (T, d), (T, d, d)


# ---------------------------------------------------------------------------
# Gaussian 2-Wasserstein distance
# ---------------------------------------------------------------------------

def gaussian_w2(mu1, Sigma1, mu2, Sigma2):
    """2-Wasserstein distance between N(mu1, Sigma1) and N(mu2, Sigma2).

    W2^2 = ||mu1 - mu2||^2 + tr(Sigma1 + Sigma2 - 2*(Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2})

    Args:
        mu1, mu2:     (d,) tensors
        Sigma1, Sigma2: (d, d) tensors
    Returns:
        scalar float
    """
    diff_mu = (mu1 - mu2).double()
    mu_sq   = (diff_mu @ diff_mu).item()

    S1 = Sigma1.double()
    S2 = Sigma2.double()

    # Sigma1^{1/2} via eigendecomposition
    L1, V1 = torch.linalg.eigh(S1)
    L1 = L1.clamp(min=1e-12)
    S1_sqrt = V1 @ torch.diag(L1.sqrt()) @ V1.T

    # M = S1^{1/2} S2 S1^{1/2}
    M = S1_sqrt @ S2 @ S1_sqrt
    Lm, Vm = torch.linalg.eigh(M)
    Lm = Lm.clamp(min=1e-12)
    M_sqrt = Vm @ torch.diag(Lm.sqrt()) @ Vm.T

    tr_term = (torch.trace(S1) + torch.trace(S2) - 2.0 * torch.trace(M_sqrt)).item()
    w2_sq   = mu_sq + tr_term
    return float(np.sqrt(max(w2_sq, 0.0)))


def _run_discrete_lowdim_sanity_check(
    model,
    x_prev_s: torch.Tensor,
    y_curr_s: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    max_points: int = 256,
) -> Dict[str, float]:
    """Lightweight sanity check for d=2/3 discrete Euler log-prob implementation.

    The check verifies finite outputs on model samples and, for d=2, compares
    binned empirical log-density trends with binned model log-density trends.
    """
    d = int(model.state_dim)
    if d not in (2, 3):
        return {"sanity/discrete_check_run": 0.0}

    n = min(max_points, x_prev_s.shape[0])
    if n == 0:
        return {"sanity/discrete_check_run": 0.0}

    with torch.no_grad():
        xp = x_prev_s[:n]
        yc = y_curr_s[:n]
        x_samp_s = model.sample(xp, yc)
        logp_disc = model.log_prob_discrete_euler(x_samp_s, xp, yc)
        finite_rate = torch.isfinite(logp_disc).float().mean().item()

    stats = {
        "sanity/discrete_check_run": 1.0,
        "sanity/discrete_logprob_finite_rate": float(finite_rate),
    }

    # Optional weak histogram consistency in 2D only.
    if d == 2:
        x_samp_u = x_samp_s * state_std + state_mean
        x_np = x_samp_u.detach().cpu().numpy()
        lp_np = logp_disc.detach().cpu().numpy()
        finite = np.isfinite(lp_np)
        x_np = x_np[finite]
        lp_np = lp_np[finite]
        if x_np.shape[0] >= 32:
            bins = 8
            hist, x_edges, y_edges = np.histogram2d(x_np[:, 0], x_np[:, 1], bins=bins)
            hist = hist / max(hist.sum(), 1.0)
            cell_mass = []
            cell_logp = []
            for i in range(bins):
                for j in range(bins):
                    x_lo, x_hi = x_edges[i], x_edges[i + 1]
                    y_lo, y_hi = y_edges[j], y_edges[j + 1]
                    mask = (
                        (x_np[:, 0] >= x_lo) & (x_np[:, 0] < x_hi) &
                        (x_np[:, 1] >= y_lo) & (x_np[:, 1] < y_hi)
                    )
                    if not np.any(mask):
                        continue
                    mass = hist[i, j]
                    if mass <= 0:
                        continue
                    cell_mass.append(float(np.log(mass + 1e-12)))
                    cell_logp.append(float(lp_np[mask].mean()))
            if len(cell_mass) >= 5:
                corr = np.corrcoef(np.array(cell_mass), np.array(cell_logp))[0, 1]
                stats["sanity/discrete_hist_logcorr"] = float(corr)
            else:
                stats["sanity/discrete_hist_logcorr"] = float("nan")
        else:
            stats["sanity/discrete_hist_logcorr"] = float("nan")
    return stats


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_lg_eval(
    checkpoint_path: str,
    data_dir: str,
    n_samples: int = 256,
    n_pairs: int = 1000,
    batch_size: int = 64,
    device: str = "cpu",
    num_sampling_steps: Optional[int] = None,
    num_likelihood_steps: Optional[int] = None,
    sampling_grid_type: Optional[str] = None,
    sampling_grid_param: Optional[Dict] = None,
    likelihood_grid_type: Optional[str] = None,
    likelihood_grid_param: Optional[Dict] = None,
    seed: Optional[int] = None,
    wandb_run=None,
) -> Dict:
    """Run the full oracle-comparison evaluation.

    Returns a dict of all metric values.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device(device)
    data_dir = Path(data_dir)
    config_path = data_dir / "config.yaml"
    config = load_config_yaml(config_path)

    # Validate system
    if config.system_params.get("system_name") != "linear_gaussian":
        raise ValueError(
            f"eval_lg.py requires a LinearGaussian dataset, "
            f"but detected system_name={config.system_params.get('system_name')!r}"
        )

    # Load model
    model = load_proposal_from_checkpoint(checkpoint_path)
    if num_sampling_steps is not None:
        _set_inference_sampling_steps(model, num_sampling_steps)
    if num_likelihood_steps is not None:
        _set_inference_likelihood_steps(model, num_likelihood_steps)
    if sampling_grid_type is not None:
        if not hasattr(model, "sampling_grid_type"):
            raise AttributeError("Loaded proposal model does not support sampling_grid_type")
        model.sampling_grid_type = sampling_grid_type
    if likelihood_grid_type is not None:
        if not hasattr(model, "likelihood_grid_type"):
            raise AttributeError("Loaded proposal model does not support likelihood_grid_type")
        model.likelihood_grid_type = likelihood_grid_type
    if sampling_grid_param is not None:
        if not hasattr(model, "sampling_grid_param"):
            raise AttributeError("Loaded proposal model does not support sampling_grid_param")
        model.sampling_grid_param = sampling_grid_param
    if likelihood_grid_param is not None:
        if not hasattr(model, "likelihood_grid_param"):
            raise AttributeError("Loaded proposal model does not support likelihood_grid_param")
        model.likelihood_grid_param = likelihood_grid_param
    if (
        sampling_grid_type is not None
        and str(sampling_grid_type).lower() == "uniform"
        and sampling_grid_param is None
        and hasattr(model, "sampling_grid_param")
    ):
        model.sampling_grid_param = None
    model.to(device)
    model.eval()

    logger.info(
        f"Loaded model: sampling_steps={_get_inference_sampling_steps(model)}, "
        f"likelihood_steps={_get_inference_likelihood_steps(model)}"
    )
    if hasattr(model, "sampling_grid_type") and hasattr(model, "likelihood_grid_type"):
        logger.info(
            "Using grids: "
            f"sampling_grid_type={model.sampling_grid_type}, "
            f"sampling_grid_param={getattr(model, 'sampling_grid_param', None)}, "
            f"likelihood_grid_type={model.likelihood_grid_type}, "
            f"likelihood_grid_param={getattr(model, 'likelihood_grid_param', None)}"
        )
    effective_sampling_steps = _get_inference_sampling_steps(model)
    effective_likelihood_steps = _get_inference_likelihood_steps(model)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "effective_num_sampling_steps": effective_sampling_steps,
                "effective_num_likelihood_steps": effective_likelihood_steps,
                "effective_sampling_grid_type": getattr(model, "sampling_grid_type", None),
                "effective_sampling_grid_param": getattr(model, "sampling_grid_param", None),
                "effective_likelihood_grid_type": getattr(model, "likelihood_grid_type", None),
                "effective_likelihood_grid_param": getattr(model, "likelihood_grid_param", None),
            },
            allow_val_change=True,
        )
        wandb_run.log(
            {
                "config/effective_num_sampling_steps": float(effective_sampling_steps),
                "config/effective_num_likelihood_steps": float(effective_likelihood_steps),
            }
        )

    # Load data
    data_module = DataAssimilationDataModule(
        config=config,
        system_class=LinearGaussian,
        data_dir=str(data_dir),
        batch_size=batch_size,
    )
    data_module.setup("test")
    system: LinearGaussian = data_module.system
    system._to_device(device)

    # Scaling stats (state)
    state_mean = system.init_mean.to(device)   # (d,)
    state_std  = system.init_std.to(device)    # (d,)

    # Obs scaling stats
    obs_mean = (
        data_module.obs_scaler_mean.to(device)
        if data_module.obs_scaler_mean is not None
        else torch.zeros(system._m, device=device)
    )
    obs_std = (
        data_module.obs_scaler_std.to(device)
        if data_module.obs_scaler_std is not None
        else torch.ones(system._m, device=device)
    )

    # Log Jacobian correction for converting log q from scaled → unscaled space:
    # log q_unscaled(x) = log q_scaled(x_scaled) - sum(log(state_std))
    log_jacobian = -state_std.log().sum().item()

    # Collect conditioning pairs (unscaled) from test set
    test_dataset = data_module.test_dataset
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    x_prevs_unscaled  = []
    x_currs_unscaled  = []
    y_currs_unscaled  = []
    x_prevs_scaled    = []
    y_currs_scaled    = []

    total_collected = 0
    for batch in loader:
        if total_collected >= n_pairs:
            break
        has_obs = batch["has_observation"].squeeze(-1)
        if not has_obs.any():
            continue

        mask = has_obs.bool()
        n_keep = min(mask.sum().item(), n_pairs - total_collected)

        x_prevs_unscaled.append(batch["x_prev"][mask][:n_keep])
        x_currs_unscaled.append(batch["x_curr"][mask][:n_keep])
        y_currs_unscaled.append(batch["y_curr"][mask][:n_keep])
        x_prevs_scaled.append(batch["x_prev_scaled"][mask][:n_keep])
        y_currs_scaled.append(batch["y_curr_scaled"][mask][:n_keep])
        total_collected += n_keep

    if total_collected == 0:
        raise RuntimeError("No observed time steps found in test dataset.")

    x_prev_u  = torch.cat(x_prevs_unscaled).to(device)   # (N, d)
    x_curr_u  = torch.cat(x_currs_unscaled).to(device)   # (N, d)
    y_curr_u  = torch.cat(y_currs_unscaled).to(device)   # (N, m)
    x_prev_s  = torch.cat(x_prevs_scaled).to(device)     # (N, d)
    y_curr_s  = torch.cat(y_currs_scaled).to(device)     # (N, m)
    N = x_prev_u.shape[0]
    logger.info(f"Collected {N} conditioning pairs for evaluation.")
    n_batches = (N + batch_size - 1) // batch_size

    # -----------------------------------------------------------------------
    # LEVEL 1: Proposal accuracy
    # -----------------------------------------------------------------------
    logger.info("Level 1: Proposal accuracy ...")

    # Oracle quantities (constant covariance, per-pair mean)
    oracle_mu, oracle_Sigma = system.optimal_proposal_params(x_prev_u, y_curr_u)
    # oracle_mu: (N, d),  oracle_Sigma: (d, d)

    mean_errors  = []
    mean_mahalanobis = []
    cov_errors   = []
    w2_distances = []
    oracle_Sigma_inv = torch.linalg.inv(oracle_Sigma)

    # Process in batches to avoid OOM
    for i in tqdm(
        range(0, N, batch_size),
        total=n_batches,
        desc="Level 1 batches",
        leave=False,
    ):
        xp_s = x_prev_s[i : i + batch_size]   # scaled
        yc_s = y_curr_s[i : i + batch_size]
        xp_u = x_prev_u[i : i + batch_size]   # unscaled (for oracle)
        yc_u = y_curr_u[i : i + batch_size]

        B = xp_s.shape[0]

        # Sample from learned proposal (scaled space)
        with torch.no_grad():
            # Expand to (B * n_samples, d)
            xp_rep = xp_s.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            yc_rep = yc_s.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            x_samp_s = model.sample(xp_rep, yc_rep)          # (B*n_samples, d) scaled

        x_samp_s = x_samp_s.reshape(B, n_samples, -1)        # (B, M, d) scaled
        x_samp_u = x_samp_s * state_std + state_mean          # unscale → (B, M, d)

        # Estimated mean and covariance
        mu_hat    = x_samp_u.mean(dim=1)                      # (B, d)
        diff_samp = x_samp_u - mu_hat.unsqueeze(1)            # (B, M, d)
        Sigma_hat = (diff_samp.unsqueeze(-1) * diff_samp.unsqueeze(-2)).mean(dim=1)  # (B,d,d)

        # Oracle mean for this batch
        om = oracle_mu[i : i + B]  # (B, d)

        # 1.1 Mean error
        diff_mu = mu_hat - om
        err = diff_mu.norm(dim=-1)   # (B,)
        mean_errors.append(err.cpu())
        # Normalized mean error in oracle units; easier to compare across datasets/scales.
        maha = torch.einsum("bi,ij,bj->b", diff_mu, oracle_Sigma_inv, diff_mu)
        mean_mahalanobis.append(maha.cpu())

        # 1.2 Covariance Frobenius error
        cov_err = (Sigma_hat - oracle_Sigma.unsqueeze(0)).norm(dim=(-2, -1))  # (B,)
        cov_errors.append(cov_err.cpu())

        # 1.3 Gaussian W2 (expensive: per-pair, subsample for speed)
        n_w2 = min(B, 16)
        for j in range(n_w2):
            w2 = gaussian_w2(
                mu_hat[j].cpu(),    oracle_Sigma.cpu(),
                om[j].cpu(),        oracle_Sigma.cpu(),
            )
            # Sigma_hat[j] vs oracle_Sigma — full W2
            w2_full = gaussian_w2(
                mu_hat[j].cpu(),   Sigma_hat[j].cpu(),
                om[j].cpu(),       oracle_Sigma.cpu(),
            )
            w2_distances.append(w2_full)

    mean_errors  = torch.cat(mean_errors).numpy()
    mean_mahalanobis = torch.cat(mean_mahalanobis).numpy()
    cov_errors   = torch.cat(cov_errors).numpy()
    w2_arr       = np.array(w2_distances)

    lvl1 = {
        "proposal/mean_error_mean":  float(mean_errors.mean()),
        "proposal/mean_error_std":   float(mean_errors.std()),
        "proposal/mean_mahalanobis_mean": float(mean_mahalanobis.mean()),
        "proposal/mean_mahalanobis_std":  float(mean_mahalanobis.std()),
        "proposal/cov_frob_error_mean": float(cov_errors.mean()),
        "proposal/cov_frob_error_std":  float(cov_errors.std()),
        "proposal/gaussian_w2_mean": float(w2_arr.mean()) if len(w2_arr) else float("nan"),
    }
    _log_metrics(lvl1, "Level 1", wandb_run)

    # -----------------------------------------------------------------------
    # LEVEL 2: Log-density accuracy
    # -----------------------------------------------------------------------
    logger.info("Level 2: Log-density accuracy ...")

    log_density_errors = []   # log q_phi(unscaled) - log q*(unscaled)
    log_density_oracle = []   # log q*(unscaled), to expose oracle scale

    for i in tqdm(
        range(0, N, batch_size),
        total=n_batches,
        desc="Level 2 batches",
        leave=False,
    ):
        xp_s = x_prev_s[i : i + batch_size]
        yc_s = y_curr_s[i : i + batch_size]
        xp_u = x_prev_u[i : i + batch_size]
        yc_u = y_curr_u[i : i + batch_size]
        xc_u = x_curr_u[i : i + batch_size]   # ground-truth x_curr (unscaled)

        B = xp_s.shape[0]

        # Scale x_curr for model
        xc_s = (xc_u - state_mean) / state_std

        with torch.no_grad():
            log_q_phi_scaled = model.log_prob(xc_s, xp_s, yc_s)   # (B,) in scaled space

        # Jacobian correction: log q_unscaled = log q_scaled - sum(log std)
        log_q_phi = log_q_phi_scaled + log_jacobian   # (B,)

        # Oracle log density
        log_q_star = system.log_optimal_proposal(xc_u, xp_u, yc_u)   # (B,)

        err = (log_q_phi - log_q_star).cpu().numpy()
        log_density_errors.append(err)
        log_density_oracle.append(log_q_star.cpu().numpy())

    log_density_errors = np.concatenate(log_density_errors)
    log_density_oracle = np.concatenate(log_density_oracle)
    log_density_abs    = np.abs(log_density_errors)
    log_density_rmse = float(np.sqrt((log_density_errors ** 2).mean()))
    oracle_std = float(log_density_oracle.std())

    lvl2 = {
        "log_density/mae":        float(log_density_abs.mean()),
        "log_density/rmse":       log_density_rmse,
        "log_density/bias":       float(log_density_errors.mean()),
        "log_density/p95_abs":    float(np.percentile(log_density_abs, 95)),
        "log_density/std":        float(log_density_errors.std()),
        # Error normalized by oracle variability is more interpretable across tasks/checkpoints.
        "log_density/oracle_mean": float(log_density_oracle.mean()),
        "log_density/oracle_std":  oracle_std,
        "log_density/error_over_oracle_std": float(log_density_rmse / (oracle_std + 1e-12)),
    }
    _log_metrics(lvl2, "Level 2", wandb_run)

    # -----------------------------------------------------------------------
    # Forward KL from oracle to learned proposal:
    # - CNF-style density (existing log_prob with divergence integration)
    # - Discrete Euler-map density (new, matched to sample() transport map)
    # Per pair: sample x ~ q*, then mean over samples of (log q*(x) - log q_phi(x)).
    # Aggregate mean/std across conditioning pairs (not over all particles pooled).
    # Small negative values can occur from MC noise; true KL is nonnegative.
    # -----------------------------------------------------------------------
    logger.info("Forward KL (oracle → learned): CNF and discrete-Euler estimates ...")

    kl_qstar_qphi_cnf_per_pair: List[np.ndarray] = []
    kl_qstar_qphi_discrete_per_pair: List[np.ndarray] = []
    discrete_invalid = 0
    discrete_total = 0
    discrete_newton_failures = 0
    discrete_solve_failures = 0
    discrete_slogdet_failures = 0

    for i in tqdm(
        range(0, N, batch_size),
        total=n_batches,
        desc="KL(q*||q_phi) batches",
        leave=False,
    ):
        xp_s = x_prev_s[i : i + batch_size]
        yc_s = y_curr_s[i : i + batch_size]
        xp_u = x_prev_u[i : i + batch_size]
        yc_u = y_curr_u[i : i + batch_size]
        B = xp_s.shape[0]

        with torch.no_grad():
            x_samp_u = system.sample_optimal_proposal(xp_u, yc_u, n_samples=n_samples)
            # (B, M, d) — same draws for log q* and log q_phi
            x_flat_u = x_samp_u.reshape(B * n_samples, -1)
            xp_rep_u = xp_u.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            yc_rep_u = yc_u.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)

            log_q_star = system.log_optimal_proposal(x_flat_u, xp_rep_u, yc_rep_u)

            x_flat_s = (x_flat_u - state_mean) / state_std
            xp_rep_s = xp_s.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            yc_rep_s = yc_s.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            log_q_phi_cnf = model.log_prob(x_flat_s, xp_rep_s, yc_rep_s) + log_jacobian

            # Discrete Euler inverse-Jacobian evaluation is expensive; chunk to limit memory.
            disc_chunk = 128
            log_q_phi_discrete_chunks = []
            for j in range(0, x_flat_s.shape[0], disc_chunk):
                x_ch = x_flat_s[j : j + disc_chunk]
                xp_ch = xp_rep_s[j : j + disc_chunk]
                yc_ch = yc_rep_s[j : j + disc_chunk]
                lp_ch = model.log_prob_discrete_euler(x_ch, xp_ch, yc_ch) + log_jacobian
                log_q_phi_discrete_chunks.append(lp_ch)
                disc_stats = getattr(model, "last_discrete_euler_stats", {})
                discrete_invalid += int(disc_stats.get("invalid_count", 0))
                discrete_total += int(disc_stats.get("batch_size", 0))
                discrete_newton_failures += int(disc_stats.get("newton_failures", 0))
                discrete_solve_failures += int(disc_stats.get("linear_solve_failures", 0))
                discrete_slogdet_failures += int(disc_stats.get("nonpositive_slogdet", 0))
            log_q_phi_discrete = torch.cat(log_q_phi_discrete_chunks, dim=0)

        kl_terms_cnf = (log_q_star - log_q_phi_cnf).reshape(B, n_samples)
        kl_pair_cnf = kl_terms_cnf.mean(dim=1)
        kl_qstar_qphi_cnf_per_pair.append(kl_pair_cnf.cpu().numpy())

        kl_terms_discrete = (log_q_star - log_q_phi_discrete).reshape(B, n_samples)
        kl_pair_discrete = torch.nanmean(kl_terms_discrete, dim=1)
        kl_qstar_qphi_discrete_per_pair.append(kl_pair_discrete.cpu().numpy())

    kl_qstar_qphi_cnf_arr = np.concatenate(kl_qstar_qphi_cnf_per_pair)
    kl_qstar_qphi_discrete_arr = np.concatenate(kl_qstar_qphi_discrete_per_pair)
    discrete_finite = np.isfinite(kl_qstar_qphi_discrete_arr)
    if not np.any(discrete_finite):
        logger.warning("All discrete-Euler KL values are non-finite.")

    # Tiny low-dim validation hook (only runs for d=2/3).
    sanity_stats = _run_discrete_lowdim_sanity_check(
        model=model,
        x_prev_s=x_prev_s,
        y_curr_s=y_curr_s,
        state_mean=state_mean,
        state_std=state_std,
    )

    kl_dist = {
        "dist/kl_qstar_qphi_cnf_mean": float(kl_qstar_qphi_cnf_arr.mean()),
        "dist/kl_qstar_qphi_cnf_std": float(kl_qstar_qphi_cnf_arr.std()),
        "dist/kl_qstar_qphi_discrete_mean": float(np.nanmean(kl_qstar_qphi_discrete_arr)),
        "dist/kl_qstar_qphi_discrete_std": float(np.nanstd(kl_qstar_qphi_discrete_arr)),
        "dist/kl_qstar_qphi_discrete_valid_rate": float(np.mean(discrete_finite.astype(np.float32))),
        "dist/discrete_inverse_invalid_count": float(discrete_invalid),
        "dist/discrete_inverse_total_count": float(discrete_total),
        "dist/discrete_inverse_invalid_rate": float(discrete_invalid / max(discrete_total, 1)),
        "dist/discrete_inverse_newton_failures": float(discrete_newton_failures),
        "dist/discrete_inverse_linear_solve_failures": float(discrete_solve_failures),
        "dist/discrete_inverse_nonpositive_slogdet": float(discrete_slogdet_failures),
        **sanity_stats,
    }
    _log_metrics(kl_dist, "KL(q*||q_phi)", wandb_run)

    # -----------------------------------------------------------------------
    # LEVEL 3: Weight-level accuracy
    # -----------------------------------------------------------------------
    logger.info("Level 3: Weight-level accuracy ...")

    incr_log_weight_errors = []
    centered_incr_log_weight_errors = []
    log_weight_oracle_values = []
    log_like_values = []
    log_trans_values = []
    log_q_phi_values = []
    log_q_star_values = []
    ess_approx_list = []
    ess_oracle_list = []

    for i in tqdm(
        range(0, N, batch_size),
        total=n_batches,
        desc="Level 3 batches",
        leave=False,
    ):
        xp_s = x_prev_s[i : i + batch_size]
        yc_s = y_curr_s[i : i + batch_size]
        xp_u = x_prev_u[i : i + batch_size]
        yc_u = y_curr_u[i : i + batch_size]
        B    = xp_s.shape[0]

        # Draw samples from learned proposal (scaled)
        with torch.no_grad():
            xp_rep = xp_s.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            yc_rep = yc_s.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            x_samp_s = model.sample(xp_rep, yc_rep)            # (B*M, d)
            xp_rep_u = xp_u.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            x_samp_u = x_samp_s * state_std + state_mean        # unscale

            # log q_phi (unscaled) for each sample
            xp_rep_s_for_lp = xp_s.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            yc_rep_s_for_lp = yc_s.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
            log_q_phi_s = model.log_prob(x_samp_s, xp_rep_s_for_lp, yc_rep_s_for_lp)  # (B*M,)
        log_q_phi_u = log_q_phi_s + log_jacobian   # (B*M,) unscaled

        yc_rep_u = yc_u.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)

        # log p(y | x_samp)
        log_like  = system.log_likelihood(yc_rep_u, x_samp_u)   # (B*M,)
        # log p(x_samp | x_prev)
        log_trans = system.log_transition(x_samp_u, xp_rep_u)   # (B*M,)
        # log q* (unscaled)
        log_q_star_s = system.log_optimal_proposal(x_samp_u, xp_rep_u, yc_rep_u)  # (B*M,)

        # Incremental log weights (unnormalized, in unscaled space)
        log_w_approx = log_like + log_trans - log_q_phi_u    # (B*M,)
        log_w_oracle = log_like + log_trans - log_q_star_s   # (B*M,)
        log_weight_oracle_values.append(log_w_oracle.cpu().numpy())
        log_like_values.append(log_like.cpu().numpy())
        log_trans_values.append(log_trans.cpu().numpy())
        log_q_phi_values.append(log_q_phi_u.cpu().numpy())
        log_q_star_values.append(log_q_star_s.cpu().numpy())

        # Reshape to (B, M)
        log_w_approx = log_w_approx.reshape(B, n_samples)
        log_w_oracle = log_w_oracle.reshape(B, n_samples)

        # Log-weight error per sample
        err = (log_w_approx - log_w_oracle).cpu().numpy()   # (B, M)
        incr_log_weight_errors.append(err)
        # Centering removes per-pair offsets and isolates weight-shape distortion.
        centered_err = (
            (log_w_approx - log_w_approx.mean(dim=1, keepdim=True))
            - (log_w_oracle - log_w_oracle.mean(dim=1, keepdim=True))
        ).cpu().numpy()
        centered_incr_log_weight_errors.append(centered_err)

        # ESS per conditioning pair: ESS = (sum w)^2 / sum(w^2)
        def compute_ess(lw_bm: torch.Tensor) -> np.ndarray:
            # lw_bm: (B, M)
            lw_bm = lw_bm - lw_bm.max(dim=1, keepdim=True).values   # stabilize
            w = torch.exp(lw_bm)  # (B, M)
            ess = w.sum(dim=1) ** 2 / (w ** 2).sum(dim=1)            # (B,)
            return ess.cpu().numpy()

        ess_approx_list.append(compute_ess(log_w_approx))
        ess_oracle_list.append(compute_ess(log_w_oracle))

    incr_errs = np.concatenate([e.ravel() for e in incr_log_weight_errors])
    centered_incr_errs = np.concatenate([e.ravel() for e in centered_incr_log_weight_errors])
    log_weight_oracle_arr = np.concatenate(log_weight_oracle_values)
    log_like_arr = np.concatenate(log_like_values)
    log_trans_arr = np.concatenate(log_trans_values)
    log_q_phi_arr = np.concatenate(log_q_phi_values)
    log_q_star_arr = np.concatenate(log_q_star_values)
    ess_approx = np.concatenate(ess_approx_list)
    ess_oracle  = np.concatenate(ess_oracle_list)
    log_weight_rmse = float(np.sqrt((incr_errs ** 2).mean()))
    log_weight_oracle_std = float(log_weight_oracle_arr.std())
    ess_gap = ess_oracle - ess_approx

    lvl3 = {
        "weights/log_weight_mae":     float(np.abs(incr_errs).mean()),
        "weights/log_weight_rmse":    log_weight_rmse,
        "weights/log_weight_variance": float(incr_errs.var()),
        "weights/log_weight_oracle_mean": float(log_weight_oracle_arr.mean()),
        "weights/log_weight_oracle_std":  log_weight_oracle_std,
        "weights/log_weight_rmse_over_oracle_std": float(log_weight_rmse / (log_weight_oracle_std + 1e-12)),
        "weights/log_like_mean": float(log_like_arr.mean()),
        "weights/log_like_std":  float(log_like_arr.std()),
        "weights/log_trans_mean": float(log_trans_arr.mean()),
        "weights/log_trans_std":  float(log_trans_arr.std()),
        "weights/log_q_phi_mean": float(log_q_phi_arr.mean()),
        "weights/log_q_phi_std":  float(log_q_phi_arr.std()),
        "weights/log_q_star_mean": float(log_q_star_arr.mean()),
        "weights/log_q_star_std":  float(log_q_star_arr.std()),
        "weights/centered_log_weight_mae":  float(np.abs(centered_incr_errs).mean()),
        "weights/centered_log_weight_rmse": float(np.sqrt((centered_incr_errs ** 2).mean())),
        "weights/ess_approx_mean":    float(ess_approx.mean()),
        "weights/ess_oracle_mean":    float(ess_oracle.mean()),
        "weights/ess_ratio_mean":     float((ess_approx / (ess_oracle + 1e-8)).mean()),
        "weights/ess_gap_mean":       float(ess_gap.mean()),
        "weights/ess_gap_std":        float(ess_gap.std()),
    }
    _log_metrics(lvl3, "Level 3", wandb_run)

    # -----------------------------------------------------------------------
    # LEVEL 4: Kalman filter comparison
    # -----------------------------------------------------------------------
    logger.info("Level 4: Kalman filter comparison ...")

    test_trajs = test_dataset.trajectories    # (n_test, T, d)
    test_obs   = test_dataset.observations    # (n_test, T_obs, m)
    n_test_trajs = min(test_trajs.shape[0], 50)   # limit for speed

    kf_rmse_list   = []
    kf_cov_err_list = []

    for traj_i in tqdm(
        range(n_test_trajs),
        total=n_test_trajs,
        desc="Level 4 trajectories",
        leave=False,
    ):
        traj_gt = torch.tensor(test_trajs[traj_i], dtype=torch.float32).to(device)  # (T, d)
        obs_seq  = torch.tensor(test_obs[traj_i], dtype=torch.float32).to(device)   # (T_obs, m)

        # For freq=1, T_obs == T; handle general case
        obs_mask_np = test_dataset.obs_mask                 # (T,)
        T  = traj_gt.shape[0]
        obs_times = np.where(obs_mask_np)[0]

        # Build full T-length obs sequence (fill non-observed steps with zeros, won't be used)
        full_obs = torch.zeros(T, system._m, device=device)
        for idx_obs, t_obs in enumerate(obs_times):
            if idx_obs < obs_seq.shape[0]:
                full_obs[t_obs] = obs_seq[idx_obs]

        # Run Kalman filter on all observed steps
        kf_means, kf_covs = kalman_filter(system, full_obs[obs_times])
        # kf_means: (T_obs, d),  kf_covs: (T_obs, d, d)

        gt_at_obs = traj_gt[obs_times]  # (T_obs, d)
        rmse = (kf_means - gt_at_obs).norm(dim=-1).mean().item()
        kf_rmse_list.append(rmse)

        # Covariance error vs oracle Sigma_star (just as a sanity check)
        # (Kalman cov is the posterior, not the proposal cov, but useful to report)
        kf_cov_err_list.append(kf_covs.norm(dim=(-2, -1)).mean().item())

    lvl4 = {
        "kalman/rmse_mean":    float(np.mean(kf_rmse_list)),
        "kalman/rmse_std":     float(np.std(kf_rmse_list)),
        "kalman/cov_norm_mean": float(np.mean(kf_cov_err_list)),
    }
    _log_metrics(lvl4, "Level 4", wandb_run)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    all_metrics = {**lvl1, **lvl2, **kl_dist, **lvl3, **lvl4}

    print("\n" + "=" * 70)
    print("LinearGaussian Oracle Evaluation Summary")
    print("=" * 70)
    for k, v in sorted(all_metrics.items()):
        print(f"  {k:<45} {v:.6f}")
    print("=" * 70)

    if wandb_run is not None:
        wandb_run.log(all_metrics)

    return all_metrics


def _log_metrics(metrics: Dict, label: str, wandb_run=None):
    logger.info(f"  [{label}]")
    for k, v in metrics.items():
        logger.info(f"    {k}: {v:.5f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oracle-comparison evaluation for LinearGaussian Benchmark B."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to proposal checkpoint (.ckpt)")
    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Path to linear_gaussian dataset directory")
    parser.add_argument("--n_samples",  type=int, default=256,
                        help="Proposal samples per conditioning pair (default: 256)")
    parser.add_argument("--n_pairs",    type=int, default=1000,
                        help="Number of conditioning pairs to evaluate (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device",     type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--num_sampling_steps",  type=int, default=None,
                        help="Override sampling steps in the proposal model")
    parser.add_argument("--num_likelihood_steps", type=int, default=None,
                        help="Override likelihood integration steps")
    parser.add_argument("--sampling_grid_type", type=str, default=None,
                        choices=["uniform", "cosine", "power_front", "power_back", "piecewise"],
                        help="Override sampling integration grid type")
    parser.add_argument("--sampling_grid_param", type=str, default=None,
                        help="JSON object for sampling grid params, e.g. '{\"gamma\": 2.0}'")
    parser.add_argument("--likelihood_grid_type", type=str, default=None,
                        choices=["uniform", "cosine", "power_front", "power_back", "piecewise"],
                        help="Override likelihood integration grid type")
    parser.add_argument("--likelihood_grid_param", type=str, default=None,
                        help="JSON object for likelihood grid params")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (skip W&B if not set)")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_level",  type=str, default="INFO")

    args = parser.parse_args()

    sampling_grid_param = None
    likelihood_grid_param = None
    if args.sampling_grid_param is not None:
        sampling_grid_param = json.loads(args.sampling_grid_param)
    if args.likelihood_grid_param is not None:
        likelihood_grid_param = json.loads(args.likelihood_grid_param)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    wandb_run = None
    if args.wandb_project is not None:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    try:
        run_lg_eval(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            n_samples=args.n_samples,
            n_pairs=args.n_pairs,
            batch_size=args.batch_size,
            device=args.device,
            num_sampling_steps=args.num_sampling_steps,
            num_likelihood_steps=args.num_likelihood_steps,
            sampling_grid_type=args.sampling_grid_type,
            sampling_grid_param=sampling_grid_param,
            likelihood_grid_type=args.likelihood_grid_type,
            likelihood_grid_param=likelihood_grid_param,
            seed=args.seed,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
