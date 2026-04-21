r"""Conditional Density Estimator (CDE) heads for the Paige-Wood
inference-network proposal.

Each head takes a feature vector ``h \in R^H`` produced by a feature
backbone (see :mod:`feature_backbones`) and parameterises a conditional
density ``q_eta(x_t | h)`` over the continuous state ``x_t \in R^D``.

All heads share the same interface:

    forward(h)       -> dict of distribution parameters
    log_prob(x, p)   -> (B,)   closed-form log-density in nats
    sample(p, n=1)   -> x      reparameterised sample of shape (B, D)
                                (or (B, n, D) when n > 1)

The implementations here are deliberately minimal / numerically sane:

  * All predicted standard deviations go through ``softplus`` and are
    clamped to a floor ``min_sigma`` to prevent the classical MDN
    variance-collapse.
  * Mixture log-weights use ``log_softmax`` (never ``log(softmax(x))``)
    and mixture densities use ``torch.logsumexp`` for numerical stability.
  * Sampling for mixture heads uses Gumbel-max on the log-weights followed
    by a single reparameterised Gaussian draw per example.

Heads provided:

  * :class:`GaussianHead`       — single diagonal Gaussian (MDN-1).
  * :class:`MDNHead`            — per-coordinate mixture of K Gaussians.
                                  Dimensions are factorised given ``h``.
  * :class:`JointMoGHead`       — joint mixture of K diagonal Gaussians
                                  over the whole state vector (default).
  * :class:`RNADEHead`          — conditional autoregressive MoG (faithful
                                  Paige-Wood, à la Uria et al. RNADE).

References
----------
  * Paige & Wood, *Inference Networks for Sequential Monte Carlo in
    Graphical Models*, ICML 2016.
  * Bishop, *Mixture Density Networks*, 1994 (§3.1 for the canonical MDN
    parameterisation).
  * Uria, Murray & Larochelle, *RNADE: The real-valued neural
    autoregressive density estimator*, NeurIPS 2013.

The shapes convention is:
  * ``B`` = batch size.
  * ``D`` = state dimension.
  * ``K`` = number of mixture components.
  * ``H`` = backbone feature dimension.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base class.
# ---------------------------------------------------------------------------


class BaseCDEHead(nn.Module):
    """Common contract for all CDE heads.

    Subclasses implement ``forward``, ``log_prob``, and ``sample``. The
    ``kind`` attribute is set by the factory; it is used by the inference
    network module when logging and when enforcing head-specific assumptions
    (e.g. "jointly-modelled dimensions").
    """

    kind: str = "base"
    feature_dim: int
    state_dim: int

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def log_prob(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def sample(
        self, params: Dict[str, torch.Tensor], n: int = 1
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def mean(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the conditional mean of ``q_eta(x_t | h)``.

        Subclasses override this when the mean has a closed form that is
        cheap to compute (it is useful for validation-time one-step RMSE
        and for plugging the head into a maximum-a-posteriori-style
        proposal).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Option 1: single diagonal Gaussian.
# ---------------------------------------------------------------------------


class GaussianHead(BaseCDEHead):
    """Single diagonal Gaussian head.

    ``q(x | h) = N(x; mu(h), diag(sigma(h)^2))``. This is the same density
    family as the NASMC :class:`~proposals.nasmc.GaussianProposal`, but
    takes a pre-computed feature vector rather than the raw
    ``(x_prev, y_t)`` conditioning. It exists here mostly as a baseline
    and sanity-check path when sweeping CDE heads.
    """

    kind = "gaussian"

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        min_sigma: float = 1e-3,
        init_log_sigma: float = -1.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.min_sigma = float(min_sigma)
        self.proj = nn.Linear(feature_dim, 2 * state_dim)
        if zero_init_output:
            nn.init.zeros_(self.proj.weight)
            with torch.no_grad():
                bias = torch.zeros(2 * state_dim)
                bias[state_dim:] = init_log_sigma
                self.proj.bias.copy_(bias)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.proj(h)
        mu, log_sigma_raw = raw.chunk(2, dim=-1)
        sigma = F.softplus(log_sigma_raw).clamp(min=self.min_sigma)
        return {"mu": mu, "sigma": sigma}

    def log_prob(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mu = params["mu"]
        sigma = params["sigma"]
        z = (x - mu) / sigma
        lp = -0.5 * (z * z).sum(dim=-1)
        lp = lp - sigma.log().sum(dim=-1)
        lp = lp - 0.5 * self.state_dim * math.log(2.0 * math.pi)
        return lp

    def sample(self, params: Dict[str, torch.Tensor], n: int = 1) -> torch.Tensor:
        mu = params["mu"]
        sigma = params["sigma"]
        if n == 1:
            eps = torch.randn_like(mu)
            return mu + sigma * eps
        shape = (n,) + tuple(mu.shape)
        eps = torch.randn(shape, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (n, B, D)

    def mean(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return params["mu"]


# ---------------------------------------------------------------------------
# Option 2: per-coordinate MoG ("MDN-K").
# ---------------------------------------------------------------------------


class MDNHead(BaseCDEHead):
    """Per-coordinate mixture of ``K`` Gaussians.

    Dimensions of ``x`` are independent given ``h``:

        q(x | h) = prod_i sum_k alpha_{i,k}(h) * N(x_i; mu_{i,k}(h),
                                                  sigma_{i,k}(h)^2)

    This captures *per-dimension* multimodality but imposes conditional
    independence across dimensions. For state-space models with observation
    coupling that's usually too restrictive; use :class:`JointMoGHead` or
    :class:`RNADEHead` when cross-dim correlation matters.
    """

    kind = "mdn_k"

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        num_components: int,
        min_sigma: float = 1e-3,
        init_log_sigma: float = -1.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_components >= 1, "num_components must be >= 1"
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.K = int(num_components)
        self.min_sigma = float(min_sigma)
        self.proj = nn.Linear(feature_dim, state_dim * self.K * 3)
        if zero_init_output:
            nn.init.zeros_(self.proj.weight)
            with torch.no_grad():
                bias = torch.zeros(state_dim * self.K * 3)
                bias.view(state_dim, self.K, 3)[:, :, 2] = init_log_sigma
                self.proj.bias.copy_(bias)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = h.shape[0]
        raw = self.proj(h).view(B, self.state_dim, self.K, 3)
        log_w = F.log_softmax(raw[..., 0], dim=-1)
        mu = raw[..., 1]
        sigma = F.softplus(raw[..., 2]).clamp(min=self.min_sigma)
        return {"log_w": log_w, "mu": mu, "sigma": sigma}

    def log_prob(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        log_w = params["log_w"]
        mu = params["mu"]
        sigma = params["sigma"]
        x_b = x.unsqueeze(-1)
        z = (x_b - mu) / sigma
        log_comp = (
            -0.5 * z * z
            - sigma.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_px_i = torch.logsumexp(log_w + log_comp, dim=-1)
        return log_px_i.sum(dim=-1)

    def sample(self, params: Dict[str, torch.Tensor], n: int = 1) -> torch.Tensor:
        log_w = params["log_w"]
        mu = params["mu"]
        sigma = params["sigma"]
        if n != 1:
            log_w = log_w.unsqueeze(0).expand(n, *log_w.shape).contiguous()
            mu = mu.unsqueeze(0).expand(n, *mu.shape).contiguous()
            sigma = sigma.unsqueeze(0).expand(n, *sigma.shape).contiguous()
        gumbel = -torch.log(-torch.log(torch.rand_like(log_w).clamp_min(1e-20)).clamp_min(1e-20))
        k = (log_w + gumbel).argmax(dim=-1, keepdim=True)
        mu_k = mu.gather(-1, k).squeeze(-1)
        sigma_k = sigma.gather(-1, k).squeeze(-1)
        eps = torch.randn_like(mu_k)
        return mu_k + sigma_k * eps

    def mean(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = params["log_w"].exp()
        return (w * params["mu"]).sum(dim=-1)


# ---------------------------------------------------------------------------
# Option 3: joint MoG with diagonal covariance — the default.
# ---------------------------------------------------------------------------


class JointMoGHead(BaseCDEHead):
    """Joint mixture of ``K`` diagonal Gaussians over the whole state.

        q(x | h) = sum_k alpha_k(h) * N(x; mu_k(h), diag(sigma_k(h)^2))

    Parameters: ``K * (2 * D + 1)``. Captures global (assignment-driven)
    multimodality and mild cross-dim correlation; closed-form sampling
    and ``log_prob`` — which is the entire point of IN vs. RF. This is
    the recommended default for the Paige-Wood baseline.
    """

    kind = "joint_mog"

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        num_components: int,
        min_sigma: float = 1e-3,
        init_log_sigma: float = -1.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_components >= 1, "num_components must be >= 1"
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.K = int(num_components)
        self.min_sigma = float(min_sigma)
        self.output_size = self.K * (2 * state_dim + 1)
        self.proj = nn.Linear(feature_dim, self.output_size)
        if zero_init_output:
            nn.init.zeros_(self.proj.weight)
            with torch.no_grad():
                bias = torch.zeros(self.output_size)
                bias_view = bias.view(self.K, 2 * state_dim + 1)
                bias_view[:, 1 + state_dim :] = init_log_sigma
                self.proj.bias.copy_(bias)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = h.shape[0]
        raw = self.proj(h).view(B, self.K, 2 * self.state_dim + 1)
        log_w = F.log_softmax(raw[..., 0], dim=-1)
        mu = raw[..., 1 : 1 + self.state_dim]
        sigma = F.softplus(raw[..., 1 + self.state_dim :]).clamp(min=self.min_sigma)
        return {"log_w": log_w, "mu": mu, "sigma": sigma}

    def log_prob(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        log_w = params["log_w"]
        mu = params["mu"]
        sigma = params["sigma"]
        x_b = x.unsqueeze(1)
        z = (x_b - mu) / sigma
        log_comp = (
            -0.5 * (z * z).sum(dim=-1)
            - sigma.log().sum(dim=-1)
            - 0.5 * self.state_dim * math.log(2.0 * math.pi)
        )
        return torch.logsumexp(log_w + log_comp, dim=-1)

    def sample(self, params: Dict[str, torch.Tensor], n: int = 1) -> torch.Tensor:
        log_w = params["log_w"]
        mu = params["mu"]
        sigma = params["sigma"]
        if n != 1:
            log_w = log_w.unsqueeze(0).expand(n, *log_w.shape).contiguous()
            mu = mu.unsqueeze(0).expand(n, *mu.shape).contiguous()
            sigma = sigma.unsqueeze(0).expand(n, *sigma.shape).contiguous()
        gumbel = -torch.log(-torch.log(torch.rand_like(log_w).clamp_min(1e-20)).clamp_min(1e-20))
        k = (log_w + gumbel).argmax(dim=-1)
        k_exp = k.unsqueeze(-1).unsqueeze(-1).expand(*k.shape, 1, mu.shape[-1])
        mu_k = mu.gather(-2, k_exp).squeeze(-2)
        sigma_k = sigma.gather(-2, k_exp).squeeze(-2)
        eps = torch.randn_like(mu_k)
        return mu_k + sigma_k * eps

    def mean(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = params["log_w"].exp().unsqueeze(-1)
        return (w * params["mu"]).sum(dim=-2)


# ---------------------------------------------------------------------------
# Option 4: conditional RNADE — the faithful Paige-Wood head.
# ---------------------------------------------------------------------------


class RNADEHead(BaseCDEHead):
    """Conditional autoregressive mixture-of-Gaussians (RNADE / MADE-MoG).

    Each coordinate is modelled as a mixture of ``K`` Gaussians conditioned
    on the backbone feature ``h`` AND the previous coordinates ``x_{<i}``:

        q(x | h) = prod_i sum_k alpha_{i,k}(h, x_{<i})
                        N(x_i; mu_{i,k}(h, x_{<i}), sigma_{i,k}(h, x_{<i})^2)

    This is the continuous-state analog of the conditional MADE used in
    Paige & Wood (see ``CODEBASE_compiled-inference/learn_smc_proposals/
    cde.py::ConditionalRealValueMADE``), but driven by a pre-computed
    feature vector rather than a raw concatenation of conditioning
    variables (which lets us reuse the existing MLP / ResNet1D backbones).

    Architecture: a single small MLP with ``num_hidden_layers`` hidden
    layers of width ``hidden_size`` and input ``[h; x]``; masking is applied
    to the input and output weights so that the output for coordinate ``i``
    only depends on ``h`` and ``x_{<i}``. Crucially, during ``forward``
    the mask ensures coordinate ``i`` does *not* see ``x_i`` itself.

    Sampling is sequential: for ``i = 1, ..., D`` draw ``x_i`` from its
    mixture, feed it back in, and compute the mixture parameters for the
    next coordinate.

    The unconditional cost is O(D * hidden_size * K); sampling adds one
    forward pass per coordinate on top of that. Still cheap compared to
    RF Euler integration + divergence for realistic D.
    """

    kind = "rnade"

    def __init__(
        self,
        feature_dim: int,
        state_dim: int,
        num_components: int,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        min_sigma: float = 1e-3,
        init_log_sigma: float = -1.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_components >= 1
        assert num_hidden_layers >= 1
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.K = int(num_components)
        self.H = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.min_sigma = float(min_sigma)

        if state_dim < 2:
            raise ValueError(
                "RNADEHead requires state_dim >= 2 for non-trivial autoregression; "
                "use 'joint_mog' or 'mdn_k' for scalar states."
            )
        m_input = torch.cat(
            [
                torch.zeros(feature_dim),
                torch.arange(1, state_dim + 1).float(),
            ]
        )
        hidden_range_max = max(state_dim - 1, 1)
        m_hidden = [
            (torch.arange(self.H).float() % hidden_range_max + 1).float()
            for _ in range(self.num_hidden_layers)
        ]
        m_out = torch.arange(1, state_dim + 1).float()
        m_out_expanded = m_out.unsqueeze(-1).expand(state_dim, 3 * self.K).reshape(-1)

        in_mask = (m_hidden[0].unsqueeze(-1) >= m_input.unsqueeze(0)).float()
        hidden_masks = [in_mask]
        for i in range(1, self.num_hidden_layers):
            hidden_masks.append(
                (m_hidden[i].unsqueeze(-1) >= m_hidden[i - 1].unsqueeze(0)).float()
            )
        out_mask = (m_out_expanded.unsqueeze(-1) > m_hidden[-1].unsqueeze(0)).float()
        skip_mask = (m_out_expanded.unsqueeze(-1) > m_input.unsqueeze(0)).float()

        self.register_buffer("_in_mask", hidden_masks[0])
        for i in range(1, self.num_hidden_layers):
            self.register_buffer(f"_hidden_mask_{i}", hidden_masks[i])
        self.register_buffer("_out_mask", out_mask)
        self.register_buffer("_skip_mask", skip_mask)

        self.in_lin = nn.Linear(feature_dim + state_dim, self.H)
        self.hidden_lins = nn.ModuleList(
            [nn.Linear(self.H, self.H) for _ in range(self.num_hidden_layers - 1)]
        )
        self.out_lin = nn.Linear(self.H, state_dim * 3 * self.K)
        self.skip_lin = nn.Linear(feature_dim + state_dim, state_dim * 3 * self.K, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.out_lin.weight)
            nn.init.zeros_(self.skip_lin.weight)
            with torch.no_grad():
                bias = torch.zeros(state_dim * 3 * self.K)
                bias_view = bias.view(state_dim, self.K, 3)
                bias_view[:, :, 2] = init_log_sigma
                self.out_lin.bias.copy_(bias)

    def _masked_forward(
        self, h: torch.Tensor, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B = h.shape[0]
        inp = torch.cat([h, x], dim=-1)
        W_in = self.in_lin.weight * self._in_mask
        z = F.linear(inp, W_in, self.in_lin.bias)
        z = F.relu(z)
        for i, lin in enumerate(self.hidden_lins, start=1):
            mask = getattr(self, f"_hidden_mask_{i}")
            W = lin.weight * mask
            z = F.linear(z, W, lin.bias)
            z = F.relu(z)
        W_out = self.out_lin.weight * self._out_mask
        W_skip = self.skip_lin.weight * self._skip_mask
        raw = F.linear(z, W_out, self.out_lin.bias) + F.linear(inp, W_skip)
        raw = raw.view(B, self.state_dim, self.K, 3)
        log_w = F.log_softmax(raw[..., 0], dim=-1)
        mu = raw[..., 1]
        sigma = F.softplus(raw[..., 2]).clamp(min=self.min_sigma)
        return {"log_w": log_w, "mu": mu, "sigma": sigma}

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"_feature": h}

    def log_prob(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        h = params["_feature"]
        mog = self._masked_forward(h, x)
        log_w = mog["log_w"]
        mu = mog["mu"]
        sigma = mog["sigma"]
        x_b = x.unsqueeze(-1)
        z = (x_b - mu) / sigma
        log_comp = (
            -0.5 * z * z
            - sigma.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_px_i = torch.logsumexp(log_w + log_comp, dim=-1)
        return log_px_i.sum(dim=-1)

    def sample(self, params: Dict[str, torch.Tensor], n: int = 1) -> torch.Tensor:
        if n != 1:
            raise NotImplementedError(
                "RNADEHead.sample only supports n=1; expand the batch instead."
            )
        h = params["_feature"]
        B = h.shape[0]
        x = torch.zeros(B, self.state_dim, device=h.device, dtype=h.dtype)
        for i in range(self.state_dim):
            mog = self._masked_forward(h, x)
            log_w_i = mog["log_w"][:, i, :]
            mu_i = mog["mu"][:, i, :]
            sigma_i = mog["sigma"][:, i, :]
            gumbel = -torch.log(-torch.log(torch.rand_like(log_w_i).clamp_min(1e-20)).clamp_min(1e-20))
            k = (log_w_i + gumbel).argmax(dim=-1, keepdim=True)
            mu_k = mu_i.gather(-1, k).squeeze(-1)
            sigma_k = sigma_i.gather(-1, k).squeeze(-1)
            eps = torch.randn_like(mu_k)
            x = x.clone()
            x[:, i] = mu_k + sigma_k * eps
        return x

    def mean(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = params["_feature"]
        B = h.shape[0]
        x = torch.zeros(B, self.state_dim, device=h.device, dtype=h.dtype)
        for i in range(self.state_dim):
            mog = self._masked_forward(h, x)
            log_w_i = mog["log_w"][:, i, :]
            mu_i = mog["mu"][:, i, :]
            w_i = log_w_i.exp()
            mean_i = (w_i * mu_i).sum(dim=-1)
            x = x.clone()
            x[:, i] = mean_i
        return x


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------


def create_cde_head(
    kind: str,
    feature_dim: int,
    state_dim: int,
    num_mixture_components: int = 1,
    min_sigma: float = 1e-3,
    init_log_sigma: float = -1.0,
    zero_init_output: bool = True,
    rnade_hidden_size: int = 128,
    rnade_num_hidden_layers: int = 2,
) -> BaseCDEHead:
    """Build a CDE head by short name.

    Args:
        kind: One of ``'gaussian'``, ``'mdn_k'``, ``'joint_mog'``, ``'rnade'``.
        feature_dim: Backbone feature dimension (H).
        state_dim: State dimension (D).
        num_mixture_components: K for mixture heads. Ignored for the single
            Gaussian head.
        min_sigma: Lower clamp on all predicted sigmas (in scaled state
            space; with unit-variance scaled data, 1e-3 is reasonable).
        init_log_sigma: Output-bias init for the log-sigma channels. A
            small negative value (e.g. ``-1.0`` so ``sigma(h=0) ~ 0.37``)
            works well in scaled space at initialisation.
        zero_init_output: If ``True``, zero out the final projection weights
            and set the log-sigma bias to ``init_log_sigma``. Equivalent to
            saying "start the head at a unit-variance Gaussian around 0";
            matches the pattern used by the existing RF / NASMC heads.
        rnade_hidden_size: Hidden width of the RNADEHead's MADE MLP.
        rnade_num_hidden_layers: Number of hidden layers in the RNADEHead.

    Returns:
        A concrete :class:`BaseCDEHead`.
    """
    if kind == "gaussian":
        return GaussianHead(
            feature_dim=feature_dim,
            state_dim=state_dim,
            min_sigma=min_sigma,
            init_log_sigma=init_log_sigma,
            zero_init_output=zero_init_output,
        )
    if kind == "mdn_k":
        return MDNHead(
            feature_dim=feature_dim,
            state_dim=state_dim,
            num_components=num_mixture_components,
            min_sigma=min_sigma,
            init_log_sigma=init_log_sigma,
            zero_init_output=zero_init_output,
        )
    if kind == "joint_mog":
        return JointMoGHead(
            feature_dim=feature_dim,
            state_dim=state_dim,
            num_components=num_mixture_components,
            min_sigma=min_sigma,
            init_log_sigma=init_log_sigma,
            zero_init_output=zero_init_output,
        )
    if kind == "rnade":
        return RNADEHead(
            feature_dim=feature_dim,
            state_dim=state_dim,
            num_components=num_mixture_components,
            hidden_size=rnade_hidden_size,
            num_hidden_layers=rnade_num_hidden_layers,
            min_sigma=min_sigma,
            init_log_sigma=init_log_sigma,
            zero_init_output=zero_init_output,
        )
    raise ValueError(
        f"Unknown CDE head kind: {kind!r}. "
        "Expected one of 'gaussian', 'mdn_k', 'joint_mog', 'rnade'."
    )


__all__ = [
    "BaseCDEHead",
    "GaussianHead",
    "MDNHead",
    "JointMoGHead",
    "RNADEHead",
    "create_cde_head",
]
