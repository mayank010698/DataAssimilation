"""Gaussian-/Gaussian-mixture-head backbones for the NASMC proposal.

These networks map ``(x_prev, y_curr, mask, t_idx?) -> raw mixture
parameters``. With ``num_components == 1`` (the default) the networks
parametrise a single diagonal Gaussian
``q_phi(x_t | x_{t-1}, y_t) = N(mu, diag(sigma^2))`` and reproduce the
original behaviour of this file bit-for-bit (modulo the output-layer
shape — see below). With ``num_components == K > 1`` they parametrise
a mixture of ``K`` diagonal Gaussians, matching the ``-MD-`` variant of
Gu, Ghahramani & Turner (2015).

Output layout
-------------
Regardless of architecture, ``forward(...)`` returns a flat tensor of
shape ``(B, K * (2 * state_dim + 1))`` which the caller is expected to
reshape to ``(B, K, 2 * state_dim + 1)``. Along the last axis:

* index ``0``                        -> mixing logit for that component
* indices ``1 .. 1 + state_dim``     -> per-coordinate ``mu_raw``
* indices ``1 + state_dim .. end``   -> per-coordinate ``log_sigma``

This matches :class:`JointMoGHead` and lets the proposal code use a
single canonical parser regardless of backbone.

Initialisation
--------------
When ``zero_init_output=True`` the *entire* output projection is
zero-initialised (weights and biases) except for the ``log_sigma`` bias
slice, which is set to ``init_log_sigma``. At step 0 this means:

* mixing logits = 0  ->  ``softmax`` gives uniform mixing weights over
  the ``K`` components, so the marginal mean collapses back onto the
  skip-connection prediction.
* ``mu_raw`` = 0     ->  every component's mean equals the skip
  prediction (``x_{t-1}`` in ``predict_delta`` mode, or the
  deterministic dynamics mean in the ``-f-`` variant).
* ``log_sigma`` = ``init_log_sigma`` (broadcast over every component).

For ``K = 1`` this is identical to the single-Gaussian initialisation
used in the original file, so a ``num_components=1`` configuration
reproduces the old ``MLPGaussianHead`` / ``ResNet1DGaussianHead``
behaviour exactly.

The backbone (trunk) itself is unchanged; only the output head widens
from ``2D`` to ``K * (2D + 1)``. This matters because we want the
NASMC-vs-FPPF/FlowDAS comparison to hold backbone capacity fixed.

Conditioning format: same ``(y_full, mask)`` inpainting-style as the
velocity networks so that sparse observations and ``obs_indices`` Just
Work.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .resnet1d import ResBlock1DAdaLN


def _zero_init_mixture_bias(
    bias: torch.Tensor,
    num_components: int,
    state_dim: int,
    init_log_sigma: float,
) -> None:
    """Write a zero-init bias with ``log_sigma`` slice set to ``init_log_sigma``.

    The bias is viewed as ``(K, 2 * state_dim + 1)`` in-place; mixing
    logit (column 0) and mu_raw (columns ``1..1+D``) stay at zero; the
    log-sigma slice (columns ``1+D..``) is broadcast-filled with
    ``init_log_sigma``.
    """
    assert bias.numel() == num_components * (2 * state_dim + 1)
    with torch.no_grad():
        view = bias.view(num_components, 2 * state_dim + 1)
        view.zero_()
        view[:, 1 + state_dim :] = init_log_sigma


class MLPGaussianHead(nn.Module):
    """MLP (Gaussian-)mixture head, conditioning via concatenation.

    Input signature mirrors :class:`MLPVelocityNetwork` but drops ``x``
    and ``s``. The output is a single ``(B, K * (2 * state_dim + 1))``
    tensor; see the module docstring for the per-component layout.

    Setting ``num_components=1`` (the default) preserves the previous
    behaviour exactly: the bias reduces to ``[0, mu_bias=0,
    log_sigma_bias=init_log_sigma]`` and the extra mixing-logit slot
    collapses to the constant ``softmax``-of-zero weight ``1``.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int = 0,
        obs_indices: Optional[List[int]] = None,
        hidden_dim: int = 128,
        depth: int = 4,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
        use_time_step: bool = False,
        zero_init_output: bool = True,
        init_log_sigma: float = -1.0,
        num_components: int = 1,
    ) -> None:
        super().__init__()

        if num_components < 1:
            raise ValueError("num_components must be >= 1")

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.time_embed_dim = time_embed_dim
        self.use_time_step = use_time_step
        self.num_components = int(num_components)
        self.init_log_sigma = float(init_log_sigma)

        if obs_indices is not None:
            self.register_buffer(
                "obs_indices", torch.tensor(obs_indices, dtype=torch.long)
            )
        else:
            self.obs_indices = None

        if self.use_time_step:
            self.traj_time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if obs_dim > 0:
            # [x_prev, y_full, mask]
            input_dim = 3 * state_dim
        else:
            input_dim = state_dim

        if self.use_time_step:
            input_dim += time_embed_dim

        layers: List[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, self.num_components * (2 * state_dim + 1))

        if zero_init_output:
            # Zero-init the entire output projection (weights + mu bias +
            # mixing-logit bias); log_sigma bias is broadcast-filled with
            # init_log_sigma across all K components.
            nn.init.zeros_(self.output.weight)
            _zero_init_mixture_bias(
                self.output.bias, self.num_components, state_dim, self.init_log_sigma,
            )

    def forward(
        self,
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_prev.shape[0]

        traj_t_embed = None
        if self.use_time_step:
            if t is None:
                raise ValueError(
                    "MLPGaussianHead configured with use_time_step=True "
                    "but t was not provided."
                )
            if t.dim() == 1:
                t = t.unsqueeze(1)
            traj_t_embed = self.traj_time_embed(t)

        if self.obs_dim > 0:
            y_full = torch.zeros(
                B, self.state_dim, device=x_prev.device, dtype=x_prev.dtype
            )
            mask = torch.zeros(
                B, self.state_dim, device=x_prev.device, dtype=x_prev.dtype
            )
            if y is not None:
                if self.obs_indices is not None:
                    y_full[:, self.obs_indices] = y
                    mask[:, self.obs_indices] = 1.0
                elif y.shape[1] == self.state_dim:
                    y_full = y
                    mask = torch.ones_like(mask)
                else:
                    y_full[:, : y.shape[1]] = y
                    mask[:, : y.shape[1]] = 1.0
            inputs = [x_prev, y_full, mask]
        else:
            inputs = [x_prev]

        if traj_t_embed is not None:
            inputs.append(traj_t_embed)

        h = self.trunk(torch.cat(inputs, dim=-1))
        return self.output(h)


class ResNet1DGaussianHead(nn.Module):
    """1D-ResNet (Gaussian-)mixture head with AdaLN conditioning.

    Mirrors :class:`ResNet1DVelocityNetwork` but drops the velocity-
    specific ``x`` and ``s`` inputs and widens the output head from 2
    channels per site to ``2 * K`` channels per site (``mu_raw`` and
    ``log_sigma`` for each of ``K`` components). ``K`` mixing logits
    are produced by a small global-pooling head so the mixture
    assignments are shared across all spatial locations, as in the
    ``-MD-`` variant of Gu et al. 2015.

    ``num_components == 1`` reproduces the original single-Gaussian
    head (with a constant mixing weight of 1 contributed by the zero
    mixing-logit bias).
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int = 0,
        obs_indices: Optional[List[int]] = None,
        channels: int = 64,
        num_blocks: int = 6,
        kernel_size: int = 5,
        time_embed_dim: int = 64,
        dropout: float = 0.0,
        use_time_step: bool = False,
        zero_init_output: bool = True,
        init_log_sigma: float = -1.0,
        num_components: int = 1,
    ) -> None:
        super().__init__()

        if num_components < 1:
            raise ValueError("num_components must be >= 1")

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.use_time_step = use_time_step
        self.init_log_sigma = float(init_log_sigma)
        self.num_components = int(num_components)

        if obs_indices is not None:
            self.register_buffer(
                "obs_indices", torch.tensor(obs_indices, dtype=torch.long)
            )
        else:
            self.obs_indices = None

        if self.use_time_step:
            self.traj_time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if obs_dim > 0:
            self.input_channels = 3  # x_prev, y_full, mask
        else:
            self.input_channels = 1  # x_prev

        self.input_proj = nn.Conv1d(
            self.input_channels, channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            padding_mode="circular",
        )
        self.blocks = nn.ModuleList(
            [
                ResBlock1DAdaLN(channels, kernel_size, time_embed_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.GroupNorm(1, channels)
        self.final_act = nn.SiLU()

        # Per-site (mu_raw, log_sigma) for each of K components.
        # Channels are grouped as [mu_1..mu_K, log_sigma_1..log_sigma_K]
        # so we can split them cleanly with a single chunk(2) call.
        self.output_proj = nn.Conv1d(channels, 2 * self.num_components, kernel_size=1)
        # Global mixing logits, one per component. Pooled over L so the
        # assignment distribution is shared across spatial locations.
        self.mix_logit_proj = nn.Linear(channels, self.num_components)

        if zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.mix_logit_proj.weight)
            with torch.no_grad():
                # output_proj bias layout: [mu_1..mu_K, log_sigma_1..log_sigma_K].
                bias = self.output_proj.bias
                bias.zero_()
                bias[self.num_components :] = self.init_log_sigma
                self.mix_logit_proj.bias.zero_()

    def forward(
        self,
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_prev.shape[0]
        device = x_prev.device
        dtype = x_prev.dtype
        K = self.num_components
        D = self.state_dim

        if self.use_time_step:
            if t is None:
                raise ValueError(
                    "ResNet1DGaussianHead configured with use_time_step=True "
                    "but t was not provided."
                )
            if t.dim() == 1:
                t = t.unsqueeze(1)
            t_embed = self.traj_time_embed(t)
        else:
            t_embed = torch.zeros(B, self.time_embed_dim, device=device, dtype=dtype)

        if self.obs_dim > 0:
            y_full = torch.zeros(B, self.state_dim, device=device, dtype=dtype)
            mask = torch.zeros(B, self.state_dim, device=device, dtype=dtype)
            if y is not None:
                if self.obs_indices is not None:
                    y_full[:, self.obs_indices] = y
                    mask[:, self.obs_indices] = 1.0
                elif y.shape[1] == self.state_dim:
                    y_full = y
                    mask = torch.ones_like(mask)
                else:
                    y_full[:, : y.shape[1]] = y
                    mask[:, : y.shape[1]] = 1.0
            net_input = torch.stack([x_prev, y_full, mask], dim=1)  # (B, 3, L)
        else:
            net_input = x_prev.unsqueeze(1)  # (B, 1, L)

        h = self.input_proj(net_input)
        for block in self.blocks:
            h = block(h, t_embed)
        h = self.final_norm(h)
        h = self.final_act(h)

        out_sites = self.output_proj(h)  # (B, 2*K, L)
        mu_raw = out_sites[:, :K, :]                  # (B, K, D)
        log_sigma = out_sites[:, K : 2 * K, :]        # (B, K, D)

        # Mixing logits from global-average-pooled features.
        pooled = h.mean(dim=-1)                       # (B, channels)
        mix_logits = self.mix_logit_proj(pooled)      # (B, K)

        # Pack as (B, K, 2D+1) and flatten to (B, K*(2D+1)) to match the
        # MLP head's layout: [logit, mu_D, log_sigma_D] per component.
        packed = torch.cat(
            [
                mix_logits.unsqueeze(-1),              # (B, K, 1)
                mu_raw,                                # (B, K, D)
                log_sigma,                             # (B, K, D)
            ],
            dim=-1,
        )                                              # (B, K, 2D+1)
        return packed.reshape(B, K * (2 * D + 1))
