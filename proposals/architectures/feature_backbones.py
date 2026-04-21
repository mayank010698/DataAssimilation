r"""Feature backbones for the Paige-Wood inference-network proposal.

These networks map the conditioning tuple ``(x_{t-1}, y_t, mask_y, t_traj?)``
to a global feature vector ``h \in R^H``. The CDE head
(:mod:`.cde_heads`) then turns ``h`` into parameters of a conditional
density ``q_eta(x_t | h)`` over the next state.

Contrast with :class:`~proposals.architectures.mlp.MLPVelocityNetwork`
and :class:`~proposals.architectures.resnet1d.ResNet1DVelocityNetwork`:

  * The velocity networks take ``(x_s, s, x_{t-1}, y_t, t_traj?)`` and
    return a per-coordinate velocity in ``R^D``. They are used inside an
    Euler integrator during sampling and require a flow-time ``s``.
  * The feature backbones take ``(x_{t-1}, y_t, t_traj?)`` and return a
    single pooled feature vector in ``R^H``. There is no flow-time and no
    per-coordinate output; the CDE head decides how to turn ``h`` into
    a joint density over ``x_t``.

Sparse-observation handling (inpainting-style [y_full, mask]) mirrors the
velocity-net code path so ``obs_indices`` Just Works. When used with the
ResNet1D backbone, the spatial axis is the state dimension and features
are produced by global average pooling over the blocks' final
representation.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .resnet1d import ResBlock1DAdaLN


class MLPFeatureBackbone(nn.Module):
    """Flat-MLP feature backbone.

    Input packing (concatenated along the last axis, identical to
    :class:`MLPVelocityNetwork` minus the flow-time / current-state
    inputs):

      * ``[x_prev, y_full, mask, t_emb]``     when ``obs_dim > 0``
      * ``[x_prev, t_emb]``                   when ``obs_dim == 0``

    ``t_emb`` is a trajectory-time embedding of width ``time_embed_dim``
    appended iff ``use_time_step=True``. Otherwise it is omitted
    entirely (the MLP input dimension shrinks accordingly).

    Output: feature vector of width ``hidden_dim``, ready to feed into
    a :class:`~proposals.architectures.cde_heads.BaseCDEHead`.
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
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.time_embed_dim = time_embed_dim
        self.use_time_step = use_time_step
        self.feature_dim = hidden_dim

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
                    "MLPFeatureBackbone configured with use_time_step=True "
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

        return self.trunk(torch.cat(inputs, dim=-1))


class ResNet1DFeatureBackbone(nn.Module):
    """1D-ResNet feature backbone with AdaLN time conditioning.

    Mirrors :class:`ResNet1DVelocityNetwork` but:

      * drops the current-state ``x`` and flow-time ``s`` inputs,
      * emits a pooled global feature vector (not a per-site velocity).

    Input channels stacked along the state (= spatial) axis:

      * ``[x_prev, y_full, mask]``  when ``obs_dim > 0``  (3 channels)
      * ``[x_prev]``                when ``obs_dim == 0`` (1 channel)

    Optional trajectory-time embedding is fed through AdaLN exactly as in
    the velocity net; when ``use_time_step=False`` we pass a zero embedding
    (the AdaLN layers are initialised to identity so a zero embedding just
    degenerates to plain LayerNorm, which is fine).

    The final feature is produced by global average pooling across the
    spatial axis followed by an ``nn.Linear`` projection to
    ``feature_dim``. The projection is important: it decouples the pooled
    channel count from the head's expected feature dim and mirrors the
    standard "feature extractor + head" split we use for RF and NASMC.
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
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.use_time_step = use_time_step
        self.feature_dim = feature_dim

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

        self.input_channels = 3 if obs_dim > 0 else 1
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
        self.feature_proj = nn.Linear(channels, feature_dim)

    def forward(
        self,
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x_prev.shape[0]
        device = x_prev.device
        dtype = x_prev.dtype

        if self.use_time_step:
            if t is None:
                raise ValueError(
                    "ResNet1DFeatureBackbone configured with use_time_step=True "
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
            net_input = torch.stack([x_prev, y_full, mask], dim=1)
        else:
            net_input = x_prev.unsqueeze(1)

        h = self.input_proj(net_input)
        for block in self.blocks:
            h = block(h, t_embed)
        h = self.final_norm(h)
        h = self.final_act(h)
        pooled = h.mean(dim=-1)
        return self.feature_proj(pooled)


__all__ = [
    "MLPFeatureBackbone",
    "ResNet1DFeatureBackbone",
]
