"""Gaussian-head backbones for the NASMC proposal.

These networks map (x_prev, y_curr, mask, t_idx?) -> (mu_raw, log_sigma),
i.e. they parametrise a diagonal Gaussian q_phi(x_t | x_{t-1}, y_t). They
mirror the input-handling logic of the existing velocity networks
(MLPVelocityNetwork and ResNet1DVelocityNetwork) but drop the flow-time
input `s` and the current-state input `x` that are specific to Rectified
Flow, and widen the output head to 2 * state_dim.

Conditioning format: same (y_full, mask) inpainting-style as the velocity
networks so that sparse observations and `obs_indices` Just Work.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .resnet1d import ResBlock1DAdaLN


class MLPGaussianHead(nn.Module):
    """MLP Gaussian head, conditioning via concatenation.

    Input signature mirrors :class:`MLPVelocityNetwork` but drops `x` and
    `s`. The output is a single ``(B, 2 * state_dim)`` tensor: the first
    ``state_dim`` coordinates are the mean ``mu_raw`` and the second
    ``state_dim`` are ``log_sigma``.
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
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.time_embed_dim = time_embed_dim
        self.use_time_step = use_time_step

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
        self.output = nn.Linear(hidden_dim, 2 * state_dim)

        if zero_init_output:
            # Initialise so that mu_raw = 0 (delta/dynamics mean is preserved)
            # and log_sigma = init_log_sigma (sigma ~ exp(init_log_sigma)).
            nn.init.zeros_(self.output.weight)
            with torch.no_grad():
                bias = torch.zeros(2 * state_dim)
                bias[state_dim:] = init_log_sigma
                self.output.bias.copy_(bias)

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
    """1D-ResNet Gaussian head with AdaLN conditioning.

    Mirrors :class:`ResNet1DVelocityNetwork` but drops the velocity-specific
    ``x`` and ``s`` inputs and outputs two channels per spatial location
    (mu_raw, log_sigma) instead of a scalar velocity.
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
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.use_time_step = use_time_step
        self.init_log_sigma = init_log_sigma

        if obs_indices is not None:
            self.register_buffer(
                "obs_indices", torch.tensor(obs_indices, dtype=torch.long)
            )
        else:
            self.obs_indices = None

        # Time-step conditioning (optional). This network does not have a
        # flow-time embedding; AdaLN expects a "time" embedding, so when
        # use_time_step=False we feed it a constant zero vector.
        self.time_embed_dim = time_embed_dim
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
        # 2 output channels: [mu_raw, log_sigma] per spatial site.
        self.output_proj = nn.Conv1d(channels, 2, kernel_size=1)

        if zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            with torch.no_grad():
                # Channel 0 -> mu_raw bias = 0; Channel 1 -> log_sigma bias.
                self.output_proj.bias.zero_()
                self.output_proj.bias[1] = init_log_sigma

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
        out = self.output_proj(h)  # (B, 2, L)

        mu_raw = out[:, 0, :]
        log_sigma = out[:, 1, :]
        return torch.cat([mu_raw, log_sigma], dim=-1)  # (B, 2*L)
