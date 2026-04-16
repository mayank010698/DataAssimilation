"""
Tiny 1D ResNet velocity network operating on spatial patches of size 2r+1.

Stacks (z_window, x_prev_window, obs_window, obs_mask) as a multi-channel 1D
signal, applies 1-2 residual blocks with small kernel size, global-average
pools, and regresses a scalar centre velocity. Flow time `s` (and optional
trajectory time `t`) are injected via AdaLN.

Unlike the global ResNet1D, patches do NOT use circular padding inside
the network: circular structure is handled by `extract_patches` at the
site level. Internal convolutions use standard zero-padding because
patch inputs are already correctly extracted windows.
"""

import torch
import torch.nn as nn
from typing import Optional


class _AdaLN1d(nn.Module):
    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Linear(embed_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        params = self.proj(embed)
        scale, shift = params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1) + 1.0
        shift = shift.unsqueeze(-1)
        return scale * x + shift


class _LocalResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, embed_dim: int):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.adaln1 = _AdaLN1d(channels, embed_dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.adaln2 = _AdaLN1d(channels, embed_dim)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, embed)
        h = self.conv1(self.act1(h))
        h = self.adaln2(h, embed)
        h = self.conv2(self.act2(h))
        return x + h


class LocalResNet1DVelocityNetwork(nn.Module):
    """
    1-2 residual blocks over a patch of size 2r+1.

    Args:
        radius: Spatial radius r.
        channels: Internal channel count.
        num_blocks: Number of residual blocks (1-2 recommended).
        kernel_size: Convolution kernel size (3 recommended).
        time_embed_dim: Flow time embedding dim.
        use_obs: Whether observations are used.
        use_time_step: Whether trajectory-time conditioning is used.
        zero_init_output: Zero-init the scalar head.
    """

    def __init__(
        self,
        radius: int,
        channels: int = 32,
        num_blocks: int = 2,
        kernel_size: int = 3,
        time_embed_dim: int = 64,
        use_obs: bool = True,
        use_time_step: bool = False,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.radius = radius
        self.window_size = 2 * radius + 1
        self.use_obs = use_obs
        self.use_time_step = use_time_step

        in_channels = 2 + (2 if use_obs else 0)  # z, x_prev, [obs, mask]
        self.in_proj = nn.Conv1d(in_channels, channels, kernel_size=1)

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if use_time_step:
            self.traj_time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            cond_dim = 2 * time_embed_dim
            self.cond_proj = nn.Linear(cond_dim, time_embed_dim)
        else:
            self.cond_proj = None

        self.blocks = nn.ModuleList([
            _LocalResBlock(channels, kernel_size, time_embed_dim)
            for _ in range(num_blocks)
        ])

        # Head: global-average pool over the window then project to scalar
        self.head_norm = nn.GroupNorm(1, channels)
        self.head_act = nn.SiLU()
        self.head = nn.Linear(channels, 1)
        if zero_init_output:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        z_window: torch.Tensor,
        x_prev_window: torch.Tensor,
        obs_window: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = z_window.shape[0]
        if s is None:
            raise ValueError("LocalResNet1DVelocityNetwork.forward: `s` is required.")
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        s_embed = self.time_embed(s)

        if self.use_time_step:
            if t is None:
                raise ValueError("use_time_step=True but t is None.")
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            t_embed = self.traj_time_embed(t)
            cond = self.cond_proj(torch.cat([s_embed, t_embed], dim=-1))
        else:
            cond = s_embed

        channels = [z_window, x_prev_window]
        if self.use_obs:
            if obs_window is None:
                obs_window = torch.zeros_like(z_window)
            if obs_mask is None:
                obs_mask = torch.ones_like(obs_window)
            channels.extend([obs_window, obs_mask])
        x = torch.stack(channels, dim=1)  # (B, C, 2r+1)

        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, cond)

        x = self.head_act(self.head_norm(x))
        x = x.mean(dim=-1)  # global avg pool over window
        return self.head(x)  # (B, 1)
