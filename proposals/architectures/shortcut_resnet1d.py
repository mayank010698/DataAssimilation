"""
Shortcut 1D ResNet velocity network for F2D2 distillation.

Extends ResNet1DVelocityNetwork with:
  - A second time embedding for step-size `dt` (added to the flow-time `t` embedding
    before it enters AdaLN, so every residual block is jointly conditioned on both)
  - A divergence accumulation head: global-average-pool over the spatial dimension
    of the final feature map → small MLP → scalar
  - A `div_head_active` flag (False = Stage 2 shortcut; True = Stage 3 F2D2)

Forward signature: (x, t, dt, x_prev, y, t_traj) → (velocity, div_scalar)
"""

import torch
import torch.nn as nn
from typing import Optional, List


# ---------------------------------------------------------------------------
# Re-use AdaLN and ResBlock from the existing resnet1d module
# ---------------------------------------------------------------------------

class AdaLN1d(nn.Module):
    """Adaptive Layer Normalisation for 1-D feature maps (B, C, L)."""

    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Linear(embed_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        params = self.proj(embed)
        scale, shift = params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1) + 1.0
        shift = shift.unsqueeze(-1)
        return scale * x_norm + shift


class ResBlock1DAdaLN(nn.Module):
    """Pre-activation residual block with AdaLN and circular convolutions."""

    def __init__(self, channels: int, kernel_size: int, time_embed_dim: int, dropout: float = 0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.adaln1 = AdaLN1d(channels, time_embed_dim)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, padding_mode="circular")
        self.adaln2 = AdaLN1d(channels, time_embed_dim)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, padding_mode="circular")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.adaln1(x, t_embed)))
        h = self.conv2(self.dropout(self.act2(self.adaln2(h, t_embed))))
        return h + x


# ---------------------------------------------------------------------------
# ShortcutResNet1DVelocityNetwork
# ---------------------------------------------------------------------------

class ShortcutResNet1DVelocityNetwork(nn.Module):
    """
    Shortcut 1D ResNet velocity network with divergence head.

    Args:
        state_dim: State-space dimension (= spatial length of the 1-D sequence).
        obs_dim: Observation dimension (0 = unconditional).
        obs_indices: Sparse observation position indices (None = dense).
        channels: Number of feature channels.
        num_blocks: Number of residual blocks.
        kernel_size: Conv kernel size (should be odd for symmetric padding).
        time_embed_dim: Dimension of time embeddings (shared by t, dt, t_traj).
        dropout: Dropout probability in residual blocks.
        use_time_step: Accept trajectory-time conditioning (t_traj).
        zero_init_output: Zero-initialise the velocity output projection.
        div_head_active: If True, divergence head is used; if False returns zeros.
        div_hidden_dim: Hidden dim inside the divergence MLP head.
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
        div_head_active: bool = False,
        div_hidden_dim: int = 64,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.use_time_step = use_time_step
        self.div_head_active = div_head_active

        if obs_indices is not None:
            self.register_buffer("obs_indices", torch.tensor(obs_indices, dtype=torch.long))
        else:
            self.obs_indices = None

        # ---- Time embeddings ----
        # t and dt are embedded separately and added together before AdaLN.
        self.t_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.dt_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        if use_time_step:
            self.traj_time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        # ---- Input channels ----
        # Stack: x (1ch) + x_prev (1ch) + [y_full (1ch) + mask (1ch) if obs_dim>0]
        self.input_channels = 4 if obs_dim > 0 else 2

        pad = (kernel_size - 1) // 2
        self.input_proj = nn.Conv1d(
            self.input_channels, channels, kernel_size, padding=pad, padding_mode="circular"
        )

        # ---- Backbone ----
        self.blocks = nn.ModuleList(
            [ResBlock1DAdaLN(channels, kernel_size, time_embed_dim, dropout) for _ in range(num_blocks)]
        )

        # ---- Velocity output ----
        self.final_norm = nn.GroupNorm(1, channels)
        self.final_act = nn.SiLU()
        self.output_proj = nn.Conv1d(channels, 1, kernel_size=1)
        if zero_init_output:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        # ---- Divergence head ----
        # Global average pool (B, C, L) → (B, C) → MLP → (B, 1)
        self.div_head = nn.Sequential(
            nn.Linear(channels, div_hidden_dim),
            nn.SiLU(),
            nn.Linear(div_hidden_dim, 1),
        )
        nn.init.zeros_(self.div_head[-1].weight)
        nn.init.zeros_(self.div_head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        x_prev: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        t_traj: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x:      (B, state_dim) — current flow position
            t:      (B, 1) or (B,) — flow time in [0,1]
            dt:     (B, 1) or scalar — step size
            x_prev: (B, state_dim) — previous state for conditioning
            y:      (B, obs_dim) or None — observation conditioning
            t_traj: (B, 1) or None — trajectory time (normalised)

        Returns:
            velocity:   (B, state_dim)
            div_scalar: (B,)
        """
        B = x.shape[0]

        # ---- Shape normalisation ----
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if isinstance(dt, torch.Tensor):
            if dt.dim() == 0:
                dt = dt.expand(B, 1)
            elif dt.dim() == 1:
                dt = dt.unsqueeze(1)
        else:
            dt = torch.full((B, 1), dt, device=x.device, dtype=x.dtype)

        # ---- Combined time embedding ----
        t_emb = self.t_embed(t) + self.dt_embed(dt)   # (B, E)
        if self.use_time_step:
            if t_traj is None:
                raise ValueError("use_time_step=True but t_traj not provided.")
            if t_traj.dim() == 1:
                t_traj = t_traj.unsqueeze(1)
            t_emb = t_emb + self.traj_time_embed(t_traj)

        # ---- Build spatial input ----
        x_in = x.unsqueeze(1)         # (B, 1, L)
        xp_in = x_prev.unsqueeze(1)   # (B, 1, L)

        if self.obs_dim > 0:
            y_full = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
            mask = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
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
            net_input = torch.cat([x_in, xp_in, y_full.unsqueeze(1), mask.unsqueeze(1)], dim=1)
        else:
            net_input = torch.cat([x_in, xp_in], dim=1)

        # ---- Forward through backbone ----
        h = self.input_proj(net_input)        # (B, C, L)
        for block in self.blocks:
            h = block(h, t_emb)

        h_out = self.final_act(self.final_norm(h))

        # ---- Velocity output ----
        velocity = self.output_proj(h_out).squeeze(1)   # (B, L) = (B, state_dim)

        # ---- Divergence head ----
        if self.div_head_active:
            h_pooled = h_out.mean(dim=-1)               # (B, C) — global avg pool
            div_scalar = self.div_head(h_pooled).squeeze(-1)   # (B,)
        else:
            div_scalar = torch.zeros(B, device=x.device, dtype=x.dtype)

        return velocity, div_scalar
