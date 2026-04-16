"""
Local MLP velocity network operating on spatial patches of size 2r+1.

Consumes a concatenated local window of
  - z_window      : current flow state at grid points [j-r, ..., j+r]
  - x_prev_window : previous state at the same window
  - obs_window    : dense observations at the same window (zero + mask when missing)
  - s             : flow time in [0,1]
  - t (optional)  : trajectory time step (normalized)

and returns the scalar velocity at the centre site j.

This network is shared across all spatial locations (spatial homogeneity of L96),
so training extracts N_x overlapping patches per transition.
"""

import torch
import torch.nn as nn
from typing import Optional


class LocalMLPVelocityNetwork(nn.Module):
    """
    Small MLP that maps a local patch to a scalar centre velocity.

    Args:
        radius: Spatial radius r. Window size is 2*r+1.
        hidden_dim: Hidden dim of the MLP.
        depth: Number of hidden layers.
        time_embed_dim: Dimension of flow-time (and trajectory-time) embeddings.
        use_obs: Whether observations are provided.
        use_time_step: Whether trajectory-time conditioning is provided.
        dropout: Dropout probability.
        zero_init_output: If True, the final linear is zero-initialized so
            the initial velocity is 0. Recommended for stable RF training.
    """

    def __init__(
        self,
        radius: int,
        hidden_dim: int = 128,
        depth: int = 4,
        time_embed_dim: int = 64,
        use_obs: bool = True,
        use_time_step: bool = False,
        dropout: float = 0.0,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.radius = radius
        self.window_size = 2 * radius + 1
        self.use_obs = use_obs
        self.use_time_step = use_time_step
        self.time_embed_dim = time_embed_dim

        # Flow time embedding (s in [0,1])
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.use_time_step:
            self.traj_time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        # Input: [z_win, x_prev_win, obs_win, obs_mask_win, flow_time_embed, (traj_time_embed)]
        #  - Observation mask channel is included so the net can distinguish
        #    "observation zero" from "no observation".
        in_dim = 2 * self.window_size + time_embed_dim
        if use_obs:
            in_dim += 2 * self.window_size  # obs + mask
        if use_time_step:
            in_dim += time_embed_dim

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        head = nn.Linear(hidden_dim, 1)
        if zero_init_output:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        layers.append(head)

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z_window: torch.Tensor,        # (B, 2r+1)
        x_prev_window: torch.Tensor,   # (B, 2r+1)
        obs_window: Optional[torch.Tensor] = None,   # (B, 2r+1) or None
        obs_mask: Optional[torch.Tensor] = None,     # (B, 2r+1) or None
        s: Optional[torch.Tensor] = None,            # (B, 1) or (B,)
        t: Optional[torch.Tensor] = None,            # (B, 1) or (B,) or None
    ) -> torch.Tensor:
        """
        Returns velocity at the centre of each patch, shape (B, 1).
        """
        B = z_window.shape[0]
        if s is None:
            raise ValueError("LocalMLPVelocityNetwork.forward: `s` is required.")
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        s_embed = self.time_embed(s)

        parts = [z_window, x_prev_window]
        if self.use_obs:
            if obs_window is None:
                obs_window = torch.zeros(B, self.window_size, device=z_window.device, dtype=z_window.dtype)
            if obs_mask is None:
                # If no mask provided, assume all observations are valid
                obs_mask = torch.ones_like(obs_window)
            parts.append(obs_window)
            parts.append(obs_mask)
        parts.append(s_embed)

        if self.use_time_step:
            if t is None:
                raise ValueError("LocalMLPVelocityNetwork configured with use_time_step=True but t is None.")
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            parts.append(self.traj_time_embed(t))

        h = torch.cat(parts, dim=-1)
        return self.net(h)
