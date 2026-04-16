"""
Shortcut MLP velocity network for F2D2 distillation.

Extends MLPVelocityNetwork with:
  - A second time embedding for step-size `dt` (combined with flow-time `t` by addition)
  - A divergence accumulation head that predicts ∫_t^{t+dt} div v_s ds
  - A `div_head_active` flag: when False the head returns zeros (Stage 2 shortcut);
    when True the head is trained (Stage 3 F2D2/likelihood)

Forward signature: (x, t, dt, x_prev, y, t_traj) → (velocity, div_scalar)

Compatible with all existing MLPVelocityNetwork conditioning options
(concat obs, sparse obs via obs_indices, trajectory-time conditioning, etc.)
"""

import torch
import torch.nn as nn
from typing import Optional, List


class ShortcutMLPVelocityNetwork(nn.Module):
    """
    Shortcut MLP velocity network with divergence head.

    Args:
        state_dim: State-space dimension.
        obs_dim: Observation dimension (0 = unconditional on obs).
        obs_indices: Sparse observation indices (None = dense).
        hidden_dim: Width of each hidden layer.
        depth: Number of hidden layers (total layers = depth + 1 output).
        time_embed_dim: Embedding dimension shared by t, dt, and (optionally) t_traj.
        dropout: Dropout probability (0 = no dropout).
        use_time_step: Whether to accept trajectory-time conditioning (t_traj).
        div_head_active: If True the divergence head is used; if False it returns zeros.
        div_hidden_dim: Hidden dim of the small divergence MLP head.
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
        div_head_active: bool = False,
        div_hidden_dim: int = 64,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.time_embed_dim = time_embed_dim
        self.use_time_step = use_time_step
        self.div_head_active = div_head_active

        if obs_indices is not None:
            self.register_buffer("obs_indices", torch.tensor(obs_indices, dtype=torch.long))
        else:
            self.obs_indices = None

        # ---- Time embeddings ----
        # Separate embeddings for flow time t and step size dt; combined by addition.
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

        # ---- Input dimension ----
        # [x (D), x_prev (D), y_full (D if obs), mask (D if obs), t+dt embed (E), traj_t embed? (E)]
        if obs_dim > 0:
            input_dim = 4 * state_dim + time_embed_dim
        else:
            input_dim = 2 * state_dim + time_embed_dim
        if use_time_step:
            input_dim += time_embed_dim

        # ---- Backbone (hidden layers) ----
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

        # ---- Output heads ----
        self.velocity_head = nn.Linear(hidden_dim, state_dim)

        self.div_head = nn.Sequential(
            nn.Linear(hidden_dim, div_hidden_dim),
            nn.SiLU(),
            nn.Linear(div_hidden_dim, 1),
        )
        # Zero-init divergence head for stable Stage 3 warm-start
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
            x:      Current flow position (B, state_dim)
            t:      Flow time in [0,1]   (B, 1) or (B,)
            dt:     Step size             (B, 1) or scalar
            x_prev: Conditioning          (B, state_dim)
            y:      Observations          (B, obs_dim) or None
            t_traj: Trajectory time (normalised) (B, 1) or None

        Returns:
            velocity:   (B, state_dim)
            div_scalar: (B,) — accumulated divergence ∫_t^{t+dt} div v ds;
                        zeros when div_head_active=False
        """
        B = x.shape[0]

        # ---- Normalise time shapes ----
        if t.dim() == 1:
            t = t.unsqueeze(1)        # (B, 1)
        if isinstance(dt, torch.Tensor):
            if dt.dim() == 0:
                dt = dt.expand(B, 1)
            elif dt.dim() == 1:
                dt = dt.unsqueeze(1)
        else:
            dt = torch.full((B, 1), dt, device=x.device, dtype=x.dtype)

        # ---- Time embeddings ----
        combined_embed = self.t_embed(t) + self.dt_embed(dt)   # (B, E)

        traj_embed = None
        if self.use_time_step:
            if t_traj is None:
                raise ValueError("use_time_step=True but t_traj was not provided.")
            if t_traj.dim() == 1:
                t_traj = t_traj.unsqueeze(1)
            traj_embed = self.traj_time_embed(t_traj)

        # ---- Observation channels ----
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
            parts = [x, x_prev, y_full, mask, combined_embed]
        else:
            parts = [x, x_prev, combined_embed]

        if traj_embed is not None:
            parts.append(traj_embed)

        h = self.backbone(torch.cat(parts, dim=-1))   # (B, hidden_dim)

        velocity = self.velocity_head(h)              # (B, state_dim)

        if self.div_head_active:
            div_scalar = self.div_head(h).squeeze(-1)  # (B,)
        else:
            div_scalar = torch.zeros(B, device=x.device, dtype=x.dtype)

        return velocity, div_scalar
