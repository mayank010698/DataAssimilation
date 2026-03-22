import math
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch

# jax_cfd 0.2.1 uses deprecated top-level jax.tree_* aliases removed in JAX 0.9+
import jax as _jax
for _fn in ("tree_map", "tree_flatten", "tree_leaves", "tree_structure", "tree_unflatten"):
    if not hasattr(_jax, _fn):
        setattr(_jax, _fn, getattr(_jax.tree_util, _fn))
del _jax, _fn

import jax.numpy as jnp
import jax.random as jrng
import jax_cfd.base as cfd


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class _InputMeta:
    input_is_torch: bool
    dtype: torch.dtype
    device: torch.device
    was_batched: bool


class KolmogorovBackend:
    """JAX/jax-cfd backend for Kolmogorov stepping and rollouts.

    Public API accepts flattened state vectors:
      - single: (D,)
      - batch: (B, D)
    where D = 2 * grid_size * grid_size.
    """

    def __init__(self, grid_size: int):
        self.grid_size = int(grid_size)
        self.state_dim = 2 * self.grid_size * self.grid_size
        self._stepper_cache = {}

    def get_coords(self) -> np.ndarray:
        x1 = np.linspace(0, 2 * math.pi, self.grid_size, dtype=np.float32)
        x2 = np.linspace(0, 2 * math.pi, self.grid_size, dtype=np.float32)
        X1, X2 = np.meshgrid(x1, x2)
        return np.stack((X1.T.ravel(), X2.T.ravel()), axis=1).astype(np.float32)

    def sample_initial_state(self, seed: int) -> np.ndarray:
        grid = cfd.grids.Grid(
            shape=(self.grid_size, self.grid_size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),
        )
        key = jrng.PRNGKey(int(seed))
        u0, v0 = cfd.initial_conditions.filtered_velocity_field(
            key, grid=grid, maximum_velocity=3.0, peak_wavenumber=4.0
        )
        yi = np.asarray(jnp.stack((u0.data, v0.data)), dtype=np.float32)
        return yi.reshape(-1)

    def step(self, x: ArrayLike, reynolds: ArrayLike, dt: float) -> ArrayLike:
        traj = self.rollout(x, reynolds, n_steps=2, dt=dt)
        if traj.ndim == 2:
            return traj[1]
        return traj[:, 1, :]

    def rollout(self, x0: ArrayLike, reynolds: ArrayLike, n_steps: int, dt: float) -> ArrayLike:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        x0_np, meta = self._to_numpy_batched_flat(x0)
        re_np = self._normalize_reynolds(reynolds, x0_np.shape[0])

        out = np.zeros((x0_np.shape[0], n_steps, self.state_dim), dtype=np.float32)
        out[:, 0, :] = x0_np

        if n_steps == 1:
            return self._restore_output(out, meta, time_dim=True)

        for b in range(x0_np.shape[0]):
            re_val = float(re_np[b])
            stepper = self._get_stepper(re_val, float(dt))
            yi = x0_np[b].reshape(2, self.grid_size, self.grid_size).astype(np.float32, copy=True)
            for t in range(1, n_steps):
                yi = stepper(yi)
                out[b, t, :] = yi.reshape(-1)

        return self._restore_output(out, meta, time_dim=True)

    def _get_stepper(self, reynolds: float, dt: float):
        key = (self.grid_size, round(float(reynolds), 6), float(dt))
        if key in self._stepper_cache:
            return self._stepper_cache[key]

        grid = cfd.grids.Grid(
            shape=(self.grid_size, self.grid_size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),
        )
        bc = cfd.boundaries.periodic_boundary_conditions(2)
        forcing = cfd.forcings.simple_turbulence_forcing(
            grid=grid,
            constant_magnitude=1.0,
            constant_wavenumber=4.0,
            linear_coefficient=-0.1,
            forcing_type="kolmogorov",
        )
        dt_min = cfd.equations.stable_time_step(
            grid=grid,
            max_velocity=5.0,
            max_courant_number=0.5,
            viscosity=1.0 / float(reynolds),
        )
        steps = 1 if dt_min > dt else math.ceil(float(dt) / dt_min)
        repeated_step = cfd.funcutils.repeated(
            f=cfd.equations.semi_implicit_navier_stokes(
                grid=grid,
                forcing=forcing,
                dt=float(dt) / steps,
                density=1.0,
                viscosity=1.0 / float(reynolds),
            ),
            steps=steps,
        )

        def step_np(yi_np: np.ndarray) -> np.ndarray:
            u, v = cfd.initial_conditions.wrap_variables(var=tuple(yi_np), grid=grid, bcs=(bc, bc))
            u, v = repeated_step((u, v))
            return np.asarray(jnp.stack((u.data, v.data)), dtype=np.float32)

        self._stepper_cache[key] = step_np
        return step_np

    def _to_numpy_batched_flat(self, x: ArrayLike) -> Tuple[np.ndarray, _InputMeta]:
        if isinstance(x, torch.Tensor):
            was_batched = x.ndim > 1
            x_t = x if was_batched else x.unsqueeze(0)
            if x_t.shape[-1] != self.state_dim:
                raise ValueError(f"Expected state_dim={self.state_dim}, got {x_t.shape[-1]}")
            x_np = x_t.detach().cpu().to(torch.float32).numpy()
            return x_np, _InputMeta(True, x.dtype, x.device, was_batched)

        x_np = np.asarray(x, dtype=np.float32)
        was_batched = x_np.ndim > 1
        x_np = x_np if was_batched else x_np[None, :]
        if x_np.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got {x_np.shape[-1]}")
        return x_np, _InputMeta(False, torch.float32, torch.device("cpu"), was_batched)

    def _normalize_reynolds(self, reynolds: ArrayLike, batch_size: int) -> np.ndarray:
        if isinstance(reynolds, torch.Tensor):
            re_np = reynolds.detach().cpu().to(torch.float32).numpy().reshape(-1)
        else:
            re_np = np.asarray(reynolds, dtype=np.float32).reshape(-1)

        if re_np.size == 1 and batch_size > 1:
            re_np = np.full((batch_size,), float(re_np[0]), dtype=np.float32)
        if re_np.size != batch_size:
            raise ValueError(
                f"Reynolds size ({re_np.size}) must be 1 or match batch size ({batch_size})."
            )
        return re_np

    def _restore_output(self, out_np: np.ndarray, meta: _InputMeta, time_dim: bool) -> ArrayLike:
        if meta.input_is_torch:
            out_t = torch.from_numpy(out_np.copy()).to(device=meta.device, dtype=meta.dtype)
            if not meta.was_batched:
                return out_t.squeeze(0)
            return out_t
        if not meta.was_batched:
            return out_np.squeeze(0)
        return out_np
