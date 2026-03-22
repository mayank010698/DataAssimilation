import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import lightning.pytorch as pl
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable, Union, Iterator, List
import h5py
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import yaml

from dynamics.kolmogorov_backend import KolmogorovBackend


class ObservationOperator:
    """Observation operator y = phi(Hx)"""
    def __init__(self, obs_components: List[int], nonlinearity: str = "arctan"):
        self.obs_components = obs_components
        self.nonlinearity = nonlinearity

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # 1. Selection (H)
        # Select components
        if x.ndim == 1:
            observed = x[self.obs_components]
        else:
            observed = x[..., self.obs_components]
        
        # 2. Nonlinearity (phi)
        if self.nonlinearity == "arctan":
            return torch.arctan(observed)
        elif self.nonlinearity == "square":
            return torch.square(observed)
        elif self.nonlinearity == "cube":
            return torch.pow(observed, 3)
        elif self.nonlinearity in ["linear_projection", "identity", "none"]:
            return observed
        elif self.nonlinearity == "quad_capped_10":
            # DBF-style: h(x) = clamp(x^4, max=10) / 10
            observed = torch.pow(observed, 4)
            observed = torch.clamp(observed, max=10.0)
            return observed / 10.0
        elif hasattr(torch, self.nonlinearity):
            # Allow other torch functions if they exist (e.g. 'tanh', 'sigmoid')
            func = getattr(torch, self.nonlinearity)
            return func(observed)
        else:
            # Check for numpy equivalents if tensor check failed or if we want generic support
            # But x is converted to tensor above.
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")


@dataclass
class DataAssimilationConfig:
    """Configuration for data assimilation experiments"""

    # Dataset generation
    num_trajectories: int = 1024
    len_trajectory: int = 1024
    warmup_steps: int = 1024
    dt: float = 0.01

    # Observation parameters
    obs_noise_std: float = 0.1
    obs_frequency: int = 1
    obs_components: list = None
    obs_nonlinearity: str = "arctan"
    
    # Process noise (for stochastic trajectory generation)
    process_noise_std: float = 0.0  # 0.0 = deterministic dynamics
    stochastic_validation_test: bool = False  # Whether to use process noise for val/test splits

    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # System parameters
    system_params: Dict[str, Any] = None

    def __post_init__(self):
        # Note: obs_components default is set by each DynamicalSystem subclass
        if self.system_params is None:
            self.system_params = {}


class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems"""

    def __init__(self, config: DataAssimilationConfig):
        self.config = config
        self.state_dim = self.get_state_dim()
        
        # Observation operator
        if config.obs_components is None:
             # Should be set by subclass, but safety fallback
             config.obs_components = [0]
             
        self.obs_dim = len(config.obs_components)
        self.observation_operator = ObservationOperator(
            config.obs_components, config.obs_nonlinearity
        )

    @abstractmethod
    def get_state_dim(self) -> int:
        """Return the dimensionality of the state space"""
        pass

    @abstractmethod
    def dynamics(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """System dynamics dx/dt = f(x, t)"""
        pass

    @abstractmethod
    def get_default_initial_state(self) -> torch.Tensor:
        """Get a default initial state"""
        pass

    def sample_initial_state(self, n_samples: int = 1) -> torch.Tensor:
        """Sample initial states. Override for custom sampling."""
        default_state = self.get_default_initial_state()
        if n_samples == 1:
            return default_state + 0.1 * torch.randn_like(default_state)
        else:
            return default_state.unsqueeze(0) + 0.1 * torch.randn(
                n_samples, *default_state.shape
            )

    def rk4_step(self, x: torch.Tensor, dt: float) -> torch.Tensor:
        """Performs a step of the fourth-order Runge-Kutta integration scheme."""
        k1 = self.dynamics(0, x)
        k2 = self.dynamics(0, x + dt * k1 / 2)
        k3 = self.dynamics(0, x + dt * k2 / 2)
        k4 = self.dynamics(0, x + dt * k3)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    def integrate(
        self,
        x0: torch.Tensor,
        n_steps: int,
        dt: float = None,
        process_noise_std: float = 0.0,
        step_start: int = 0,
        static_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Integrate the system forward in time.
        
        Args:
            x0: Initial state(s), shape (state_dim,) or (batch, state_dim)
            n_steps: Number of time steps to integrate
            dt: Time step size (defaults to config.dt)
            process_noise_std: Standard deviation of additive Gaussian process noise.
                              If > 0, noise is added after each RK4 step: x_{t+1} = RK4(x_t) + σ·ε
            step_start: The absolute step index corresponding to x0 (used for time-dependent dynamics/jumps).
        """
        if dt is None:
            dt = self.config.dt

        # Ensure input is tensor
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32)
            
        # Handle single state vs batch
        is_batch = x0.ndim > 1
        if not is_batch:
            x0 = x0.unsqueeze(0)

        trajectories = [x0]
        x_curr = x0
        
        # We start from t=0, so we need n_steps-1 more steps to get n_steps total points
        for i in range(n_steps - 1):
            x_curr = self.rk4_step(x_curr, dt)
            # Add process noise if specified (stochastic dynamics)
            if process_noise_std > 0:
                x_curr = x_curr + process_noise_std * torch.randn_like(x_curr)
            trajectories.append(x_curr)
            
        # Stack along time dimension: (batch, time, state)
        result = torch.stack(trajectories, dim=1)
        
        if not is_batch:
            return result.squeeze(0)
        return result

    def apply_observation_operator(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the observation operator h(x)"""
        return self.observation_operator(x)

    def observe(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Apply observation operator h(x) + noise"""
        observed = self.apply_observation_operator(x)

        if add_noise:
            noise = self.config.obs_noise_std * torch.randn_like(observed)
            observed = observed + noise

        return observed



class DoubleWell(DynamicalSystem):
    """
    Double Well system with manual state jumps.
    Dynamics: dx = -4x(x^2 - 1)dt + sigma*dW
    Manual jumps (sign flips) applied periodically to replicate specific experiments.
    """

    def __init__(self, config: DataAssimilationConfig):
        # Default to observing component 0
        if config.obs_components is None:
            config.obs_components = [0]
            
        # Add identifier
        if config.system_params is None:
            config.system_params = {}
        config.system_params["system_name"] = "double_well"
            
        super().__init__(config)
        self.init_mean = torch.tensor([-1.0], dtype=torch.float32)
        self.init_std = torch.tensor([1.0], dtype=torch.float32)

    def get_state_dim(self) -> int:
        return 1

    def dynamics(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Double Well deterministic dynamics: f(x) = -V'(x)
        V(x) = (x^2 - 1)^2
        f(x) = -4x(x^2 - 1) = 4x(1 - x^2)
        """
        return 4 * x * (1 - x**2)

    def preprocess(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Scale data using init_mean and init_std (loaded from data file)."""
        if isinstance(x, np.ndarray):
            init_mean_np = self.init_mean.numpy()
            init_std_np = self.init_std.numpy()
            return (x - init_mean_np) / init_std_np

        device = x.device
        return (x - self.init_mean.to(device)) / self.init_std.to(device)

    def postprocess(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Unscale data using init_mean and init_std (loaded from data file)."""
        if isinstance(x, np.ndarray):
            init_mean_np = self.init_mean.numpy()
            init_std_np = self.init_std.numpy()
            return init_mean_np + init_std_np * x

        device = x.device
        return self.init_mean.to(device) + self.init_std.to(device) * x

    def get_default_initial_state(self) -> torch.Tensor:
        return torch.tensor([-1.0])

    def sample_initial_state(self, n_samples: int = 1) -> torch.Tensor:
        # Replicate original: -1 + randn * 0.02
        default = self.get_default_initial_state()
        if n_samples == 1:
            return default + 0.02 * torch.randn_like(default)
        else:
            return default.unsqueeze(0) + 0.02 * torch.randn(n_samples, *default.shape)
            
    def integrate(
        self,
        x0: torch.Tensor,
        n_steps: int,
        dt: float = None,
        process_noise_std: float = 0.0,
        step_start: int = 0,
        static_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Integrate with manual jumps to match original experiment.
        Jumps occur at steps 21, 41, 61, 81... (indices where loop i=20, 40...)
        """
        if dt is None:
            dt = self.config.dt
            
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32)
        is_batch = x0.ndim > 1
        if not is_batch:
            x0 = x0.unsqueeze(0)
            
        trajectories = [x0]
        x_curr = x0
        
        # Stability clamping threshold
        CLAMP_MIN, CLAMP_MAX = -5.0, 5.0
        
        for i in range(n_steps - 1):
            real_i = i + step_start
            
            x_curr = self.rk4_step(x_curr, dt)
            if process_noise_std > 0:
                x_curr = x_curr + process_noise_std * torch.randn_like(x_curr)
            
            # Manual jumps logic from original code:
            if real_i > 0 and real_i % 20 == 0:
                 x_curr = -x_curr
            
            # Clamp to prevent explosion
            x_curr = torch.clamp(x_curr, CLAMP_MIN, CLAMP_MAX)
                 
            trajectories.append(x_curr)
            
        result = torch.stack(trajectories, dim=1)
        if not is_batch:
            return result.squeeze(0)
        return result


class Lorenz63(DynamicalSystem):
    """Lorenz 63 system implementation"""

    def __init__(self, config: DataAssimilationConfig):
        default_params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
        if config.system_params is None:
            config.system_params = default_params
        else:
            for key, val in default_params.items():
                if key not in config.system_params:
                    config.system_params[key] = val

        # Default to observing only first component for Lorenz63
        if config.obs_components is None:
            config.obs_components = [0]

        super().__init__(config)
        self.sigma = config.system_params["sigma"]
        self.rho = config.system_params["rho"]
        self.beta = config.system_params["beta"]

        self.init_mean = torch.tensor([0.0, 0.0, 25.0], dtype=torch.float32)
        self.init_cov = torch.tensor(
            [
                [64.0, 50.0, 0.0],
                [50.0, 81.0, 0.0],
                [0.0, 0.0, 75.0],
            ],
            dtype=torch.float32,
        )
        self.init_std = torch.sqrt(torch.diag(self.init_cov))

    def get_state_dim(self) -> int:
        return 3

    def dynamics(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """Lorenz 63 dynamics"""
        # x shape: (batch, 3) or (3,)
        
        if x.ndim == 1:
            x0, x1, x2 = x[0], x[1], x[2]
            dx0 = self.sigma * (x1 - x0)
            dx1 = x0 * (self.rho - x2) - x1
            dx2 = x0 * x1 - self.beta * x2
            return torch.stack([dx0, dx1, dx2])
        else:
            x0 = x[..., 0]
            x1 = x[..., 1]
            x2 = x[..., 2]
            
            dx0 = self.sigma * (x1 - x0)
            dx1 = x0 * (self.rho - x2) - x1
            dx2 = x0 * x1 - self.beta * x2
            
            return torch.stack([dx0, dx1, dx2], dim=-1)

    def get_default_initial_state(self) -> torch.Tensor:
        return torch.tensor([1.0, 1.0, 1.0])

    def sample_initial_state(self, n_samples: int = 1) -> torch.Tensor:
        if n_samples == 1:
            dist = torch.distributions.MultivariateNormal(self.init_mean, self.init_cov)
            return dist.sample()
        else:
            dist = torch.distributions.MultivariateNormal(self.init_mean, self.init_cov)
            return dist.sample((n_samples,))

    def preprocess(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            # Fallback for legacy numpy support
            init_mean_np = self.init_mean.numpy()
            init_std_np = self.init_std.numpy()
            return (x - init_mean_np) / init_std_np
        
        device = x.device
        return (x - self.init_mean.to(device)) / self.init_std.to(device)

    def postprocess(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            # Fallback for legacy numpy support
            init_mean_np = self.init_mean.numpy()
            init_std_np = self.init_std.numpy()
            return init_mean_np + init_std_np * x
            
        device = x.device
        return self.init_mean.to(device) + self.init_std.to(device) * x


class Lorenz96(DynamicalSystem):
    """Lorenz 96 system implementation"""

    def __init__(self, config: DataAssimilationConfig):
        # Default parameters for Lorenz 96
        # F=8 is standard for chaotic behavior
        # dim=40 is standard, but can be scaled up to 1M+ (see arXiv:2309.00983)
        # dt=0.01 is used here, though 0.05 (6 hours) is also common in literature
        default_params = {
            "F": 8,
            "dim": 40,
            "init_std": 1.0,
            "init_sampling": "gaussian",
            "init_low": -10.0,
            "init_high": 10.0,
        }
        if config.system_params is None:
            config.system_params = default_params
        else:
            for key, val in default_params.items():
                if key not in config.system_params:
                    config.system_params[key] = val

        # Default to observing all components for Lorenz96
        if config.obs_components is None:
            dim = config.system_params["dim"]
            config.obs_components = list(range(dim))

        super().__init__(config)
        self.forcing = config.system_params["F"]
        self.dim = config.system_params["dim"]
        self.init_sampling = config.system_params["init_sampling"]
        self.init_low = config.system_params["init_low"]
        self.init_high = config.system_params["init_high"]

        if self.init_sampling == "uniform":
            # For preprocess/postprocess use mean and std of Uniform(low, high)
            self.init_mean = torch.zeros(self.dim) + (self.init_low + self.init_high) / 2.0
            self.init_std = (self.init_high - self.init_low) / (12.0 ** 0.5)
        else:
            self.init_mean = torch.zeros(self.dim)
            self.init_std = config.system_params["init_std"]
        self.init_cov = (self.init_std ** 2) * torch.eye(self.dim)

    def get_state_dim(self) -> int:
        return self.config.system_params["dim"]

    def dynamics(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """Lorenz 96 dynamics"""
        x_p1 = x.roll(-1, -1)
        x_m2 = x.roll(2, -1)
        x_m1 = x.roll(1, -1)
        return (x_p1 - x_m2) * x_m1 - x + self.forcing
        

    def get_default_initial_state(self) -> torch.Tensor:
        initial_state = torch.zeros((1,self.dim))
        return initial_state
        
    def sample_initial_state(self, n_samples: int = 1) -> torch.Tensor:
        if self.init_sampling == "uniform":
            # DBF-style: Uniform(init_low, init_high) per component
            if n_samples == 1:
                u = torch.rand(self.dim, device=self.init_mean.device)
                return (self.init_low + (self.init_high - self.init_low) * u)
            else:
                u = torch.rand(n_samples, self.dim, device=self.init_mean.device)
                return self.init_low + (self.init_high - self.init_low) * u
        if n_samples == 1:
            return self.init_mean + self.init_std * torch.randn(
                self.dim, device=self.init_mean.device
            )
        else:
            eps = torch.randn(n_samples, self.dim, device=self.init_mean.device)
            return self.init_mean.unsqueeze(0) + self.init_std * eps

    def preprocess(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Scale data using init_mean and init_std (loaded from data file)."""
        if isinstance(x, np.ndarray):
            init_mean_np = self.init_mean.numpy() if isinstance(self.init_mean, torch.Tensor) else np.asarray(self.init_mean)
            init_std_np = self.init_std.numpy() if isinstance(self.init_std, torch.Tensor) else np.asarray(self.init_std)
            return (x - init_mean_np) / init_std_np
        
        device = x.device
        init_mean = self.init_mean.to(device) if isinstance(self.init_mean, torch.Tensor) else torch.as_tensor(self.init_mean, device=device)
        init_std = self.init_std.to(device) if isinstance(self.init_std, torch.Tensor) else torch.as_tensor(self.init_std, device=device)
        return (x - init_mean) / init_std

    def postprocess(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Unscale data using init_mean and init_std (loaded from data file)."""
        if isinstance(x, np.ndarray):
            init_mean_np = self.init_mean.numpy() if isinstance(self.init_mean, torch.Tensor) else np.asarray(self.init_mean)
            init_std_np = self.init_std.numpy() if isinstance(self.init_std, torch.Tensor) else np.asarray(self.init_std)
            return init_mean_np + init_std_np * x
            
        device = x.device
        init_mean = self.init_mean.to(device) if isinstance(self.init_mean, torch.Tensor) else torch.as_tensor(self.init_mean, device=device)
        init_std = self.init_std.to(device) if isinstance(self.init_std, torch.Tensor) else torch.as_tensor(self.init_std, device=device)
        return init_mean + init_std * x


class KuramotoSivashinsky(DynamicalSystem):
    """Kuramoto-Sivashinsky system implementation using ETD-RK4 integration"""

    def __init__(self, config: DataAssimilationConfig):
        # Default parameters
        # J = 64 (spatial resolution)
        # L = 32 * pi (domain size)
        default_params = {"J": 64, "L": 16 * np.pi, "init_std": 1.0}
        if config.system_params is None:
            config.system_params = default_params
        else:
            for key, val in default_params.items():
                if key not in config.system_params:
                    config.system_params[key] = val

        # Default to observing all components if not specified
        if config.obs_components is None:
            config.obs_components = list(range(config.system_params["J"]))

        self.J = config.system_params["J"]
        self.L = config.system_params["L"]
        self.init_std = config.system_params["init_std"]

        super().__init__(config)
        
        # Mean/Std for normalization (updated from data if available, else defaults)
        self.init_mean = torch.zeros(self.J)
        # init_std from config is a scalar multiplier, but self.init_std can be vector
        # We start with ones * scalar
        self.init_std_vector = torch.ones(self.J) * self.init_std
        # Note: DynamicalSystem doesn't enforce self.init_std name, but Lorenz uses it.
        # I will shadow it with self.normalization_mean/std to avoid confusion with the scalar param
        self.normalization_mean = torch.zeros(self.J)
        self.normalization_std = torch.ones(self.J)

        # Spectral Setup
        # Wave numbers for Real FFT
        # rfftfreq returns [0, 1, ..., n/2]
        # We need to scale by 2*pi/L
        # k = 2*pi/L * [0, 1, ..., J/2]
        self.k = (2 * np.pi / self.L) * torch.arange(0, self.J // 2 + 1, dtype=torch.float64)
        
        # Precompute ETD-RK4 coefficients (on CPU initially)
        self.precompute_etd_rk4_coefficients()

    def get_state_dim(self) -> int:
        return self.J

    def precompute_etd_rk4_coefficients(self):
        """Precompute ETD-RK4 coefficients using complex contour integration"""
        # Based on Kassam and Trefethen (2005) and DAFM implementation
        
        # Use float64 for precision in coefficient calculation
        dtype = torch.float64
        device = torch.device("cpu") # Compute on CPU
        
        h = self.config.dt
        k = self.k.to(device=device, dtype=dtype)
        
        # Linear operator L = k^2 - k^4 (diagonal in spectral space)
        # Note: signs matter. 
        # KS eq: u_t + u_xx + u_xxxx + u u_x = 0
        # => u_t = -u_xx - u_xxxx - u u_x
        # Linear part: -u_xx - u_xxxx
        # In Fourier: -(-k^2) - (k^4) = k^2 - k^4
        L = k**2 - k**4
        
        # Precompute ETD-RK4 scalar quantities
        self.E = torch.exp(h * L).to(dtype=dtype)
        self.E2 = torch.exp(h * L / 2).to(dtype=dtype)
        
        # Roots of unity for contour integration
        n_roots = 16
        roots = np.exp(1j * np.pi * (0.5 + np.arange(n_roots)) / n_roots)
        roots = torch.from_numpy(roots).to(device=device)
        
        # Contour for each element of L
        # CL shape: (n_roots, len(k))
        CL = h * L.unsqueeze(0) + roots.unsqueeze(1)
        
        # Function to compute mean over contour
        def contour_mean(func_of_CL):
            return func_of_CL.mean(dim=0).real
            
        # Q = h * mean( (exp(CL/2) - 1) / CL )
        self.Q = h * contour_mean((torch.exp(CL / 2) - 1) / CL)
        
        # f1 = h * mean( (-4 - CL + exp(CL)(4 - 3CL + CL^2)) / CL^3 )
        self.f1 = h * contour_mean(
            (-4 - CL + torch.exp(CL) * (4 - 3 * CL + CL ** 2)) / CL ** 3
        )
        
        # f2 = h * mean( (2 + CL + exp(CL)(-2 + CL)) / CL^3 )
        self.f2 = h * contour_mean(
            (2 + CL + torch.exp(CL) * (-2 + CL)) / CL ** 3
        )
        
        # f3 = h * mean( (-4 - 3CL - CL^2 + exp(CL)(4 - CL)) / CL^3 )
        self.f3 = h * contour_mean(
            (-4 - 3 * CL - CL ** 2 + torch.exp(CL) * (4 - CL)) / CL ** 3
        )
        
        self.E = self.E.float()
        self.E2 = self.E2.float()
        self.Q = self.Q.float()
        self.f1 = self.f1.float()
        self.f2 = self.f2.float()
        self.f3 = self.f3.float()
        self.k = self.k.float()

    def dynamics(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Compute only the nonlinear term N(u) = -u u_x = -0.5 (u^2)_x.
        Note: This is NOT the full dynamics dx/dt. It is used internally by ETD-RK4.
        
        Input: x is in SPECTRAL space (u_hat) if called internally?
        Wait, DynamicalSystem expects x in PHYSICAL space.
        
        If this is called by rk4_step (from base class), it will be wrong because it misses linear part.
        Since we override integrate, this shouldn't be called for integration.
        But for completeness/debugging, we might want to throw error or implement full dynamics.
        
        Given we override integrate, let's implement the nonlinear term helper here,
        but expect input in SPECTRAL space.
        """
        raise NotImplementedError("Use integrate() for KS system. Standard dynamics() not implemented.")

    def nonlinear_term(self, u_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute nonlinear term N(u) = -0.5 * d/dx (u^2) in Fourier space.
        
        Args:
            u_hat: State in spectral domain (batch, J//2 + 1)
        """
        # Transform to physical space
        u = torch.fft.irfft(u_hat, n=self.J, dim=-1)
        
        # Compute u^2
        u2 = u ** 2
        
        # Transform back to spectral
        u2_hat = torch.fft.rfft(u2, n=self.J, dim=-1)
        
        # Derivative in spectral space: d/dx -> * (i k)
        # N(u_hat) = -0.5 * (i k) * u2_hat
        # Note: k is shape (J//2+1,)
        # Ensure k is on correct device
        if self.k.device != u_hat.device:
            self.k = self.k.to(u_hat.device)
            
        ik = 1j * self.k
        
        return -0.5 * ik * u2_hat

    def integrate(
        self,
        x0: torch.Tensor,
        n_steps: int,
        dt: float = None,
        process_noise_std: float = 0.0,
        step_start: int = 0,
        static_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Integrate using ETD-RK4."""
        if dt is None:
            dt = self.config.dt
            
        # Ensure coefficients are on correct device
        device = x0.device
        if self.E.device != device:
            self.E = self.E.to(device)
            self.E2 = self.E2.to(device)
            self.Q = self.Q.to(device)
            self.f1 = self.f1.to(device)
            self.f2 = self.f2.to(device)
            self.f3 = self.f3.to(device)
            self.k = self.k.to(device)

        # Handle batch
        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32)
        is_batch = x0.ndim > 1
        if not is_batch:
            x0 = x0.unsqueeze(0)
            
        # Initial state is in physical space. Transform to spectral.
        u_hat = torch.fft.rfft(x0, n=self.J, dim=-1)
        
        trajectories = [x0]
        
        for _ in range(n_steps - 1):
            # ETD-RK4 Step
            
            # N1 = N(u_n)
            N1 = self.nonlinear_term(u_hat)
            
            # a = E2 * u_n + Q * N1
            a = self.E2 * u_hat + self.Q * N1
            
            # N2 = N(a)
            N2 = self.nonlinear_term(a)
            
            # b = E2 * u_n + Q * N2
            b = self.E2 * u_hat + self.Q * N2
            
            # N3 = N(b)
            N3 = self.nonlinear_term(b)
            
            # c = E2 * a + Q * (2*N3 - N1) 
            # DAFM:
            # v = u_hat (u in code)
            # v1 = E2*v + Q*N1
            # v2a = E2*v + Q*N2a  (where N2a = NL(v1))
            # v2b = E2*v1 + Q*(2*N2b - N1) (where N2b = NL(v2a))
            # Final: v_new = E*v + f1*N1 + 2*f2*(N2a + N2b) + f3*N3 (where N3 = NL(v2b))
            
            # Mapping names:
            # v -> u_hat
            # N1 -> N1
            # v1 -> a
            # N2a -> N2
            # v2a -> b
            # N2b -> N3 (different from my simple N3 above)
            
            # Correct Sequence:
            # 1. N1 = NL(u_hat)
            # 2. a = E2*u_hat + Q*N1
            # 3. N2 = NL(a)
            # 4. b = E2*u_hat + Q*N2
            # 5. N3 = NL(b)
            # 6. c = E2*a + Q*(2*N3 - N1)
            # 7. N4 = NL(c)
            # 8. u_next = E*u_hat + f1*N1 + 2*f2*(N2 + N3) + f3*N4
            
            # Note: DAFM code uses:
            # v1 = E2 * v + Q * N1
            # v2a = E2 * v + Q * N2a (N2a = NL(v1))
            # v2b = E2 * v1 + Q * (2 * N2b - N1) (N2b = NL(v2a))
            # v = E * v + N1 * f1 + 2 * (N2a + N2b) * f2 + N3 * f3 (N3 = NL(v2b))
            
            # So my "a" is v1.
            # My "b" is v2a.
            # My "c" is v2b.
            
            v = u_hat
            
            # Step 1
            N1 = self.nonlinear_term(v)
            v1 = self.E2 * v + self.Q * N1
            
            # Step 2
            N2a = self.nonlinear_term(v1)
            v2a = self.E2 * v + self.Q * N2a
            
            # Step 3
            N2b = self.nonlinear_term(v2a)
            v2b = self.E2 * v1 + self.Q * (2 * N2b - N1)
            
            # Step 4
            N3 = self.nonlinear_term(v2b)
            
            # Update
            u_next = self.E * v + N1 * self.f1 + 2 * (N2a + N2b) * self.f2 + N3 * self.f3
            
            # Process Noise handling (Physical space)
            if process_noise_std > 0:
                # Transform to physical
                u_phys = torch.fft.irfft(u_next, n=self.J, dim=-1)
                # Add noise
                u_phys = u_phys + process_noise_std * torch.randn_like(u_phys)
                # Transform back
                u_next = torch.fft.rfft(u_phys, n=self.J, dim=-1)
            
            u_hat = u_next
            
            # Save trajectory in physical space
            trajectories.append(torch.fft.irfft(u_hat, n=self.J, dim=-1))
            
        result = torch.stack(trajectories, dim=1)
        
        if not is_batch:
            return result.squeeze(0)
        return result

    def get_default_initial_state(self) -> torch.Tensor:
        # Standard initial condition from papers often involves cosine/sine sum
        # u(x, 0) = cos(x/L) * (1 + sin(x/L)) is a common one, 
        # but DAFM uses noise on spectral modes?
        # DAFM KS: x0 from dapper.mods.KS.Model.x0.
        # DAPPER default x0: cos(x/16 * (2*pi/L) * x)? No.
        
        # Let's use a simple random start with some smoothness, 
        # or the one from the plan:
        # "Seed Profile: Initialize with u(x) = cos(x/16) * (1 + sin(x/16))"
        # Wait, plan says: "Seed Profile: Initialize with u = cos(x) * (1 + sin(x))"
        # and "L = 32 pi".
        # If L=32pi, domain is [0, 32pi).
        # x = np.arange(J) * L / J.
        
        # Let's implement this profile.
        x = torch.arange(self.J, dtype=torch.float32) * self.L / self.J
        # Rescale argument to match "cos(x/16)" for domain size?
        # If domain is 32pi, x goes 0 to ~100.
        # cos(x) oscillates 16 times in 32pi? No, 32pi/2pi = 16 periods.
        # "cos(x/16) * (1 + sin(x/16))" -> Period is 32pi. 
        # This matches the domain size L=32pi (one full period).
        
        # x/16 goes from 0 to 2pi.
        arg = x / 16.0
        u0 = torch.cos(arg) * (1.0 + torch.sin(arg))
        return u0

    def sample_initial_state(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample initial state with burn-in.
        Plan: 
        1. Seed Profile: u = cos(x/16)*(1+sin(x/16))
        2. Burn-in: Discard first 2,000 steps of the generated trajectory.
        """
        
        # 1. Base seed
        u0 = self.get_default_initial_state() # (J,)
        
        # Create batch
        if n_samples > 1:
            u0 = u0.unsqueeze(0).repeat(n_samples, 1)
            # Add small noise to differentiate trajectories
            u0 = u0 + 1e-4 * torch.randn_like(u0)
        else:
            u0 = u0.unsqueeze(0) # (1, J)
            
        # 2. Burn-in
        # "Discard the first 2,000 steps"
        # We integrate for 2000 steps.
        # We assume dt is default.
        burn_in_steps = 2000
        
        # We can use our integrate method, but we only need the last state.
        # integrate returns full trajectory, which is memory expensive for 2000 steps if we only need last.
        # I'll implement a loop here to save memory.
        
        # Move to device if needed (currently CPU)
        u_curr = u0
        
        # Spectral transform
        u_hat = torch.fft.rfft(u_curr, n=self.J, dim=-1)
        
        # Ensure coeffs on correct device
        if self.E.device != u_hat.device:
            self.E = self.E.to(u_hat.device)
            self.E2 = self.E2.to(u_hat.device)
            self.Q = self.Q.to(u_hat.device)
            self.f1 = self.f1.to(u_hat.device)
            self.f2 = self.f2.to(u_hat.device)
            self.f3 = self.f3.to(u_hat.device)
            self.k = self.k.to(u_hat.device)
            
        # Burn-in loop
        # We can do this in chunks if needed, but 2000 is fast for 1D.
        # We assume deterministic burn-in to reach attractor? Or with noise?
        # Usually attractor is intrinsic, noise helps exploration.
        # The plan doesn't specify noise during burn-in, but usually KS is chaotic enough.
        
        for _ in range(burn_in_steps):
             # Copying single step logic from integrate to avoid overhead/memory of full storage
             v = u_hat
             N1 = self.nonlinear_term(v)
             v1 = self.E2 * v + self.Q * N1
             N2a = self.nonlinear_term(v1)
             v2a = self.E2 * v + self.Q * N2a
             N2b = self.nonlinear_term(v2a)
             v2b = self.E2 * v1 + self.Q * (2 * N2b - N1)
             N3 = self.nonlinear_term(v2b)
             u_hat = self.E * v + N1 * self.f1 + 2 * (N2a + N2b) * self.f2 + N3 * self.f3
             
        # Transform back to physical
        u_final = torch.fft.irfft(u_hat, n=self.J, dim=-1)
        
        if n_samples == 1:
            return u_final.squeeze(0)
        return u_final

    def preprocess(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            init_mean = self.normalization_mean.numpy()
            init_std = self.normalization_std.numpy()
            return (x - init_mean) / init_std
            
        device = x.device
        init_mean = self.normalization_mean.to(device)
        init_std = self.normalization_std.to(device)
        return (x - init_mean) / init_std

    def postprocess(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(x, np.ndarray):
            init_mean = self.normalization_mean.numpy()
            init_std = self.normalization_std.numpy()
            return init_mean + init_std * x
            
        device = x.device
        init_mean = self.normalization_mean.to(device)
        init_std = self.normalization_std.to(device)
        return init_mean + init_std * x


def generate_dataset_splits(system: DynamicalSystem, config: DataAssimilationConfig, initial_states: Optional[Dict[str, torch.Tensor]] = None, precomputed_splits: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    """
    Generate trajectory data (splits).
    
    Args:
        system: DynamicalSystem instance
        config: DataAssimilationConfig instance
        initial_states: Optional dictionary with keys 'train', 'val', 'test' containing initial states (tensors).
                       If provided, these are used instead of sampling new ones.
        precomputed_splits: Optional dictionary with keys 'train', 'val', 'test' containing tuples (trajectories, observations).
                            If provided, these are used directly instead of generating new data.
                       
    Returns:
        splits_unscaled: Dictionary containing train/val/test splits, each with 'trajectories' and 'observations'
        obs_mask: Boolean mask indicating which time steps are observed
    """
    total_steps = config.warmup_steps + config.len_trajectory
    
    # Calculate split sizes
    n_train = int(config.train_ratio * config.num_trajectories)
    n_val = int(config.val_ratio * config.num_trajectories)
    n_test = config.num_trajectories - n_train - n_val
    
    process_noise_std = config.process_noise_std
    if process_noise_std > 0:
        logging.info(f"Generating training data with process noise std={process_noise_std}")
    
    # Generate observations mask (same for all splits)
    obs_mask = np.zeros(config.len_trajectory, dtype=bool)
    obs_mask[:: config.obs_frequency] = True
    obs_time_indices = np.where(obs_mask)[0]
    
    def generate_split(n_samples: int, use_process_noise: bool, x0_given: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trajectories and observations for a split."""
        if n_samples == 0:
            state_dim = system.state_dim
            obs_dim = system.obs_dim
            return (
                np.zeros((0, config.len_trajectory, state_dim)),
                np.zeros((0, len(obs_time_indices), obs_dim)),
            )
        
        if x0_given is not None:
            x0 = x0_given
            # Check if x0 size matches n_samples (allow mismatch if we just want to use provided x0 regardless of n_samples? 
            # Better to be strict or allow slicing if x0 is larger)
            if len(x0) != n_samples:
                 # If provided x0 is larger, slice it?
                 if len(x0) > n_samples:
                     x0 = x0[:n_samples]
                 else:
                     raise ValueError(f"Provided initial states size {len(x0)} is smaller than split size {n_samples}")
        else:
            x0 = system.sample_initial_state(n_samples)
            
        if n_samples == 1 and x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        noise_std = process_noise_std if use_process_noise else 0.0
        trajectories = system.integrate(x0, total_steps, config.dt, process_noise_std=noise_std)
        
        # Remove warmup steps
        trajectories = trajectories[:, config.warmup_steps:, :]
        
        # Generate observations
        observations = []
        for t in obs_time_indices:
            obs_t = system.observe(trajectories[:, t, :], add_noise=True)
            observations.append(obs_t)
        observations = torch.stack(observations, dim=1)
        
        return trajectories.cpu().numpy(), observations.cpu().numpy()
    
    # Get initial states if provided
    x0_train = initial_states.get("train") if initial_states else None
    x0_val = initial_states.get("val") if initial_states else None
    x0_test = initial_states.get("test") if initial_states else None

    # Generate each split separately
    
    # Training data: with process noise (if configured)
    if precomputed_splits and "train" in precomputed_splits:
        train_traj, train_obs = precomputed_splits["train"]
    else:
        train_traj, train_obs = generate_split(n_train, use_process_noise=True, x0_given=x0_train)
        
    # Validation and test data
    val_use_noise = config.stochastic_validation_test
    test_use_noise = config.stochastic_validation_test
    
    if precomputed_splits and "val" in precomputed_splits:
        val_traj, val_obs = precomputed_splits["val"]
    else:
        val_traj, val_obs = generate_split(n_val, use_process_noise=val_use_noise, x0_given=x0_val)
        
    if precomputed_splits and "test" in precomputed_splits:
        test_traj, test_obs = precomputed_splits["test"]
    else:
        test_traj, test_obs = generate_split(n_test, use_process_noise=test_use_noise, x0_given=x0_test)
    
    splits_unscaled = {
        "train": {"trajectories": train_traj, "observations": train_obs},
        "val": {"trajectories": val_traj, "observations": val_obs},
        "test": {"trajectories": test_traj, "observations": test_obs},
    }
    
    return splits_unscaled, obs_mask


def observations_from_trajectories(
    system: DynamicalSystem,
    trajectories: np.ndarray,
    obs_frequency: int,
    len_trajectory: int,
) -> tuple:
    """
    Compute observations from state trajectories using the system's observation operator.
    Used to produce multiple datasets (different obs operators) from the same trajectories.

    Args:
        system: DynamicalSystem instance (must have the desired obs_nonlinearity and obs_noise_std)
        trajectories: (N, T, state_dim) state trajectories
        obs_frequency: observation frequency (every this many steps)
        len_trajectory: trajectory length T

    Returns:
        observations: (N, num_obs_times, obs_dim) numpy array
        obs_mask: (T,) boolean mask of observed time indices
    """
    obs_mask = np.zeros(len_trajectory, dtype=bool)
    obs_mask[::obs_frequency] = True
    obs_time_indices = np.where(obs_mask)[0]

    if trajectories.size == 0:
        state_dim = system.state_dim
        obs_dim = system.obs_dim
        return (
            np.zeros((0, len(obs_time_indices), obs_dim)),
            obs_mask,
        )

    obs_list = []
    for t in obs_time_indices:
        state_t = torch.from_numpy(trajectories[:, t, :].astype(np.float32))
        obs_t = system.observe(state_t, add_noise=True)
        obs_list.append(obs_t)
    observations = torch.stack(obs_list, dim=1).cpu().numpy()
    return observations, obs_mask


def save_generated_data(data_dir: Path, splits_unscaled: Dict, obs_mask: np.ndarray):
    """
    Calculate scalers and save data to disk.
    
    Args:
        data_dir: Directory to save data to
        splits_unscaled: Dictionary containing train/val/test splits (from generate_dataset_splits)
        obs_mask: Boolean mask indicating which time steps are observed
    """
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        
    # Get state dim from first non-empty split for scaler computation
    n_train = len(splits_unscaled["train"]["trajectories"])
    train_traj = splits_unscaled["train"]["trajectories"]
    train_obs = splits_unscaled["train"]["observations"]
    
    if n_train > 0:
        state_dim = train_traj.shape[-1]
        obs_dim = train_obs.shape[-1]
    else:
        # Fallback to val or test
        if len(splits_unscaled["val"]["trajectories"]) > 0:
            state_dim = splits_unscaled["val"]["trajectories"].shape[-1]
            obs_dim = splits_unscaled["val"]["observations"].shape[-1]
        elif len(splits_unscaled["test"]["trajectories"]) > 0:
            state_dim = splits_unscaled["test"]["trajectories"].shape[-1]
            obs_dim = splits_unscaled["test"]["observations"].shape[-1]
        else:
             # Empty dataset?
             state_dim = 1
             obs_dim = 1

    # Fit scaler on train trajectories
    if n_train > 0:
        train_traj_flat = train_traj.reshape(-1, state_dim)
        scaler_mean = np.mean(train_traj_flat, axis=0)
        scaler_std = np.std(train_traj_flat, axis=0)
        
        train_obs_flat = train_obs.reshape(-1, obs_dim)
        obs_scaler_mean = np.mean(train_obs_flat, axis=0)
        obs_scaler_std = np.std(train_obs_flat, axis=0)
    else:
        # Use defaults if no training data
        scaler_mean = np.zeros(state_dim)
        scaler_std = np.ones(state_dim)
        obs_scaler_mean = np.zeros(obs_dim)
        obs_scaler_std = np.ones(obs_dim)
        
    # Avoid division by zero
    scaler_std = np.where(scaler_std < 1e-8, 1.0, scaler_std)
    obs_scaler_std = np.where(obs_scaler_std < 1e-8, 1.0, obs_scaler_std)

    # Apply scaler to all splits
    splits_scaled = {}
    for split_name, split_data in splits_unscaled.items():
        traj = split_data["trajectories"]
        obs = split_data["observations"]
        
        if len(traj) == 0:
             splits_scaled[split_name] = {
                "trajectories": traj,
                "observations": obs,
            }
             continue
        
        # Scale trajectories
        traj_flat = traj.reshape(-1, traj.shape[-1])
        traj_scaled_flat = (traj_flat - scaler_mean) / scaler_std
        traj_scaled = traj_scaled_flat.reshape(traj.shape)
        
        # Scale observations
        obs_flat = obs.reshape(-1, obs.shape[-1])
        obs_scaled_flat = (obs_flat - obs_scaler_mean) / obs_scaler_std
        obs_scaled = obs_scaled_flat.reshape(obs.shape)
        
        splits_scaled[split_name] = {
            "trajectories": traj_scaled,
            "observations": obs_scaled,
        }

    # Save unscaled data
    data_file = data_dir / "data.h5"
    with h5py.File(data_file, "w") as f:
        f.create_dataset("obs_mask", data=obs_mask)
        # Save trajectory scalers for reference even in unscaled file
        f.create_dataset("scaler_mean", data=scaler_mean)
        f.create_dataset("scaler_std", data=scaler_std)
        # Save observation scalers
        f.create_dataset("obs_scaler_mean", data=obs_scaler_mean)
        f.create_dataset("obs_scaler_std", data=obs_scaler_std)

        for split_name, split_data in splits_unscaled.items():
            group = f.create_group(split_name)
            group.create_dataset("trajectories", data=split_data["trajectories"])
            group.create_dataset("observations", data=split_data["observations"])

    logging.info(f"Unscaled data saved to {data_file}")

    # Save scaled data
    data_scaled_file = data_dir / "data_scaled.h5"
    with h5py.File(data_scaled_file, "w") as f:
        f.create_dataset("obs_mask", data=obs_mask)
        f.create_dataset("scaler_mean", data=scaler_mean)
        f.create_dataset("scaler_std", data=scaler_std)
        f.create_dataset("obs_scaler_mean", data=obs_scaler_mean)
        f.create_dataset("obs_scaler_std", data=obs_scaler_std)

        for split_name, split_data in splits_scaled.items():
            group = f.create_group(split_name)
            group.create_dataset("trajectories", data=split_data["trajectories"])
            group.create_dataset("observations", data=split_data["observations"])

    logging.info(f"Scaled data saved to {data_scaled_file}")
    
    for split_name in ["train", "val", "test"]:
        traj_shape = splits_unscaled[split_name]["trajectories"].shape
        obs_shape = splits_unscaled[split_name]["observations"].shape
        logging.info(
            f"{split_name}: trajectories {traj_shape}, observations {obs_shape}"
        )


class DataAssimilationDataset(Dataset):
    """Dataset for data assimilation experiments"""

    def __init__(
        self,
        system: DynamicalSystem,
        trajectories: np.ndarray,
        observations: np.ndarray,
        obs_mask: np.ndarray,
        trajectories_scaled: Optional[np.ndarray] = None,
        observations_scaled: Optional[np.ndarray] = None,
        mode: str = "inference",
        obs_components: Optional[List[int]] = None,
    ):
        """
        Args:
            system: DynamicalSystem instance
            trajectories: Shape (n_trajectories, n_steps, state_dim)
            observations: Shape (n_trajectories, n_obs_steps, full_obs_dim)
            obs_mask: Shape (n_steps,) - boolean mask where True indicates observation available
            trajectories_scaled: Scaled trajectories (optional)
            observations_scaled: Scaled observations (optional)
            mode: 'train', 'inference'
            obs_components: List of component indices to observe. If None, uses system.config.obs_components.
                            This allows loading dense observations from disk but only exposing a subset.
        """
        self.system = system
        self.trajectories = trajectories
        self.observations = observations
        self.obs_mask = obs_mask
        self.trajectories_scaled = trajectories_scaled
        self.observations_scaled = observations_scaled
        self.mode = mode
        
        # Determine which components to return
        if obs_components is not None:
            self.obs_components = obs_components
        else:
            self.obs_components = system.config.obs_components
            
        self.n_trajectories, self.n_steps, self.state_dim = trajectories.shape
        self.obs_dim = len(self.obs_components) # Dimension of *observed* part

        self.obs_time_indices = np.where(obs_mask)[0]

        if mode == "inference":
            self.items = [
                (traj, t)
                for traj in range(self.n_trajectories)
                for t in range(1, self.n_steps)
            ]

    def __len__(self):
        if self.mode == "inference":
            return len(self.items)
        else:
            return self.n_trajectories

    def __getitem__(self, idx):
        if self.mode == "inference":
            return self._get_inference_item(idx)
        else:
            return self._get_training_item(idx)
            
    def _slice_obs(self, obs_tensor):
        """Slice observations to select only requested components"""
        if self.obs_components is None:
            return obs_tensor
        return obs_tensor[..., self.obs_components]

    def _get_inference_item(self, idx):
        """Get single time step for inference"""
        trajectory_idx, time_idx = self.items[idx]

        x_prev = self.trajectories[trajectory_idx, time_idx - 1]
        x_curr = self.trajectories[trajectory_idx, time_idx]

        # Get scaled versions if available
        if self.trajectories_scaled is not None:
            x_prev_scaled = self.trajectories_scaled[trajectory_idx, time_idx - 1]
            x_curr_scaled = self.trajectories_scaled[trajectory_idx, time_idx]
        else:
            x_prev_scaled = np.zeros_like(x_prev) # Placeholder
            x_curr_scaled = np.zeros_like(x_curr) # Placeholder

        has_observation = self.obs_mask[time_idx]
        if has_observation:
            obs_idx = np.where(self.obs_time_indices == time_idx)[0][0]
            
            # Slice observations here
            y_curr = self.observations[trajectory_idx, obs_idx]
            y_curr = y_curr[self.obs_components] # Apply sparsity
            
            if self.observations_scaled is not None:
                y_curr_scaled = self.observations_scaled[trajectory_idx, obs_idx]
                y_curr_scaled = y_curr_scaled[self.obs_components] # Apply sparsity
            else:
                y_curr_scaled = np.zeros_like(y_curr)
        else:
            y_curr = np.zeros(self.obs_dim)
            y_curr_scaled = np.zeros(self.obs_dim)

        # Convert to torch if they are numpy arrays (from HDF5)
        return {
            "trajectory_idx": torch.LongTensor([trajectory_idx]),
            "time_idx": torch.LongTensor([time_idx]),
            "x_prev": torch.as_tensor(x_prev, dtype=torch.float32),
            "x_curr": torch.as_tensor(x_curr, dtype=torch.float32),
            "y_curr": torch.as_tensor(y_curr, dtype=torch.float32),
            "x_prev_scaled": torch.as_tensor(x_prev_scaled, dtype=torch.float32),
            "x_curr_scaled": torch.as_tensor(x_curr_scaled, dtype=torch.float32),
            "y_curr_scaled": torch.as_tensor(y_curr_scaled, dtype=torch.float32),
            "has_observation": torch.BoolTensor([has_observation]),
        }

    def _get_training_item(self, idx):
        """Get full trajectory for training"""
        obs = self.observations[idx]
        obs = obs[..., self.obs_components] # Apply sparsity
        
        item = {
            "trajectory_idx": torch.LongTensor([idx]),
            "trajectories": torch.as_tensor(self.trajectories[idx], dtype=torch.float32),
            "observations": torch.as_tensor(obs, dtype=torch.float32),
            "obs_mask": torch.as_tensor(self.obs_mask, dtype=torch.bool),
        }
        
        if self.trajectories_scaled is not None:
            item["trajectories_scaled"] = torch.as_tensor(
                self.trajectories_scaled[idx], dtype=torch.float32
            )
        if self.observations_scaled is not None:
            obs_s = self.observations_scaled[idx]
            obs_s = obs_s[..., self.obs_components] # Apply sparsity
            item["observations_scaled"] = torch.as_tensor(
                obs_s, dtype=torch.float32
            )
            
        return item


class TimeAlignedBatchSampler(Sampler[List[int]]):
    """
    Sampler that yields batches of indices corresponding to the same time step
    across multiple trajectories.
    """
    
    def __init__(self, 
                 data_source_len: int, 
                 num_trajectories: int, 
                 traj_len: int, 
                 batch_size: int,
                 shuffle: bool = False):
        self.data_source_len = data_source_len
        self.num_trajectories = num_trajectories
        self.traj_len = traj_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Calculate number of trajectory groups
        self.num_traj_groups = (num_trajectories + batch_size - 1) // batch_size

    def __iter__(self) -> Iterator[List[int]]:
        # Determine order of trajectory groups
        if self.shuffle:
            group_indices = torch.randperm(self.num_traj_groups).tolist()
        else:
            group_indices = list(range(self.num_traj_groups))
            
        for group_idx in group_indices:
            # Determine which trajectories are in this group
            start_traj_idx = group_idx * self.batch_size
            end_traj_idx = min(start_traj_idx + self.batch_size, self.num_trajectories)
            
            current_batch_trajs = list(range(start_traj_idx, end_traj_idx))
            
            # Iterate through time steps for this group of trajectories
            for t in range(self.traj_len):
                batch_indices = [
                    traj_idx * self.traj_len + t 
                    for traj_idx in current_batch_trajs
                ]
                yield batch_indices

    def __len__(self) -> int:
        return self.num_traj_groups * self.traj_len


class DataAssimilationDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for data assimilation"""

    def __init__(
        self,
        config: DataAssimilationConfig,
        system_class: type,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.config = config
        self.system_class = system_class
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.system = system_class(config)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Generate and save data if it doesn't exist"""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True)

        data_file = self.data_dir / "data.h5"
        if not data_file.exists():
            logging.info("Generating data...")
            self._generate_data()
        else:
            logging.info(f"Data already exists at {data_file}")

    def _generate_data(self):
        """Generate trajectory data (both unscaled and scaled versions)"""
        splits_unscaled, obs_mask = generate_dataset_splits(self.system, self.config)
        save_generated_data(self.data_dir, splits_unscaled, obs_mask)

    def setup(self, stage: Optional[str] = None):
        """Load data and create datasets"""
        data_file = self.data_dir / "data.h5"
        data_scaled_file = self.data_dir / "data_scaled.h5"
        
        if not data_file.exists() or not data_scaled_file.exists():
            # Try to generate if missing (though prepare_data should have handled it)
            self._generate_data()

        # Open both files
        with h5py.File(data_file, "r") as f, h5py.File(data_scaled_file, "r") as f_scaled:
            obs_mask = f["obs_mask"][:]
            
            # Load scaler statistics and update system
            if "scaler_mean" in f:
                scaler_mean = torch.from_numpy(f["scaler_mean"][:]).float()
                scaler_std = torch.from_numpy(f["scaler_std"][:]).float()
                
                # Update system's normalization parameters
                # For Lorenz63/96/KS:
                if hasattr(self.system, 'init_mean'):
                    self.system.init_mean = scaler_mean
                if hasattr(self.system, 'init_std'):
                    # Be careful if init_std is scalar vs vector
                    # If it was scalar in __init__, this overwrites with vector.
                    self.system.init_std = scaler_std
                    
                if hasattr(self.system, 'normalization_mean'):
                    self.system.normalization_mean = scaler_mean
                if hasattr(self.system, 'normalization_std'):
                    self.system.normalization_std = scaler_std

                logging.info(f"Updated system normalization stats from data.h5")

            if "obs_scaler_mean" in f:
                self.obs_scaler_mean = torch.from_numpy(f["obs_scaler_mean"][:]).float()
                self.obs_scaler_std = torch.from_numpy(f["obs_scaler_std"][:]).float()
                logging.info(f"Loaded observation scaler stats from data.h5")
            else:
                self.obs_scaler_mean = None
                self.obs_scaler_std = None

            if stage == "fit" or stage is None:
                train_traj = f["train/trajectories"][:]
                train_obs = f["train/observations"][:]
                train_traj_scaled = f_scaled["train/trajectories"][:]
                train_obs_scaled = f_scaled["train/observations"][:]
                
                self.train_dataset = DataAssimilationDataset(
                    self.system, 
                    train_traj, 
                    train_obs, 
                    obs_mask, 
                    trajectories_scaled=train_traj_scaled,
                    observations_scaled=train_obs_scaled,
                    mode="train",
                    obs_components=self.config.obs_components
                )

                val_traj = f["val/trajectories"][:]
                val_obs = f["val/observations"][:]
                val_traj_scaled = f_scaled["val/trajectories"][:]
                val_obs_scaled = f_scaled["val/observations"][:]
                
                self.val_dataset = DataAssimilationDataset(
                    self.system, 
                    val_traj, 
                    val_obs, 
                    obs_mask, 
                    trajectories_scaled=val_traj_scaled,
                    observations_scaled=val_obs_scaled,
                    mode="train",
                    obs_components=self.config.obs_components
                )

            if stage == "test" or stage is None:
                test_traj = f["test/trajectories"][:]
                test_obs = f["test/observations"][:]
                test_traj_scaled = f_scaled["test/trajectories"][:]
                test_obs_scaled = f_scaled["test/observations"][:]
                
                self.test_dataset = DataAssimilationDataset(
                    self.system, 
                    test_traj, 
                    test_obs, 
                    obs_mask, 
                    trajectories_scaled=test_traj_scaled,
                    observations_scaled=test_obs_scaled,
                    mode="inference",
                    obs_components=self.config.obs_components
                )

    def train_dataloader(self):
        """Training dataloader"""
        if self.train_dataset is None:
            raise ValueError("Train dataset not set up. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Validation dataloader"""
        if self.val_dataset is None:
            raise ValueError("Val dataset not set up. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Test dataloader for inference"""
        if self.test_dataset is None:
            raise ValueError("Test dataset not set up. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


# Config serialization utilities

def config_to_dict(config: DataAssimilationConfig) -> Dict[str, Any]:
    """Convert DataAssimilationConfig to serializable dictionary"""
    config_dict = asdict(config)
    
    # Remove complex objects if any (none currently, but good practice)
    # config_dict.pop("observation_operator", None) # Removed from class
    
    # Convert numpy arrays in system_params to lists
    if config_dict.get("system_params"):
        system_params = {}
        for k, v in config_dict["system_params"].items():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                if isinstance(v, torch.Tensor):
                    system_params[k] = v.cpu().numpy().tolist()
                else:
                    system_params[k] = v.tolist()
            else:
                system_params[k] = v
        config_dict["system_params"] = system_params
    
    return config_dict


def dict_to_config(config_dict: Dict[str, Any]) -> DataAssimilationConfig:
    """Reconstruct DataAssimilationConfig from dictionary"""
    # Make a copy to avoid modifying the original
    config_dict = config_dict.copy()
    
    # Ensure obs_components has a default for backward compatibility with old configs
    if config_dict.get("obs_components") is None:
        config_dict["obs_components"] = [0]
    
    # Backward compatibility: default process_noise_std if not present
    if "process_noise_std" not in config_dict:
        config_dict["process_noise_std"] = 0.0
        
    # Backward compatibility: handle old keys if present
    config_dict.pop("observation_operator_type", None)
    config_dict.pop("observation_operator_matrix", None)
    
    # Convert system_params lists back to numpy arrays if needed
    if config_dict.get("system_params"):
        system_params = {}
        for k, v in config_dict["system_params"].items():
            if isinstance(v, list):
                system_params[k] = np.array(v)
            else:
                system_params[k] = v
        config_dict["system_params"] = system_params
    
    return DataAssimilationConfig(**config_dict)


def generate_dataset_directory_name(
    config: DataAssimilationConfig, system_name: str = "lorenz63"
) -> str:
    """Generate comprehensive directory name from config parameters"""
    # For many obs_components (like Lorenz96), just indicate count
    if len(config.obs_components) > 4:
        dim = config.system_params.get('dim')
        if dim is None:
            dim = config.system_params.get('J') # KS
        if dim is None:
            dim = len(config.obs_components)
            
        obs_comp_str = f"{len(config.obs_components)}of{dim}"
    else:
        obs_comp_str = ",".join(map(str, config.obs_components))
    
    # Build directory name with key parameters
    parts = [
        system_name,
        f"n{config.num_trajectories}",
        f"len{config.len_trajectory}",
        f"dt{config.dt:.4f}".replace(".", "p"),
        f"obs{config.obs_noise_std:.3f}".replace(".", "p"),
        f"freq{config.obs_frequency}",
        f"comp{obs_comp_str}",
        config.obs_nonlinearity,
    ]
    
    # Add process noise to directory name if non-zero
    if config.process_noise_std > 0:
        parts.append(f"pnoise{config.process_noise_std:.3f}".replace(".", "p"))

    # Add init std to directory name if not one (Gaussian); uniform ICs get initUni
    init_sampling = config.system_params.get("init_sampling", "gaussian")
    if init_sampling == "uniform":
        init_low = config.system_params.get("init_low", -10.0)
        init_high = config.system_params.get("init_high", 10.0)
        parts.append(f"initUni{init_low:.0f}_{init_high:.0f}".replace(".", "p").replace("-", "m"))
    else:
        init_std = config.system_params.get("init_std", 1.0)
        if init_std != 1.0:
            parts.append(f"init{init_std:.3f}".replace(".", "p"))

    # Add KS specific parameters
    if "J" in config.system_params and "L" in config.system_params:
        parts.append(f"J{config.system_params['J']}")
        parts.append(f"L{config.system_params['L']:.2f}".replace(".", "p"))
    
    return "_".join(parts)


def save_config_yaml(config: DataAssimilationConfig, path: Path):
    """Save DataAssimilationConfig as YAML file"""
    config_dict = config_to_dict(config)
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def load_config_yaml(path: Path) -> DataAssimilationConfig:
    """Load DataAssimilationConfig from YAML file"""
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return dict_to_config(config_dict)


# =============================================================================
# Kolmogorov Flow (2D Navier-Stokes)
# =============================================================================


def generate_kolmogorov_data(
    size: int = 150,
    num_trajectories: int = 200,
    num_steps: int = 200,
    warmup_steps: int = 100,
    dt: float = 0.04,
    re_min: float = 500.0,
    re_max: float = 1500.0,
    seed: int = 42,
    output_path: Union[str, Path] = None,
) -> Path:
    """Generate Kolmogorov flow trajectories using the shared backend."""
    np.random.seed(seed)
    reynoldses = np.random.uniform(re_min, re_max, num_trajectories).astype(np.float32)
    backend = KolmogorovBackend(grid_size=size)
    x0_flat = backend.sample_initial_state(seed=seed)
    coords = backend.get_coords()

    y = np.zeros((num_trajectories, num_steps, 2, size, size), dtype=np.float32)

    for ni in range(num_trajectories):
        reynolds = float(reynoldses[ni])
        traj_flat = backend.rollout(x0_flat, reynolds=reynolds, n_steps=warmup_steps + num_steps, dt=dt)
        y[ni] = traj_flat[warmup_steps:].reshape(num_steps, 2, size, size)
        logging.info(f"Trajectory {ni + 1}/{num_trajectories} done (Re={reynolds:.1f})")

    # Resolve output path
    output_path = Path(output_path) if output_path is not None else Path(".")
    if output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "kolmogorov_data.npz"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = output_path

    np.savez(
        save_path,
        y=y,
        u=reynoldses.astype(np.float32),
        x=coords,
        dt=np.float32(dt),
    )
    logging.info(f"Kolmogorov data saved to {save_path} — shape: {y.shape}")
    return save_path


def generate_kolmogorov_data_subset(
    size: int = 150,
    num_trajectories: int = 200,
    num_steps: int = 200,
    warmup_steps: int = 100,
    dt: float = 0.04,
    re_min: float = 500.0,
    re_max: float = 1500.0,
    seed: int = 42,
    trajectory_indices: Union[List[int], np.ndarray] = None,
    output_part_path: Union[str, Path] = None,
) -> Path:
    """Generate a subset of Kolmogorov flow trajectories (by global trajectory index).

    This is designed for schedule-independent parallelization: each trajectory index
    is computed deterministically from (seed, trajectory_index) by reproducing the
    same Reynolds-number sampling used in `generate_kolmogorov_data`.
    """
    if trajectory_indices is None:
        raise ValueError("trajectory_indices must be provided")
    if output_part_path is None:
        raise ValueError("output_part_path must be provided (path to the part .npz file)")

    trajectory_indices = np.asarray(trajectory_indices, dtype=np.int64)
    if trajectory_indices.ndim != 1:
        raise ValueError("trajectory_indices must be a 1D array of global indices")

    output_part_path = Path(output_part_path)
    output_part_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    reynoldses = np.random.uniform(re_min, re_max, num_trajectories).astype(np.float32)
    backend = KolmogorovBackend(grid_size=size)
    x0_flat = backend.sample_initial_state(seed=seed)
    coords = backend.get_coords()

    y_part = np.zeros((len(trajectory_indices), num_steps, 2, size, size), dtype=np.float32)
    u_part = reynoldses[trajectory_indices].astype(np.float32)

    for local_ni, global_ni in enumerate(trajectory_indices.tolist()):
        reynolds = float(reynoldses[global_ni])
        traj_flat = backend.rollout(x0_flat, reynolds=reynolds, n_steps=warmup_steps + num_steps, dt=dt)
        y_part[local_ni] = traj_flat[warmup_steps:].reshape(num_steps, 2, size, size)

        logging.info(
            f"Kolmogorov subset traj {local_ni + 1}/{len(trajectory_indices)} "
            f"(global idx={global_ni}) done (Re={reynolds:.1f})"
        )

    np.savez(
        output_part_path,
        y=y_part,
        u=u_part,
        trajectory_indices=trajectory_indices.astype(np.int64),
        x=coords,
        dt=np.float32(dt),
    )
    logging.info(
        f"Kolmogorov subset saved to {output_part_path} — shape: {y_part.shape}"
    )
    return output_part_path


def merge_kolmogorov_data_parts(
    part_paths: List[Union[str, Path]],
    num_trajectories: int,
    output_npz_path: Union[str, Path],
) -> Path:
    """Merge per-index Kolmogorov `.npz` parts into a single final `kolmogorov_data.npz`.

    Expects each part to contain:
      - `y`: [Ni_part, num_steps, 2, size, size]
      - `u`: [Ni_part]
      - `trajectory_indices`: global indices for each part entry
      - `x`: coordinates array
      - `dt`: scalar dt
    """
    if len(part_paths) == 0:
        raise ValueError("part_paths must contain at least one part .npz")

    output_npz_path = Path(output_npz_path)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Load shape/metadata from the first part.
    first = np.load(str(part_paths[0]), allow_pickle=True)
    y0 = first["y"].astype(np.float32)
    num_steps = int(y0.shape[1])
    size = int(y0.shape[3])
    x = first["x"].astype(np.float32)
    dt = np.float32(first["dt"])

    y_full = np.zeros((num_trajectories, num_steps, 2, size, size), dtype=np.float32)
    u_full = np.zeros((num_trajectories,), dtype=np.float32)

    covered = np.zeros((num_trajectories,), dtype=bool)

    for part_path in part_paths:
        part = np.load(str(part_path), allow_pickle=True)
        y_part = part["y"].astype(np.float32)
        u_part = part["u"].astype(np.float32)
        idx_part = part["trajectory_indices"].astype(np.int64)

        if y_part.shape[1:] != (num_steps, 2, size, size):
            raise ValueError(
                f"Part {part_path} has incompatible shape {y_part.shape}; "
                f"expected (?, {num_steps}, 2, {size}, {size})"
            )
        if len(idx_part) != y_part.shape[0] or len(idx_part) != len(u_part):
            raise ValueError(
                f"Part {part_path} has mismatched trajectory_indices/u/y lengths."
            )

        if np.any(covered[idx_part]):
            dupes = np.where(covered[idx_part])[0]
            raise ValueError(
                f"Part overlap detected while merging: {len(dupes)} duplicate indices."
            )

        y_full[idx_part] = y_part
        u_full[idx_part] = u_part
        covered[idx_part] = True

    missing = np.where(~covered)[0]
    if len(missing) != 0:
        raise ValueError(
            f"Missing trajectories while merging Kolmogorov parts: {len(missing)} indices "
            f"e.g. {missing[:10].tolist()}"
        )

    np.savez(
        output_npz_path,
        y=y_full,
        u=u_full,
        x=x,
        dt=dt,
    )
    logging.info(
        f"Merged Kolmogorov data to {output_npz_path} — shape: {y_full.shape}"
    )
    return output_npz_path


@dataclass
class KolmogorovConfig:
    """Configuration for Kolmogorov flow (2D Navier-Stokes) experiments."""

    # Simulation / generation parameters
    grid_size: int = 150
    num_trajectories: int = 200
    num_steps: int = 200
    warmup_steps: int = 100
    dt: float = 0.04
    re_min: float = 500.0
    re_max: float = 1500.0
    seed: int = 42

    # Observation parameters
    obs_frequency: int = 5       # observe every N steps → num_steps//obs_frequency assimilation steps
    obs_grid_size: int = 150     # obs_grid_size x obs_grid_size equispaced grid;
                                 # 150 = fully observed, 10 = paper's sparse 100-point grid
    obs_noise_std: float = 0.0   # noise-free by default (paper uses 0 for encoder training)
    # RNG seed used to generate deterministic observation noise in `KolmogorovDataset`.
    # If left as None, it falls back to `seed`.
    obs_noise_seed: Optional[int] = None

    # Data splits (applied in trajectory-index order)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    def __post_init__(self):
        if self.grid_size % self.obs_grid_size != 0:
            raise ValueError(
                f"obs_grid_size ({self.obs_grid_size}) must evenly divide "
                f"grid_size ({self.grid_size})"
            )
        if self.obs_noise_seed is None:
            self.obs_noise_seed = int(self.seed)


def save_kolmogorov_config_yaml(config: KolmogorovConfig, path: Path):
    """Save KolmogorovConfig as YAML file."""
    config_dict = asdict(config)
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


class KolmogorovSystem(DynamicalSystem):
    """Kolmogorov flow (2D incompressible Navier-Stokes) system wrapper.

    The physical state is the 2D velocity field (u, v) on a periodic
    (grid_size x grid_size) domain, stored as a flat vector of length
    2 * grid_size^2. Layout: [u-channel (grid_size^2 values), v-channel (grid_size^2 values)],
    where within each channel the spatial ordering is C-order row-major
    (index k = x * grid_size + y).

    Online PDE simulation requires JAX + jax-cfd; calling dynamics() raises
    NotImplementedError. Use generate_kolmogorov_data() to produce data offline.

    The observation operator is a reused ObservationOperator with precomputed flat
    indices for an equispaced obs_grid_size x obs_grid_size subsampling grid.
    Observation layout: [u at grid pts (obs_grid_size^2), v at grid pts (obs_grid_size^2)],
    matching the observ_idx convention in CODEBASE_ld-ensf/kolmogorov_flow/observation_data.py.
    """

    def __init__(self, config: KolmogorovConfig):
        # Bypass DynamicalSystem.__init__: it expects DataAssimilationConfig and builds
        # an ObservationOperator from integer component indices, which doesn't fit here.
        # We set all required attributes manually instead.
        self.config = config
        self.grid_size = config.grid_size
        self.obs_grid_size = config.obs_grid_size
        self.state_dim = self.get_state_dim()

        # Build obs indices matching observ_idx() in observation_data.py:
        #   flat_idx = grid_inx_y + grid_inx_x * Ny  (= y + x * grid_size = x*grid_size + y)
        stride = config.grid_size // config.obs_grid_size
        gx, gy = np.meshgrid(
            np.arange(0, config.grid_size, stride),
            np.arange(0, config.grid_size, stride),
        )
        flat_idx = (gy + gx * config.grid_size).ravel()  # (obs_grid_size^2,) spatial indices
        obs_components = np.concatenate(
            [flat_idx, flat_idx + config.grid_size ** 2]
        ).tolist()  # u-channel then v-channel → 2 * obs_grid_size^2 total

        self.obs_dim = len(obs_components)
        self.observation_operator = ObservationOperator(
            obs_components, nonlinearity="identity"
        )
        self.requires_static_params = True
        self.allow_persistence_fallback = False
        self.backend = KolmogorovBackend(grid_size=self.grid_size)

    def get_state_dim(self) -> int:
        return 2 * self.config.grid_size * self.config.grid_size

    def dynamics(self, t: float, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "KolmogorovSystem online dynamics use JAX/jax-cfd. "
            "Use generate_kolmogorov_data() to produce trajectory data offline."
        )

    def get_default_initial_state(self) -> torch.Tensor:
        return torch.zeros(self.state_dim)

    def integrate(
        self,
        x0: torch.Tensor,
        n_steps: int,
        dt: float = None,
        process_noise_std: float = 0.0,
        step_start: int = 0,
        static_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if dt is None:
            dt = self.config.dt
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        if not isinstance(x0, torch.Tensor):
            x0 = torch.tensor(x0, dtype=torch.float32)
        is_batch = x0.ndim > 1
        if not is_batch:
            x0 = x0.unsqueeze(0)

        if static_params is None:
            raise ValueError(
                "KolmogorovSystem.integrate requires static_params with Reynolds "
                "(key 'reynolds' or 'u')."
            )
        reynolds = static_params.get("reynolds", static_params.get("u"))
        if reynolds is None:
            raise ValueError("Missing Reynolds static param ('reynolds' or 'u').")
        if process_noise_std > 0:
            # Keep stochastic semantics: additive noise after each transition
            # should affect all subsequent states.
            re = torch.as_tensor(reynolds, dtype=torch.float32, device=x0.device).reshape(-1)
            if re.numel() == 1 and x0.shape[0] > 1:
                re = re.expand(x0.shape[0])
            if re.numel() != x0.shape[0]:
                raise ValueError(
                    f"Reynolds batch size ({re.numel()}) does not match x0 batch size ({x0.shape[0]})."
                )

            states = [x0]
            x_curr = x0
            for _ in range(n_steps - 1):
                x_next = self.backend.step(x_curr, reynolds=re, dt=dt)
                x_next = x_next + process_noise_std * torch.randn_like(x_next)
                states.append(x_next)
                x_curr = x_next
            result = torch.stack(states, dim=1)
        else:
            result = self.backend.rollout(x0, reynolds=reynolds, n_steps=n_steps, dt=dt)

        if not is_batch:
            return result.squeeze(0)
        return result


class KolmogorovDataset(Dataset):
    """Dataset for Kolmogorov flow experiments.

    Loads a pre-generated kolmogorov_data.npz file and exposes items in the
    same format as DataAssimilationDataset for compatibility with EnsembleScoreFilter
    and other models.

    Observations are pre-computed in __init__ using the system's observation operator
    and optional Gaussian noise.

    In 'inference' mode, __getitem__ returns per-timestep dicts
    {trajectory_idx, time_idx, x_prev, x_curr, y_curr, has_observation}.
    In 'train' mode, __getitem__ returns per-trajectory dicts
    {trajectory_idx, x, y, u, dt, obs_mask}.
    """

    def __init__(
        self,
        npz_path: Union[str, Path],
        config: KolmogorovConfig,
        system: KolmogorovSystem,
        split: str = "train",
        mode: str = "inference",
    ):
        data = np.load(npz_path, allow_pickle=True)
        y_all = data["y"].astype(np.float32)  # [Ni, Nt, 2, H, W]
        u_all = data["u"].astype(np.float32)  # [Ni] Reynolds numbers
        self.dt = float(data["dt"])

        Ni = y_all.shape[0]
        n_train = int(config.train_ratio * Ni)
        n_val = int(config.val_ratio * Ni)

        if split == "train":
            self.y = y_all[:n_train]
            self.u = u_all[:n_train]
        elif split == "val":
            self.y = y_all[n_train : n_train + n_val]
            self.u = u_all[n_train : n_train + n_val]
        else:  # test
            self.y = y_all[n_train + n_val :]
            self.u = u_all[n_train + n_val :]

        self.config = config
        self.system = system
        self.mode = mode

        Ni_split, Nt, C, H, W = self.y.shape
        self.n_trajectories = Ni_split
        self.n_steps = Nt
        self.state_dim = C * H * W

        # Observation mask and time indices
        self.obs_mask = np.zeros(Nt, dtype=bool)
        self.obs_mask[:: config.obs_frequency] = True
        self.obs_time_indices = np.where(self.obs_mask)[0]  # [Nt_obs]

        # Precompute observations: apply observation operator at every obs timestep.
        # Flatten (2, H, W) → (2*H*W,) then index with obs_components.
        flat = self.y.reshape(Ni_split, Nt, -1)             # [Ni, Nt, 2*H*W]
        obs_flat = flat[:, self.obs_time_indices, :]         # [Ni, Nt_obs, 2*H*W]
        Ni_s, Nt_obs, D = obs_flat.shape

        obs_tensor = torch.from_numpy(obs_flat.reshape(-1, D))  # [Ni*Nt_obs, 2*H*W]
        with torch.no_grad():
            obs_computed = system.apply_observation_operator(obs_tensor)  # [Ni*Nt_obs, obs_dim]
        if config.obs_noise_std > 0.0:
            # Deterministic per-dataset observation noise so different dataset
            # variants (different obs_noise_std / seed) are reproducible.
            gen = torch.Generator()
            gen.manual_seed(int(config.obs_noise_seed))
            noise = torch.randn(
                obs_computed.shape,
                dtype=obs_computed.dtype,
                device=obs_computed.device,
                generator=gen,
            )
            obs_computed = obs_computed + config.obs_noise_std * noise
        self.observations = obs_computed.numpy().reshape(Ni_s, Nt_obs, -1)  # [Ni, Nt_obs, obs_dim]
        self.obs_dim = self.observations.shape[-1]

        if mode == "inference":
            self.items = [
                (traj_idx, t)
                for traj_idx in range(Ni_split)
                for t in range(1, Nt)
            ]

    def __len__(self):
        if self.mode == "inference":
            return len(self.items)
        return self.n_trajectories

    def __getitem__(self, idx):
        if self.mode == "inference":
            return self._get_inference_item(idx)
        return self._get_train_item(idx)

    def _get_inference_item(self, idx):
        traj_idx, t = self.items[idx]

        x_prev = self.y[traj_idx, t - 1].ravel()  # [2*H*W]
        x_curr = self.y[traj_idx, t].ravel()       # [2*H*W]

        has_observation = bool(self.obs_mask[t])
        if has_observation:
            obs_pos = int(np.searchsorted(self.obs_time_indices, t))
            y_curr = self.observations[traj_idx, obs_pos]  # [obs_dim]
        else:
            y_curr = np.zeros(self.obs_dim, dtype=np.float32)

        return {
            "trajectory_idx": torch.LongTensor([traj_idx]),
            "time_idx": torch.LongTensor([t]),
            "x_prev": torch.as_tensor(x_prev, dtype=torch.float32),
            "x_curr": torch.as_tensor(x_curr, dtype=torch.float32),
            "y_curr": torch.as_tensor(y_curr, dtype=torch.float32),
            # Keep both names for backward compatibility and clarity.
            "u": torch.as_tensor(self.u[traj_idx], dtype=torch.float32),
            "reynolds": torch.as_tensor(self.u[traj_idx], dtype=torch.float32),
            "has_observation": torch.BoolTensor([has_observation]),
        }

    def _get_train_item(self, idx):
        x = self.y[idx].reshape(self.n_steps, -1)  # [Nt, 2*H*W]

        return {
            "trajectory_idx": torch.LongTensor([idx]),
            "x": torch.as_tensor(x, dtype=torch.float32),             # [Nt, state_dim]
            "y": torch.as_tensor(self.observations[idx], dtype=torch.float32),  # [Nt_obs, obs_dim]
            "u": torch.as_tensor(self.u[idx], dtype=torch.float32),   # scalar Re
            "dt": torch.as_tensor(self.dt, dtype=torch.float32),
            "obs_mask": torch.as_tensor(self.obs_mask, dtype=torch.bool),
        }


class KolmogorovDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Kolmogorov flow experiments."""

    def __init__(
        self,
        npz_path: Union[str, Path],
        config: KolmogorovConfig,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        super().__init__()
        self.npz_path = Path(npz_path)
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.system = KolmogorovSystem(config)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = KolmogorovDataset(
                self.npz_path, self.config, self.system, split="train", mode="train"
            )
            self.val_dataset = KolmogorovDataset(
                self.npz_path, self.config, self.system, split="val", mode="train"
            )
        if stage == "test" or stage is None:
            self.test_dataset = KolmogorovDataset(
                self.npz_path, self.config, self.system, split="test", mode="inference"
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not set up. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Val dataset not set up. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not set up. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing observation operators...")

    print("\n1. Linear projection (Identity nonlinearity):")
    config_linear = DataAssimilationConfig(
        num_trajectories=10,
        len_trajectory=50,
        warmup_steps=50,
        dt=0.01,
        obs_noise_std=0.1,
        obs_frequency=5,
        obs_components=[0, 2],
        obs_nonlinearity="identity",
        system_params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    )

    system = Lorenz63(config_linear)
    test_state = torch.tensor([1.5, -2.3, 25.6])
    result = system.apply_observation_operator(test_state)
    print(f"Test state: {test_state}")
    print(f"Observed (linear): {result}")
    print(f"Expected: [{test_state[0]:.1f}, {test_state[2]:.1f}]")

    print("\n2. Nonlinear arctan:")
    config_arctan = DataAssimilationConfig(
        obs_components=[0],
        obs_nonlinearity="arctan",
    )
    system_arctan = Lorenz63(config_arctan)
    result_arctan = system_arctan.apply_observation_operator(test_state)
    print(f"Test state: {test_state}")
    print(f"Observed (arctan): {result_arctan}")
    print(f"Expected: {torch.arctan(test_state[0]):.3f}")
    
    print("\n3. Testing integration (RK4)")
    x0 = system.sample_initial_state(1).squeeze(0)
    print(f"Initial state: {x0}")
    traj = system.integrate(x0, 10)
    print(f"Trajectory shape: {traj.shape}")
    print(f"Last state: {traj[-1]}")

    print("\n4. Testing Lorenz 96 (High-dim + Nonlinear Obs)")
    config_l96 = DataAssimilationConfig(
        obs_components=[0, 1, 2],
        obs_nonlinearity="arctan",
        system_params={"dim": 100, "F": 8}
    )
    system_l96 = Lorenz96(config_l96)
    print(f"State dim: {system_l96.state_dim} (Expected: 100)")
    
    x0_l96 = system_l96.sample_initial_state(1).squeeze(0)
    print(f"Initial state shape: {x0_l96.shape}")
    
    # Test dynamics
    traj_l96 = system_l96.integrate(x0_l96, 10)
    print(f"Trajectory shape: {traj_l96.shape}")
    
    # Test observation
    obs_l96 = system_l96.observe(x0_l96)
    print(f"Observation shape: {obs_l96.shape} (Expected: 3)")
    print(f"Observation values: {obs_l96}")

    print("\n5. Testing Kuramoto-Sivashinsky (ETD-RK4)")
    config_ks = DataAssimilationConfig(
        dt=0.05, # typical for KS
        system_params={"J": 64, "L": 32 * np.pi},
        obs_nonlinearity="arctan"
    )
    system_ks = KuramotoSivashinsky(config_ks)
    print(f"KS State dim: {system_ks.state_dim} (Expected: 64)")
    
    x0_ks = system_ks.sample_initial_state(1).squeeze(0)
    print(f"KS Initial state shape: {x0_ks.shape}")
    
    # Test dynamics integration
    traj_ks = system_ks.integrate(x0_ks, 20)
    print(f"KS Trajectory shape: {traj_ks.shape}")
    print(f"KS Last state mean: {traj_ks[-1].mean():.3f}")

    print("\n6. Testing Double Well (Jumps)")
    config_dw = DataAssimilationConfig(
        dt=0.1,
        system_params={},
        obs_nonlinearity="identity"
    )
    system_dw = DoubleWell(config_dw)
    x0_dw = system_dw.get_default_initial_state() # -1
    print(f"DW Initial: {x0_dw}")
    
    # Run long enough to see a jump (at i=20 -> step 21)
    # We need n_steps > 21. Let's do 25.
    traj_dw = system_dw.integrate(x0_dw, 25)
    print(f"DW Trajectory shape: {traj_dw.shape}")
    
    # Check values around jump
    # i=20 corresponds to index 21 in trajectory (since traj[0] is x0)
    # Loop i=19 computes traj[20] -> No jump
    # Loop i=20 computes traj[21] -> Jump
    print(f"Step 20 (i=19): {traj_dw[20].item():.4f}")
    print(f"Step 21 (i=20): {traj_dw[21].item():.4f} (Should be flipped/positive)")
    print(f"Step 22 (i=21): {traj_dw[22].item():.4f}")

    print("\nAll tests passed!")
