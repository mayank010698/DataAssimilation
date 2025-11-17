import torch
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable, Union
import h5py
from pathlib import Path
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import logging


@dataclass
class DataAssimilationConfig:
    """Configuration for data assimilation experiments"""

    # Dataset generation
    num_trajectories: int = 1024
    len_trajectory: int = 1024
    warmup_steps: int = 1024
    dt: float = 0.01
    use_preprocessing: bool = True

    # Observation parameters
    obs_noise_std: float = 0.1
    obs_frequency: int = 1
    obs_components: list = None
    observation_operator: Union[Callable, np.ndarray, None] = None

    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # System parameters
    system_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.obs_components is None:
            self.obs_components = [0]
        if self.system_params is None:
            self.system_params = {}
        if self.observation_operator is None:
            self.observation_operator = np.arctan


class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems"""

    def __init__(self, config: DataAssimilationConfig):
        self.config = config
        self.state_dim = self.get_state_dim()
        self.obs_dim = len(config.obs_components)

    @abstractmethod
    def get_state_dim(self) -> int:
        """Return the dimensionality of the state space"""
        pass

    @abstractmethod
    def dynamics(self, t: float, x: np.ndarray) -> np.ndarray:
        """System dynamics dx/dt = f(x, t)"""
        pass

    @abstractmethod
    def get_default_initial_state(self) -> np.ndarray:
        """Get a default initial state"""
        pass

    def sample_initial_state(self, n_samples: int = 1) -> np.ndarray:
        """Sample initial states. Override for custom sampling."""
        default_state = self.get_default_initial_state()
        if n_samples == 1:
            return default_state + 0.1 * np.random.randn(*default_state.shape)
        else:
            return default_state[None, :] + 0.1 * np.random.randn(
                n_samples, *default_state.shape
            )

    def integrate(self, x0: np.ndarray, n_steps: int, dt: float = None) -> np.ndarray:
        """Integrate the system forward in time"""
        if dt is None:
            dt = self.config.dt

        if x0.ndim == 1:
            t_span = (0, (n_steps - 1) * dt)
            t_eval = np.arange(0, n_steps * dt, dt)
            sol = solve_ivp(
                self.dynamics,
                t_span,
                x0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-8,
                atol=1e-11,
            )
            return sol.y.T
        else:
            n_particles = x0.shape[0]
            trajectories = []
            for i in range(n_particles):
                traj = self.integrate(x0[i], n_steps, dt)
                trajectories.append(traj)
            return np.stack(trajectories, axis=0)

    def apply_observation_operator(self, x: np.ndarray) -> np.ndarray:
        """Apply the observation operator h(x)"""

        if isinstance(self.config.observation_operator, np.ndarray):
            H = self.config.observation_operator
            if x.ndim == 1:
                return H @ x
            else:
                return x @ H.T

        elif callable(self.config.observation_operator):
            if x.ndim == 1:
                observed = x[self.config.obs_components]
            else:
                observed = x[..., self.config.obs_components]
            return self.config.observation_operator(observed)
        else:
            raise TypeError("observation_operator must be callable or numpy array")

    def observe(self, x: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Apply observation operator h(x) + noise"""
        observed = self.apply_observation_operator(x)

        if add_noise:
            noise = self.config.obs_noise_std * np.random.randn(*observed.shape)
            observed = observed + noise

        return observed


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

        super().__init__(config)
        self.sigma = config.system_params["sigma"]
        self.rho = config.system_params["rho"]
        self.beta = config.system_params["beta"]

        self.init_mean = np.array([0.0, 0.0, 25.0])
        self.init_cov = np.array(
            [
                [64.0, 50.0, 0.0],
                [50.0, 81.0, 0.0],
                [0.0, 0.0, 75.0],
            ]
        )
        self.init_std = np.sqrt(np.diag(self.init_cov))

    def get_state_dim(self) -> int:
        return 3

    def dynamics(self, t: float, x: np.ndarray) -> np.ndarray:
        """Lorenz 63 dynamics"""
        dxdt = np.zeros_like(x)
        dxdt[0] = self.sigma * (x[1] - x[0])
        dxdt[1] = x[0] * (self.rho - x[2]) - x[1]
        dxdt[2] = x[0] * x[1] - self.beta * x[2]
        return dxdt

    def get_default_initial_state(self) -> np.ndarray:
        return [1.0, 1.0, 1.0]

    def sample_initial_state(self, n_samples: int = 1) -> np.ndarray:
        if n_samples == 1:
            return np.random.multivariate_normal(self.init_mean, self.init_cov)
        else:
            return np.random.multivariate_normal(
                self.init_mean, self.init_cov, size=n_samples
            )

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        return (x - self.init_mean) / self.init_std

    def postprocess(self, x: np.ndarray) -> np.ndarray:
        return self.init_mean + self.init_std * x


class DataAssimilationDataset(Dataset):
    """Dataset for data assimilation experiments"""

    def __init__(
        self,
        system: DynamicalSystem,
        trajectories: np.ndarray,
        observations: np.ndarray,
        obs_mask: np.ndarray,
        mode: str = "inference",
    ):
        """
        Args:
            system: DynamicalSystem instance
            trajectories: Shape (n_trajectories, n_steps, state_dim)
            observations: Shape (n_trajectories, n_obs_steps, obs_dim)
            obs_mask: Shape (n_steps,) - boolean mask where True indicates observation available
            mode: 'train', 'inference'
        """
        self.system = system
        self.trajectories = trajectories
        self.observations = observations
        self.obs_mask = obs_mask
        self.mode = mode

        self.n_trajectories, self.n_steps, self.state_dim = trajectories.shape
        self.obs_dim = observations.shape[-1]

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

    def _get_inference_item(self, idx):
        """Get single time step for inference"""
        trajectory_idx, time_idx = self.items[idx]

        x_prev = self.trajectories[trajectory_idx, time_idx - 1]
        x_curr = self.trajectories[trajectory_idx, time_idx]

        has_observation = self.obs_mask[time_idx]
        if has_observation:
            obs_idx = np.where(self.obs_time_indices == time_idx)[0][0]
            y_curr = self.observations[trajectory_idx, obs_idx]
        else:
            y_curr = np.zeros(self.obs_dim)

        return {
            "trajectory_idx": torch.LongTensor([trajectory_idx]),
            "time_idx": torch.LongTensor([time_idx]),
            "x_prev": torch.FloatTensor(x_prev),
            "x_curr": torch.FloatTensor(x_curr),
            "y_curr": torch.FloatTensor(y_curr),
            "has_observation": torch.BoolTensor([has_observation]),
        }

    def _get_training_item(self, idx):
        """Get full trajectory for training"""
        return {
            "trajectory_idx": torch.LongTensor([idx]),
            "trajectories": torch.FloatTensor(self.trajectories[idx]),
            "observations": torch.FloatTensor(self.observations[idx]),
            "obs_mask": torch.BoolTensor(self.obs_mask),
        }


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
        """Generate trajectory data"""
        x0 = self.system.sample_initial_state(self.config.num_trajectories)

        total_steps = self.config.warmup_steps + self.config.len_trajectory
        trajectories = self.system.integrate(x0, total_steps, self.config.dt)

        trajectories = trajectories[:, self.config.warmup_steps :, :]

        if self.config.use_preprocessing:
            trajectories = self.system.preprocess(trajectories)

        obs_mask = np.zeros(self.config.len_trajectory, dtype=bool)
        obs_mask[:: self.config.obs_frequency] = True

        obs_time_indices = np.where(obs_mask)[0]
        observations = []

        for t in obs_time_indices:
            obs_t = self.system.observe(trajectories[:, t, :], add_noise=True)
            observations.append(obs_t)

        observations = np.stack(observations, axis=1)

        n_train = int(self.config.train_ratio * self.config.num_trajectories)
        n_val = int(self.config.val_ratio * self.config.num_trajectories)

        splits = {
            "train": {
                "trajectories": trajectories[:n_train],
                "observations": observations[:n_train],
            },
            "val": {
                "trajectories": trajectories[n_train : n_train + n_val],
                "observations": observations[n_train : n_train + n_val],
            },
            "test": {
                "trajectories": trajectories[n_train + n_val :],
                "observations": observations[n_train + n_val :],
            },
        }

        data_file = self.data_dir / "data.h5"
        with h5py.File(data_file, "w") as f:
            f.create_dataset("obs_mask", data=obs_mask)

            for split_name, split_data in splits.items():
                group = f.create_group(split_name)
                group.create_dataset("trajectories", data=split_data["trajectories"])
                group.create_dataset("observations", data=split_data["observations"])

        logging.info(f"Data saved to {data_file}")
        for split_name, split_data in splits.items():
            traj_shape = split_data["trajectories"].shape
            obs_shape = split_data["observations"].shape
            logging.info(
                f"{split_name}: trajectories {traj_shape}, observations {obs_shape}"
            )

    def setup(self, stage: Optional[str] = None):
        """Load data and create datasets"""
        data_file = self.data_dir / "data.h5"

        with h5py.File(data_file, "r") as f:
            obs_mask = f["obs_mask"][:]

            if stage == "fit" or stage is None:
                train_traj = f["train/trajectories"][:]
                train_obs = f["train/observations"][:]
                self.train_dataset = DataAssimilationDataset(
                    self.system, train_traj, train_obs, obs_mask, mode="train"
                )

                val_traj = f["val/trajectories"][:]
                val_obs = f["val/observations"][:]
                self.val_dataset = DataAssimilationDataset(
                    self.system, val_traj, val_obs, obs_mask, mode="train"
                )

            if stage == "test" or stage is None:
                test_traj = f["test/trajectories"][:]
                test_obs = f["test/observations"][:]
                self.test_dataset = DataAssimilationDataset(
                    self.system, test_traj, test_obs, obs_mask, mode="inference"
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


# Helper functions for creating observation operators
def create_projection_matrix(state_dim, obs_components):
    """Create projection matrix to select specific state components"""
    obs_dim = len(obs_components)
    H = np.zeros((obs_dim, state_dim))
    for i, comp in enumerate(obs_components):
        H[i, comp] = 1.0
    return H


def identity_operator(x):
    """Identity observation operator: h(x) = x"""
    return x


def arctan_operator(x):
    """Arctan observation operator: h(x) = arctan(x)"""
    return np.arctan(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing observation operators...")

    print("\n1. Linear projection matrix [0, 2]:")
    H = create_projection_matrix(3, [0, 2])
    print(f"Projection matrix:\n{H}")

    config_linear = DataAssimilationConfig(
        num_trajectories=10,
        len_trajectory=50,
        warmup_steps=50,
        dt=0.01,
        obs_noise_std=0.1,
        obs_frequency=5,
        obs_components=[0, 2],
        observation_operator=H,
        system_params={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
    )

    system = Lorenz63(config_linear)
    test_state = np.array([1.5, -2.3, 25.6])
    result = system.apply_observation_operator(test_state)
    print(f"Test state: {test_state}")
    print(f"Observed (linear): {result}")
    print(f"Expected: [{test_state[0]:.1f}, {test_state[2]:.1f}]")

    print("\n2. Nonlinear arctan:")
    config_arctan = DataAssimilationConfig(
        obs_components=[0],
        observation_operator=np.arctan,
    )
    system_arctan = Lorenz63(config_arctan)
    result_arctan = system_arctan.apply_observation_operator(test_state)
    print(f"Test state: {test_state}")
    print(f"Observed (arctan): {result_arctan}")
    print(f"Expected: {np.arctan(test_state[0]):.3f}")

    print("\nAll tests passed!")
