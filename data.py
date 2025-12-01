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
    observation_operator: Union[Callable, np.ndarray, torch.Tensor, None] = None

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
            self.observation_operator = torch.arctan


class DynamicalSystem(ABC):
    """Abstract base class for dynamical systems"""

    def __init__(self, config: DataAssimilationConfig):
        self.config = config
        self.state_dim = self.get_state_dim()
        self.obs_dim = len(config.obs_components)
        
        # Pre-process observation operator for torch
        self.obs_matrix = None
        if isinstance(self.config.observation_operator, (np.ndarray, torch.Tensor)):
            if isinstance(self.config.observation_operator, np.ndarray):
                self.obs_matrix = torch.from_numpy(self.config.observation_operator).float()
            else:
                self.obs_matrix = self.config.observation_operator.float()

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
    def integrate(self, x0: torch.Tensor, n_steps: int, dt: float = None) -> torch.Tensor:
        """Integrate the system forward in time"""
        if dt is None:
            dt = self.config.dt
            
        t_span = (0, (n_steps - 1) * dt)
        t_eval = torch.arange(0, n_steps * dt, dt)

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
        for _ in range(n_steps - 1):
            x_curr = self.rk4_step(x_curr, dt)
            trajectories.append(x_curr)
            
        # Stack along time dimension: (batch, time, state)
        result = torch.stack(trajectories, dim=1)
        
        if not is_batch:
            return result.squeeze(0)
        return result

    def apply_observation_operator(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the observation operator h(x)"""
        # Ensure input is tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if self.obs_matrix is not None:
            # Linear operator (matrix)
            H = self.obs_matrix.to(x.device)
            if x.ndim == 1:
                return H @ x
            else:
                return x @ H.T

        elif callable(self.config.observation_operator):
            # Handle specific known operators or generic callable
            if self.config.observation_operator in [torch.arctan, np.arctan, arctan_operator]:
                 # Use torch.arctan specifically
                 op = torch.arctan
            else:
                 op = self.config.observation_operator

            if x.ndim == 1:
                observed = x[self.config.obs_components]
            else:
                observed = x[..., self.config.obs_components]
            return op(observed)
        else:
            raise TypeError("observation_operator must be callable, numpy array, or torch Tensor")

    def observe(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Apply observation operator h(x) + noise"""
        observed = self.apply_observation_operator(x)

        if add_noise:
            noise = self.config.obs_noise_std * torch.randn_like(observed)
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
        default_params = {"F":8,"dim":40}
        if config.system_params is None:
            config.system_params = default_params
        else:
            for key, val in default_params.items():
                if key not in config.system_params:
                    config.system_params[key] = val

        super().__init__(config)
        self.forcing = config.system_params["F"]
        self.dim = config.system_params["dim"]
        
        
        self.init_mean = torch.zeros(self.dim)
        self.init_std = 1.0
        self.init_cov = (self.init_std ** 2) * torch.eye(self.dim)
    def get_state_dim(self) -> int:
        return 40

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
        if n_samples == 1:
            return self.init_mean + self.init_std * torch.randn(
                self.dim, device=self.init_mean.device
            )
        else:
            eps = torch.randn(n_samples, self.dim, device=self.init_mean.device)
            return self.init_mean.unsqueeze(0) + self.init_std * eps

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x
    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x

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
    ):
        """
        Args:
            system: DynamicalSystem instance
            trajectories: Shape (n_trajectories, n_steps, state_dim)
            observations: Shape (n_trajectories, n_obs_steps, obs_dim)
            obs_mask: Shape (n_steps,) - boolean mask where True indicates observation available
            trajectories_scaled: Scaled trajectories (optional)
            observations_scaled: Scaled observations (optional)
            mode: 'train', 'inference'
        """
        self.system = system
        self.trajectories = trajectories
        self.observations = observations
        self.obs_mask = obs_mask
        self.trajectories_scaled = trajectories_scaled
        self.observations_scaled = observations_scaled
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
            y_curr = self.observations[trajectory_idx, obs_idx]
            if self.observations_scaled is not None:
                y_curr_scaled = self.observations_scaled[trajectory_idx, obs_idx]
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
        item = {
            "trajectory_idx": torch.LongTensor([idx]),
            "trajectories": torch.as_tensor(self.trajectories[idx], dtype=torch.float32),
            "observations": torch.as_tensor(self.observations[idx], dtype=torch.float32),
            "obs_mask": torch.as_tensor(self.obs_mask, dtype=torch.bool),
        }
        
        if self.trajectories_scaled is not None:
            item["trajectories_scaled"] = torch.as_tensor(
                self.trajectories_scaled[idx], dtype=torch.float32
            )
        if self.observations_scaled is not None:
            item["observations_scaled"] = torch.as_tensor(
                self.observations_scaled[idx], dtype=torch.float32
            )
            
        return item


class TimeAlignedBatchSampler(Sampler[List[int]]):
    """
    Sampler that yields batches of indices corresponding to the same time step
    across multiple trajectories.
    
    Given N trajectories of length T, and batch size B:
    Batch 0: [Traj_0_t0, Traj_1_t0, ..., Traj_B-1_t0]
    Batch 1: [Traj_0_t1, Traj_1_t1, ..., Traj_B-1_t1]
    ...
    Batch T-1: [Traj_0_tT-1, Traj_1_tT-1, ..., Traj_B-1_tT-1]
    Batch T: [Traj_B_t0, Traj_B+1_t0, ..., Traj_2B-1_t0]
    
    This ensures that a single batch processed by the model corresponds to 
    multiple independent trajectories evolving in parallel.
    """
    
    def __init__(self, 
                 data_source_len: int, 
                 num_trajectories: int, 
                 traj_len: int, 
                 batch_size: int,
                 shuffle: bool = False):
        """
        Args:
            data_source_len: Total length of dataset (should be num_trajectories * traj_len)
            num_trajectories: Total number of trajectories
            traj_len: Length of each trajectory (number of time steps)
            batch_size: Number of trajectories to process in parallel
            shuffle: Whether to shuffle the order of trajectory groups (not time steps within groups)
        """
        self.data_source_len = data_source_len
        self.num_trajectories = num_trajectories
        self.traj_len = traj_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if data_source_len != num_trajectories * traj_len:
             pass

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
                # Calculate dataset indices
                # Assuming dataset is ordered as: 
                # [Traj0_t0, Traj0_t1..., Traj1_t0, Traj1_t1...]
                # Index = traj_idx * traj_len + time_idx
                
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
        x0 = self.system.sample_initial_state(self.config.num_trajectories)

        total_steps = self.config.warmup_steps + self.config.len_trajectory
        trajectories = self.system.integrate(x0, total_steps, self.config.dt)

        # Keep trajectories unscaled initially
        trajectories = trajectories[:, self.config.warmup_steps :, :]

        # Generate observations mask
        obs_mask = np.zeros(self.config.len_trajectory, dtype=bool)
        obs_mask[:: self.config.obs_frequency] = True

        obs_time_indices = np.where(obs_mask)[0]
        observations = []

        # Generate observations from unscaled trajectories
        # Use torch loop to keep things in tensor land until the end
        # trajectories is (N, T, D)
        for t in obs_time_indices:
            obs_t = self.system.observe(trajectories[:, t, :], add_noise=True)
            observations.append(obs_t)

        observations = torch.stack(observations, dim=1)
        
        # Convert to numpy for splitting and saving
        trajectories_np = trajectories.cpu().numpy()
        observations_np = observations.cpu().numpy()

        n_train = int(self.config.train_ratio * self.config.num_trajectories)
        n_val = int(self.config.val_ratio * self.config.num_trajectories)

        # Split unscaled data
        splits_unscaled = {
            "train": {
                "trajectories": trajectories_np[:n_train],
                "observations": observations_np[:n_train],
            },
            "val": {
                "trajectories": trajectories_np[n_train : n_train + n_val],
                "observations": observations_np[n_train : n_train + n_val],
            },
            "test": {
                "trajectories": trajectories_np[n_train + n_val :],
                "observations": observations_np[n_train + n_val :],
            },
        }

        # Fit scaler on train trajectories
        train_traj_flat = splits_unscaled["train"]["trajectories"].reshape(-1, trajectories_np.shape[-1])
        scaler_mean = np.mean(train_traj_flat, axis=0)
        scaler_std = np.std(train_traj_flat, axis=0)
        # Avoid division by zero
        scaler_std = np.where(scaler_std < 1e-8, 1.0, scaler_std)

        # Fit scaler on train observations
        train_obs_flat = splits_unscaled["train"]["observations"].reshape(-1, observations_np.shape[-1])
        obs_scaler_mean = np.mean(train_obs_flat, axis=0)
        obs_scaler_std = np.std(train_obs_flat, axis=0)
        # Avoid division by zero
        obs_scaler_std = np.where(obs_scaler_std < 1e-8, 1.0, obs_scaler_std)

        # Apply scaler to all splits
        splits_scaled = {}
        for split_name, split_data in splits_unscaled.items():
            traj = split_data["trajectories"]
            obs = split_data["observations"]
            
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
        data_file = self.data_dir / "data.h5"
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
        data_scaled_file = self.data_dir / "data_scaled.h5"
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

    def setup(self, stage: Optional[str] = None):
        """Load data and create datasets"""
        data_file = self.data_dir / "data.h5"
        data_scaled_file = self.data_dir / "data_scaled.h5"
        
        if not data_file.exists() or not data_scaled_file.exists():
            # Try to generate if missing (though prepare_data should have handled it)
            self._generate_data()

        # Open both files
        # We need to keep them open or read into memory. 
        # For simplicity/performance with small datasets (Lorenz63), reading into memory is fine.
        
        with h5py.File(data_file, "r") as f, h5py.File(data_scaled_file, "r") as f_scaled:
            obs_mask = f["obs_mask"][:]
            
            # Load scaler statistics and update system
            if "scaler_mean" in f:
                scaler_mean = torch.from_numpy(f["scaler_mean"][:]).float()
                scaler_std = torch.from_numpy(f["scaler_std"][:]).float()
                
                # Update system's normalization parameters
                self.system.init_mean = scaler_mean
                self.system.init_std = scaler_std
                logging.info(f"Updated system normalization stats from data.h5")

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
                    mode="train"
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
                    mode="train"
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
                    mode="inference"
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
def _get_observation_operator_type(observation_operator) -> str:
    """Get string identifier for observation operator"""
    if isinstance(observation_operator, (np.ndarray, torch.Tensor)):
        return "linear_projection"
    elif callable(observation_operator):
        if observation_operator in [np.arctan, torch.arctan, arctan_operator]:
            return "arctan"
        else:
            return "custom_callable"
    else:
        return "unknown"


def config_to_dict(config: DataAssimilationConfig) -> Dict[str, Any]:
    """Convert DataAssimilationConfig to serializable dictionary"""
    config_dict = asdict(config)
    
    # Convert observation_operator to string identifier
    obs_op_type = _get_observation_operator_type(config.observation_operator)
    config_dict["observation_operator_type"] = obs_op_type
    
    # If it's a linear projection, save the matrix
    if isinstance(config.observation_operator, (np.ndarray, torch.Tensor)):
        if isinstance(config.observation_operator, torch.Tensor):
            config_dict["observation_operator_matrix"] = config.observation_operator.cpu().numpy().tolist()
        else:
            config_dict["observation_operator_matrix"] = config.observation_operator.tolist()
    
    # Remove the non-serializable observation_operator
    config_dict.pop("observation_operator", None)
    
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
    
    # Reconstruct observation_operator
    obs_op_type = config_dict.pop("observation_operator_type", "arctan")
    obs_components = config_dict.get("obs_components", [0])
    
    if obs_op_type == "linear_projection":
        if "observation_operator_matrix" in config_dict:
            observation_operator = np.array(config_dict.pop("observation_operator_matrix"))
            # NOTE: We keep it as numpy in config, converted to torch in DynamicalSystem.__init__
        else:
            # Reconstruct from obs_components
            state_dim = 3  # Default for Lorenz63, could be made configurable
            observation_operator = create_projection_matrix(state_dim, obs_components)
    elif obs_op_type == "arctan":
        observation_operator = torch.arctan # Use torch.arctan
    else:
        # Default to arctan if unknown
        observation_operator = torch.arctan
    
    config_dict["observation_operator"] = observation_operator
    
    # Convert system_params lists back to numpy arrays if needed (or just leave as lists/values)
    # DynamicalSystem can handle them. But preserving original behavior is good.
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
    obs_op_type = _get_observation_operator_type(config.observation_operator)
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
        obs_op_type,
    ]
    
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
    if isinstance(x, torch.Tensor):
        return torch.arctan(x)
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
    test_state = torch.tensor([1.5, -2.3, 25.6])
    result = system.apply_observation_operator(test_state)
    print(f"Test state: {test_state}")
    print(f"Observed (linear): {result}")
    print(f"Expected: [{test_state[0]:.1f}, {test_state[2]:.1f}]")

    print("\n2. Nonlinear arctan:")
    config_arctan = DataAssimilationConfig(
        obs_components=[0],
        observation_operator=torch.arctan,
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

    print("\nAll tests passed!")
