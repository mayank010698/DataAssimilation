"""
Dataset for Rectified Flow Training

Provides (x_{t-1}, x_t) pairs from trajectories for learning the transition distribution.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import h5py
from pathlib import Path
from typing import Optional
import logging


class RFTransitionDataset(Dataset):
    """
    Dataset for learning p(x_t | x_{t-1}) with Rectified Flow
    
    Extracts consecutive state pairs from trajectories.
    """
    
    def __init__(
        self, 
        trajectories: np.ndarray, 
        observations: Optional[np.ndarray] = None,
        window: int = 1
    ):
        """
        Args:
            trajectories: Shape (n_trajectories, n_steps, state_dim)
            observations: Shape (n_trajectories, n_steps, obs_dim) or None
            window: Window size for conditioning (for future extension, currently must be 1)
        """
        self.trajectories = trajectories
        self.observations = observations
        self.window = window
        
        if window != 1:
            raise NotImplementedError("Window size > 1 not yet implemented")
        
        self.n_trajectories, self.n_steps, self.state_dim = trajectories.shape
        
        if observations is not None:
            assert observations.shape[:2] == trajectories.shape[:2], \
                f"Observations shape {observations.shape} doesn't match trajectories {trajectories.shape}"
        
        # Create list of all valid (trajectory_idx, time_idx) pairs
        # We need at least window+1 steps to get a pair
        self.pairs = []
        for traj_idx in range(self.n_trajectories):
            for t in range(self.window, self.n_steps):
                self.pairs.append((traj_idx, t))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        traj_idx, t = self.pairs[idx]
        
        # Get x_{t-1} and x_t
        x_prev = self.trajectories[traj_idx, t - 1]
        x_curr = self.trajectories[traj_idx, t]
        
        result = {
            'x_prev': torch.FloatTensor(x_prev),
            'x_curr': torch.FloatTensor(x_curr),
            'trajectory_idx': traj_idx,
            'time_idx': t,
        }
        
        # Add observations if available
        if self.observations is not None:
            y_curr = self.observations[traj_idx, t]
            result['y_curr'] = torch.FloatTensor(y_curr)
        
        return result


class RFDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Rectified Flow training
    
    Loads pre-generated trajectory data and creates train/val/test dataloaders
    with (x_prev, x_curr) pairs.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        window: int = 1,
        use_preprocessing: bool = False,
        use_observations: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window = window
        self.use_preprocessing = use_preprocessing
        self.use_observations = use_observations
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Load data and create datasets"""
        filename = "data_scaled.h5" if self.use_preprocessing else "data.h5"
        data_file = self.data_dir / filename
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}. "
                "Run data generation first using DataAssimilationDataModule."
            )
        
        with h5py.File(data_file, "r") as f:
            if stage == "fit" or stage is None:
                train_traj = f["train/trajectories"][:]
                val_traj = f["val/trajectories"][:]
                
                train_obs = None
                val_obs = None
                if self.use_observations:
                    train_obs = f["train/observations"][:]
                    val_obs = f["val/observations"][:]
                
                self.train_dataset = RFTransitionDataset(
                    train_traj, 
                    observations=train_obs,
                    window=self.window
                )
                self.val_dataset = RFTransitionDataset(
                    val_traj,
                    observations=val_obs,
                    window=self.window
                )
                
                logging.info(f"Loaded RF training data: {len(self.train_dataset)} pairs")
                logging.info(f"Loaded RF validation data: {len(self.val_dataset)} pairs")
            
            if stage == "test" or stage is None:
                test_traj = f["test/trajectories"][:]
                
                test_obs = None
                if self.use_observations:
                    test_obs = f["test/observations"][:]
                
                self.test_dataset = RFTransitionDataset(
                    test_traj,
                    observations=test_obs,
                    window=self.window
                )
                logging.info(f"Loaded RF test data: {len(self.test_dataset)} pairs")
    
    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Train dataset not set up. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("Val dataset not set up. Call setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not set up. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def test_rf_dataset():
    """Test the RF dataset"""
    print("Testing RF Dataset...")
    
    # Create dummy trajectory data
    n_trajectories = 10
    n_steps = 100
    state_dim = 3
    
    trajectories = np.random.randn(n_trajectories, n_steps, state_dim)
    
    # Create dataset
    dataset = RFTransitionDataset(trajectories, window=1)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Expected size: {n_trajectories * (n_steps - 1)} = {n_trajectories * (n_steps - 1)}")
    
    # Test getting an item
    item = dataset[0]
    print(f"\nSample item keys: {item.keys()}")
    print(f"x_prev shape: {item['x_prev'].shape}")
    print(f"x_curr shape: {item['x_curr'].shape}")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    
    print(f"\nBatch x_prev shape: {batch['x_prev'].shape}")
    print(f"Batch x_curr shape: {batch['x_curr'].shape}")
    
    # Verify the pairs are consecutive
    traj_idx = batch['trajectory_idx'][0].item()
    time_idx = batch['time_idx'][0].item()
    expected_prev = trajectories[traj_idx, time_idx - 1]
    expected_curr = trajectories[traj_idx, time_idx]
    
    assert np.allclose(batch['x_prev'][0].numpy(), expected_prev)
    assert np.allclose(batch['x_curr'][0].numpy(), expected_curr)
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_rf_dataset()