import torch
import numpy as np
from typing import Dict, Any, Optional, Union, Callable, List
import logging
import time

from .base_pf import FilteringMethod

# =============================================================================
# Utility Functions
# =============================================================================

def center_ensemble(E: torch.Tensor, rescale: bool = False):
    """
    Centers the ensemble E along the second dimension (dim=1).
    Args:
        E (torch.Tensor): Ensemble tensor (batch_size x N_particles x d_state).
        rescale (bool): If True, rescale anomalies for unbiased covariance estimate.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Centered anomalies, ensemble mean.
    """
    x = torch.mean(E, dim=1, keepdims=True)
    X_centered = E - x

    if rescale:
        N = E.shape[1]
        if N > 1:
            X_centered *= torch.sqrt(torch.tensor(N / (N - 1), device=E.device, dtype=E.dtype))

    return X_centered, x

def apply_inflation(ensemble: torch.Tensor, inflation_factor: float):
    """
    Applies multiplicative inflation to ensemble anomalies.
    """
    if inflation_factor is None or inflation_factor == 1.0:
        return ensemble

    anomalies, mean_ens = center_ensemble(ensemble, rescale=False)
    inflated_ensemble = mean_ens + inflation_factor * anomalies
    return inflated_ensemble

def dist2coeff(dist: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Gaspari-Cohn function for localization.
    Args:
        dist: Distance tensor.
        radius: Localization radius.
    Returns:
        Coefficients in [0, 1].
    """
    r = dist / radius
    r = torch.abs(r)
    
    # Initialize with zeros
    coeffs = torch.zeros_like(r)
    
    # Case 0 <= r <= 1
    mask1 = (r <= 1.0)
    r1 = r[mask1]
    coeffs[mask1] = 1.0 - (5.0/3.0) * r1**2 + (5.0/8.0) * r1**3 + (1.0/2.0) * r1**4 - (1.0/4.0) * r1**5
    
    # Case 1 < r <= 2
    mask2 = (r > 1.0) & (r <= 2.0)
    r2 = r[mask2]
    coeffs[mask2] = 4.0 - 5.0 * r2 + (5.0/3.0) * r2**2 + (5.0/8.0) * r2**3 - (1.0/2.0) * r2**4 + (1.0/12.0) * r2**5 - (2.0/3.0) * r2**(-1)
    
    # Case r > 2 is already 0.0
    
    return coeffs

def pairwise_distances(x: torch.Tensor, y: torch.Tensor, domain_length: Optional[float] = None) -> torch.Tensor:
    """
    Compute pairwise distances between x and y.
    Supports 1D periodic domain if domain_length is provided.
    Args:
        x: (Nx, D) or (Nx,)
        y: (Ny, D) or (Ny,)
        domain_length: Length of periodic domain (assumed 1D if provided).
    Returns:
        (Nx, Ny) distance matrix.
    """
    if x.ndim == 1: x = x.unsqueeze(1)
    if y.ndim == 1: y = y.unsqueeze(1)
    
    # x: (Nx, 1, D), y: (1, Ny, D)
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    
    if domain_length is not None:
        # 1D periodic distance: min(|d|, L-|d|)
        # Assumes D=1 or distances are component-wise independent (usually D=1 for grid points)
        diff = torch.abs(diff)
        diff = torch.min(diff, domain_length - diff)
        
    dist = torch.norm(diff, dim=-1)
    return dist


# =============================================================================
# Ensemble Kalman Filter (Stochastic / Perturbed Observations)
# =============================================================================

class EnsembleKalmanFilter(FilteringMethod):
    """
    Stochastic Ensemble Kalman Filter (EnKF) with perturbed observations.
    """
    def __init__(
        self,
        system,
        n_particles: int = 50,
        state_dim: int = 3,
        obs_dim: int = 1,
        process_noise_std: float = 0.0, # Usually handled by model dynamics + additive noise if needed
        obs_noise_std: float = 1.0, # Can be overridden by system config
        inflation_factor: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__(system, state_dim, obs_dim, device)
        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        self.inflation_factor = inflation_factor
        
        # Override obs_noise_std from system config if available
        if hasattr(system, "config") and hasattr(system.config, "obs_noise_std"):
            self.obs_noise_std = system.config.obs_noise_std
        else:
            self.obs_noise_std = obs_noise_std

        self.particles = None
        
        logging.info(f"Initialized EnKF with {n_particles} particles, inflation={inflation_factor}")

    @property
    def weights(self) -> torch.Tensor:
        """
        Return uniform weights for compatibility with particle filter interfaces.
        Returns: (Batch, N_particles)
        """
        if self.particles is None:
            return None
        batch_size = self.particles.shape[0]
        return torch.full((batch_size, self.n_particles), 1.0 / self.n_particles, device=self.device)

    def initialize_filter(self, x0: torch.Tensor, init_std: Optional[Union[float, torch.Tensor]] = None) -> None:
        """Initialize ensemble around x0"""
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        
        batch_size = x0.shape[0]
        x0 = x0.to(self.device)
        
        # Initial spread
        if init_std is None:
            init_std = getattr(self.system, "init_std", 1.0)
            
        if isinstance(init_std, torch.Tensor):
            init_std = init_std.to(self.device)
            
        noise = torch.randn(batch_size, self.n_particles, self.state_dim, device=self.device)
        if isinstance(init_std, torch.Tensor) and init_std.ndim > 0:
             noise = noise * init_std.unsqueeze(0).unsqueeze(0) # Broadcast
        else:
             noise = noise * init_std
             
        self.particles = x0.unsqueeze(1) + noise
        
    def predict_step(self, dt: float, y_curr: Optional[torch.Tensor] = None, step_start: int = 0) -> None:
        """Propagate ensemble forward"""
        batch_size = self.particles.shape[0]
        
        # Flatten: (B*N, D)
        particles_flat = self.particles.reshape(-1, self.state_dim)
        
        # Integrate
        # Assuming system.integrate returns (Batch, Steps, D)
        # We take the last step.
        integration_result = self.system.integrate(particles_flat, 2, dt, step_start=step_start)
        x_next_flat = integration_result[:, 1, :]
        
        # Add process noise if specified
        if self.process_noise_std > 0:
            x_next_flat += torch.randn_like(x_next_flat) * self.process_noise_std
            
        self.particles = x_next_flat.reshape(batch_size, self.n_particles, self.state_dim)
        
        # Apply inflation after forecast (common practice)
        # self.particles = apply_inflation(self.particles, self.inflation_factor)

    def update_step(self, observation: torch.Tensor) -> None:
        """Analysis step with perturbed observations"""
        # Apply inflation before analysis
        self.particles = apply_inflation(self.particles, self.inflation_factor)
        
        batch_size = self.particles.shape[0]
        observation = observation.to(self.device) # (B, Obs_dim)
        
        # 1. Compute Y_f (predicted observations)
        particles_flat = self.particles.reshape(-1, self.state_dim)
        y_f_flat = self.system.apply_observation_operator(particles_flat)
        y_f = y_f_flat.reshape(batch_size, self.n_particles, self.obs_dim)
        
        # 2. Perturb observations
        # R = sigma^2 * I
        sigma_y = self.obs_noise_std
        obs_perturbations = sigma_y * torch.randn(batch_size, self.n_particles, self.obs_dim, device=self.device)
        y_perturbed = observation.unsqueeze(1) + obs_perturbations # (B, N, Obs_dim)
        
        # 3. Compute Covariances
        # Center ensembles
        Ef, mean_Ef = center_ensemble(self.particles) # (B, N, State_dim)
        Yf, mean_Yf = center_ensemble(y_f) # (B, N, Obs_dim)
        
        scaling = 1.0 / (self.n_particles - 1)
        
        # Pxy: (B, State_dim, Obs_dim)
        Pxy = torch.bmm(Ef.transpose(1, 2), Yf) * scaling
        
        # Pyy: (B, Obs_dim, Obs_dim)
        Pyy = torch.bmm(Yf.transpose(1, 2), Yf) * scaling
        
        # R: (Obs_dim, Obs_dim)
        R = (sigma_y**2) * torch.eye(self.obs_dim, device=self.device)
        
        # Innovation covariance: S = Pyy + R
        S = Pyy + R.unsqueeze(0) # Broadcast R to batch
        
        # 4. Kalman Gain: K = Pxy @ S^-1
        # Solve S @ K.T = Pxy.T -> K = (S^-1 @ Pxy.T).T
        # torch.linalg.solve(A, B) solves AX = B
        # We want K = Pxy @ inv(S)
        # Transpose trick: K.T = inv(S).T @ Pxy.T = inv(S) @ Pxy.T (since S symmetric)
        
        try:
            # (B, Obs_dim, State_dim)
            K_T = torch.linalg.solve(S, Pxy.transpose(1, 2))
            K = K_T.transpose(1, 2) # (B, State_dim, Obs_dim)
        except RuntimeError:
            # Fallback for singular S
            K = torch.bmm(Pxy, torch.linalg.pinv(S))
            
        # 5. Update
        # Innovations: (B, N, Obs_dim)
        innovations = y_perturbed - y_f
        
        # Update: x_a = x_f + K @ innovation
        # (B, N, State_dim) + (B, State_dim, Obs_dim) @ (B, N, Obs_dim).T -> shapes need alignment
        # We want for each particle i: x_a[i] = x_f[i] + K @ innov[i]
        # Batched mul: (B, N, State_dim) + bmm(innov, K.T) ?
        # innov: (B, N, Obs_dim)
        # K.T: (B, Obs_dim, State_dim)
        # innov @ K.T -> (B, N, State_dim)
        
        update_term = torch.bmm(innovations, K.transpose(1, 2))
        self.particles = self.particles + update_term

    def get_state_estimate(self) -> torch.Tensor:
        return torch.mean(self.particles, dim=1)

    def get_state_covariance(self) -> torch.Tensor:
        _, mean = center_ensemble(self.particles)
        centered = self.particles - mean
        cov = torch.matmul(centered.transpose(1, 2), centered) / (self.n_particles - 1)
        return cov

    def sample_posterior(self, n_samples: int) -> torch.Tensor:
        indices = torch.randint(0, self.n_particles, (n_samples,), device=self.device)
        # If batched, this is tricky. We'll just return first batch or handle logic upstream.
        # Base class assumes single trajectory or handles batching differently.
        # For now, just return random samples from the first batch element if called naively,
        # or implement proper batched sampling if needed.
        return self.particles[:, indices, :] # (B, n_samples, D)

    def compute_log_likelihood(self, observation: torch.Tensor) -> float:
        # Approximate using Gaussian assumption on predicted observations
        # This is expensive for EnKF usually, often just RMSE is tracked.
        # We can implement a basic Gaussian approximation.
        
        particles_flat = self.particles.reshape(-1, self.state_dim)
        y_f_flat = self.system.apply_observation_operator(particles_flat)
        y_f = y_f_flat.reshape(self.particles.shape[0], self.n_particles, self.obs_dim)
        
        y_mean = torch.mean(y_f, dim=1)
        y_centered = y_f - y_mean.unsqueeze(1)
        Pyy = torch.bmm(y_centered.transpose(1, 2), y_centered) / (self.n_particles - 1)
        
        sigma_y = self.obs_noise_std
        R = (sigma_y**2) * torch.eye(self.obs_dim, device=self.device)
        S = Pyy + R.unsqueeze(0)
        
        diff = observation - y_mean
        
        # Log likelihood: -0.5 * (log|S| + diff.T @ S^-1 @ diff + k log(2pi))
        try:
            dist = torch.distributions.MultivariateNormal(loc=y_mean, covariance_matrix=S)
            return dist.log_prob(observation).mean().item()
        except:
            return -1.0e8 # Numerical failure

    def step(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        trajectory_idxs: torch.Tensor,
        time_idxs: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        EnKF step for a BATCH of trajectories.
        Overrides base step to handle batching and return list of metrics.
        """
        start_time = time.perf_counter()
        
        # Check for re-initialization
        # Assumes synchronized batch start
        if self.current_trajectory_idx != trajectory_idxs[0].item():
            self.initialize_filter(x_prev)
            self.current_trajectory_idx = trajectory_idxs[0].item()
            self.current_time_idx = time_idxs[0].item()
            
        # Predict
        step_start = 0
        # We want the filter to be unaware of time-dependent system changes (like manual jumps),
        # so we do NOT pass the actual time index to the integrator.
        # if time_idxs is not None and len(time_idxs) > 0:
        #    step_start = int(time_idxs[0].item()) - 1
        
        self.predict_step(dt, y_curr, step_start=step_start)
        
        # Update
        log_likelihoods = torch.zeros(x_curr.shape[0], device=self.device)
        
        if y_curr is not None:
             # Compute batched log likelihood
             # Re-implementing logic from compute_log_likelihood but returning tensor
             particles_flat = self.particles.reshape(-1, self.state_dim)
             y_f_flat = self.system.apply_observation_operator(particles_flat)
             y_f = y_f_flat.reshape(self.particles.shape[0], self.n_particles, self.obs_dim)
             
             y_mean = torch.mean(y_f, dim=1)
             y_centered = y_f - y_mean.unsqueeze(1)
             Pyy = torch.bmm(y_centered.transpose(1, 2), y_centered) / (self.n_particles - 1)
             
             sigma_y = self.obs_noise_std
             R = (sigma_y**2) * torch.eye(self.obs_dim, device=self.device)
             S = Pyy + R.unsqueeze(0)
             
             try:
                 dist = torch.distributions.MultivariateNormal(loc=y_mean, covariance_matrix=S)
                 log_likelihoods = dist.log_prob(y_curr) # (B,)
             except:
                 log_likelihoods = torch.ones(x_curr.shape[0], device=self.device) * -1.0e8

             self.update_step(y_curr)
             
        # Estimates
        x_est = self.get_state_estimate()
        P_est = self.get_state_covariance()
        
        # Metrics
        elapsed_time = (time.perf_counter() - start_time) / x_curr.shape[0]
        
        results = []
        x_curr = x_curr.to(self.device)
        error = x_est - x_curr
        rmse = torch.sqrt(torch.mean(error**2, dim=1))
        
        # ESS is N for EnKF
        ess = float(self.n_particles)
        
        for i in range(x_curr.shape[0]):
            metrics = {
                "rmse": rmse[i].item(),
                "log_likelihood": log_likelihoods[i].item() if y_curr is not None else 0.0,
                "x_est": x_est[i].cpu().numpy(),
                "P_est": P_est[i].cpu().numpy(),
                "error": error[i].cpu().numpy(),
                "trajectory_idx": trajectory_idxs[i].item(),
                "time_idx": time_idxs[i].item(),
                "resampled": False,
                "ess_pre_resample": ess,
                "ess": ess,
                "step_time": elapsed_time,
            }
            results.append(metrics)
            
        return results


# =============================================================================
# Local Ensemble Transform Kalman Filter (LETKF)
# =============================================================================

class LocalEnsembleTransformKalmanFilter(FilteringMethod):
    """
    Local Ensemble Transform Kalman Filter (LETKF).
    Assumes Lorenz96-style 1D periodic domain for localization.
    """
    def __init__(
        self,
        system,
        n_particles: int = 50,
        state_dim: int = 40,
        obs_dim: int = 40,
        process_noise_std: float = 0.0,
        obs_noise_std: float = 1.0,
        inflation_factor: float = 1.0,
        localization_radius: float = 4.0,
        device: str = "cpu",
    ):
        super().__init__(system, state_dim, obs_dim, device)
        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        self.inflation_factor = inflation_factor
        self.localization_radius = localization_radius
        
        if hasattr(system, "config") and hasattr(system.config, "obs_noise_std"):
            self.obs_noise_std = system.config.obs_noise_std
        else:
            self.obs_noise_std = obs_noise_std

        self.particles = None
        
        # Define coordinates for localization (assuming Lorenz96 1D grid)
        # State coords: 0, 1, ..., state_dim-1
        self.state_coords = torch.arange(state_dim, dtype=torch.float32, device=device)
        
        # Obs coords: need to know which state variables are observed.
        # We assume system.config.obs_components tells us indices.
        if hasattr(system, "config") and hasattr(system.config, "obs_components"):
            self.obs_coords = torch.tensor(system.config.obs_components, dtype=torch.float32, device=device)
        else:
            # Assume full observation if not specified
            self.obs_coords = torch.arange(obs_dim, dtype=torch.float32, device=device)
            
        self.domain_length = float(state_dim) # For periodic distance

        logging.info(f"Initialized LETKF with {n_particles} particles, infl={inflation_factor}, loc_rad={localization_radius}")

    @property
    def weights(self) -> torch.Tensor:
        """
        Return uniform weights for compatibility with particle filter interfaces.
        Returns: (Batch, N_particles)
        """
        if self.particles is None:
            return None
        batch_size = self.particles.shape[0]
        return torch.full((batch_size, self.n_particles), 1.0 / self.n_particles, device=self.device)

    def initialize_filter(self, x0: torch.Tensor, init_std: Optional[Union[float, torch.Tensor]] = None) -> None:
        if x0.dim() == 1: x0 = x0.unsqueeze(0)
        batch_size = x0.shape[0]
        x0 = x0.to(self.device)
        
        if init_std is None:
            init_std = getattr(self.system, "init_std", 1.0)
        # Ensure init_std is on the correct device if it's a tensor
        if isinstance(init_std, torch.Tensor):
            init_std = init_std.to(self.device)
            
        noise = torch.randn(batch_size, self.n_particles, self.state_dim, device=self.device)
        if isinstance(init_std, torch.Tensor) and init_std.ndim > 0:
             noise = noise * init_std.unsqueeze(0).unsqueeze(0) # Broadcast
        else:
             noise = noise * init_std
             
        self.particles = x0.unsqueeze(1) + noise

    def predict_step(self, dt: float, y_curr: Optional[torch.Tensor] = None, step_start: int = 0) -> None:
        batch_size = self.particles.shape[0]
        particles_flat = self.particles.reshape(-1, self.state_dim)
        integration_result = self.system.integrate(particles_flat, 2, dt, step_start=step_start)
        x_next_flat = integration_result[:, 1, :]
        
        if self.process_noise_std > 0:
            x_next_flat += torch.randn_like(x_next_flat) * self.process_noise_std
            
        self.particles = x_next_flat.reshape(batch_size, self.n_particles, self.state_dim)
        # self.particles = apply_inflation(self.particles, self.inflation_factor)

    def update_step(self, observation: torch.Tensor) -> None:
        """LETKF Analysis Step"""
        # Apply inflation before analysis
        self.particles = apply_inflation(self.particles, self.inflation_factor)
        
        batch_size = self.particles.shape[0]
        observation = observation.to(self.device)
        
        # 1. Compute Y_f
        particles_flat = self.particles.reshape(-1, self.state_dim)
        y_f_flat = self.system.apply_observation_operator(particles_flat)
        y_f = y_f_flat.reshape(batch_size, self.n_particles, self.obs_dim)
        
        # 2. Center ensembles
        Ef, mean_Ef = center_ensemble(self.particles) # (B, N, D)
        Yf, mean_Yf = center_ensemble(y_f) # (B, N, Obs_dim)
        
        # 3. Local Analysis Loop
        # We update each state variable independently (in parallel across batch)
        
        # Precompute distances between all state vars and all obs vars
        # dist_matrix: (State_dim, Obs_dim)
        dist_matrix = pairwise_distances(self.state_coords, self.obs_coords, domain_length=self.domain_length)
        
        # Precompute localization weights
        # weights: (State_dim, Obs_dim)
        loc_weights = dist2coeff(dist_matrix, self.localization_radius)
        
        # Prepare analysis ensemble container
        E_a = torch.zeros_like(self.particles)
        
        # R inverse (diagonal assumption)
        sigma_y = self.obs_noise_std
        R_inv_diag = 1.0 / (sigma_y**2)
        
        # Identity matrix for (N-1)I
        eye_N = torch.eye(self.n_particles, device=self.device)
        
        # Loop over state variables
        for i in range(self.state_dim):
            # Find local observations
            # weights[i]: (Obs_dim,)
            local_obs_mask = loc_weights[i] > 1e-6
            local_obs_indices = torch.where(local_obs_mask)[0]
            
            if len(local_obs_indices) == 0:
                # No observations affect this state variable
                E_a[:, :, i] = self.particles[:, :, i]
                continue
                
            # Extract local quantities
            # Yf_local: (B, N, N_local_obs)
            Yf_local = Yf[:, :, local_obs_indices]
            
            # y_local: (B, N_local_obs)
            y_local = observation[:, local_obs_indices]
            
            # mean_Yf_local: (B, 1, N_local_obs)
            mean_Yf_local = mean_Yf[:, :, local_obs_indices]
            
            # Localization weights for these obs: (N_local_obs,)
            rho = loc_weights[i, local_obs_indices]
            sqrt_rho = torch.sqrt(rho).unsqueeze(0).unsqueeze(0) # (1, 1, N_local_obs)
            
            # Weight the transformed observations
            # We effectively scale R_inv by rho. 
            # Equivalently, scale Yf and innovation by sqrt(rho) and use standard R_inv
            
            Yf_local_weighted = Yf_local * sqrt_rho
            
            # Innovation: y - mean_Yf
            innovation = y_local.unsqueeze(1) - mean_Yf_local # (B, 1, N_local_obs)
            innovation_weighted = innovation * sqrt_rho
            
            # Compute P_tilde (inverse of covariance in ensemble space)
            # P_tilde = (N-1)I + Y_local^T @ R_local^-1 @ Y_local
            # Here R_local^-1 is just 1/sigma^2 * I (since we absorbed rho into Y)
            
            # Yf_local_weighted: (B, N, N_loc)
            # C = Yf_local_weighted @ Yf_local_weighted.T * R_inv_diag
            # (B, N, N)
            C = torch.bmm(Yf_local_weighted, Yf_local_weighted.transpose(1, 2)) * R_inv_diag
            
            P_tilde = (self.n_particles - 1) * eye_N.unsqueeze(0) + C
            
            # Compute Wa (transform matrix) and wa (weight vector)
            # We need P_tilde^-1
            # Eigen decomposition for stability: P_tilde = V D V^T
            
            evals, evecs = torch.linalg.eigh(P_tilde)
            
            # Clamp evals
            evals = torch.clamp(evals, min=1e-9)
            
            # P_tilde^-1 = V D^-1 V^T
            # Pa = (N-1) * P_tilde^-1
            # Wa = [(N-1) * P_tilde^-1]^(1/2) = sqrt(N-1) * V D^-0.5 V^T
            
            D_inv = 1.0 / evals
            D_inv_sqrt = torch.sqrt(D_inv)
            
            # Wa: (B, N, N)
            # V @ diag(d) @ V.T
            Wa = torch.bmm(evecs * D_inv_sqrt.unsqueeze(1), evecs.transpose(1, 2)) * np.sqrt(self.n_particles - 1)
            
            # wa = P_tilde^-1 @ Y_local^T @ R^-1 @ innovation
            # Y_local^T @ R^-1 @ innovation -> (B, N, N_loc) @ (B, N_loc, 1) -> (B, N, 1)
            # Note: innovation is (B, 1, N_loc), transpose to (B, N_loc, 1)
            
            # term1 = Yf_local_weighted.transpose(1, 2) @ innovation_weighted.transpose(1, 2) * R_inv_diag
            # Actually: Yf^T @ R^-1 @ d
            # Yf_local_weighted has shape (B, N, N_loc)
            # innovation_weighted has shape (B, 1, N_loc)
            
            # (B, N, N_loc) @ (B, N_loc, 1) -> (B, N, 1)
            term1 = torch.bmm(Yf_local_weighted, innovation_weighted.transpose(1, 2)) * R_inv_diag
            
            # wa = P_tilde^-1 @ term1
            # P_tilde^-1 = V D^-1 V^T
            P_tilde_inv = torch.bmm(evecs * D_inv.unsqueeze(1), evecs.transpose(1, 2))
            wa = torch.bmm(P_tilde_inv, term1) # (B, N, 1)
            
            # Update weights: W = Wa + wa
            # But wait, LETKF update is: x_a = mean_f + Ef @ (wa + Wa)
            # Ef_local for this state variable i is just Ef[:, :, i] (B, N)
            
            Ef_i = Ef[:, :, i].unsqueeze(2) # (B, N, 1)
            mean_Ef_i = mean_Ef[:, :, i].unsqueeze(2) # (B, 1, 1)
            
            # Total weight matrix W_tot = Wa + wa (broadcasting wa)
            # x_a_i = mean_Ef_i + Ef_i @ (wa + Wa) ? No.
            # The formula is X_a = mean_f + X_f @ (w 1^T + W)
            # Here Ef is X_f (anomalies).
            # So we compute T = wa + Wa.
            # Then E_a_i = mean_Ef_i + Ef_i @ T ??
            # Ef_i is (B, N, 1). We need (B, N).
            # Actually Ef is (B, N). We want E_a (B, N).
            # The transform is in ensemble space.
            # X_a = X_f @ T? No.
            # Standard LETKF:
            # w_a (mean weight) = P_tilde^-1 Y^T R^-1 d
            # W_a (perturbation weight) = [(N-1) P_tilde^-1]^1/2
            # x_mean_a = x_mean_f + X_f @ w_a
            # X_a (anomalies) = X_f @ W_a
            # E_a = x_mean_a + X_a
            
            # Calculate mean update
            # Ef_i: (B, N, 1). wa: (B, N, 1)
            # mean_update = Ef_i^T @ wa ?? No.
            # mean_update = sum(Ef_i * wa) ?
            # X_f is (M, N). w_a is (N, 1). X_f @ w_a -> (M, 1).
            # Here Ef_i is (B, N). We treat it as (B, 1, N) for matmul?
            # Ef_i (B, N). wa (B, N).
            # We want dot product over N.
            
            mean_update = torch.sum(Ef_i * wa, dim=1, keepdim=True) # (B, 1, 1)
            
            # Calculate anomaly update
            # X_a = X_f @ W_a
            # Ef_i: (B, 1, N) (transposed for matmul)
            # Wa: (B, N, N)
            # Ef_i @ Wa -> (B, 1, N)
            
            Ef_i_T = Ef_i.transpose(1, 2) # (B, 1, N)
            anom_update_T = torch.bmm(Ef_i_T, Wa) # (B, 1, N)
            anom_update = anom_update_T.transpose(1, 2) # (B, N, 1)
            
            # Combine
            E_a_i = mean_Ef_i + mean_update + anom_update # (B, N, 1)
            
            E_a[:, :, i] = E_a_i.squeeze(2)
            
        self.particles = E_a

    def get_state_estimate(self) -> torch.Tensor:
        return torch.mean(self.particles, dim=1)

    def get_state_covariance(self) -> torch.Tensor:
        _, mean = center_ensemble(self.particles)
        centered = self.particles - mean
        cov = torch.matmul(centered.transpose(1, 2), centered) / (self.n_particles - 1)
        return cov

    def sample_posterior(self, n_samples: int) -> torch.Tensor:
        indices = torch.randint(0, self.n_particles, (n_samples,), device=self.device)
        return self.particles[:, indices, :]

    def compute_log_likelihood(self, observation: torch.Tensor) -> float:
        return -1.0 # Not implemented efficiently for LETKF

    def step(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        trajectory_idxs: torch.Tensor,
        time_idxs: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        LETKF step for a BATCH of trajectories.
        Overrides base step to handle batching and return list of metrics.
        """
        start_time = time.perf_counter()
        
        # Check for re-initialization
        if self.current_trajectory_idx != trajectory_idxs[0].item():
            self.initialize_filter(x_prev)
            self.current_trajectory_idx = trajectory_idxs[0].item()
            self.current_time_idx = time_idxs[0].item()
            
        # Predict
        step_start = 0
        # We want the filter to be unaware of time-dependent system changes (like manual jumps),
        # so we do NOT pass the actual time index to the integrator.
        # if time_idxs is not None and len(time_idxs) > 0:
        #    step_start = int(time_idxs[0].item()) - 1
        
        self.predict_step(dt, y_curr, step_start=step_start)
        
        # Update
        log_likelihoods = torch.ones(x_curr.shape[0], device=self.device) * -1.0
        
        if y_curr is not None:
             self.update_step(y_curr)
             
        # Estimates
        x_est = self.get_state_estimate()
        P_est = self.get_state_covariance()
        
        # Metrics
        elapsed_time = (time.perf_counter() - start_time) / x_curr.shape[0]
        
        results = []
        x_curr = x_curr.to(self.device)
        error = x_est - x_curr
        rmse = torch.sqrt(torch.mean(error**2, dim=1))
        
        # ESS is N for LETKF
        ess = float(self.n_particles)
        
        for i in range(x_curr.shape[0]):
            metrics = {
                "rmse": rmse[i].item(),
                "log_likelihood": log_likelihoods[i].item(),
                "x_est": x_est[i].cpu().numpy(),
                "P_est": P_est[i].cpu().numpy(),
                "error": error[i].cpu().numpy(),
                "trajectory_idx": trajectory_idxs[i].item(),
                "time_idx": time_idxs[i].item(),
                "resampled": False,
                "ess_pre_resample": ess,
                "ess": ess,
                "step_time": elapsed_time,
            }
            results.append(metrics)
            
        return results
