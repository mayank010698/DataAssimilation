import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import time

from .base_pf import FilteringMethod
from .proposals import ProposalDistribution, TransitionProposal


class BootstrapParticleFilterUnbatched(FilteringMethod):
    """
    Bootstrap Particle Filter with CORRECT arbitrary proposal distribution support.
    (Sequential/Unbatched version)

    This version properly implements the importance sampling weights:
    w_t^(i) = w_{t-1}^(i) * [p(y_t | x_t^(i)) * p(x_t^(i) | x_{t-1}^(i))] / q(x_t^(i) | x_{t-1}^(i), y_t)
    """

    def __init__(
        self,
        system,
        proposal_distribution: Optional[ProposalDistribution] = None,
        n_particles: int = 1000,
        state_dim: int = 3,
        obs_dim: int = 1,
        process_noise_std: float = 0.25,
        device: str = "cpu",
        use_optimal_weight_update: bool = False,
        resampling_threshold_ratio: float = 0.5,
    ):
        super().__init__(system, state_dim, obs_dim, device)
        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        self.use_optimal_weight_update = use_optimal_weight_update
        self.resampling_threshold_ratio = resampling_threshold_ratio

        if proposal_distribution is None:
            self.proposal = TransitionProposal(system, process_noise_std)
        else:
            self.proposal = proposal_distribution

        self.transition_prior = TransitionProposal(system, process_noise_std)

        self.particles = None
        self.particles_prev = None  # Store previous particles for weight computation
        self.weights = None
        self.log_weights = None

        # Debug stats
        self.last_proposal_log_probs = None
        self.last_obs_log_likelihoods = None

        self.step_count = 0
        self.resampling_history = []

        logging.info("Initialized BootstrapParticleFilterUnbatched")
        logging.info(
            f"n_particles={n_particles}, proposal={type(self.proposal).__name__}, "
            f"process_noise_std={process_noise_std}, obs_noise_std={system.config.obs_noise_std}, "
            f"obs_dim={obs_dim}"
        )

    def initialize_filter(self, x0: torch.Tensor, init_std: Optional[Union[float, torch.Tensor]] = None) -> None:
        """Initialize particles around initial state"""
        # x0 should be (1, State_dim) for unbatched or (Batch, State_dim) for batched
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        
        # Ensure x0 is on correct device
        x0 = x0.to(self.device)

        if init_std is None:
            init_std = self.system.config.obs_noise_std
        
        # Handle if init_std is a tensor (e.g. climatological std)
        if isinstance(init_std, torch.Tensor):
            init_std = init_std.to(self.device)
            # Expand to match batch/particles if necessary, or just rely on broadcasting
            # noise shape: (Batch, N_particles, State_dim)
            # init_std shape: (State_dim,) or scalar
            if init_std.dim() > 0:
                # Reshape to (1, 1, State_dim) for broadcasting
                scale = init_std.reshape(1, 1, -1)
            else:
                scale = init_std
        else:
            scale = init_std

        # Noise: (N_particles, State_dim)
        noise = torch.randn(
            self.n_particles, self.state_dim, device=self.device
        )
        
        # If batched, we need (Batch, N, D)
        batch_size = x0.shape[0]
        noise = noise.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply scaling
        noise = noise * scale
        
        # x0: (1, State_dim), noise: (N, State_dim) -> particles: (N, State_dim)
        self.particles = x0.unsqueeze(1) + noise
        self.particles_prev = self.particles.clone()  # Initialize previous particles

        self.weights = (
            torch.ones(self.n_particles, device=self.device) / self.n_particles
        )
        self.log_weights = torch.log(self.weights)

        self.step_count = 0
        logging.info(
            f"Filter initialized with {self.n_particles} particles; initial spread std="
            f"{torch.std(self.particles, dim=0).cpu().numpy()}"
        )

    def predict_step(self, dt: float, y_curr: Optional[torch.Tensor] = None) -> None:
        """
        CORRECTED: Propagate particles using the PROPOSAL DISTRIBUTION
        If y_curr is None, falls back to transition prior (diffusion).
        """
        # Store previous particles for weight computation
        self.particles_prev = self.particles.clone()

        # Vectorized proposal sampling
        new_particles = []
        
        # Select proposal: if observation missing, use transition prior
        proposal = self.proposal
        if y_curr is None:
            proposal = self.transition_prior

        for i in range(self.n_particles):
            # Use the proposal distribution to sample new particles
            x_new = proposal.sample(self.particles[i], y_curr, dt)
            new_particles.append(x_new)

        self.particles = torch.stack(new_particles)

        if self.step_count % 1 == 0:
            mean_pos = torch.mean(self.particles, dim=0)
            std_pos = torch.std(self.particles, dim=0)
            logging.debug(
                f"Step {self.step_count}: Particle mean = {mean_pos}, std = {std_pos}"
            )

    def compute_transition_log_prob(
        self, x_curr: torch.Tensor, x_prev: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Compute log p(x_t | x_{t-1}) for the transition dynamics
        """
        # Fully vectorized implementation using PyTorch system in physical space
        integration_result = self.system.integrate(x_prev, 2, dt)
        x_expected = integration_result[:, 1, :]
        noise_std = torch.tensor(self.process_noise_std, device=self.device)

        # Compute Gaussian log-likelihood of the transition
        # x_curr: (N, D), x_expected: (N, D)
        diff = x_curr - x_expected
        noise_var = noise_std**2
        
        # Sum over state dimensions (D)
        # If noise_std is scalar or vector (D,), broadcasting works.
        log_prob = -0.5 * torch.sum(diff**2 / noise_var, dim=1)
        
        if noise_std.ndim == 0:
            log_det = self.state_dim * torch.log(noise_std)
        else:
            log_det = torch.sum(torch.log(noise_std))
            
        log_prob -= (0.5 * self.state_dim * np.log(2 * np.pi) + log_det)

        return log_prob

    def compute_predictive_log_likelihood(
        self, x_prev: torch.Tensor, observation: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Compute approx log p(y_t | x_{t-1}) using Inflated Noise approximation.
        1. x_hat = dynamics(x_{t-1}) (deterministic)
        2. y_hat = obs_op(x_hat)
        3. cov = obs_noise^2 * I + process_noise^2 * I (approx)
        """
        # 1. Deterministic propagation
        # integration result shape: (N, steps, D) -> get last step
        integration_result = self.system.integrate(x_prev, 2, dt)
        x_hat = integration_result[:, 1, :]

        # 2. Observation projection
        y_hat = self.system.apply_observation_operator(x_hat)
        
        # 3. Variance inflation
        obs_noise_var = self.system.config.obs_noise_std**2
        process_noise_var = self.process_noise_std**2
        total_var = obs_noise_var + process_noise_var

        # 4. Compute Gaussian Log-Likelihood
        # observation: (obs_dim,) -> expand to (1, obs_dim) for broadcasting
        # y_hat: (N, obs_dim)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            
        diff = observation - y_hat
        
        if self.obs_dim == 1:
            log_prob = -0.5 * (diff.squeeze()**2) / total_var - 0.5 * np.log(
                2 * np.pi * total_var
            )
        else:
            # Multivariate independent
            # log p(y) = sum log p(y_i)
            log_prob = -0.5 * torch.sum(diff**2, dim=1) / total_var
            log_prob -= 0.5 * self.obs_dim * np.log(2 * np.pi * total_var)
            
        return log_prob

    def update_step(self, observation: torch.Tensor) -> None:
        """
        CORRECTED: Update particle weights using IMPORTANCE SAMPLING
        w_t^(i) = w_{t-1}^(i) * [p(y_t | x_t^(i)) * p(x_t^(i) | x_{t-1}^(i))] / q(x_t^(i) | x_{t-1}^(i), y_t)
        
        OR (if use_optimal_weight_update=True):
        w_t^(i) = w_{t-1}^(i) * p(y_t | x_{t-1}^(i))
        """
        if observation.dim() == 0:
            observation = observation.unsqueeze(0)
            
        # Ensure observation is on device
        observation = observation.to(self.device)

        # Apply observation operator (vectorized)
        # self.particles is (N, D), expected_observations is (N, obs_dim)
        expected_observations = self.system.apply_observation_operator(self.particles)

        logging.debug(f"observation shape: {observation.shape}")
        logging.debug(f"expected_observations shape: {expected_observations.shape}")
        logging.debug(f"obs_dim: {self.obs_dim}")

        # Compute observation log-likelihoods: log p(y_t | x_t^(i))
        obs_noise_var = self.system.config.obs_noise_std**2
        
        # Vectorized calculation
        # observation: (obs_dim,) or (1, obs_dim)
        # expected_observations: (N, obs_dim)
        diff = observation - expected_observations # Broadcasting (1, D) vs (N, D) -> (N, D)

        if self.obs_dim == 1:
            obs_log_likelihoods = -0.5 * (diff.squeeze()**2) / obs_noise_var - 0.5 * np.log(
                2 * np.pi * obs_noise_var
            )
        else:
            # Multivariate Gaussian
            obs_cov = obs_noise_var * torch.eye(self.obs_dim, device=self.device)
            inv_cov = torch.inverse(obs_cov)
            log_det_2pi_cov = torch.logdet(obs_cov) + self.obs_dim * np.log(2 * np.pi)
            
            # Quadratic form: (x-mu)^T Sigma^-1 (x-mu)
            # We need diag(diff @ inv_cov @ diff^T)
            # Or better: sum((diff @ inv_cov) * diff, dim=1)
            
            # diff: (N, D)
            # inv_cov: (D, D)
            # diff @ inv_cov: (N, D)
            # term: (N, D) * (N, D) -> sum dim 1 -> (N,)
            quadratic_form = torch.sum((diff @ inv_cov) * diff, dim=1)
            obs_log_likelihoods = -0.5 * (quadratic_form + log_det_2pi_cov)
        
        # Save obs log likelihoods for metrics regardless of update type
        self.last_obs_log_likelihoods = obs_log_likelihoods

        # OPTIMAL WEIGHT UPDATE BRANCH
        if self.use_optimal_weight_update:
            # log w_t = log w_{t-1} + log p(y_t | x_{t-1})
            predictive_log_probs = self.compute_predictive_log_likelihood(
                self.particles_prev, observation, self.system.config.dt
            )
            importance_weights = predictive_log_probs
            
            # Compute other terms just for debugging/logging purposes (optional, but good for comparison)
            # Compute transition log-probabilities: log p(x_t^(i) | x_{t-1}^(i))
            transition_log_probs = self.compute_transition_log_prob(
                self.particles, self.particles_prev, self.system.config.dt
            )
            # Compute proposal log-probabilities: log q(x_t^(i) | x_{t-1}^(i), y_t)
            # (Can be expensive, so maybe skip if performance is critical, but good for now)
            proposal_log_probs = torch.zeros(self.n_particles, device=self.device)
            for i in range(self.n_particles):
                prop_log_prob = self.proposal.log_prob(
                    self.particles[i],
                    self.particles_prev[i],
                    observation,
                    self.system.config.dt,
                )
                proposal_log_probs[i] = prop_log_prob
            
            self.last_proposal_log_probs = proposal_log_probs
            
        else:
            # STANDARD WEIGHT UPDATE BRANCH
            # log w_t^(i) = log w_{t-1}^(i) + log p(y_t | x_t^(i)) + log p(x_t^(i) | x_{t-1}^(i)) - log q(x_t^(i) | x_{t-1}^(i), y_t)
            
            # Compute transition log-probabilities: log p(x_t^(i) | x_{t-1}^(i))
            transition_log_probs = self.compute_transition_log_prob(
                self.particles, self.particles_prev, self.system.config.dt
            )

            # Compute proposal log-probabilities: log q(x_t^(i) | x_{t-1}^(i), y_t)
            proposal_log_probs = torch.zeros(self.n_particles, device=self.device)
            for i in range(self.n_particles):
                prop_log_prob = self.proposal.log_prob(
                    self.particles[i],
                    self.particles_prev[i],
                    observation,
                    self.system.config.dt,
                )
                proposal_log_probs[i] = prop_log_prob

            importance_weights = (
                obs_log_likelihoods + transition_log_probs - proposal_log_probs
            )
            
            self.last_proposal_log_probs = proposal_log_probs

        # --- DEBUGGING ---
        if self.step_count % 10 == 0:  # Reduce spam
            print(f"Step {self.step_count} [{'Opt' if self.use_optimal_weight_update else 'Std'}]:")
            if self.last_proposal_log_probs is not None:
                print(f"  proposal_log_probs (first 5): {self.last_proposal_log_probs[:5].detach().cpu().numpy()}")
            print(f"  obs_log_probs (first 5): {obs_log_likelihoods[:5].detach().cpu().numpy()}")
            # print(f"  transition_log_probs (first 5): {transition_log_probs[:5].detach().cpu().numpy()}")
            print(f"  log_weights before norm (first 5): {self.log_weights[:5].detach().cpu().numpy()}")
            print(f"  log_weights stats: min={self.log_weights.min().item():.4f}, max={self.log_weights.max().item():.4f}, std={self.log_weights.std().item():.4f}")
        # -----------------
        
        self.log_weights += importance_weights

        max_ll = torch.max(obs_log_likelihoods).item()
        mean_ll = torch.mean(obs_log_likelihoods).item()
        max_iw = torch.max(importance_weights).item()
        mean_iw = torch.mean(importance_weights).item()

        logging.debug(
            f"Observation update: max_obs_ll = {max_ll:.3f}, mean_obs_ll = {mean_ll:.3f}"
        )
        logging.debug(
            f"Importance weights: max_iw = {max_iw:.3f}, mean_iw = {mean_iw:.3f}"
        )

    def resample(self) -> Tuple[bool, float]:
        """Systematic resampling of particles"""
        max_log_weight = torch.max(self.log_weights)
        log_weights_normalized = self.log_weights - max_log_weight
        weights_unnormalized = torch.exp(log_weights_normalized)
        weight_sum = torch.sum(weights_unnormalized)

        if weight_sum == 0:
            logging.warning("All weights are zero; resetting to uniform distribution.")
            self.weights = (
                torch.ones(self.n_particles, device=self.device) / self.n_particles
            )
            self.log_weights = torch.log(self.weights)
            return False, self.n_particles

        self.weights = weights_unnormalized / weight_sum
        ess = (1.0 / torch.sum(self.weights**2)).item()
        resample_threshold = self.n_particles * self.resampling_threshold_ratio

        if ess < resample_threshold:
            cumsum = torch.cumsum(self.weights, dim=0)
            u = torch.rand(1, device=self.device) / self.n_particles
            indices = torch.searchsorted(
                cumsum,
                u
                + torch.arange(
                    self.n_particles, device=self.device, dtype=torch.float32
                )
                / self.n_particles,
            )

            indices = torch.clamp(indices, 0, self.n_particles - 1)
            self.particles = self.particles[indices]
            self.particles_prev = self.particles_prev[
                indices
            ]  # Also resample previous particles

            self.weights = (
                torch.ones(self.n_particles, device=self.device) / self.n_particles
            )
            self.log_weights = torch.log(self.weights)

            self.resampling_history.append(self.step_count)
            logging.debug(f"Resampled at step {self.step_count}, ESS={ess:.1f}")
            return True, ess

        logging.debug(f"No resampling needed, ESS = {ess:.1f}")
        return False, ess

    def get_state_estimate(self) -> torch.Tensor:
        """Get current state estimate (weighted mean)"""
        return torch.sum(self.weights.unsqueeze(1) * self.particles, dim=0)

    def get_state_covariance(self) -> torch.Tensor:
        """Get current state covariance estimate"""
        mean = self.get_state_estimate()
        centered = self.particles - mean.unsqueeze(0)
        cov = torch.sum(
            self.weights.unsqueeze(1).unsqueeze(2)
            * centered.unsqueeze(2)
            * centered.unsqueeze(1),
            dim=0,
        )
        return cov

    def sample_posterior(self, n_samples: int) -> torch.Tensor:
        """Sample from current particle approximation"""
        indices = torch.multinomial(self.weights, n_samples, replacement=True)
        return self.particles[indices]

    def compute_log_likelihood(self, observation: torch.Tensor) -> float:
        """
        CORRECTED: Compute log-likelihood of observation given current state
        This should be computed BEFORE the weight update
        """
        if observation.dim() == 0:
            observation = observation.unsqueeze(0)
        observation = observation.to(self.device)

        expected_observations = self.system.apply_observation_operator(self.particles)
        obs_noise_var = self.system.config.obs_noise_std**2

        # Vectorized log likelihood computation
        diff = observation - expected_observations

        if self.obs_dim == 1:
             log_likelihoods = -0.5 * (diff.squeeze()**2) / obs_noise_var - 0.5 * np.log(
                2 * np.pi * obs_noise_var
            )
        else:
            obs_cov = obs_noise_var * torch.eye(self.obs_dim, device=self.device)
            inv_cov = torch.inverse(obs_cov)
            log_det_2pi_cov = torch.logdet(obs_cov) + self.obs_dim * np.log(2 * np.pi)
            quadratic_form = torch.sum((diff @ inv_cov) * diff, dim=1)
            log_likelihoods = -0.5 * (quadratic_form + log_det_2pi_cov)

        # Weighted average of log-likelihoods using current weights
        log_marginal_likelihood = torch.logsumexp(
            self.log_weights + log_likelihoods, dim=0
        ) - torch.logsumexp(self.log_weights, dim=0)
        return log_marginal_likelihood.item()

    def step(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        y_curr: Optional[torch.Tensor],
        dt: float,
        trajectory_idx: int,
        time_idx: int,
    ) -> Dict[str, Any]:
        """Particle filter step"""

        start_time = time.perf_counter()
        
        # Initialize filter if this is the start of a new trajectory
        if self.current_trajectory_idx != trajectory_idx:
            self.initialize_filter(x_prev)
            self.current_trajectory_idx = trajectory_idx
            self.current_time_idx = time_idx
            
        self.step_count += 1
        
        # Propagate
        self.predict_step(dt, y_curr)

        # Update and Resample ONLY if observation is present
        if y_curr is not None:
            # Likelihood
            log_likelihood = self.compute_log_likelihood(y_curr)
            
            # Update
            self.update_step(y_curr)
            
            # Resample
            resampled, ess_pre = self.resample()
            
            # Record metrics dependent on update
            if self.last_proposal_log_probs is not None:
                 metrics_extra = {"proposal_log_prob_mean": self.last_proposal_log_probs.mean().item()}
            else:
                 metrics_extra = {}
                 
            if self.last_obs_log_likelihoods is not None:
                 metrics_extra["obs_log_prob_mean"] = self.last_obs_log_likelihoods.mean().item()
                 metrics_extra["obs_log_prob_std"] = self.last_obs_log_likelihoods.std().item()
                 
            metrics_extra["log_likelihood"] = log_likelihood 
        else:
            # Missing observation
            resampled = False
            # ESS doesn't change if weights don't change
            ess_pre = (1.0 / torch.sum(self.weights**2)).item()
            metrics_extra = {"log_likelihood": 0.0}

        # Calculate RMSE
        x_est = self.get_state_estimate()
        x_est_np = x_est.detach().cpu().numpy()
        error = x_est - x_curr.to(self.device)
        rmse = torch.sqrt(torch.mean(error**2)).item()
        
        elapsed_time = time.perf_counter() - start_time
        
        metrics = {
            "rmse": rmse,
            "x_est": x_est_np,
            "error": error.cpu().numpy(),
            "trajectory_idx": trajectory_idx,
            "time_idx": time_idx,
            "resampled": resampled,
            "ess_pre_resample": ess_pre,
            "ess": (1.0 / torch.sum(self.weights**2)).item(),
            "n_effective_particles": (1.0 / torch.sum(self.weights**2)).item(),
            "max_weight": torch.max(self.weights).item(),
            "min_weight": torch.min(self.weights).item(),
            "particle_spread": torch.std(self.particles, dim=0).cpu().numpy(),
            "proposal_type": type(self.proposal).__name__,
            "step_time": elapsed_time,
            **metrics_extra
        }

        if self.step_count % 1 == 0:
            logging.debug(
                f"Step {self.step_count}: RMSE={metrics['rmse']:.4f}, ESS={metrics['ess']:.1f}, Time={elapsed_time:.4f}s"
            )
            logging.debug(f"  Proposal: {metrics['proposal_type']}")
            logging.debug(f"  True state: {x_curr.cpu().numpy()}")
            logging.debug(f"  Estimated:  {metrics['x_est']}")
            if y_curr is not None:
                logging.debug(f"  Observation: {y_curr.cpu().numpy()}")
                # Use torch version of apply_observation_operator logic manually or via system
                expected_obs = self.system.apply_observation_operator(x_curr)
                logging.debug(f"  Expected obs: {expected_obs.cpu().numpy()}")

        return metrics


class BootstrapParticleFilter(FilteringMethod):
    """
    Batched Bootstrap Particle Filter with arbitrary proposal distribution support.

    Implements particle filter updates for a batch of independent trajectories simultaneously.
    
    Shapes:
    - particles: (Batch, N_particles, State_dim)
    - weights: (Batch, N_particles)
    """

    def __init__(
        self,
        system,
        proposal_distribution: Optional[ProposalDistribution] = None,
        n_particles: int = 1000,
        state_dim: int = 3,
        obs_dim: int = 1,
        process_noise_std: float = 0.25,
        device: str = "cpu",
        use_optimal_weight_update: bool = False,
        resampling_threshold_ratio: float = 0.5,
    ):
        super().__init__(system, state_dim, obs_dim, device)
        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        self.use_optimal_weight_update = use_optimal_weight_update
        self.resampling_threshold_ratio = resampling_threshold_ratio

        if proposal_distribution is None:
            self.proposal = TransitionProposal(system, process_noise_std)
        else:
            self.proposal = proposal_distribution

        self.transition_prior = TransitionProposal(system, process_noise_std)

        self.particles = None
        self.particles_prev = None
        self.weights = None
        self.log_weights = None

        # Debug stats
        self.last_proposal_log_probs = None
        self.last_obs_log_likelihoods = None

        # Per-batch metrics
        self.step_count = 0
        self.resampling_history = []

        logging.info("Initialized BootstrapParticleFilter (Batched)")

    def initialize_filter(self, x0: torch.Tensor, init_std: Optional[Union[float, torch.Tensor]] = None) -> None:
        """
        Initialize particles around initial state.
        Args:
            x0: Initial states for batch, shape (Batch, State_dim)
            init_std: Optional standard deviation for initialization. 
                      If None, uses obs_noise_std.
                      Can be scalar or tensor (State_dim,).
        """
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
            
        batch_size = x0.shape[0]
        
        # Ensure x0 is on correct device
        x0 = x0.to(self.device)

        if init_std is None:
            init_std = self.system.config.obs_noise_std

        # Handle if init_std is a tensor (e.g. climatological std)
        if isinstance(init_std, torch.Tensor):
            init_std = init_std.to(self.device)
            if init_std.dim() > 0:
                # Reshape to (1, 1, State_dim) for broadcasting: (Batch, N, D)
                scale = init_std.reshape(1, 1, -1)
            else:
                scale = init_std
        else:
            scale = init_std

        # Shape: (Batch, N_particles, State_dim)
        noise = torch.randn(
            batch_size, self.n_particles, self.state_dim, device=self.device
        )
        
        noise = noise * scale
        
        # Expand x0: (Batch, 1, State_dim) -> (Batch, N, State_dim)
        self.particles = x0.unsqueeze(1) + noise
        self.particles_prev = self.particles.clone()

        self.weights = (
            torch.ones(batch_size, self.n_particles, device=self.device) / self.n_particles
        )
        self.log_weights = torch.log(self.weights)

        self.step_count = 0
        logging.info(
            f"Filter initialized with batch size {batch_size}, {self.n_particles} particles"
        )

    def predict_step(self, dt: float, y_curr: Optional[torch.Tensor] = None) -> None:
        """
        Propagate particles using the PROPOSAL DISTRIBUTION (Batched)
        """
        self.particles_prev = self.particles.clone()
        
        batch_size = self.particles.shape[0]
        
        # Flatten batch and particle dimensions for proposal sampling
        # (Batch, N, D) -> (Batch*N, D)
        particles_flat = self.particles.reshape(-1, self.state_dim)
        
        # Handle y_curr if it exists: (Batch, Obs_dim) -> (Batch*N, Obs_dim)
        y_curr_flat = None
        if y_curr is not None:
             y_curr_expanded = y_curr.unsqueeze(1).expand(batch_size, self.n_particles, -1)
             # Explicitly specify first dimension to avoid ambiguity with 0-dim observations
             y_curr_flat = y_curr_expanded.reshape(batch_size * self.n_particles, y_curr.shape[-1])

        # Sample new particles
        proposal = self.proposal
        if y_curr is None:
            proposal = self.transition_prior
            
        # proposal.sample should handle the flat batch of particles
        x_new_flat = proposal.sample(particles_flat, y_curr_flat, dt)
        
        # Reshape back to (Batch, N, D)
        self.particles = x_new_flat.reshape(batch_size, self.n_particles, self.state_dim)

    def compute_transition_log_prob(
        self, x_curr: torch.Tensor, x_prev: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Compute log p(x_t | x_{t-1}) for the transition dynamics
        Supports shapes (Batch, N, D)
        """
        # Flatten for system integration
        batch_size, n_particles, _ = x_curr.shape
        x_curr_flat = x_curr.reshape(-1, self.state_dim)
        x_prev_flat = x_prev.reshape(-1, self.state_dim)
        
        integration_result = self.system.integrate(x_prev_flat, 2, dt)
        x_expected_flat = integration_result[:, 1, :]
        noise_std = torch.tensor(self.process_noise_std, device=self.device)

        # Compute Gaussian log-likelihood
        diff = x_curr_flat - x_expected_flat
        noise_var = noise_std**2
        
        # log_prob: (Batch*N,)
        log_prob_flat = -0.5 * torch.sum(diff**2 / noise_var, dim=1)
        # Adjust for log det
        if noise_std.ndim == 0:
             log_det = self.state_dim * torch.log(noise_std)
        else:
             log_det = torch.sum(torch.log(noise_std))
             
        log_prob_flat -= (0.5 * self.state_dim * np.log(2 * np.pi) + log_det)

        return log_prob_flat.reshape(batch_size, n_particles)

    def compute_predictive_log_likelihood(
        self, x_prev: torch.Tensor, observation: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Compute approx log p(y_t | x_{t-1}) using Inflated Noise approximation (Batched).
        Supports shapes (Batch, N, D)
        """
        batch_size, n_particles, _ = x_prev.shape
        
        # 1. Deterministic propagation
        # Flatten for system integration
        x_prev_flat = x_prev.reshape(-1, self.state_dim)
        integration_result = self.system.integrate(x_prev_flat, 2, dt)
        x_hat_flat = integration_result[:, 1, :] # (Batch*N, D)
        
        # 2. Observation projection
        y_hat_flat = self.system.apply_observation_operator(x_hat_flat)
        y_hat = y_hat_flat.reshape(batch_size, n_particles, self.obs_dim)
        
        # 3. Variance inflation
        obs_noise_var = self.system.config.obs_noise_std**2
        process_noise_var = self.process_noise_std**2
        total_var = obs_noise_var + process_noise_var

        # 4. Compute Gaussian Log-Likelihood
        # observation: (Batch, obs_dim) -> expand to (Batch, 1, obs_dim)
        obs_expanded = observation.unsqueeze(1)
        
        diff = obs_expanded - y_hat # (Batch, N, obs_dim)
        
        if self.obs_dim == 1:
            log_prob = -0.5 * (diff.squeeze(-1)**2) / total_var - 0.5 * np.log(
                2 * np.pi * total_var
            )
        else:
            log_prob = -0.5 * torch.sum(diff**2, dim=2) / total_var
            log_prob -= 0.5 * self.obs_dim * np.log(2 * np.pi * total_var)
            
        return log_prob # (Batch, N)

    def update_step(self, observation: torch.Tensor) -> None:
        """
        Update particle weights using IMPORTANCE SAMPLING (Batched)
        observation: (Batch, Obs_dim)
        """
        batch_size = self.particles.shape[0]
        observation = observation.to(self.device)

        # Flatten particles: (Batch*N, D)
        particles_flat = self.particles.reshape(-1, self.state_dim)
        
        # Apply observation operator
        expected_obs_flat = self.system.apply_observation_operator(particles_flat)
        expected_observations = expected_obs_flat.reshape(batch_size, self.n_particles, self.obs_dim)

        # Compute observation log-likelihoods: log p(y_t | x_t)
        # observation: (Batch, Obs_dim) -> (Batch, 1, Obs_dim)
        obs_expanded = observation.unsqueeze(1)
        
        diff = obs_expanded - expected_observations # (Batch, N, Obs_dim)
        obs_noise_var = self.system.config.obs_noise_std**2

        if self.obs_dim == 1:
            obs_log_likelihoods = -0.5 * (diff.squeeze(-1)**2) / obs_noise_var - 0.5 * np.log(
                2 * np.pi * obs_noise_var
            )
        else:
            obs_log_likelihoods = -0.5 * torch.sum(diff**2, dim=2) / obs_noise_var
            obs_log_likelihoods -= 0.5 * (self.obs_dim * np.log(2 * np.pi * obs_noise_var))

        self.last_obs_log_likelihoods = obs_log_likelihoods

        # OPTIMAL WEIGHT UPDATE BRANCH
        if self.use_optimal_weight_update:
            # log w_t = log w_{t-1} + log p(y_t | x_{t-1})
            predictive_log_probs = self.compute_predictive_log_likelihood(
                self.particles_prev, observation, self.system.config.dt
            )
            importance_weights = predictive_log_probs

            # Compute other terms for logging (optional)
            transition_log_probs = self.compute_transition_log_prob(
                self.particles, self.particles_prev, self.system.config.dt
            )
            
            # Proposal log probs
            particles_flat = self.particles.reshape(-1, self.state_dim)
            particles_prev_flat = self.particles_prev.reshape(-1, self.state_dim)
            obs_flat = observation.unsqueeze(1).expand(batch_size, self.n_particles, -1).reshape(batch_size * self.n_particles, self.obs_dim)
            
            proposal_log_probs_flat = self.proposal.log_prob(
                particles_flat,
                particles_prev_flat,
                obs_flat,
                self.system.config.dt,
            )
            proposal_log_probs = proposal_log_probs_flat.reshape(batch_size, self.n_particles)
            self.last_proposal_log_probs = proposal_log_probs
            
        else:
            # STANDARD WEIGHT UPDATE BRANCH
            
            # Compute transition log-probabilities: log p(x_t | x_{t-1})
            transition_log_probs = self.compute_transition_log_prob(
                self.particles, self.particles_prev, self.system.config.dt
            )

            # Compute proposal log-probabilities: log q(x_t | x_{t-1}, y_t)
            particles_flat = self.particles.reshape(-1, self.state_dim)
            particles_prev_flat = self.particles_prev.reshape(-1, self.state_dim)
            
            # Expand observation for proposal: (Batch, Obs_dim) -> (Batch, N, Obs_dim) -> (Batch*N, Obs_dim)
            obs_flat = observation.unsqueeze(1).expand(batch_size, self.n_particles, -1).reshape(batch_size * self.n_particles, self.obs_dim)
            
            proposal_log_probs_flat = self.proposal.log_prob(
                particles_flat,
                particles_prev_flat,
                obs_flat,
                self.system.config.dt,
            )
            proposal_log_probs = proposal_log_probs_flat.reshape(batch_size, self.n_particles)

            # Update weights: log w_t = log w_{t-1} + log p(y|x) + log p(x|x_prev) - log q(x|x_prev, y)
            importance_weights = (
                obs_log_likelihoods + transition_log_probs - proposal_log_probs
            )
            
            self.last_proposal_log_probs = proposal_log_probs

        # --- DEBUGGING ---
        print(f"Step {self.step_count} (Batch 0) [{'Opt' if self.use_optimal_weight_update else 'Std'}]:")
        print(f"  proposal_log_probs (first 5): {proposal_log_probs[0, :5].detach().cpu().numpy()}")
        print(f"  obs_log_probs (first 5): {obs_log_likelihoods[0, :5].detach().cpu().numpy()}")
        print(f"  transition_log_probs (first 5): {transition_log_probs[0, :5].detach().cpu().numpy()}")
        print(f"  log_weights before norm (first 5): {self.log_weights[0, :5].detach().cpu().numpy()}")
        # -----------------
        
        self.log_weights += importance_weights

    def resample(self) -> Tuple[List[bool], torch.Tensor]:
        """
        Systematic resampling of particles (Batched)
        Returns:
            - list of booleans indicating if resampling happened for each batch item
            - tensor of ESS values before resampling (Batch,)
        """
        batch_size = self.log_weights.shape[0]
        
        # Normalize weights
        max_log_weight = torch.max(self.log_weights, dim=1, keepdim=True)[0]
        log_weights_normalized = self.log_weights - max_log_weight
        weights_unnormalized = torch.exp(log_weights_normalized)
        weight_sum = torch.sum(weights_unnormalized, dim=1, keepdim=True)
        
        # Avoid div by zero
        mask_zero = (weight_sum.squeeze() == 0)
        if mask_zero.any():
             # For failed batches, reset to uniform
             weights_unnormalized[mask_zero] = 1.0
             weight_sum[mask_zero] = self.n_particles
        
        self.weights = weights_unnormalized / weight_sum
        
        # Calculate ESS
        ess = 1.0 / torch.sum(self.weights**2, dim=1)
        resample_threshold = self.n_particles * self.resampling_threshold_ratio
        
        # Identify which batches need resampling
        needs_resample = ess < resample_threshold
        resampled_flags = needs_resample.cpu().tolist()
        
        if not needs_resample.any():
             self.log_weights = torch.log(self.weights) # Just update log weights
             return resampled_flags, ess

        # Perform resampling for those that need it
        # Vectorized systematic resampling:
        
        cumsum = torch.cumsum(self.weights, dim=1) # (Batch, N)
        
        # Systematic noise: (Batch, 1)
        u = torch.rand(batch_size, 1, device=self.device) / self.n_particles
        
        # (Batch, N)
        positions = u + torch.arange(self.n_particles, device=self.device, dtype=torch.float32).unsqueeze(0) / self.n_particles
        
        # Perform searchsorted for each batch row
        indices = torch.searchsorted(cumsum, positions)
        indices = torch.clamp(indices, 0, self.n_particles - 1)
        
        # Use gather to select particles
        # indices is (Batch, N). particles is (Batch, N, D)
        # We need to expand indices to (Batch, N, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.state_dim)
        
        particles_resampled = torch.gather(self.particles, 1, indices_expanded)
        particles_prev_resampled = torch.gather(self.particles_prev, 1, indices_expanded)
        
        # Only apply to batches that needed resampling
        mask_resample = needs_resample.unsqueeze(-1).unsqueeze(-1) # (Batch, 1, 1)
        
        self.particles = torch.where(mask_resample, particles_resampled, self.particles)
        self.particles_prev = torch.where(mask_resample, particles_prev_resampled, self.particles_prev)
        
        # Reset weights for resampled batches
        new_weights = torch.where(
            needs_resample.unsqueeze(-1),
            torch.ones_like(self.weights) / self.n_particles,
            self.weights
        )
        self.weights = new_weights
        self.log_weights = torch.log(self.weights)
        
        return resampled_flags, ess

    def get_state_estimate(self) -> torch.Tensor:
        """Get current state estimate (weighted mean) - (Batch, D)"""
        # weights: (Batch, N), particles: (Batch, N, D)
        return torch.sum(self.weights.unsqueeze(-1) * self.particles, dim=1)

    def get_state_covariance(self) -> torch.Tensor:
        """Get current state covariance estimate - (Batch, D, D)"""
        mean = self.get_state_estimate() # (Batch, D)
        centered = self.particles - mean.unsqueeze(1) # (Batch, N, D)
        
        # (Batch, N, D, 1) * (Batch, N, 1, D) -> (Batch, N, D, D)
        outer_prod = centered.unsqueeze(3) @ centered.unsqueeze(2)
        
        # Weighted sum over particles
        # weights: (Batch, N, 1, 1)
        cov = torch.sum(self.weights.unsqueeze(-1).unsqueeze(-1) * outer_prod, dim=1)
        return cov

    def compute_log_likelihood(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood of observation given current state (Batched)
        Returns (Batch,)
        """
        batch_size = self.particles.shape[0]
        observation = observation.to(self.device)
        
        particles_flat = self.particles.reshape(-1, self.state_dim)
        expected_obs_flat = self.system.apply_observation_operator(particles_flat)
        expected_observations = expected_obs_flat.reshape(batch_size, self.n_particles, self.obs_dim)
        
        obs_expanded = observation.unsqueeze(1)
        diff = obs_expanded - expected_observations
        obs_noise_var = self.system.config.obs_noise_std**2
        
        if self.obs_dim == 1:
            log_likelihoods = -0.5 * (diff.squeeze(-1)**2) / obs_noise_var - 0.5 * np.log(
                2 * np.pi * obs_noise_var
            )
        else:
            log_likelihoods = -0.5 * torch.sum(diff**2, dim=2) / obs_noise_var
            log_likelihoods -= 0.5 * (self.obs_dim * np.log(2 * np.pi * obs_noise_var))
            
        # Weighted average of log-likelihoods
        # log sum exp (log_weights + log_ll) - log sum exp (log_weights)
        
        log_marginal = torch.logsumexp(self.log_weights + log_likelihoods, dim=1) - \
                       torch.logsumexp(self.log_weights, dim=1)
                       
        return log_marginal

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
        Particle filter step for a BATCH of trajectories
        """
        start_time = time.perf_counter()
        self.step_count += 1
        
        # Check for re-initialization (if new trajectories started)
        # Assumes if one changes, all change (based on Sampler logic)
        # Or we rely on external caller to call initialize_filter
        
        # Predict
        self.predict_step(dt, y_curr)
        
        # Update
        log_likelihoods = torch.zeros(x_curr.shape[0], device=self.device)
        resampled_flags = [False] * x_curr.shape[0]
        ess_pre = (1.0 / torch.sum(self.weights**2, dim=1))
        
        if y_curr is not None:
             # Check which indices have observations
             # If passed as None, none have it. If passed as tensor, check masks.
             # Current interface assumes y_curr is None if NO obs.
             # If we have mixed batch (some obs, some not), logic needs to be robust.
             # We assume here y_curr contains data for ALL batch items, or placeholder zeros.
             
             # In Batched PF, we compute LL for all, assuming valid y_curr.
             # If some don't have obs, they should have been masked out before calling update_step
             # or handled via specific logic.
             # For now, assuming uniform observation availability in batch (TimeAlignedSampler helps this)
             log_likelihoods = self.compute_log_likelihood(y_curr)
             self.update_step(y_curr)
             
             # Resample
             resampled_flags, ess_pre = self.resample()
             
        # Estimates
        x_est = self.get_state_estimate()
        P_est = self.get_state_covariance()
        
        # Metrics
        elapsed_time = (time.perf_counter() - start_time) / x_curr.shape[0] # Average time per item
        
        results = []
        
        # Ensure x_curr is on correct device
        x_curr = x_curr.to(self.device)
        
        error = x_est - x_curr
        rmse = torch.sqrt(torch.mean(error**2, dim=1)) # (Batch,)
        
        ess = 1.0 / torch.sum(self.weights**2, dim=1)
        
        prop_means = self.last_proposal_log_probs.mean(dim=1) if self.last_proposal_log_probs is not None else torch.zeros(x_curr.shape[0], device=self.device)
        obs_means = self.last_obs_log_likelihoods.mean(dim=1) if self.last_obs_log_likelihoods is not None else torch.zeros(x_curr.shape[0], device=self.device)
        obs_stds = self.last_obs_log_likelihoods.std(dim=1) if self.last_obs_log_likelihoods is not None else torch.zeros(x_curr.shape[0], device=self.device)
        
        for i in range(x_curr.shape[0]):
            metrics = {
                "rmse": rmse[i].item(),
                "log_likelihood": log_likelihoods[i].item(),
                "x_est": x_est[i].cpu().numpy(),
                "P_est": P_est[i].cpu().numpy(),
                "error": error[i].cpu().numpy(),
                "trajectory_idx": trajectory_idxs[i].item(),
                "time_idx": time_idxs[i].item(),
                "resampled": resampled_flags[i],
                "ess_pre_resample": ess_pre[i].item(),
                "ess": ess[i].item(),
                "step_time": elapsed_time,
                "proposal_log_prob_mean": prop_means[i].item(),
                "obs_log_prob_mean": obs_means[i].item(),
                "obs_log_prob_std": obs_stds[i].item(),
            }
            results.append(metrics)
            
        return results

    # Required abstract methods (not used in this batched flow directly but needed for inheritance)
    def sample_posterior(self, n_samples: int) -> torch.Tensor:
        raise NotImplementedError("Not implemented for batched filter yet")
