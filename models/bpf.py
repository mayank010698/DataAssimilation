import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
import time

from .base_pf import FilteringMethod
from .proposals import ProposalDistribution, TransitionProposal


class BootstrapParticleFilter(FilteringMethod):
    """
    Bootstrap Particle Filter with CORRECT arbitrary proposal distribution support.

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
    ):
        super().__init__(system, state_dim, obs_dim, device)
        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        self.use_preprocessing = system.config.use_preprocessing

        if proposal_distribution is None:
            self.proposal = TransitionProposal(system, process_noise_std)
        else:
            self.proposal = proposal_distribution

        self.particles = None
        self.particles_prev = None  # Store previous particles for weight computation
        self.weights = None
        self.log_weights = None

        self.step_count = 0
        self.resampling_history = []

        logging.info("Initialized BootstrapParticleFilter")
        logging.info(
            f"n_particles={n_particles}, proposal={type(self.proposal).__name__}, "
            f"process_noise_std={process_noise_std}, obs_noise_std={system.config.obs_noise_std}, "
            f"obs_dim={obs_dim}, use_preprocessing={self.use_preprocessing}"
        )

    def initialize_filter(self, x0: torch.Tensor) -> None:
        """Initialize particles around initial state"""
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)

        init_std = 0.5
        noise = init_std * torch.randn(
            self.n_particles, self.state_dim, device=self.device
        )
        self.particles = x0 + noise
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
        if self.use_preprocessing:
            logging.info("Operating in normalized space due to preprocessing")

    def predict_step(self, dt: float, y_curr: Optional[torch.Tensor] = None) -> None:
        """
        CORRECTED: Propagate particles using the PROPOSAL DISTRIBUTION
        """
        # Store previous particles for weight computation
        self.particles_prev = self.particles.clone()

        new_particles = []

        for i in range(self.n_particles):
            # Use the proposal distribution to sample new particles
            x_new = self.proposal.sample(self.particles[i], y_curr, dt)
            new_particles.append(x_new)

        self.particles = torch.stack(new_particles)

        if self.step_count % 1 == 0:
            mean_pos = torch.mean(self.particles, dim=0).cpu().numpy()
            std_pos = torch.std(self.particles, dim=0).cpu().numpy()
            logging.debug(
                f"Step {self.step_count}: Particle mean = {mean_pos}, std = {std_pos}"
            )

    def compute_transition_log_prob(
        self, x_curr: torch.Tensor, x_prev: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Compute log p(x_t | x_{t-1}) for the transition dynamics
        """
        log_probs = torch.zeros(self.n_particles, device=self.device)

        for i in range(self.n_particles):
            # Get expected next state from deterministic dynamics
            x_prev_np = x_prev[i].cpu().numpy()

            if self.use_preprocessing:
                # Convert to original space, integrate, convert back
                x_prev_orig = self.system.postprocess(x_prev_np)
                x_expected_orig = self.system.integrate(x_prev_orig, 2, dt)[1]
                x_expected = self.system.preprocess(x_expected_orig.reshape(1, 1, -1))[
                    0, 0
                ]
                noise_std = self.process_noise_std / self.system.init_std
            else:
                x_expected = self.system.integrate(x_prev_np, 2, dt)[1]
                noise_std = self.process_noise_std

            # Convert everything to tensors for consistency
            noise_std = torch.tensor(noise_std, dtype=torch.float32, device=self.device)
            x_expected_tensor = torch.tensor(
                x_expected, dtype=torch.float32, device=self.device
            )

            # Compute Gaussian log-likelihood of the transition
            diff = x_curr[i] - x_expected_tensor
            noise_var = noise_std**2

            log_prob = -0.5 * torch.sum(diff**2 / noise_var)
            log_prob -= 0.5 * torch.sum(torch.log(2 * np.pi * noise_var))

            log_probs[i] = log_prob

        return log_probs

    def apply_observation_operator_torch(self, particles: torch.Tensor) -> torch.Tensor:
        """Apply observation operator to particles"""
        particles_np = particles.cpu().numpy()

        if particles_np.ndim == 1:
            expected_obs = self.system.apply_observation_operator(particles_np)
        else:
            expected_obs_list = []
            for i in range(particles_np.shape[0]):
                obs = self.system.apply_observation_operator(particles_np[i])
                expected_obs_list.append(obs)
            expected_obs = np.array(expected_obs_list)

        return torch.tensor(expected_obs, dtype=torch.float32, device=self.device)

    def update_step(self, observation: torch.Tensor) -> None:
        """
        CORRECTED: Update particle weights using IMPORTANCE SAMPLING
        w_t^(i) = w_{t-1}^(i) * [p(y_t | x_t^(i)) * p(x_t^(i) | x_{t-1}^(i))] / q(x_t^(i) | x_{t-1}^(i), y_t)
        """
        if observation.dim() == 0:
            observation = observation.unsqueeze(0)

        expected_observations = self.apply_observation_operator_torch(self.particles)

        logging.debug(f"observation shape: {observation.shape}")
        logging.debug(f"expected_observations shape: {expected_observations.shape}")
        logging.debug(f"obs_dim: {self.obs_dim}")

        # Compute observation log-likelihoods: log p(y_t | x_t^(i))
        obs_log_likelihoods = torch.zeros(self.n_particles, device=self.device)
        obs_noise_var = self.system.config.obs_noise_std**2

        for i in range(self.n_particles):
            expected_obs = expected_observations[i]

            if expected_obs.dim() == 0:
                expected_obs = expected_obs.unsqueeze(0)
            if observation.dim() == 0:
                observation = observation.unsqueeze(0)

            diff = observation - expected_obs

            if self.obs_dim == 1:
                log_likelihood = -0.5 * (diff**2) / obs_noise_var - 0.5 * np.log(
                    2 * np.pi * obs_noise_var
                )
            else:
                obs_cov = obs_noise_var * torch.eye(self.obs_dim, device=self.device)
                inv_cov = torch.inverse(obs_cov)
                quadratic_form = diff @ inv_cov @ diff
                log_det_2pi_cov = torch.logdet(obs_cov) + self.obs_dim * np.log(
                    2 * np.pi
                )
                log_likelihood = -0.5 * (quadratic_form + log_det_2pi_cov)

            obs_log_likelihoods[i] = log_likelihood

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

        # CORRECTED importance sampling weight update:
        # log w_t^(i) = log w_{t-1}^(i) + log p(y_t | x_t^(i)) + log p(x_t^(i) | x_{t-1}^(i)) - log q(x_t^(i) | x_{t-1}^(i), y_t)
        importance_weights = (
            obs_log_likelihoods + transition_log_probs - proposal_log_probs
        )
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

    def resample(self) -> bool:
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
            return False

        self.weights = weights_unnormalized / weight_sum
        ess = 1.0 / torch.sum(self.weights**2)
        resample_threshold = self.n_particles / 3

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
            return True

        logging.debug(f"No resampling needed, ESS = {ess:.1f}")
        return False

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

        expected_observations = self.apply_observation_operator_torch(self.particles)
        obs_noise_var = self.system.config.obs_noise_std**2

        log_likelihoods = torch.zeros(self.n_particles, device=self.device)

        for i in range(self.n_particles):
            expected_obs = expected_observations[i]

            if expected_obs.dim() == 0:
                expected_obs = expected_obs.unsqueeze(0)

            diff = observation - expected_obs

            if self.obs_dim == 1:
                log_likelihood = -0.5 * (diff**2) / obs_noise_var - 0.5 * np.log(
                    2 * np.pi * obs_noise_var
                )
            else:
                obs_cov = obs_noise_var * torch.eye(self.obs_dim, device=self.device)
                inv_cov = torch.inverse(obs_cov)
                quadratic_form = diff @ inv_cov @ diff
                log_det_2pi_cov = torch.logdet(obs_cov) + self.obs_dim * np.log(
                    2 * np.pi
                )
                log_likelihood = -0.5 * (quadratic_form + log_det_2pi_cov)

            log_likelihoods[i] = log_likelihood

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
        self.step_count += 1
        metrics = super().step(x_prev, x_curr, y_curr, dt, trajectory_idx, time_idx)
        resampled = self.resample()

        metrics["resampled"] = resampled
        metrics["ess"] = (1.0 / torch.sum(self.weights**2)).item()
        metrics["n_effective_particles"] = metrics["ess"]
        metrics["max_weight"] = torch.max(self.weights).item()
        metrics["min_weight"] = torch.min(self.weights).item()
        metrics["particle_spread"] = torch.std(self.particles, dim=0).cpu().numpy()
        metrics["proposal_type"] = type(self.proposal).__name__
        
        elapsed_time = time.perf_counter() - start_time
        metrics["step_time"] = elapsed_time

        if self.step_count % 1 == 0:
            logging.info(
                f"Step {self.step_count}: RMSE={metrics['rmse']:.4f}, ESS={metrics['ess']:.1f}, Time={elapsed_time:.4f}s"
            )
            logging.info(f"  Proposal: {metrics['proposal_type']}")
            logging.info(f"  True state: {x_curr.cpu().numpy()}")
            logging.info(f"  Estimated:  {metrics['x_est']}")
            if y_curr is not None:
                logging.info(f"  Observation: {y_curr.cpu().numpy()}")
                expected_obs = self.system.apply_observation_operator(
                    x_curr.cpu().numpy()
                )
                logging.info(f"  Expected obs: {expected_obs}")

        return metrics
