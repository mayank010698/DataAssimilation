import torch
import numpy as np
import logging
import time
from typing import Optional, Dict, Any, Union, List

from .base_pf import FilteringMethod, compute_unweighted_spread
from .proposals import ProposalDistribution, TransitionProposal


class NoiseSchedule:
    """
    Linear noise schedule for diffusion model.
    alpha_t(0) = 1, alpha_t(1) = epsilon
    sigma2_t(0) = 0, sigma2_t(1) = 1
    """
    def __init__(self, eps_a=0.001, eps_b=0.001, device="cpu"):
        self.eps_a = eps_a
        self.eps_b = eps_b
        self.device = device

    def get_values(self, t):
        """
        Returns alpha_t, beta_sq_t (sigma^2), and their derivatives.
        Using linear schedule from reference.
        """
        # Ensure t is tensor for operations if needed, or float
        # Based on reference:
        # alpha_t = 1 - (1 - eps_a) * t
        # beta_sq_t = eps_b + (1 - eps_b) * t
        
        alpha_t = 1 - (1 - self.eps_a) * t
        beta_sq_t = self.eps_b + (1 - self.eps_b) * t
        
        d_log_alpha_dt = -(1 - self.eps_a) / alpha_t
        d_beta_sq_dt = (1 - self.eps_b)
        
        return alpha_t, beta_sq_t, d_log_alpha_dt, d_beta_sq_dt

    def get_sde_coef(self, t):
        """
        Drift and diffusion coefficients for the Reverse SDE.
        dx = [f(t)x - g^2(t)score]dt + g(t)dw
        """
        alpha_t, beta_sq_t, d_log_alpha_dt, d_beta_sq_dt = self.get_values(t)
        
        # f(t) = d_log_alpha_dt
        sde_a = d_log_alpha_dt
        
        # g^2(t) = d_beta_sq_dt - 2 * d_log_alpha_dt * beta_sq_t
        sde_b_sq = d_beta_sq_dt - 2 * d_log_alpha_dt * beta_sq_t
        
        return sde_a, sde_b_sq


class EnsembleScoreFilterWithProposal(FilteringMethod):
    """
    Ensemble Score Filter (EnSF) with observation-informed proposal support.

    This variant uses an external proposal distribution q_φ to generate the
    forecast ensemble, and then applies the standard EnSF reverse-SDE
    transport to obtain the posterior ensemble.

    The proposal samples themselves define the informed prior; no importance
    weights or transition-prior correction are used. The total score guiding
    the reverse SDE is

        score(x_t, t) = score_prior(x_t, t) + score_likelihood(x_t, t),

    where the prior score is a Gaussian-mixture score over the proposal
    forecast ensemble and the likelihood score is computed via autograd on
    the observation operator with a time-dependent damping factor.
    """

    def __init__(
        self,
        system,
        proposal_distribution: Optional[ProposalDistribution] = None,
        n_particles: int = 100,
        state_dim: int = 3,
        obs_dim: int = 1,
        process_noise_std: float = 0.25,
        device: str = "cpu",
        # EnSF specific parameters
        ensf_time_steps: int = 50,
        ensf_eps_a: float = 0.5,  # Small time cutoff / alpha_min
        ensf_eps_b: float = 0.025, # Initial noise variance
        ensf_score_clip: float = 50.0,
        ensf_score_type: str = "mixture", # 'mixture' or 'diagonal'
        fallback_to_physical_when_no_obs: bool = False,
    ):
        super().__init__(system, state_dim, obs_dim, device)
        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        
        if proposal_distribution is None:
            self.proposal = TransitionProposal(system, process_noise_std)
        else:
            self.proposal = proposal_distribution

        self.fallback_to_physical_when_no_obs = fallback_to_physical_when_no_obs
        self.transition_prior = TransitionProposal(system, process_noise_std)

        self.time_steps = ensf_time_steps
        self.score_clip = ensf_score_clip
        self.score_type = ensf_score_type
        
        self.noise_schedule = NoiseSchedule(
            eps_a=ensf_eps_a, 
            eps_b=ensf_eps_b, 
            device=device
        )
        
        self.particles = None
        self.weights = None
        
        logging.info(f"Initialized EnsembleScoreFilterWithProposal with {n_particles} particles")
        logging.info(f"EnSF params: steps={ensf_time_steps}, type={ensf_score_type}")

    def initialize_filter(self, x0: torch.Tensor, init_std: Optional[Union[float, torch.Tensor]] = None) -> None:
        """Initialize particles around initial state"""
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0) # (1, D)
            
        batch_size = x0.shape[0]
        x0 = x0.to(self.device)
        
        if init_std is None:
            init_std = self.system.config.obs_noise_std
            
        if isinstance(init_std, torch.Tensor):
            init_std = init_std.to(self.device)
            scale = init_std.reshape(1, 1, -1) if init_std.dim() > 0 else init_std
        else:
            scale = init_std
            
        noise = torch.randn(batch_size, self.n_particles, self.state_dim, device=self.device)
        self.particles = x0.unsqueeze(1) + noise * scale
        
        # EnSF uses unweighted samples (weights are uniform)
        self.weights = torch.ones(batch_size, self.n_particles, device=self.device) / self.n_particles

    def predict_step(self, dt: float, y_curr: Optional[torch.Tensor] = None) -> None:
        """Propagate particles forward"""
        batch_size = self.particles.shape[0]
        
        # Flatten for proposal
        particles_flat = self.particles.reshape(-1, self.state_dim)
        
        y_curr_flat = None
        if y_curr is not None:
             # Expand obs to match particles
             y_curr_expanded = y_curr.unsqueeze(1).expand(batch_size, self.n_particles, -1)
             y_curr_flat = y_curr_expanded.reshape(batch_size * self.n_particles, -1)

        # Select proposal: when fallback is on and no observation, use physical process
        if self.fallback_to_physical_when_no_obs and y_curr is None:
            proposal = self.transition_prior
        else:
            proposal = self.proposal

        # Sample new particles from the proposal q_φ. These samples define the
        # informed predictive prior in the hybrid EnSF–q_φ formulation.
        x_new_flat = proposal.sample(particles_flat, y_curr_flat, dt)
        
        # Reshape back
        self.particles = x_new_flat.reshape(batch_size, self.n_particles, self.state_dim)

        # Weights remain uniform in EnSF forecast step
        self.weights = torch.ones(batch_size, self.n_particles, device=self.device) / self.n_particles

    def update_step(self, observation: torch.Tensor) -> None:
        """
        EnSF Update: Transform forecast ensemble to analysis ensemble via Reverse SDE.
        """
        if observation is None:
            return

        batch_size = self.particles.shape[0]
        observation = observation.to(self.device) # (Batch, Obs_dim)
        
        # The forecast ensemble acts as the prior reference (x0 in the diffusion formulas)
        # Shape: (Batch, N, D)
        x_prior = self.particles.clone() 
        
        # Initial state for reverse SDE (at t=1)
        xt = torch.randn_like(self.particles)
        
        # Time discretization (from 1 to 0)
        t_steps = torch.linspace(1.0, 0.0, self.time_steps + 1, device=self.device)
        
        dt_step = 1.0 / self.time_steps
        
        for i in range(self.time_steps):
            t_now = t_steps[i]
            t_next = t_steps[i+1]
            dt = t_now - t_next # Positive dt
            
            # 1. Compute Score: Prior + Likelihood
            # score(xt, t) = score_prior(xt, t) + score_likelihood(xt, t)
            
            # Prior score: based on x_prior
            if self.score_type == 'mixture':
                score_p = self._score_gaussian_mixture(xt, x_prior, t_now)
            else:
                score_p = self._score_diagonal_gaussian(xt, x_prior, t_now)
                
            # Likelihood score
            score_l = self._score_likelihood(xt, observation, t_now)
            
            total_score = score_p + score_l
            
            # Clip score
            total_score = torch.clamp(total_score, -self.score_clip, self.score_clip)
            
            # 2. SDE Step (Euler-Maruyama)
            sde_a, sde_b_sq = self.noise_schedule.get_sde_coef(t_now)
            
            noise = torch.randn_like(xt)
            
            drift = sde_a * xt - sde_b_sq * total_score
            diffusion = torch.sqrt(dt * sde_b_sq)
            
            xt = xt - dt * drift + diffusion * noise
            
        self.particles = xt
        # Weights remain uniform
        self.weights = torch.ones(batch_size, self.n_particles, device=self.device) / self.n_particles
        
    def _score_gaussian_mixture(self, xt, x_prior, t):
        r"""
        Compute score of Gaussian Mixture defined by x_prior at time t.
        p_t(x) = \sum w_i N(x | alpha_t * x_prior_i, beta_sq_t * I)
        """
        batch_size, n_particles, dim = xt.shape
        alpha_t, beta_sq_t, _, _ = self.noise_schedule.get_values(t)
        
        # Means of the mixture components at time t
        mu_t = x_prior * alpha_t # (B, N, D)
        var_t = beta_sq_t # scalar (or tensor if handling vector noise)
        
        # We need to compute score for each particle xt_j against the whole mixture mu_t.
        # Efficient computation:
        # score(x) = \sum (x - mu_i)/var * weight_i(x)
        # weight_i(x) = softmax( -0.5 * |x - mu_i|^2 / var )
        
        # xt: (B, N, D) -> need (B, N, 1, D) for broadcasting against mixture components
        xt_expanded = xt.unsqueeze(2) 
        
        # mu_t: (B, N, D) -> need (B, 1, N, D)
        mu_expanded = mu_t.unsqueeze(1)
        
        # Diff: (B, N_eval, N_component, D)
        diff = xt_expanded - mu_expanded
        
        # Squared distance: (B, N_eval, N_component)
        dist_sq = torch.sum(diff**2, dim=-1)
        
        # Log probabilities (unnormalized component likelihood): -0.5 * dist_sq / var
        log_probs = -0.5 * dist_sq / (var_t + 1e-8)
        
        # Weights: softmax over component dimension (dim=2)
        # Numerical stability with log-sum-exp trick inside softmax
        weights = torch.softmax(log_probs, dim=2) # (B, N_eval, N_component)
        
        # Weighted sum of scores
        # Individual score component: - (x - mu_i) / var
        # Total score: \sum w_i * [-(x - mu_i)/var] = -1/var * \sum w_i * (x - mu_i)
        
        # Weighted diff sum: (B, N_eval, D)
        # weights: (B, N_eval, N_comp) -> (B, N_eval, N_comp, 1)
        weighted_diff = torch.sum(weights.unsqueeze(-1) * diff, dim=2)
        
        score = -weighted_diff / (var_t + 1e-8)
        return score

    def _score_diagonal_gaussian(self, xt, x_prior, t):
        """
        Simplified score assuming each particle only cares about its corresponding prior particle (independent).
        Effectively p(x) approx N(alpha_t * x_prior, beta_sq_t * I) element-wise.
        This is a mean-field approximation to the full Gaussian mixture prior.
        """
        alpha_t, beta_sq_t, _, _ = self.noise_schedule.get_values(t)
        mu_t = x_prior * alpha_t
        
        score = -(xt - mu_t) / (beta_sq_t + 1e-8)
        return score

    def _score_likelihood(self, xt, observation, t):
        r"""
        Compute \nabla_x \log p(y|x) at xt.
        Using autograd on the observation operator.
        """
        # Enable grad for xt
        xt_in = xt.detach().clone().requires_grad_(True)
        
        # Need to handle how observation operator is applied.
        # Observation operator expects physical space.
        # Diffusion runs in a latent-like space defined by alpha_t.
        # x_t \approx \alpha_t * x_0.
        # So we should estimate x_0 from x_t before applying H.
        # Simple estimate: x_0_hat = x_t / alpha_t (if alpha_t > epsilon)
        
        alpha_t, _, _, _ = self.noise_schedule.get_values(t)
        
        # Avoid division by zero
        if alpha_t < 1e-4:
            x_0_hat = xt_in
        else:
            x_0_hat = xt_in / alpha_t
            
        # Apply Observation Operator
        # x_0_hat: (B, N, D) -> flatten to (B*N, D)
        batch_size, n_particles, dim = xt_in.shape
        x_flat = x_0_hat.reshape(-1, dim)
        
        y_pred_flat = self.system.apply_observation_operator(x_flat)
        y_pred = y_pred_flat.reshape(batch_size, n_particles, -1)
        
        # Calculate log likelihood
        # log p(y|x) = -0.5 * (y - H(x))^2 / R
        
        # Observation: (B, Obs_dim) -> (B, 1, Obs_dim)
        obs_expanded = observation.unsqueeze(1)
        
        diff = observation.unsqueeze(1) - y_pred
        obs_noise_var = self.system.config.obs_noise_std**2
        
        # Sum over obs dimensions
        log_lik = -0.5 * torch.sum(diff**2, dim=-1) / obs_noise_var
        
        # We need the gradient of log_lik w.r.t xt_in (summed is fine since batch items independent)
        # Sum log_lik to scalar
        total_log_lik = torch.sum(log_lik)
        
        grads = torch.autograd.grad(total_log_lik, xt_in, create_graph=False)[0]
        
        # The paper/code sometimes scales this gradient.
        # In latent diffusion: score_likelihood = score_y * damp_fn(t)
        # The reference uses damp_fn(t) = 1 - t (roughly, for linear schedule this matches alpha_t?)
        # Actually in EnSF.py: "tau = self.g_tau(t) ... return tau*score_x" where g_tau(t) = 1-t.
        # Let's check NoiseSchedule alpha: alpha = 1 - (1-eps)*t. Approx 1-t.
        # So it seems we scale by alpha_t? 
        # Or simply use the gradient as is?
        # If we derived x_0_hat = x_t / alpha_t, then d(x_0_hat)/d(x_t) = 1/alpha_t.
        # So \nabla_{x_t} log p(y|x_0_hat) = \nabla_{x_0} ... * (1/alpha_t).
        # But reference scales DOWN by (1-t) approx alpha_t.
        # This implies they might be doing something else or I am misinterpreting.
        # Let's stick to the reference "damp_fn" approach.
        # damp_fn(t) = 1 - t.
        
        damp_factor = 1.0 - t
        return grads * damp_factor

    def compute_log_likelihood(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute approximate log-likelihood of observation given current ensemble.
        Returns tensor of shape (Batch,).
        """
        # Ensure observation is on device
        observation = observation.to(self.device)
        
        # Particles are (Batch, N, D)
        # Propagate to observation space
        batch_size, n_particles, dim = self.particles.shape
        particles_flat = self.particles.reshape(-1, dim)
        y_pred_flat = self.system.apply_observation_operator(particles_flat)
        y_pred = y_pred_flat.reshape(batch_size, n_particles, -1)
        
        obs_noise_var = self.system.config.obs_noise_std**2
        
        # diff: (Batch, N, Obs_dim)
        # observation: (Batch, Obs_dim) -> (Batch, 1, Obs_dim)
        diff = observation.unsqueeze(1) - y_pred
        
        # Sum over obs dimensions
        log_probs = -0.5 * torch.sum(diff**2, dim=-1) / obs_noise_var
        
        # Log mean exp over particles
        log_lik = torch.logsumexp(log_probs, dim=1) - np.log(n_particles)
        
        return log_lik # (Batch,)

    def get_state_estimate(self) -> torch.Tensor:
        """Mean of particles"""
        return torch.mean(self.particles, dim=1)

    def get_state_covariance(self) -> torch.Tensor:
        """Covariance of particles"""
        mean = self.get_state_estimate().unsqueeze(1)
        centered = self.particles - mean
        # (B, N, D, 1) @ (B, N, 1, D) -> (B, N, D, D)
        cov = torch.mean(centered.unsqueeze(3) @ centered.unsqueeze(2), dim=1)
        return cov

    def sample_posterior(self, n_samples: int) -> torch.Tensor:
        """Sample from particles uniformly"""
        indices = torch.randint(0, self.n_particles, (self.particles.shape[0], n_samples), device=self.device)
        # Gather (B, N, D) -> (B, K, D)
        batch_indices = torch.arange(self.particles.shape[0], device=self.device).unsqueeze(1).expand(-1, n_samples)
        return self.particles[batch_indices, indices]

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
        EnSF step for a BATCH of trajectories.
        Returns a list of metrics dictionaries (one per batch item).
        """
        start_time = time.perf_counter()
        
        # Predict
        self.predict_step(dt, y_curr)
        
        # Update
        batch_size = x_curr.shape[0]
        log_likelihoods = torch.zeros(batch_size, device=self.device)
        updated_flags = [False] * batch_size
        
        if y_curr is not None:
             # Compute LL before update (using forecast ensemble)
             log_likelihoods = self.compute_log_likelihood(y_curr)
             
             self.update_step(y_curr)
             updated_flags = [True] * batch_size
             
        # Estimates
        x_est = self.get_state_estimate()
        P_est = self.get_state_covariance()
        
        # Metrics
        elapsed_time = (time.perf_counter() - start_time) / batch_size
        
        results = []
        x_curr = x_curr.to(self.device)
        
        error = x_est - x_curr
        rmse = torch.sqrt(torch.mean(error**2, dim=1)) # (Batch,)
        
        # Uniform weights for EnSF implies ESS = N
        ess = float(self.n_particles)
        # Ensemble spread (unweighted, analysis ensemble)
        spread = compute_unweighted_spread(self.particles)  # (Batch,)
        
        for i in range(batch_size):
            metrics = {
                "rmse": rmse[i].item(),
                "log_likelihood": log_likelihoods[i].item(),
                "x_est": x_est[i].cpu().numpy(),
                "P_est": P_est[i].cpu().numpy(),
                "error": error[i].cpu().numpy(),
                "trajectory_idx": trajectory_idxs[i].item(),
                "time_idx": time_idxs[i].item(),
                "resampled": updated_flags[i], 
                "ess_pre_resample": ess,
                "ess": ess,
                "ensemble_spread": spread[i].item(),
                "step_time": elapsed_time,
            }
            results.append(metrics)
            
        return results

