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


class EnsembleScoreFilter(FilteringMethod):
    """
    Ensemble Score Filter (EnSF).
    Uses a score-based diffusion model to transform the forecast ensemble 
    into the analysis ensemble by integrating a reverse SDE in pseudo-time.
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
    ):
        super().__init__(system, state_dim, obs_dim, device)
        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        
        if proposal_distribution is None:
            self.proposal = TransitionProposal(system, process_noise_std)
        else:
            self.proposal = proposal_distribution

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
        
        logging.info(f"Initialized EnsembleScoreFilter with {n_particles} particles")
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

        # Sample new particles
        x_new_flat = self.proposal.sample(particles_flat, y_curr_flat, dt)
        
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
        
        # Initial state for reverse SDE (at t=1) is samples from N(0, I) (conceptually)
        # But in EnSF, we often start from a noised version of the prior or standard noise.
        # The paper (Algorithm 1) says: 
        # "Initialize x_T ~ N(0, I)"
        # However, we want to maintain the correlation structure or "guide" the specific particles.
        # The reference implementation creates x_T from standard normal, normalized to standard deviation 1.
        # "x_T = (x_T - mean) / std"
        
        # Let's generate fresh noise for the reverse process
        xt = torch.randn_like(self.particles)
        
        # Normalize xt to be strictly standard normal (optional but in reference)
        # xt = (xt - xt.mean(dim=1, keepdim=True)) / xt.std(dim=1, keepdim=True)
        # Standard random normal is usually sufficient if N is large.
        
        # Time discretization (from 1 to 0)
        t_steps = torch.linspace(1.0, 0.0, self.time_steps + 1, device=self.device)
        
        dt_step = 1.0 / self.time_steps
        
        for i in range(self.time_steps):
            t_now = t_steps[i]
            t_next = t_steps[i+1]
            dt = t_now - t_next # Positive dt
            
            # 1. Compute Score: Prior + Likelihood
            # score(xt, t) = score_prior(xt, t) + score_likelihood(xt, t)
            
            # Prior score: based on x_prior (the ensemble at t=0)
            if self.score_type == 'mixture':
                score_p = self._score_gaussian_mixture(xt, x_prior, t_now)
            else:
                score_p = self._score_diagonal_gaussian(xt, x_prior, t_now)
                
            # Likelihood score: based on observation
            # We need to compute grad log p(y | x_0_hat) where x_0_hat is estimated from xt?
            # Or use the analytic approximation.
            # Reference uses: analytic gradient scaled by dampening or transition.
            # Paper Eq 3.5: \nabla_{x_t} \log p(y|x_t) \approx \nabla_{x} \log p(y|x)|_{x=\hat{x}_0(x_t)} * \frac{\partial \hat{x}_0}{\partial x_t}
            # Or simplified: score_L = \nabla_x \log p(y|x) evaluated at x_t (approx) or \hat{x}_0.
            # The reference code uses:
            # score_x = -(atan(xt) - obs)/sigma^2 * (1/(1+xt^2)) * damp_fn(t)
            # damp_fn(t) = 1-t (roughly).
            
            # We will use autograd to compute \nabla_x \log p(y|x) at xt.
            # Ideally we should evaluate at E[x_0|x_t], but evaluating at xt is a common approximation in score-based DA 
            # (assuming xt is close to x0 for small t, or using the bridge property).
            # Actually, diffusion models relate x_t approx alpha_t * x_0.
            # So we should evaluate likelihood at x_t / alpha_t ?
            # Reference implementation evaluates at xt directly but scales the result.
            
            # Let's compute gradient at xt, and scale if necessary.
            # The likelihood score needs to be weighted by the noise level usually.
            
            score_l = self._score_likelihood(xt, observation, t_now)
            
            total_score = score_p + score_l
            
            # Clip score
            total_score = torch.clamp(total_score, -self.score_clip, self.score_clip)
            
            # 2. SDE Step (Euler-Maruyama)
            sde_a, sde_b_sq = self.noise_schedule.get_sde_coef(t_now)
            
            # dx = [f(t)x - g^2(t)score]dt + g(t)dw
            # In our backward time notation (t goes 1->0):
            # x_{t-1} = x_t - [f(t)x_t - g^2(t)score] * (-dt) + g(t) * sqrt(dt) * z
            # Note signs: dt in loop is positive (step size). 
            # The reverse SDE equation usually has a standard form.
            # Reference: xt_next = xt - dt * (sde_a * xt - sde_b_sq * score) + sqrt(dt * sde_b_sq) * noise
            
            noise = torch.randn_like(xt)
            
            drift = sde_a * xt - sde_b_sq * total_score
            diffusion = torch.sqrt(dt * sde_b_sq)
            
            xt = xt - dt * drift + diffusion * noise
            
        # The final xt (at t=0) is the posterior ensemble
        # We need to unscale it? 
        # The noise schedule defines x_t = alpha_t * x_0 + ...
        # At t=0, alpha_0 = 1. So xt converges to x_0.
        
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
        
        # Log probabilities (unnormalized): -0.5 * dist_sq / var
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
