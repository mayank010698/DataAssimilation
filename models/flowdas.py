import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Callable, List, Tuple, Union

# Try to import create_velocity_network from proposals.architectures
try:
    from proposals.architectures import create_velocity_network
except ImportError:
    # If running as script or different path structure
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from proposals.architectures import create_velocity_network
    except ImportError:
        pass # Fallback or error later if needed

class MultiGaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding multiple inputs (e.g., time and extra elements)."""
    def __init__(self, input_dim, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights for each input dimension
        self.W = nn.Parameter(torch.randn(input_dim, embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        # self.W shape: [input_dim, embed_dim // 2]
        
        # Element-wise projection logic to match dimensions expected by ScoreNet
        # x[..., None] shape: [batch, input_dim, 1]
        # self.W[None, :, :] shape: [1, input_dim, embed_dim // 2]
        x_proj = x[..., None] * self.W[None, :, :] * 2 * np.pi
        
        # x_proj shape: [batch, input_dim, embed_dim // 2]
        # Concatenate sin and cos on the last dimension
        # Result shape: [batch, input_dim, embed_dim]
        x_proj_cat = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        # Flatten input_dim and embed_dim to get [batch, input_dim * embed_dim]
        return x_proj_cat.view(x.shape[0], -1)


class ScoreNet(nn.Module):
    """
    A time-dependent score-based model.
    Supports both the original MLP architecture and other architectures via create_velocity_network.
    """
    def __init__(self, marginal_prob_std, x_dim, extra_dim=0, hidden_depth=2, embed_dim=128, use_bn=True,
                 architecture='mlp', **kwargs):
        super().__init__()
        self.architecture = architecture
        self.x_dim = x_dim
        self.extra_dim = extra_dim
        self.marginal_prob_std = marginal_prob_std

        if architecture == 'mlp':
            self.hidden_depth = hidden_depth
            self.embed_dim = embed_dim
            self.use_bn = use_bn

            # Adjusted embedding layer to handle time and extra elements
            input_dim = 1 + extra_dim  # 1 for time t, plus extra elements
            
            # Note: The original implementation used a specific projection that mixed dims.
            # We'll use a standard projection here for simplicity if exact reproduction isn't strictly required,
            # but let's try to match the original spirit if possible.
            # Original: MultiGaussianFourierProjection(input_dim=input_dim, embed_dim=embed_dim)
            self.embed = nn.Sequential(
                MultiGaussianFourierProjection(input_dim=input_dim, embed_dim=embed_dim),
                nn.Linear(embed_dim, embed_dim) # Original had input_dim * embed_dim input to linear?
                # The original code: 
                # self.W shape: (input_dim, embed_dim // 2)
                # x_proj shape: (batch, input_dim * embed_dim) ? No.
                # x_proj = x[..., None] * self.W[None, :, :] -> (batch, input_dim, embed_dim//2)
                # flattened -> (batch, input_dim * embed_dim // 2)
                # sin/cos cat -> (batch, input_dim * embed_dim)
                # Linear input dim: input_dim * embed_dim
            )
            # Re-implementing the specific projection to match dimensions
            self.projection = MultiGaussianFourierProjection(input_dim=input_dim, embed_dim=embed_dim)
            self.embed_linear = nn.Linear(input_dim * embed_dim, embed_dim)

            # Input layer
            self.input = nn.Linear(x_dim, embed_dim)

            # Hidden layers
            self.fc_all = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(self.hidden_depth)])

            # Batch normalization layers
            if self.use_bn:
                self.bn = nn.ModuleList([nn.BatchNorm1d(num_features=embed_dim) for _ in range(self.hidden_depth)])

            # Output layer
            self.output = nn.Linear(embed_dim, x_dim)

            # Activation function
            self.act = lambda x: x * torch.sigmoid(x)

        else:
            # Use external architecture factory
            # Map extra_dim to conditioning size
            # create_velocity_network expects x, s, x_prev. 
            # Here extra_elements is x_prev (condition).
            self.net = create_velocity_network(
                architecture=architecture,
                state_dim=x_dim,
                # conditioning_method='concat' is usually default or passed in kwargs
                # We assume kwargs contains necessary params for the factory
                **kwargs
            )

    def forward(self, x, t, extra_elements=None):
        if self.architecture == 'mlp':
            # Combine time t and extra elements
            if extra_elements is not None:
                if extra_elements.dim() == 1:
                    extra_elements = extra_elements.unsqueeze(-1)
                if t.dim() == 1:
                    t = t.unsqueeze(-1)
                te = torch.cat([t, extra_elements], dim=-1)  # Shape: [batch_size, input_dim]
            else:
                te = t.unsqueeze(-1)

            # Embed time + condition
            # Projection returns (batch, input_dim * embed_dim)
            # We need to access projection directly
            x_proj = self.projection(te)
            x_proj = x_proj.view(x_proj.shape[0], -1) # Flatten if not already
            embed = self.act(self.embed_linear(x_proj))

            # Process input x
            h = self.input(x)

            # Residual connections with embedding
            for i in range(self.hidden_depth):
                h = h + self.act(self.fc_all[i](h)) + embed
                if self.use_bn:
                    h = self.bn[i](h)

            return self.output(h)
        else:
            # External architecture
            # Map inputs: x -> x, t -> s, extra_elements -> x_prev
            # t needs to be (batch, 1)
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            return self.net(x, t, x_prev=extra_elements)

# =============================================================================
# Loss Function (Stochastic Interpolant)
# =============================================================================

def loss_fn(model, x):
    """
    The loss function for training score-based generative models (FlowDAS/SI).
    """
    zt_squeezed = x['zt'] # Stochastic Interpolant
    cond_squeezed = x['cond']
    target = x['drift_target']
    
    score = model(zt_squeezed, x['t'], extra_elements=cond_squeezed)
    loss = (score - target).pow(2).sum(-1).mean() # Mean Squared Error

    return loss, score.shape[0]

def prepare_batch(batch, device='cuda:0', config=None):
    """
    Process batch data and prepare for training/sampling.
    Expected batch is a dict from DataAssimilationDataModule/Dataset.
    Or raw tensor if using legacy loader.
    
    If using DataAssimilationDataset:
    batch is dict with 'x_prev', 'x_curr', 'x_prev_scaled', 'x_curr_scaled'
    """
    # Check if batch is dict (from new dataloader) or tensor (legacy)
    if isinstance(batch, dict) and 'x_prev' in batch:
        # Using new DataAssimilationDataset
        # We prefer scaled data for training if available
        if 'x_prev_scaled' in batch and batch['x_prev_scaled'] is not None:
            z0 = batch['x_prev_scaled'].to(device)
            z1 = batch['x_curr_scaled'].to(device)
        else:
            z0 = batch['x_prev'].to(device)
            z1 = batch['x_curr'].to(device)
        
        N = z0.shape[0]
        # Ensure shapes are flat (N, D)
        if z0.dim() > 2:
            z0 = z0.view(N, -1)
            z1 = z1.view(N, -1)
            
    else:
        # Legacy tensor batch support (N, L-1, 6) or similar
        # Assuming input is (N, 2*D) where first half is prev, second is curr
        if isinstance(batch, dict): 
            # Maybe it's the dict from 'to(x)' in legacy code but containing tensors
             raise ValueError("Unexpected batch format")
             
        batch = batch.to(device)
        half_dim = batch.shape[-1] // 2
        z0 = batch[..., :half_dim].view(-1, half_dim)
        z1 = batch[..., half_dim:].view(-1, half_dim)
        N = z0.shape[0]

    # Initialize dictionary for SI components
    sigma_coef = 1 # Could be configurable
    D = {
        'z0': z0, 
        'z1': z1, 
        'N': N
    }

    # Interpolant functions
    def wide(t):
        return t

    def alpha(t):
        return wide(1-t)

    def alpha_dot(t): 
        return wide(-1.0 * torch.ones_like(t))

    def beta(t):
        return wide(t.pow(2))

    def beta_dot(t):
        return wide(2.0 * t)

    def sigma(t):
        return sigma_coef * wide(1-t)

    def sigma_dot(t):
        return sigma_coef * wide(-torch.ones_like(t))
    
    def gamma(t):
        return wide(t.sqrt()) * sigma(t)

    def compute_zt(D):
        return D['at'] * D['z0'] + D['bt'] * D['z1'] + D['gamma_t'] * D['noise']

    def compute_target(D):
        return D['adot'] * D['z0'] + D['bdot'] * D['z1'] + (D['sdot'] * D['root_t']) * D['noise']

    # Sample time
    D['t'] = torch.distributions.Uniform(low=0, high=1).sample(sample_shape=(N, 1)).to(device)
    
    D['cond'] = z0
    D['noise'] = torch.randn_like(D['z0'])
    D['at'] = alpha(D['t'])
    D['bt'] = beta(D['t'])
    D['adot'] = alpha_dot(D['t'])
    D['bdot'] = beta_dot(D['t'])
    D['root_t'] = wide(D['t'].sqrt())
    D['gamma_t'] = gamma(D['t'])
    D['st'] = sigma(D['t'])
    D['sdot'] = sigma_dot(D['t'])
    D['zt'] = compute_zt(D)
    D['drift_target'] = compute_target(D)

    return D

# =============================================================================
# Sampling / Inference
# =============================================================================

def marginal_prob_std(t, sigma):
    # Used for ScoreNet initialization if needed
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def observe(x):
    # Default observation operator (legacy)
    return torch.atan(x)

def grad_and_value_NOEST(x_prev, x1_hat, measurement, observation_operator=None, 
                         scaler_mean=None, scaler_std=None, sigma_obs=0.25):
    """
    Compute guidance gradient based on observation likelihood.
    
    Args:
        x_prev: Current state in the sampling process (xt). Shape (B, D).
        x1_hat: Estimated x1 (target/data) from Taylor expansion. List of tensors or Tensor.
        measurement: Actual observation y. Shape (B, obs_dim).
        observation_operator: Callable that maps state to observation.
        scaler_mean, scaler_std: For unscaling if working in scaled space.
        sigma_obs: Observation noise standard deviation.
    """
    if isinstance(x1_hat, list):
        x1_hat = torch.cat(x1_hat, dim=0) # (B*MC_times, D)
    
    # Enable gradient tracking on x1_hat if not already
    if not x1_hat.requires_grad:
        x1_hat = x1_hat.requires_grad_(True)
        
    # Unscale if needed
    if scaler_mean is not None and scaler_std is not None:
        # Assuming x1_hat is scaled, we unscale it to apply physical observation operator
        x1_phys = x1_hat * scaler_std + scaler_mean
    else:
        x1_phys = x1_hat

    # Apply observation operator
    if observation_operator is None:
        y_pred = observe(x1_phys)
        # Default legacy behavior: only use -3 dim if it looks like (..., 3*window)
        # But here x1_phys is likely (B*MC, D)
        # The legacy code: observe(x1_hat)[:,-3] implies scalar observation of one variable?
        # We will assume observation_operator handles dimensions correctly.
    else:
        y_pred = observation_operator(x1_phys)

    # Compute difference
    # measurement shape: (B, obs_dim)
    # y_pred shape: (B*MC, obs_dim)
    # We need to broadcast measurement
    B = measurement.shape[0]
    MC = x1_hat.shape[0] // B
    
    measurement_expanded = measurement.repeat_interleave(MC, dim=0)
    
    # Norm over observation dimensions
    differences = torch.linalg.norm(measurement_expanded - y_pred, dim=-1) # (B*MC,)
    
    # Compute weights
    # 0.25 is hardcoded sigma in original code. Use sigma_obs.
    weights = -differences / (2 * (sigma_obs)**2)
    
    # Detach weights
    weights_detached = weights.detach()
    
    # Reshape to (B, MC) to do softmax over MC samples for each batch item
    weights_reshaped = weights_detached.view(B, MC)
    softmax_weights = torch.softmax(weights_reshaped, dim=1).view(-1) # Flatten back to (B*MC,)
    
    # Result
    result = softmax_weights * differences
    final_result = result.sum()
    
    # Gradients
    norm_grad_tuple = torch.autograd.grad(outputs=final_result, inputs=x_prev, allow_unused=True)
    norm_grad = norm_grad_tuple[0]
    
    return norm_grad, final_result

def MC_taylor_est2rd_x1(model, xt, t, bF, g, cond=None, MC_times=1):
    """
    Estimate x1 using 2nd order Taylor expansion.
    """
    # Note: t is a tensor of shape (B, 1) or similar
    
    # Constants/Coefficients for Taylor expansion
    # Matches original code logic
    
    # Generate random noise for MC integration
    # hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
    
    t1 = torch.FloatTensor([1-1e-5]).to(xt.device)
    # We need bF at t=1 (approx)
    # In original code: bF2 = model(xt, t1, extra_elements=cond)
    # Note: Using xt as input at t=1 might be an approximation since xt is at t
    bF2 = model(xt, t1.expand(xt.shape[0], 1), extra_elements=cond).requires_grad_(True)
    
    # Noise coef
    noise_coef = (2/3 - t.sqrt() + (1/3) * (t.sqrt())**3)
    
    hat_x1_list = []
    for _ in range(MC_times):
        # Taylor expansion formula from FlowDAS
        hat_x1 = xt + (bF + bF2)/2 * (1-t) + torch.randn_like(xt) * noise_coef
        hat_x1_list.append(hat_x1)
        
    return hat_x1_list

def EM(model, base, cond=None, num_steps=50, measurement=None, 
       sigma_obs=0.25, MC_times=1, step_size=0.1, 
       observation_operator=None, scaler_mean=None, scaler_std=None):
    """
    Euler-Maruyama sampler with guidance.
    """
    device = base.device
    steps = num_steps
    tmin, tmax = 0, 1
    ts = torch.linspace(tmin, tmax, steps).to(device)
    dt = ts[1] - ts[0]
    
    xt = base.clone().requires_grad_(True)
    
    # History for debugging/analysis
    diff_list = []
    
    for i, tscalar in enumerate(ts):
        if i == len(ts) - 1: # Last step
            break
            
        t_batch = torch.full((xt.shape[0], 1), tscalar.item(), device=device)
        
        # Avoid t=0 or t=1 singularities if any
        if tscalar == 0: t_batch += 1e-5
        if tscalar == 1: t_batch -= 1e-5
            
        # 1. Compute Drift (bF)
        bF = model(xt, t_batch, extra_elements=cond)
        
        # 2. Diffusion coef (sigma)
        # Original: sigma(t) = 1-t
        sigma_t = 1 - t_batch
        
        f = bF
        g = sigma_t
        
        # 3. Guidance (if measurement provided)
        norm_grad = 0
        if measurement is not None:
            es_x1 = MC_taylor_est2rd_x1(model, xt, t_batch, bF, g, cond=cond, MC_times=MC_times)
            
            grad, val = grad_and_value_NOEST(
                x_prev=xt, 
                x1_hat=es_x1, 
                measurement=measurement,
                observation_operator=observation_operator,
                scaler_mean=scaler_mean,
                scaler_std=scaler_std,
                sigma_obs=sigma_obs
            )
            
            if grad is not None:
                norm_grad = grad
            diff_list.append(val.detach())
            
        # 4. Update
        # SDE: dx = f dt + g dw
        # Guidance: - scale * grad
        noise = torch.randn_like(xt)
        xt = xt + f * dt + g * noise * dt.sqrt() - step_size * norm_grad
        
        # Optional: detach to save memory if history not needed for grad
        xt = xt.detach().requires_grad_(True)
        
    return xt, diff_list

def Euler_Maruyama_sampler(model, base, cond=None, measurement=None,
                           num_steps=50, sigma_obs=0.25, MC_times=1, step_size=0.1,
                           observation_operator=None, scaler_mean=None, scaler_std=None):
    """
    Wrapper for EM sampler.
    """
    model.eval()
    return EM(model, base, cond, num_steps, measurement, 
              sigma_obs, MC_times, step_size, 
              observation_operator, scaler_mean, scaler_std)


