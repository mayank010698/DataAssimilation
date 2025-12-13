# Neural Network Surrogate Model Architecture
## Based on Brajard et al. (2020) - "Combining data assimilation and machine learning to emulate a dynamical model"

This document provides implementation details for the neural network architecture used as a one-step predictor (surrogate model) for chaotic dynamical systems like Lorenz 96. The architecture is suitable for use as a **proposal model** in particle filtering / ensemble flow filtering contexts.

---

## 1. Core Concept: Residual One-Step Predictor

The neural network learns the **resolvent** (one-step forward map) of the dynamics, NOT the ODE right-hand-side directly. The key formulation is:

```
x_{k+1} = M_W(x_k) = x_k + f_nn(x_k, W)
```

Where:
- `x_k` is the state at time `t_k` (dimension `m`)
- `f_nn` is a neural network with weights `W`
- The **residual structure** is critical: the network predicts the **increment**, not the full next state

This is a **one-block residual network** inspired by He et al. (2016). The residual structure is natural because the dynamics equation `x_{k+1} = x_k + ∫f(x)dt` is already in incremental form.

---

## 2. Architecture Overview

### 2.1 High-Level Structure

```
Input: x_k ∈ R^m  (e.g., m=40 for Lorenz 96)
    │
    ▼
┌─────────────────┐
│  Batch Norm     │  Layer 1: Standardize inputs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bilinear Conv  │  Layer 2: Capture multiplicative interactions
│  (3 branches)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Conv1D + ReLU  │  Layer 3: Learn local patterns
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Conv1D Linear  │  Layer 4: Project to output dimension
└────────┬────────┘
         │
         ▼
    f_nn(x_k)
         │
    ┌────┴────┐
    │   ADD   │◄──── x_k (skip connection)
    └────┬────┘
         │
         ▼
Output: x_{k+1} = x_k + f_nn(x_k)
```

### 2.2 Detailed Layer Specifications

| Layer | Type | Output Channels/Size | Kernel Size | Activation | Notes |
|-------|------|---------------------|-------------|------------|-------|
| Input | - | 40 (state dim) | - | - | 1D periodic domain |
| L1 | BatchNorm1D | 40 | - | - | Standardizes input data |
| L2 | Bilinear Conv | 72 (24×3) | 5 | ReLU | See bilinear structure below |
| L3 | Conv1D | 37 | 5 | ReLU | Standard convolution |
| L4 | Conv1D | 1 | 1 | Linear | L2 regularization: 1e-4 |
| Output | Add | 40 | - | - | Skip connection from input |

**Total trainable parameters: 9,389**

---

## 3. Critical Implementation Details

### 3.1 The Bilinear Layer (Layer 2)

This is the key architectural innovation. Layer 2 has **three parallel branches**:

```
            x (from BatchNorm)
           /    |    \
          /     |     \
    ┌────┐  ┌────┐  ┌────┐
    │CNN2a│  │CNN2b│  │CNN2c│   Three parallel Conv1D branches
    │24ch │  │24ch │  │24ch │   Each: Conv1D(in, 24, kernel=5) + ReLU
    └──┬──┘  └──┬──┘  └──┬──┘
       │        │        │
       │        └───×────┘      Element-wise multiplication
       │             │
       └─────concat──┘          Concatenate: [CNN2a, CNN2b × CNN2c]
              │
              ▼
        72 channels (24 + 48 virtual but actually 24+24=48? -> 72 total)
```

**Rationale**: The bilinear layer captures **multiplicative interactions** which are ubiquitous in ODE-based geophysical models. For Lorenz 96, the dynamics include terms like `(x_{n+1} - x_{n-2}) * x_{n-1}`.

**PyTorch Implementation Sketch:**
```python
class BilinearConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch, kernel_size):
        super().__init__()
        # Three parallel branches
        self.conv_a = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size, padding='same')
        self.conv_b = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size, padding='same')
        self.conv_c = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size, padding='same')
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Three branches
        a = self.activation(self.conv_a(x))  # Linear branch
        b = self.activation(self.conv_b(x))  # Multiplicative branch 1
        c = self.activation(self.conv_c(x))  # Multiplicative branch 2
        
        # Bilinear: multiply b and c
        bc = b * c  # Element-wise multiplication
        
        # Concatenate linear and bilinear outputs
        out = torch.cat([a, bc], dim=1)  # 24 + 24 = 48 channels
        return out
```

**Note**: The paper says "Size of layer 2: 24 × 3 = 72" which suggests three 24-channel branches concatenated to give 72 channels total.

### 3.2 Circular/Periodic Convolutions

**Critical**: Since Lorenz 96 has periodic boundaries (`x_m = x_0`), the convolutions must use **circular padding**:

```python
# Option 1: Manual circular padding
def circular_pad(x, pad_size):
    # x shape: (batch, channels, length)
    return torch.cat([x[..., -pad_size:], x, x[..., :pad_size]], dim=-1)

# Option 2: Use padding='circular' in newer PyTorch versions
conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2, padding_mode='circular')
```

### 3.3 Batch Normalization (Layer 1)

The batch norm is applied **only to the input layer** and serves primarily to standardize the input data. This is simpler than using batch norm throughout:

```python
self.input_bn = nn.BatchNorm1d(num_features=1)  # Or state_dim if treating as channels
```

### 3.4 L2 Regularization on Final Layer

Only the final layer (Layer 4) has L2 regularization with coefficient `λ = 1e-4`:

```python
# In optimizer or loss:
l2_reg = 1e-4 * sum(p.pow(2).sum() for p in model.layer4.parameters())
loss = mse_loss + l2_reg

# Or use weight_decay in optimizer for that layer specifically
```

---

## 4. Training Configuration

### 4.1 Loss Function

The loss function is a **weighted MSE** over one-step predictions:

```
L(W) = Σ_k || M_W(x_k) - x_{k+1} ||²_{P_k^{-1}}
```

Where `P_k` is a diagonal matrix of uncertainties (from data assimilation analysis covariance). In the simplest case:

```python
def loss_fn(pred, target, weights=None):
    """
    pred: predicted x_{k+1}
    target: true x_{k+1}
    weights: 1/variance per state dimension (optional)
    """
    mse = (pred - target).pow(2)
    if weights is not None:
        mse = mse * weights
    return mse.mean()
```

### 4.2 Optimizer and Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adagrad | Adaptive learning rate per parameter |
| Mini-batch size | 256 | |
| Epochs per cycle | 20 | In iterative DA-ML algorithm |
| Forecast lead time (N_f) | 1 | One-step prediction |
| L2 regularization | 1e-4 | Only on final layer |

**Adagrad Optimizer:**
```python
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
```

### 4.3 Initialization Strategy

**Important**: Random initialization can cause convergence issues. The authors initialize by:

1. Create a training set from **cubic interpolation** of sparse observations
2. Train the neural network with:
   - `N_f = 4` (forecast 4 steps ahead in loss)
   - `epochs = 40` (more than normal)
3. For interpolated values, set loss weight to 0 (only use as input, not target)

```python
# Pseudo-code for initialization training
def init_training(observations, model):
    # Interpolate sparse observations to full field
    interpolated_states = cubic_interpolate(observations)
    
    # Create masks: 1 where observed, 0 where interpolated
    masks = create_observation_masks(observations)
    
    # Train with N_f=4 steps ahead
    for epoch in range(40):
        for batch in dataloader:
            x_k, x_targets, mask = batch  # targets are [x_{k+1}, x_{k+2}, x_{k+3}, x_{k+4}]
            
            pred = x_k
            loss = 0
            for i in range(4):
                pred = model(pred)
                step_loss = ((pred - x_targets[i]).pow(2) * mask).sum() / mask.sum()
                loss += step_loss
            
            loss.backward()
            optimizer.step()
```

---

## 5. Complete Model Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearConvBlock(nn.Module):
    """Bilinear convolutional layer with three branches."""
    
    def __init__(self, in_channels, branch_channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        
        # Three parallel branches with circular padding
        self.conv_a = nn.Conv1d(in_channels, branch_channels, kernel_size, 
                                 padding=pad, padding_mode='circular')
        self.conv_b = nn.Conv1d(in_channels, branch_channels, kernel_size,
                                 padding=pad, padding_mode='circular')
        self.conv_c = nn.Conv1d(in_channels, branch_channels, kernel_size,
                                 padding=pad, padding_mode='circular')
        
    def forward(self, x):
        a = F.relu(self.conv_a(x))
        b = F.relu(self.conv_b(x))
        c = F.relu(self.conv_c(x))
        
        # Bilinear interaction
        bc = b * c
        
        # Concatenate: linear path + bilinear path
        return torch.cat([a, bc], dim=1)


class BrajardSurrogateModel(nn.Module):
    """
    One-step predictor surrogate model based on Brajard et al. (2020).
    
    Predicts x_{k+1} = x_k + f_nn(x_k) using a residual CNN architecture
    with bilinear layers to capture multiplicative dynamics.
    
    Args:
        state_dim: Dimension of state vector (e.g., 40 for L96)
        bilinear_channels: Channels per branch in bilinear layer (default: 24)
        conv3_channels: Channels in conv layer 3 (default: 37)
        kernel_size: Convolution kernel size (default: 5)
    """
    
    def __init__(self, state_dim=40, bilinear_channels=24, 
                 conv3_channels=37, kernel_size=5):
        super().__init__()
        
        self.state_dim = state_dim
        pad = kernel_size // 2
        
        # Layer 1: Batch normalization on input
        self.bn_input = nn.BatchNorm1d(1)
        
        # Layer 2: Bilinear convolutional block
        # Output: bilinear_channels * 2 (from concat of linear + bilinear paths)
        self.bilinear = BilinearConvBlock(1, bilinear_channels, kernel_size)
        bilinear_out_channels = bilinear_channels * 2  # 48
        
        # Layer 3: Standard convolution
        self.conv3 = nn.Conv1d(bilinear_out_channels, conv3_channels, kernel_size,
                               padding=pad, padding_mode='circular')
        
        # Layer 4: 1x1 convolution to output (with L2 reg applied externally)
        self.conv4 = nn.Conv1d(conv3_channels, 1, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass computing x_{k+1} = x_k + f_nn(x_k)
        
        Args:
            x: State tensor of shape (batch, state_dim) or (batch, 1, state_dim)
            
        Returns:
            Next state prediction of shape (batch, state_dim)
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, state_dim)
        
        identity = x  # For residual connection
        
        # Layer 1: Batch norm
        h = self.bn_input(x)
        
        # Layer 2: Bilinear convolution
        h = self.bilinear(h)
        
        # Layer 3: Conv + ReLU
        h = F.relu(self.conv3(h))
        
        # Layer 4: 1x1 conv (linear activation)
        h = self.conv4(h)
        
        # Residual connection
        out = identity + h
        
        return out.squeeze(1)  # (batch, state_dim)
    
    def get_l2_reg_loss(self, lambda_reg=1e-4):
        """Get L2 regularization loss for conv4 layer only."""
        l2_loss = 0
        for param in self.conv4.parameters():
            l2_loss += param.pow(2).sum()
        return lambda_reg * l2_loss
    
    def multi_step_forward(self, x, n_steps):
        """
        Roll out model for multiple steps.
        
        Args:
            x: Initial state (batch, state_dim)
            n_steps: Number of steps to predict
            
        Returns:
            Trajectory of shape (batch, n_steps+1, state_dim)
        """
        trajectory = [x]
        current = x
        for _ in range(n_steps):
            current = self.forward(current)
            trajectory.append(current)
        return torch.stack(trajectory, dim=1)


# Training loop example
def train_surrogate(model, train_data, epochs=20, batch_size=256, lr=0.01, l2_reg=1e-4):
    """
    Train the surrogate model.
    
    Args:
        model: BrajardSurrogateModel instance
        train_data: Tuple of (x_k, x_k_plus_1) tensors
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate for Adagrad
        l2_reg: L2 regularization coefficient for final layer
    """
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    
    x_data, y_data = train_data  # (N, state_dim) each
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(x_batch)
            
            # MSE loss + L2 regularization on final layer
            mse_loss = F.mse_loss(pred, y_batch)
            reg_loss = model.get_l2_reg_loss(l2_reg)
            loss = mse_loss + reg_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    
    return model


# Usage example for Lorenz 96 (dim=40)
if __name__ == "__main__":
    # Create model
    model = BrajardSurrogateModel(state_dim=40)
    
    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params}")  # Should be ~9389
    
    # Test forward pass
    x = torch.randn(32, 40)  # Batch of 32 states
    x_next = model(x)
    print(f"Input shape: {x.shape}, Output shape: {x_next.shape}")
```

---

## 6. Adapting for Higher Dimensions / Your Use Case

### 6.1 Scaling to Higher Dimensions (e.g., Lorenz 96 with D > 40)

For larger state dimensions:
1. The convolutional architecture naturally scales (kernel is local)
2. May need more channels in intermediate layers
3. Consider kernel size relative to system's locality scale

### 6.2 Using as Proposal Model in Particle Filtering

To use this architecture as a **learned proposal distribution** in your flow-based particle filter:

1. **Train on transition data**: Pairs of (x_k, x_{k+1}) from your dynamics
2. **Condition on observations**: Modify architecture to include observation conditioning:
   - FiLM: Add affine transformation conditioned on y
   - AdaLN: Adaptive layer norm conditioned on y
   - Cross-attention: Attend to observation features

```python
class ConditionalSurrogate(BrajardSurrogateModel):
    """Surrogate conditioned on observations for proposal generation."""
    
    def __init__(self, state_dim, obs_dim, **kwargs):
        super().__init__(state_dim, **kwargs)
        
        # FiLM conditioning: map observations to scale/shift
        self.film_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim * 2)  # gamma and beta
        )
    
    def forward(self, x, y_obs):
        # Get FiLM parameters
        film_params = self.film_net(y_obs)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # ... apply FiLM after batch norm or intermediate layers
```

### 6.3 Key Takeaways for Implementation

1. **Residual structure is essential**: The skip connection from input to output
2. **Bilinear layer matters**: Captures the multiplicative dynamics common in physical systems
3. **Circular padding required**: For periodic boundary conditions
4. **One-step prediction**: Train on (x_k → x_{k+1}), not the ODE derivative
5. **Careful initialization**: Use interpolated data or pretrain before joint optimization

---

## 7. References

- Brajard, J., Carrassi, A., Bocquet, M., & Bertino, L. (2020). Combining data assimilation and machine learning to emulate a dynamical model from sparse and noisy observations: A case study with the Lorenz 96 model. *Journal of Computational Science*, 44, 101171.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *CVPR*.
- Code repository: https://github.com/brajard/GMD-code