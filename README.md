# Data Assimilation with Particle Filters

## How to Run for Lorenz 96
* Use commands such as in `scripts/gen_96.sh` to generate datasets
* Use commands such as in `scripts/train_96.sh` to train RF proposals
* Use commands such as in `scripts/eval_proposal_96.sh` to optionally evaluate proposals in an autoregressive manner (like in FlowDAS)
* Use commands such as in `scripts/eval_96.sh` to run a BPF with a standard transition proposal or with learned flow proposal

## Project Structure

```
├── data.py                  # Core data generation
├── generate.py              # Data generation script
├── generate_l96.py          # Data generation for Lorenz 96
├── run.py                   # Main evaluation script
├── eval.py                  # Evaluation script
├── eval_proposal_dist.py    # Evaluation of proposal distributions
├── models/                  # Particle filter implementations
│   ├── base_pf.py          # Abstract base class
│   ├── bpf.py              # Bootstrap Particle Filter
│   ├── flow_pf.py          # Flow-based Particle Filter
│   ├── score_pf.py         # Score-based Particle Filter
│   └── proposals.py        # Proposal distribution classes
├── proposals/              # Proposal training and architectures
│   ├── architectures/      # Neural network architectures
│   │   ├── mlp.py
│   │   ├── resnet1d.py
│   │   └── ...
│   ├── rectified_flow.py   # Rectified Flow model
│   ├── deterministic_model.py # Deterministic model
│   ├── train_rf.py         # Training script for RF
│   ├── train_deterministic.py # Training script for deterministic models
│   └── ...
├── scripts/                # Shell scripts for running experiments
│   ├── gen_*.sh            # Data generation scripts
│   ├── train_*.sh          # Training scripts
│   ├── eval_*.sh           # Evaluation scripts
│   └── ...
```

## Features

### Dynamical System
- **Lorenz 63**: Classic chaotic system with configurable parameters (σ, ρ, β)
- **Preprocessing**: Optional state space normalization for improved numerical stability
- **Initial conditions**: Configurable sampling from multivariate Gaussian distributions

### Observation Models
- **Linear Projection**: Observing specific state components with projection matrices
- **Nonlinear Arctan**: Nonlinear observation operator h(x) = arctan(x)
- **Configurable noise**: Additive Gaussian noise with adjustable standard deviation
- **Variable frequency**: Configurable observation intervals

### Particle Filter Implementation
- **Bootstrap Particle Filter**: Importance sampling with resampling
- **Multiple proposals**: Support for different proposal distributions
- **Adaptive resampling**: ESS-based resampling decisions
- **Performance metrics**: RMSE, log-likelihood, ESS tracking

### Proposal Distributions
1. **TransitionProposal**: Uses system dynamics with process noise
2. **RectifiedFlowProposal**: Neural network learned from trajectory data

## Quick Start

### 1. Generate Data and Run Evaluation
```bash
python run.py
```
This will:
- Generate Lorenz 63 trajectory data for different configurations
- Train particle filters with various proposal types
- Create visualization plots comparing performance

### 2. Comprehensive Evaluation
```bash
python eval.py --proposal-type transition --observation-operator arctan --use-preprocessing
```

### 3. Train Rectified Flow Models
```bash
cd proposals
python train_rf.py --data_dir "../lorenz63_data_*" --output_dir "../rf_runs/model_name"
```

## Configuration Options

### Data Generation
- `num_trajectories`: Number of trajectory samples (default: 1024)
- `len_trajectory`: Length of each trajectory (default: 100)
- `dt`: Integration time step (default: 0.01)
- `obs_frequency`: Observation frequency (default: 2)
- `obs_noise_std`: Observation noise level (default: 0.25)

### Particle Filter
- `n_particles`: Number of particles (default: 100)
- `process_noise_std`: Process noise standard deviation (default: 0.25)
- `proposal_type`: "transition" or "rf"

### System Parameters
- `sigma`: Lorenz σ parameter (default: 10.0)
- `rho`: Lorenz ρ parameter (default: 28.0)  
- `beta`: Lorenz β parameter (default: 8/3)

## Experimental Configurations

The project includes predefined experimental setups:

1. **Linear Projection (Preprocessing ON/OFF)**
   - Observes state components [0, 2]
   - Uses linear projection matrix

2. **Arctan Nonlinear (Preprocessing ON/OFF)**
   - Observes component [0] with arctan transformation
   - Tests nonlinear observation operators

Each configuration can be run with either transition or rectified flow proposals.

## HPC Usage

### SLURM Job Submission
```bash
# Train all RF models
sbatch train.sbatch

# Run evaluations  
sbatch run.sbatch
```

The project is configured for GPU clusters with:
- A100 GPU support
- Conda environment: `da`
- Custom Python path setup

## Dependencies

- PyTorch + PyTorch Lightning
- NumPy, SciPy
- Matplotlib (visualization)
- H5py (data storage)
- Weights & Biases (logging)
- SLURM (HPC scheduling)

## Results

The project generates:
- **Performance plots**: 3D trajectories, state components, RMSE evolution
- **Metrics**: Mean RMSE, log-likelihood, effective sample size
- **Wandb logs**: Comprehensive experiment tracking
- **Checkpoints**: Trained RF models for reuse

## TODO

See `TODO.md` for current development tasks and known issues.
