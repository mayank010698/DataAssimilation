import argparse
import sys
import os
import torch
import time
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress lightning logs
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("models").setLevel(logging.ERROR)

def setup_args():
    parser = argparse.ArgumentParser(description='Estimate GPU memory usage for BPF')
    parser.add_argument('--state_dim', type=int, required=True)
    parser.add_argument('--n_particles', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--device', type=int, default=0, help='GPU device index to use for estimation')
    return parser.parse_args()

def estimate_bpf_memory(args, device):
    from models.bpf import BootstrapParticleFilter
    from data import Lorenz96, DataAssimilationConfig

    # Config
    config = DataAssimilationConfig(
        system_params={"dim": args.state_dim, "F": 8},
        obs_components=list(range(args.state_dim)),
        obs_nonlinearity="arctan", # Standard for these datasets
        dt=0.01
    )
    
    # System
    system = Lorenz96(config)
    
    # Filter
    pf = BootstrapParticleFilter(
        system=system,
        proposal_distribution=None, # Transition proposal
        n_particles=args.n_particles,
        state_dim=args.state_dim,
        obs_dim=args.state_dim,
        device=device
    )
    
    # Dummy data
    x_prev = torch.randn(args.batch_size, args.state_dim).to(device)
    x_curr = torch.randn(args.batch_size, args.state_dim).to(device)
    y_curr = torch.randn(args.batch_size, args.state_dim).to(device)
    
    # Needed for step
    trajectory_idxs = torch.zeros(args.batch_size, dtype=torch.long).to(device)
    time_idxs = torch.ones(args.batch_size, dtype=torch.long).to(device)
    
    # Initialization
    pf.initialize_filter(x_prev)
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    
    # Run a few steps
    # We want to capture the peak memory during resampling/update
    # We force resampling by keeping weights weird? 
    # Actually random weights will trigger resampling eventually if threshold is high.
    # But step() checks threshold.
    
    # Run step
    for _ in range(5):
        # Generate new random data to prevent static graph optimizations (unlikely here but good practice)
        x_curr = torch.randn(args.batch_size, args.state_dim).to(device)
        y_curr = torch.randn(args.batch_size, args.state_dim).to(device)
        
        pf.step(x_prev, x_curr, y_curr, 0.01, trajectory_idxs, time_idxs)
        x_prev = x_curr

    return torch.cuda.max_memory_allocated(device)

def main():
    args = setup_args()
    
    if not torch.cuda.is_available():
        print("0")
        return

    device = torch.device(f"cuda:{args.device}")
    
    try:
        peak_mem = estimate_bpf_memory(args, device)
        # Convert to MiB
        peak_mem_mib = peak_mem / (1024 * 1024)
        print(f"{int(peak_mem_mib)}")
        
    except Exception as e:
        print(f"Error estimating memory: {e}", file=sys.stderr)
        # If OOM, return a huge number
        if "out of memory" in str(e).lower():
            print("999999")
        else:
            print("0")

if __name__ == "__main__":
    main()

