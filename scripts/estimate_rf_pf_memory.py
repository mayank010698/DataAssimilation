import argparse
import sys
import os
import torch
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress lightning logs
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("models").setLevel(logging.ERROR)

def setup_args():
    parser = argparse.ArgumentParser(description='Estimate GPU memory usage for RF-BPF')
    parser.add_argument('--state_dim', type=int, required=True)
    parser.add_argument('--n_particles', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--device', type=int, default=0, help='GPU device index to use for estimation')
    return parser.parse_args()

def estimate_rf_bpf_memory(args, device):
    from models.bpf import BootstrapParticleFilter
    from models.proposals import RectifiedFlowProposal
    from data import Lorenz96, DataAssimilationConfig

    # Config - we need enough to initialize the system
    config = DataAssimilationConfig(
        system_params={"dim": args.state_dim, "F": 8},
        obs_components=list(range(args.state_dim)),
        obs_nonlinearity="arctan", 
        dt=0.01
    )
    
    # System
    system = Lorenz96(config)
    
    # Initialize RF Proposal
    # This loads the model weights which is a significant part of memory usage
    try:
        proposal = RectifiedFlowProposal(
            checkpoint_path=args.checkpoint_path,
            device=str(device), # RFProposal expects string device like 'cuda:0'
            system=system,
            # Use smaller steps for estimation speed, but enough to trigger allocations
            num_sampling_steps=10, 
            num_likelihood_steps=10,
            obs_components=list(range(args.state_dim))
        )
    except Exception as e:
        print(f"Error loading RF proposal: {e}", file=sys.stderr)
        raise e

    # Filter
    pf = BootstrapParticleFilter(
        system=system,
        proposal_distribution=proposal,
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
    
    # Run a step
    # This involves:
    # 1. Proposal sampling (RF generation) -> High memory
    # 2. Likelihood computation (RF likelihood) -> High memory (O(D^2) if exact trace, or just heavy backprop/grad if optimized, but here we are in inference mode)
    # Note: RF likelihood is computed in log_prob. 
    # BPF step calls sample() then log_prob() (for weight update if proposal != transition, wait, standard BPF uses proposal=transition usually, but here we are using RF as proposal).
    # If proposal is learned, BPF weight update is w_t = w_{t-1} * p(y|x) * p(x|x_{t-1}) / q(x|x_{t-1}, y).
    # If q is RF, we need to evaluate q.log_prob().
    
    # However, RFProposal wrapper handles log_prob.
    
    with torch.no_grad(): # Inference mode
        pf.step(x_prev, x_curr, y_curr, 0.01, trajectory_idxs, time_idxs)
        
    return torch.cuda.max_memory_allocated(device)

def main():
    args = setup_args()
    
    if not torch.cuda.is_available():
        print("0")
        return

    device = torch.device(f"cuda:{args.device}")
    
    try:
        peak_mem = estimate_rf_bpf_memory(args, device)
        # Convert to MiB
        peak_mem_mib = peak_mem / (1024 * 1024)
        print(f"{int(peak_mem_mib)}")
        
    except Exception as e:
        print(f"Error estimating memory: {e}", file=sys.stderr)
        # If OOM, return a huge number
        if "out of memory" in str(e).lower():
            print("999999")
        else:
            # Fallback high value if generic error
            print("999999")

if __name__ == "__main__":
    main()

