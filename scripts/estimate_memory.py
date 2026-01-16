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

def setup_args():
    parser = argparse.ArgumentParser(description='Estimate GPU memory usage for training')
    parser.add_argument('--model_type', type=str, required=True, choices=['rf', 'flowdas'])
    parser.add_argument('--state_dim', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=10)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--device', type=int, default=0, help='GPU device index to use for estimation')
    return parser.parse_args()

def estimate_rf_memory(args, device):
    from proposals.rectified_flow import RFProposal
    
    # Initialize model
    model = RFProposal(
        state_dim=args.state_dim,
        architecture='resnet1d',
        channels=args.channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        train_cond_method='adaln',
        cond_embed_dim=128,
        obs_dim=args.state_dim, # Assuming full observation for worst case/standard case
        obs_indices=list(range(args.state_dim))
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data
    # RFProposal expects x_prev, x_curr, y_curr
    x_prev = torch.randn(args.batch_size, args.state_dim).to(device)
    x_curr = torch.randn(args.batch_size, args.state_dim).to(device)
    y_curr = torch.randn(args.batch_size, args.state_dim).to(device) # Obs dim same as state dim for now
    
    batch = {
        'x_prev': x_prev,
        'x_curr': x_curr,
        'y_curr': y_curr
    }
    
    model.train()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    
    # Run a few steps
    for _ in range(5):
        optimizer.zero_grad()
        loss = model.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        
    return torch.cuda.max_memory_allocated(device)

def estimate_flowdas_memory(args, device):
    from models.flowdas import ScoreNet, loss_fn, prepare_batch
    
    # Initialize model
    # Note: ScoreNet initialization args in train_flowdas.py:
    # marginal_prob_std=None, x_dim=state_dim, extra_dim=state_dim, ...
    model = ScoreNet(
        marginal_prob_std=None,
        x_dim=args.state_dim,
        extra_dim=args.state_dim, # Condition on x_prev
        architecture='resnet1d',
        channels=args.channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        obs_dim=args.state_dim,
        obs_indices=list(range(args.state_dim)),
        embed_dim=128, # Matching width arg default
        use_bn=False # Default in parse_args is False for use_bn
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data for FlowDAS
    # prepare_batch expects dict with x_prev, x_curr (and optionally scaled versions)
    # It converts them to z0, z1
    
    batch_raw = {
        'x_prev': torch.randn(args.batch_size, args.state_dim),
        'x_curr': torch.randn(args.batch_size, args.state_dim),
        'y_curr': torch.randn(args.batch_size, args.state_dim)
    }
    
    model.train()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    
    # Run a few steps
    for _ in range(5):
        # Prepare batch moves to device and creates SI targets
        batch_si = prepare_batch(batch_raw, device=device)
        
        optimizer.zero_grad()
        loss, _ = loss_fn(model, batch_si)
        loss.backward()
        optimizer.step()
        
    return torch.cuda.max_memory_allocated(device)

def main():
    args = setup_args()
    
    if not torch.cuda.is_available():
        print("0") # Return 0 if no GPU
        return

    device = torch.device(f"cuda:{args.device}")
    
    try:
        if args.model_type == 'rf':
            peak_mem = estimate_rf_memory(args, device)
        elif args.model_type == 'flowdas':
            peak_mem = estimate_flowdas_memory(args, device)
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
            
        # Convert to MiB
        peak_mem_mib = peak_mem / (1024 * 1024)
        print(f"{int(peak_mem_mib)}")
        
    except Exception as e:
        # In case of OOM or other error, print error to stderr and 0 to stdout or handle in caller
        print(f"Error estimating memory: {e}", file=sys.stderr)
        print("0")

if __name__ == "__main__":
    main()

