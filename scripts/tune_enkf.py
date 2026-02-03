import argparse
import logging
import os
import sys
import json
import itertools
from pathlib import Path
import numpy as np
import torch
import csv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    DataAssimilationConfig,
    DataAssimilationDataModule,
    KuramotoSivashinsky,
    Lorenz63,
    Lorenz96,
    DoubleWell,
    load_config_yaml,
)
from models.enkf import EnsembleKalmanFilter, LocalEnsembleTransformKalmanFilter
from models.proposals import TransitionProposal
from eval import run_batched_eval, run_sequential_eval, aggregate_metrics_across_trajectories

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for EnKF/LETKF")
    
    # Dataset/DataModule configuration
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-eval-trajectories", type=int, default=1, help="Number of trajectories to evaluate on (subset for speed)")
    
    # Method configuration
    parser.add_argument("--method", type=str, required=True, choices=["enkf", "letkf"], help="Method to tune")
    parser.add_argument("--n-particles", type=int, default=50)
    parser.add_argument("--process-noise-std", type=float, default=0.0)
    parser.add_argument("--init-mode", type=str, default="truth", choices=["truth", "climatology"], help="Initialization mode")
    
    # Tuning Grid
    parser.add_argument("--inflation-values", type=str, default="0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8", help="Comma-separated inflation values")
    parser.add_argument("--localization-values", type=str, default="1,2,3,4,5,6,7,8", help="Comma-separated localization radius values (LETKF only)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="tuning_results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

def parse_float_list(value: str) -> list:
    return [float(v) for v in value.split(",") if v.strip() != ""]

def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Setup Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Config & System
    data_dir = Path(args.data_dir)
    config_path = data_dir / "config.yaml"
    config = load_config_yaml(config_path)
    
    # Setup System
    if "J" in config.system_params:
        system_class = KuramotoSivashinsky
    elif "dim" in config.system_params or "F" in config.system_params:
        system_class = Lorenz96
    elif config.system_params.get("system_name") == "double_well":
        system_class = DoubleWell
    else:
        system_class = Lorenz63
    
    system = system_class(config)
    
    # Load Data
    data_module = DataAssimilationDataModule(
        config=config,
        system_class=system_class,
        data_dir=str(data_dir),
        batch_size=args.batch_size,
    )
    data_module.setup("test")
    
    # Limit evaluation trajectories
    if args.num_eval_trajectories is not None:
        test_dataset = data_module.test_dataset
        if args.num_eval_trajectories < test_dataset.n_trajectories:
            logging.info(f"Limiting evaluation to first {args.num_eval_trajectories} trajectories")
            test_dataset.items = [item for item in test_dataset.items if item[0] < args.num_eval_trajectories]
            test_dataset.n_trajectories = args.num_eval_trajectories
            
    # System normalization stats
    system = data_module.system
    
    # Define Grid
    inflation_grid = parse_float_list(args.inflation_values)
    
    if args.method == "letkf":
        localization_grid = parse_float_list(args.localization_values)
    else:
        localization_grid = [None] # Not used for EnKF
        
    # Results storage
    results = []
    
    # Iterate
    combinations = list(itertools.product(inflation_grid, localization_grid))
    logging.info(f"Starting tuning for {args.method} with {len(combinations)} combinations...")
    
    best_rmse = float("inf")
    best_config = None
    
    # Dummy proposal (EnKF doesn't use it but eval script might expect it passed)
    proposal = TransitionProposal(system, process_noise_std=args.process_noise_std)
    obs_dim = len(config.obs_components)
    
    for i, (infl, loc) in enumerate(combinations):
        logging.info(f"[{i+1}/{len(combinations)}] Testing Inflation={infl}, Localization={loc}")
        
        # Run Evaluation
        # We construct a minimal args object for run_batched_eval
        class EvalArgs:
            batch_size = args.batch_size
            n_particles = args.n_particles
            process_noise_std = args.process_noise_std
            obs_noise_std = config.obs_noise_std
            device = args.device
            use_opt_weight_update = False # Not relevant for EnKF
            resampling_threshold = 0.5 # Not relevant
            save_dir = str(output_dir / "temp_trajs") # Dummy
            obs_frequency = config.obs_frequency # From config
            method = args.method
            inflation = infl
            localization_radius = loc
            init_mode = args.init_mode
            rf_checkpoint = None # Not used
            rf_likelihood_steps = None
            rf_sampling_steps = None
            
        eval_args = EvalArgs()
        
        # Suppress prints from eval
        # sys.stdout = open(os.devnull, 'w')
        try:
            # EnKF and LETKF are implemented as batched filters, so we use run_batched_eval
            # even if batch_size is 1.
            use_batched = (args.batch_size > 1) or (args.method in ["enkf", "letkf"])
            
            if use_batched:
                traj_results = run_batched_eval(
                    eval_args, config, system, data_module, obs_dim, proposal, wandb_run=None, vis_indices=set()
                )
            else:
                traj_results = run_sequential_eval(
                    eval_args, config, system, data_module, obs_dim, proposal, wandb_run=None, vis_indices=set()
                )
                
            metrics = aggregate_metrics_across_trajectories(traj_results)
            mean_rmse = metrics["mean_rmse_across_trajectories"]
            mean_crps = metrics.get("mean_crps_across_trajectories", float("nan"))
            
            logging.info(f"  -> RMSE: {mean_rmse:.4f}, CRPS: {mean_crps:.4f}")
            
            result_entry = {
                "inflation": infl,
                "localization": loc,
                "rmse": mean_rmse,
                "crps": mean_crps
            }
            results.append(result_entry)
            
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_config = result_entry
                logging.info(f"  -> New Best RMSE!")
                
        except Exception as e:
            logging.error(f"Evaluation failed for config {infl}, {loc}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # sys.stdout = sys.__stdout__
            pass

    # Save Results
    result_name = f"tune_{args.method}"
    if args.init_mode == "climatology":
        result_name += "_clim"
    csv_path = output_dir / f"{result_name}_results.csv"
    
    if results:
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
    
    logging.info("=" * 80)
    logging.info(f"Tuning Complete. Results saved to {csv_path}")
    if best_config:
        logging.info(f"Best Configuration:")
        logging.info(f"  Inflation: {best_config['inflation']}")
        if args.method == "letkf":
            logging.info(f"  Localization: {best_config['localization']}")
        logging.info(f"  RMSE: {best_config['rmse']:.4f}")
        
        # Save best config to JSON
        with open(output_dir / f"best_config_{args.method}.json", "w") as f:
            json.dump(best_config, f, indent=4)
    else:
        logging.warning("No successful evaluations found. Best configuration is None.")
    logging.info("=" * 80)

if __name__ == "__main__":
    main()

