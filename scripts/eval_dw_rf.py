import subprocess
import sys
from pathlib import Path
import logging


def pick_checkpoint(checkpoint_dir: Path, preferred_name: str) -> Path:
    preferred = checkpoint_dir / preferred_name
    if preferred.exists():
        return preferred
    fallback = checkpoint_dir / "last.ckpt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def run_eval(data_dir: Path, rf_checkpoint: Path, n_particles: int, process_noise_std: float, run_name: str, num_eval_trajectories: int = 50):
    python_executable = "/home/cnagda/miniconda3/envs/da/bin/python"
    cmd = [
        python_executable, "eval.py",
        "--data-dir", str(data_dir),
        "--method", "bpf",
        "--proposal-type", "rf",
        "--rf-checkpoint", str(rf_checkpoint),
        "--n-particles", str(n_particles),
        "--process-noise-std", str(process_noise_std),
        "--batch-size", "100",
        "--num-eval-trajectories", str(num_eval_trajectories),
        "--wandb-project", "pf-eval-dw-rf",
        "--run-name", run_name,
        "--experiment-label", "dw_rf_eval",
        "--device", "cuda"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    logging.basicConfig(level=logging.INFO)

    base_data_dir = Path("datasets")
    base_rf_dir = Path("rf_runs")

    configs = [
        {
            "name": "dw_std",
            "data_dir": base_data_dir / "double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p200",
            "rf_dir": base_rf_dir / "dw_std_mlp" / "checkpoints",
            "preferred_ckpt": "rf-epoch=062-val_loss=0.136742.ckpt",
            "process_noise_std": 0.2,
        },
        {
            "name": "dw_large",
            "data_dir": base_data_dir / "double_well_n1000_len100_dt0p1000_obs0p100_freq1_comp0_identity_pnoise0p300",
            "rf_dir": base_rf_dir / "dw_large_mlp" / "checkpoints",
            "preferred_ckpt": "rf-epoch=029-val_loss=0.150516.ckpt",
            "process_noise_std": 0.3,
        },
    ]

    for cfg in configs:
        data_dir = cfg["data_dir"]
        rf_ckpt = pick_checkpoint(cfg["rf_dir"], cfg["preferred_ckpt"])
        run_name = f"bpf_rf_{cfg['name']}"
        run_eval(
            data_dir=data_dir,
            rf_checkpoint=rf_ckpt,
            n_particles=1000,
            process_noise_std=cfg["process_noise_std"],
            run_name=run_name,
        )


if __name__ == "__main__":
    main()

