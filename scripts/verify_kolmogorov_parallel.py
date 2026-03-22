import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def find_kolmogorov_dataset_npz(base_dir: Path) -> Path:
    """Pick the per-`obs_grid_size` dataset npz (exclude the cached PDE rollout).

    `generate.py` now writes:
      - a shared cached PDE rollout: `kolmogorov_pde_*/kolmogorov_data.npz`
      - a per-dataset symlink/copy: `kolmogorov_n*/kolmogorov_data.npz`
    """
    matches = list(base_dir.rglob("kolmogorov_data.npz"))
    dataset_matches = [p for p in matches if "kolmogorov_n" in p.as_posix() and "kolmogorov_pde_" not in p.as_posix()]
    if len(dataset_matches) != 1:
        raise RuntimeError(
            f"Expected exactly 1 per-dataset kolmogorov_data.npz under {base_dir}, "
            f"found {len(dataset_matches)} (total kolmogorov_data.npz found: {len(matches)})"
        )
    return dataset_matches[0]


def run_generation(generate_py: Path, out_dir: Path, parallel_gpus: int, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(generate_py),
        "--system",
        "kolmogorov",
        "--num-trajectories",
        "8",
        "--kol-grid-size",
        "32",
        "--kol-num-steps",
        "10",
        "--kol-warmup-steps",
        "5",
        "--kol-dt",
        "0.04",
        "--re-min",
        "500",
        "--re-max",
        "520",
        "--obs-frequency",
        "2",
        "--obs-grid-size",
        "8",
        "--obs-noise-std",
        "0.0",
        "--train-ratio",
        "0.6",
        "--val-ratio",
        "0.2",
        "--test-ratio",
        "0.2",
        "--seed",
        str(seed),
        "--output-dir",
        str(out_dir),
        "--kol-parallel-gpus",
        str(parallel_gpus),
        "--force",
    ]
    subprocess.run(cmd, cwd=str(generate_py.parent), check=True)
    npz_path = find_kolmogorov_dataset_npz(out_dir)
    return npz_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    generate_py = repo_root / "generate.py"

    run_root = Path("/tmp") / f"kolmogorov_verify_{int(time.time())}"
    out_serial = run_root / "serial"
    out_parallel1 = run_root / "parallel_2g_1"
    out_parallel2 = run_root / "parallel_2g_2"

    print(f"Running verification in: {run_root}")

    y_serial_path = run_generation(generate_py, out_serial, parallel_gpus=1, seed=args.seed)
    y_par1_path = run_generation(generate_py, out_parallel1, parallel_gpus=2, seed=args.seed)
    y_par2_path = run_generation(generate_py, out_parallel2, parallel_gpus=2, seed=args.seed)

    y_serial = np.load(y_serial_path, allow_pickle=True)["y"].astype(np.float32)
    y_par1 = np.load(y_par1_path, allow_pickle=True)["y"].astype(np.float32)
    y_par2 = np.load(y_par2_path, allow_pickle=True)["y"].astype(np.float32)

    u_serial = np.load(y_serial_path, allow_pickle=True)["u"].astype(np.float32)
    u_par1 = np.load(y_par1_path, allow_pickle=True)["u"].astype(np.float32)
    u_par2 = np.load(y_par2_path, allow_pickle=True)["u"].astype(np.float32)

    def assert_close(a, b, name_a: str, name_b: str, rtol: float, atol: float):
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            max_abs = float(np.max(np.abs(a - b)))
            max_rel = float(np.max(np.abs(a - b) / (np.abs(a) + 1e-12)))
            raise AssertionError(
                f"Mismatch between {name_a} and {name_b}: allclose failed "
                f"(max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, rtol={rtol}, atol={atol})"
            )

    # Reproducibility contract (schedule-independent): parallel run twice should match closely.
    assert_close(
        y_par1,
        y_par2,
        "y_parallel_run1",
        "y_parallel_run2",
        rtol=1e-5,
        atol=1e-6,
    )
    assert_close(
        u_par1,
        u_par2,
        "u_parallel_run1",
        "u_parallel_run2",
        rtol=0.0,
        atol=0.0,
    )

    # Serial vs parallel: allow slightly looser tolerance if GPU nondeterminism exists.
    assert_close(
        y_serial,
        y_par1,
        "y_serial",
        "y_parallel",
        rtol=1e-4,
        atol=1e-5,
    )
    assert_close(
        u_serial,
        u_par1,
        "u_serial",
        "u_parallel",
        rtol=0.0,
        atol=0.0,
    )

    print("Kolmogorov parallel verification PASSED.")
    print(f"serial npz:    {y_serial_path}")
    print(f"parallel npz1: {y_par1_path}")
    print(f"parallel npz2: {y_par2_path}")


if __name__ == "__main__":
    main()

