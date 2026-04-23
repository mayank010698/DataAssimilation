#!/usr/bin/env python
"""Aggregate NeurIPS L96 grid results (BPF / APF / EnKF / FPPF-RF) into a single CSV.

Each per-method grid already produces a ``summary.csv`` under
``logs/eval_<method>_l96_neurips_<jobid>/``. This script reads the chosen
summaries, filters to the "canonical" config per method (e.g. EnKF with a
50-member ensemble), and writes a tidy CSV with a consistent schema:

    method, dim, obs_noise, obs_op,
    mean_rmse, std_rmse, mean_crps, std_crps,
    rmse_str, crps_str, n_particles, source

``rmse_str``/``crps_str`` are pre-formatted "mean ± std" strings handy for
dropping into paper tables.

Usage:
    python scripts/aggregate_l96_neurips_results.py \
        [--out evaluation_outputs/l96_neurips_aggregate.csv]

Override individual summary paths via --bpf-csv / --apf-csv / --enkf-csv /
--rf-csv if you rerun a grid and want to point at a new job directory.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCES: dict[str, dict] = {
    # method -> {csv, filter (predicate on row dict), n_particles_label}
    "BPF": {
        "csv": PROJECT_ROOT / "logs/eval_bpf_l96_neurips_8197261/summary.csv",
        "filter": lambda r: True,
    },
    "APF": {
        "csv": PROJECT_ROOT / "logs/eval_apf_l96_neurips_8192996/summary.csv",
        "filter": lambda r: True,
    },
    "EnKF": {
        # Skip the 20-member ensemble per user request; keep the 50-member one.
        "csv": PROJECT_ROOT / "logs/eval_enkf_l96_neurips_8193368/summary.csv",
        "filter": lambda r: int(r["n_particles"]) == 50,
    },
    "FPPF": {
        "csv": PROJECT_ROOT / "logs/eval_rf_l96_neurips_8183193/summary.csv",
        "filter": lambda r: True,
    },
}

OUT_COLUMNS = [
    "method",
    "dim",
    "obs_noise",
    "obs_op",
    "mean_rmse",
    "std_rmse",
    "mean_crps",
    "std_crps",
    "rmse_str",
    "crps_str",
    "n_particles",
    "source",
]


def _f(x: str) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(mean: Optional[float], std: Optional[float], digits: int = 4) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def read_summary(
    csv_path: Path,
    method: str,
    row_filter=lambda r: True,
) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary.csv for {method}: {csv_path}")

    rows: list[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not row_filter(r):
                continue
            mean_rmse = _f(r.get("mean_rmse", ""))
            std_rmse = _f(r.get("std_rmse", ""))
            mean_crps = _f(r.get("mean_crps", ""))
            std_crps = _f(r.get("std_crps", ""))
            if mean_rmse is None and mean_crps is None:
                # task crashed with no metrics; skip
                continue
            rows.append(
                {
                    "method": method,
                    "dim": int(r["dim"]),
                    "obs_noise": _f(r["obs_noise"]),
                    "obs_op": r["obs_op"],
                    "mean_rmse": mean_rmse,
                    "std_rmse": std_rmse,
                    "mean_crps": mean_crps,
                    "std_crps": std_crps,
                    "rmse_str": _fmt(mean_rmse, std_rmse),
                    "crps_str": _fmt(mean_crps, std_crps),
                    "n_particles": int(r["n_particles"]) if r.get("n_particles") else None,
                    "source": str(csv_path.relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def write_aggregate(rows: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in OUT_COLUMNS})


def write_pivot(rows: list[dict], methods: list[str], out_path: Path) -> None:
    """Side-by-side pivot: one row per (dim, obs_noise, obs_op), one pair of
    columns per method (``<method>_rmse`` and ``<method>_crps`` as mean±std
    strings). Missing cells stay empty."""
    by_cell: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        key = (r["dim"], r["obs_noise"], r["obs_op"])
        by_cell.setdefault(key, {})[r["method"]] = r

    header = ["dim", "obs_noise", "obs_op"]
    for m in methods:
        header += [f"{m}_rmse", f"{m}_crps"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key in sorted(by_cell):
            dim, noise, op = key
            row = [dim, noise, op]
            for m in methods:
                r = by_cell[key].get(m)
                row.append(r["rmse_str"] if r else "")
                row.append(r["crps_str"] if r else "")
            writer.writerow(row)


def print_coverage(rows: list[dict]) -> None:
    by_method: dict[str, set] = {}
    for r in rows:
        by_method.setdefault(r["method"], set()).add((r["dim"], r["obs_noise"], r["obs_op"]))
    expected_dims = [5, 10, 15, 20, 25, 50]
    expected_noises = [0.2, 0.5, 1.0, 3.0, 5.0]
    expected_ops = ["arctan", "quad_capped"]
    full_grid = {
        (d, n, o) for d in expected_dims for n in expected_noises for o in expected_ops
    }
    print(f"{'method':<6} {'cells':>6} {'missing':>8}  missing_cells")
    for method, cells in by_method.items():
        missing = sorted(full_grid - cells)
        if missing:
            missing_str = ", ".join(f"(d{d}, σ={n}, {o})" for d, n, o in missing[:5])
            if len(missing) > 5:
                missing_str += f", ... (+{len(missing) - 5} more)"
        else:
            missing_str = "-"
        print(f"{method:<6} {len(cells):>6} {len(missing):>8}  {missing_str}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation_outputs" / "l96_neurips_aggregate.csv",
    )
    p.add_argument(
        "--pivot-out",
        type=Path,
        default=PROJECT_ROOT / "evaluation_outputs" / "l96_neurips_aggregate_pivot.csv",
    )
    p.add_argument("--bpf-csv", type=Path, default=None)
    p.add_argument("--apf-csv", type=Path, default=None)
    p.add_argument("--enkf-csv", type=Path, default=None)
    p.add_argument("--rf-csv", type=Path, default=None)
    args = p.parse_args()

    overrides = {
        "BPF": args.bpf_csv,
        "APF": args.apf_csv,
        "EnKF": args.enkf_csv,
        "FPPF": args.rf_csv,
    }

    all_rows: list[dict] = []
    for method, cfg in DEFAULT_SOURCES.items():
        csv_path = overrides[method] or cfg["csv"]
        rows = read_summary(csv_path, method, cfg["filter"])
        rows.sort(key=lambda r: (r["dim"], r["obs_noise"], r["obs_op"]))
        print(f"[{method}] {len(rows)} rows  <- {csv_path.relative_to(PROJECT_ROOT)}")
        all_rows.extend(rows)

    # Stable order: method first, then (dim, obs_noise, obs_op).
    method_order = {m: i for i, m in enumerate(DEFAULT_SOURCES)}
    all_rows.sort(key=lambda r: (method_order[r["method"]], r["dim"], r["obs_noise"], r["obs_op"]))

    write_aggregate(all_rows, args.out)
    print(f"\nWrote {len(all_rows)} rows -> {args.out.relative_to(PROJECT_ROOT)}")

    methods = list(DEFAULT_SOURCES.keys())
    write_pivot(all_rows, methods, args.pivot_out)
    print(f"Wrote pivot       -> {args.pivot_out.relative_to(PROJECT_ROOT)}")
    print()
    print_coverage(all_rows)


if __name__ == "__main__":
    main()
