"""Analyze the hyperparameter sweep CSV produced by
`scripts/tune_rf_proposal_steps_grid.sbatch`.

Produces:
  1) Text report: top-k by RMSE and by CRPS, and summary of failed runs.
  2) Per-metric "performance vs likelihood_steps" plots, one subplot per
     sampling_steps, with one colored line per (sampling_grid, likelihood_grid)
     pair. Error bars use the std columns emitted by eval.py's aggregated
     output.
  3) Per-pair heatmaps over the (sampling_steps, likelihood_steps) grid for
     each metric (RMSE, CRPS, ESS, wall_time_sec) so the Pareto frontier is
     easy to eyeball.
  4) Pareto scatter: wall_time_sec vs RMSE, colored by pair.

Usage:
  python scripts/analyze_tune_rf_steps.py \
      --csv   logs/tune_rf_steps_8176812/summary.csv \
      --out   figures/tune_rf_steps_8176812 \
      [--top-k 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PAIR_ORDER = [
    ("uniform",    "uniform"),
    ("uniform",    "cosine"),
    ("uniform",    "power_back"),
    ("cosine",     "cosine"),
    ("power_back", "power_back"),
]
PAIR_COLORS = {
    ("uniform",    "uniform"):    "#1f77b4",  # blue
    ("uniform",    "cosine"):     "#2ca02c",  # green
    ("uniform",    "power_back"): "#9467bd",  # purple
    ("cosine",     "cosine"):     "#ff7f0e",  # orange
    ("power_back", "power_back"): "#d62728",  # red
}
PAIR_MARKERS = {
    ("uniform",    "uniform"):    "o",
    ("uniform",    "cosine"):     "s",
    ("uniform",    "power_back"): "D",
    ("cosine",     "cosine"):     "^",
    ("power_back", "power_back"): "v",
}


def pair_label(samp_grid: str, lik_grid: str) -> str:
    return f"samp={samp_grid} / lik={lik_grid}"


def load_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split into valid vs failed (blank-metric) rows."""
    df = pd.read_csv(csv_path)
    # Coerce numerics; blanks become NaN.
    for col in ("mean_rmse", "std_rmse", "mean_crps", "std_crps",
                "mean_ess", "wall_time_sec"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    failed = df[df["mean_rmse"].isna()].copy()
    ok = df[~df["mean_rmse"].isna()].copy()
    ok = ok.sort_values(["sampling_grid", "likelihood_grid",
                         "sampling_steps", "likelihood_steps"]).reset_index(drop=True)
    return ok, failed


def report_top_k(ok: pd.DataFrame, metric: str, k: int, ascending: bool) -> str:
    cols = ["sampling_grid", "likelihood_grid", "sampling_steps",
            "likelihood_steps", metric, "wall_time_sec", "task_id"]
    sub = ok.sort_values(metric, ascending=ascending).head(k)[cols]
    return sub.to_string(index=False, float_format=lambda x: f"{x:.4f}")


def report_bottom_k(ok: pd.DataFrame, metric: str, k: int) -> str:
    return report_top_k(ok, metric, k, ascending=False)


def plot_metric_vs_steps(
    ok: pd.DataFrame,
    metric_mean: str,
    metric_std: str | None,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """One figure per metric. Rows = sampling_steps, single column.

    Each subplot: x = likelihood_steps, y = metric_mean, colored lines per pair.
    """
    samp_steps_vals = sorted(ok["sampling_steps"].unique())
    n_rows = len(samp_steps_vals)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(8, 3.0 * n_rows + 0.5), sharex=True, sharey=True
    )
    if n_rows == 1:
        axes = [axes]

    for ax, s_samp in zip(axes, samp_steps_vals):
        sub = ok[ok["sampling_steps"] == s_samp]
        for pair in PAIR_ORDER:
            samp_grid, lik_grid = pair
            grp = sub[(sub["sampling_grid"] == samp_grid) &
                      (sub["likelihood_grid"] == lik_grid)]
            if grp.empty:
                continue
            grp = grp.sort_values("likelihood_steps")
            x = grp["likelihood_steps"].to_numpy()
            y = grp[metric_mean].to_numpy()
            color = PAIR_COLORS[pair]
            marker = PAIR_MARKERS[pair]
            if metric_std is not None and metric_std in grp.columns:
                yerr = grp[metric_std].to_numpy()
                ax.errorbar(x, y, yerr=yerr, color=color, marker=marker,
                            label=pair_label(samp_grid, lik_grid),
                            capsize=3, lw=1.5, ms=6, alpha=0.9)
            else:
                ax.plot(x, y, color=color, marker=marker,
                        label=pair_label(samp_grid, lik_grid),
                        lw=1.5, ms=6, alpha=0.9)
        ax.set_title(f"sampling_steps = {s_samp}", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("likelihood_steps")
    # One legend outside the top.
    handles = [
        Line2D([0], [0], color=PAIR_COLORS[p], marker=PAIR_MARKERS[p],
               lw=1.5, ms=6, label=pair_label(*p))
        for p in PAIR_ORDER
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0 - 0.02 / n_rows),
               ncol=3, frameon=False)
    fig.suptitle(title, y=1.005, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_pair_heatmaps(
    ok: pd.DataFrame,
    metric: str,
    cmap: str,
    title: str,
    out_path: Path,
    lower_is_better: bool = True,
) -> None:
    """One row per pair, one column for each metric. Here we generate one
    heatmap per pair (single metric). The grid is (sampling_steps rows,
    likelihood_steps cols). Value is metric mean, annotated in each cell.
    """
    present_pairs = [
        p for p in PAIR_ORDER
        if not ok[(ok["sampling_grid"] == p[0]) & (ok["likelihood_grid"] == p[1])].empty
    ]
    n_pairs = len(present_pairs)
    if n_pairs == 0:
        return

    samp_steps_vals = sorted(ok["sampling_steps"].unique())
    lik_steps_vals  = sorted(ok["likelihood_steps"].unique())

    # Determine a shared color scale so cross-pair comparison is honest.
    vals = ok[metric].dropna().to_numpy()
    vmin, vmax = float(np.min(vals)), float(np.max(vals))

    fig, axes = plt.subplots(
        1, n_pairs, figsize=(3.2 * n_pairs + 1.0, 3.5), sharey=True
    )
    if n_pairs == 1:
        axes = [axes]

    for ax, pair in zip(axes, present_pairs):
        samp_grid, lik_grid = pair
        sub = ok[(ok["sampling_grid"] == samp_grid) &
                 (ok["likelihood_grid"] == lik_grid)]
        # Build matrix indexed [row=samp_steps, col=lik_steps]
        mat = np.full((len(samp_steps_vals), len(lik_steps_vals)), np.nan)
        for _, row in sub.iterrows():
            i = samp_steps_vals.index(int(row["sampling_steps"]))
            j = lik_steps_vals.index(int(row["likelihood_steps"]))
            mat[i, j] = row[metric]
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", aspect="auto")
        ax.set_xticks(range(len(lik_steps_vals)), lik_steps_vals)
        ax.set_yticks(range(len(samp_steps_vals)), samp_steps_vals)
        ax.set_xlabel("likelihood_steps")
        if ax is axes[0]:
            ax.set_ylabel("sampling_steps")
        ax.set_title(pair_label(samp_grid, lik_grid), fontsize=9)

        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    txt = "—"
                else:
                    txt = f"{v:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="black")

        # Highlight best cell in the pair
        finite = np.where(np.isfinite(mat))
        if finite[0].size > 0:
            vals_ok = mat[finite]
            best = np.argmin(vals_ok) if lower_is_better else np.argmax(vals_ok)
            bi, bj = finite[0][best], finite[1][best]
            ax.add_patch(plt.Rectangle(
                (bj - 0.5, bi - 0.5), 1, 1,
                fill=False, edgecolor="lime", lw=2.5))

    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label(metric)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(ok: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for pair in PAIR_ORDER:
        samp_grid, lik_grid = pair
        sub = ok[(ok["sampling_grid"] == samp_grid) &
                 (ok["likelihood_grid"] == lik_grid)]
        if sub.empty:
            continue
        ax.scatter(
            sub["wall_time_sec"], sub["mean_rmse"],
            color=PAIR_COLORS[pair], marker=PAIR_MARKERS[pair],
            s=55, alpha=0.85, label=pair_label(samp_grid, lik_grid),
            edgecolors="black", linewidths=0.4,
        )
    ax.set_xlabel("wall_time_sec  (one PF run, 10 trajs x 100 steps)")
    ax.set_ylabel("mean_rmse")
    ax.set_title("Speed / accuracy trade-off (one dot per (grid_pair, s_samp, s_lik))")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True,
                    help="Path to the summary.csv produced by the tuning sweep.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Directory to write plots and text report.")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ok, failed = load_csv(args.csv)

    print(f"Loaded {len(ok)} valid rows, {len(failed)} failed rows from {args.csv}")
    print()
    if len(failed) > 0:
        print("=== Failed / blank-metric runs ===")
        print(failed[["task_id", "sampling_grid", "likelihood_grid",
                      "sampling_steps", "likelihood_steps",
                      "wall_time_sec"]].to_string(index=False))
        print()

    print(f"=== Top {args.top_k} by mean_rmse (lower is better) ===")
    print(report_top_k(ok, "mean_rmse", args.top_k, ascending=True))
    print()
    print(f"=== Top {args.top_k} by mean_crps (lower is better) ===")
    print(report_top_k(ok, "mean_crps", args.top_k, ascending=True))
    print()
    print(f"=== Top {args.top_k} by mean_ess (higher is better) ===")
    print(report_bottom_k(ok, "mean_ess", args.top_k))
    print()

    # Per-pair mean across the whole step grid (helps summarize "which pair is
    # best on average" independent of step choice).
    print("=== Per-pair averages across the (ss, ls) grid ===")
    agg = ok.groupby(["sampling_grid", "likelihood_grid"]).agg(
        rmse_mean=("mean_rmse", "mean"),
        rmse_std =("mean_rmse", "std"),
        crps_mean=("mean_crps", "mean"),
        crps_std =("mean_crps", "std"),
        ess_mean =("mean_ess",  "mean"),
        wall_mean=("wall_time_sec", "mean"),
        n_cells  =("mean_rmse", "count"),
    ).reset_index().sort_values("rmse_mean")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # --- Plots ---
    plot_metric_vs_steps(
        ok, "mean_rmse", "std_rmse", ylabel="mean RMSE",
        title="RMSE vs likelihood_steps (rows = sampling_steps; error bars = std across trajectories)",
        out_path=args.out / "rmse_vs_likelihood_steps.png",
    )
    plot_metric_vs_steps(
        ok, "mean_crps", "std_crps", ylabel="mean CRPS",
        title="CRPS vs likelihood_steps",
        out_path=args.out / "crps_vs_likelihood_steps.png",
    )
    plot_metric_vs_steps(
        ok, "mean_ess", None, ylabel="mean ESS",
        title="ESS vs likelihood_steps",
        out_path=args.out / "ess_vs_likelihood_steps.png",
    )

    plot_per_pair_heatmaps(
        ok, "mean_rmse", cmap="viridis_r",
        title="mean RMSE heatmaps per (samp_grid, lik_grid) pair  (lime box = best in pair)",
        out_path=args.out / "heatmap_rmse.png", lower_is_better=True,
    )
    plot_per_pair_heatmaps(
        ok, "mean_crps", cmap="viridis_r",
        title="mean CRPS heatmaps per (samp_grid, lik_grid) pair",
        out_path=args.out / "heatmap_crps.png", lower_is_better=True,
    )
    plot_per_pair_heatmaps(
        ok, "mean_ess", cmap="viridis",
        title="mean ESS heatmaps per (samp_grid, lik_grid) pair",
        out_path=args.out / "heatmap_ess.png", lower_is_better=False,
    )
    plot_per_pair_heatmaps(
        ok, "wall_time_sec", cmap="magma",
        title="wall_time_sec heatmaps per (samp_grid, lik_grid) pair  (lower = faster)",
        out_path=args.out / "heatmap_wall_time.png", lower_is_better=True,
    )

    plot_pareto(ok, args.out / "pareto_walltime_vs_rmse.png")

    print(f"Plots written to {args.out}/")
    for p in sorted(args.out.glob("*.png")):
        print("  ", p)


if __name__ == "__main__":
    main()
