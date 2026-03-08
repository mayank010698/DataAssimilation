import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT = "ml-climate/eval-96-prev-corr"


@dataclass
class RunRecord:
    run_id: str
    name: str
    state: str
    model_type: str
    p_corrupt: Optional[float]
    sigma: Optional[float]
    mask_ratio: Optional[float]
    p_min: Optional[float]
    mean_rmse: Optional[float]
    mean_crps: Optional[float]
    mean_obs_error: Optional[float]


def _parse_name(name: str) -> dict:
    """
    Parse corruption hyperparameters from run / RF checkpoint name.
    Examples (from sweep scripts):
      - l96_dbf_prevcorr_p030_sig050
      - l96_dbf_prevcorr_p020_sig050
      - l96_dbf_prevcorr_p030_sig025
      - l96_dbf_prevcorr_p030_mask020_only
      - l96_dbf_prevcorr_p030_mask040_only
      - l96_dbf_prevcorr_p030_sig050_mask020
      - l96_dbf_prevcorr_p030_sig050_mask040
      - l96_dbf_prevcorr_p030_sig100
      - l96_dbf_prevcorr_p030_sig075
      - l96_dbf_prevcorr_p030_mask030_only
      - l96_dbf_prevcorr_p030_sig050_mask030
      - l96_dbf_prevcorr_p020_sig025
      - l96_dbf_prevcorr_p030_sig050_pmin0
      - l96_dbf_prevcorr_p040_sig050
      - l96_dbf_prevcorr_p030_sig050_mask050
      - l96_dbf_prevcorr_p020_sig050_mask020
    """
    if name is None:
        return {"p_corrupt": None, "sigma": None, "mask_ratio": None, "p_min": None}

    parts = name.split("_")

    p_corrupt = None
    sigma = None
    mask_ratio = None
    p_min = None

    for p in parts:
        if p.startswith("p0") and len(p) == 4:
            # e.g. p020 -> 0.20
            try:
                p_corrupt = float(p[1:]) / 100.0
            except ValueError:
                pass
        elif p.startswith("sig") and len(p) == 6:
            # e.g. sig050, sig025, sig075, sig100
            try:
                sigma = float(p[3:]) / 100.0
            except ValueError:
                pass
        elif p.startswith("mask") and len(p) == 7:
            # e.g. mask020, mask040, mask030, mask050
            try:
                mask_ratio = float(p[4:]) / 100.0
            except ValueError:
                pass
        elif p.startswith("pmin"):
            # e.g. pmin0 -> 0.0
            try:
                p_min = float(p[4:]) / 10.0
            except ValueError:
                pass

    return {
        "p_corrupt": p_corrupt,
        "sigma": sigma,
        "mask_ratio": mask_ratio,
        "p_min": p_min,
    }


def _infer_model_type(run_name: str) -> str:
    # Matches eval_96_dbf_rf_prev_corr_sweep.sh convention:
    #   DATE_MODEL_LABEL_pnoise0p2
    # where MODEL in {enkf, ensf, ensf_rf}
    if run_name is None:
        return "unknown"
    parts = run_name.split("_")
    # Expect: YYYYMMDD, model, label, ...
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def fetch_runs() -> List[RunRecord]:
    api = wandb.Api()
    runs = api.runs(PROJECT)

    records: List[RunRecord] = []
    for r in runs:
        # Global metrics logged by eval loop
        s = dict(r.summary or {})
        # Eval script logs plain mean_* keys (not eval/global_*)
        rmse = s.get("mean_rmse")
        crps = s.get("mean_crps")
        obs_err = s.get("mean_obs_error")

        model_type = _infer_model_type(r.name or r.id)

        # Hyperparameters are mainly encoded in the LABEL part of the run name,
        # which for ensf_rf runs should match the RF checkpoint dir name.
        # Example: 20250301_ensf_rf_l96_dbf_prevcorr_p030_sig050_pnoise0p2
        label = None
        if r.name:
            name_parts = r.name.split("_")
            if len(name_parts) >= 3:
                # Join middle parts from index 2 to -1 (strip date + model + pnoise suffix)
                label = "_".join(name_parts[2:-1]) if len(name_parts) > 3 else name_parts[2]

        hp = _parse_name(label or r.name or r.id)

        rec = RunRecord(
            run_id=r.id,
            name=r.name or r.id,
            state=r.state,
            model_type=model_type,
            p_corrupt=hp["p_corrupt"],
            sigma=hp["sigma"],
            mask_ratio=hp["mask_ratio"],
            p_min=hp["p_min"],
            mean_rmse=float(rmse) if rmse is not None else None,
            mean_crps=float(crps) if crps is not None else None,
            mean_obs_error=float(obs_err) if obs_err is not None else None,
        )
        records.append(rec)

    return records


def to_dataframe(records: List[RunRecord]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in records])
    # Make sure numeric columns are numeric
    for col in ["p_corrupt", "sigma", "mask_ratio", "p_min", "mean_rmse", "mean_crps", "mean_obs_error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summarize_configs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate metrics per configuration and compare against baselines.
    Returns:
      - config_df: per-configuration metrics for ensf_rf runs
      - comparison_df: same table with deltas vs baselines
    """
    # Baselines (single runs)
    enkf = df[df["model_type"] == "enkf"].copy()
    ensf = df[(df["model_type"] == "ensf") & (~df["name"].str.contains("prevcorr", na=False))].copy()

    enkf_rmse = float(enkf["mean_rmse"].iloc[0]) if not enkf.empty and enkf["mean_rmse"].notna().any() else np.nan
    ensf_rmse = float(ensf["mean_rmse"].iloc[0]) if not ensf.empty and ensf["mean_rmse"].notna().any() else np.nan

    enkf_crps = float(enkf["mean_crps"].iloc[0]) if not enkf.empty and enkf["mean_crps"].notna().any() else np.nan
    ensf_crps = float(ensf["mean_crps"].iloc[0]) if not ensf.empty and ensf["mean_crps"].notna().any() else np.nan

    # Filter to RF runs
    rf_df = df[(df["model_type"] == "ensf") & (df["name"].str.contains("prevcorr", na=False))].copy()

    # Group by corruption hyperparameters
    group_cols = ["p_corrupt", "sigma", "mask_ratio", "p_min"]
    config_df = (
        rf_df.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            mean_rmse=("mean_rmse", "mean"),
            std_rmse=("mean_rmse", "std"),
            mean_crps=("mean_crps", "mean"),
            std_crps=("mean_crps", "std"),
        )
        .reset_index()
    )

    # Standard error (per configuration, over runs; trajectory-level variation is already inside mean_rmse)
    config_df["se_rmse"] = config_df["std_rmse"] / np.sqrt(config_df["n_runs"].clip(lower=1))
    config_df["se_crps"] = config_df["std_crps"] / np.sqrt(config_df["n_runs"].clip(lower=1))

    # Deltas vs baselines
    config_df["delta_rmse_vs_ensf"] = config_df["mean_rmse"] - ensf_rmse
    config_df["delta_rmse_vs_enkf"] = config_df["mean_rmse"] - enkf_rmse
    config_df["delta_crps_vs_ensf"] = config_df["mean_crps"] - ensf_crps
    config_df["delta_crps_vs_enkf"] = config_df["mean_crps"] - enkf_crps

    return config_df, config_df.copy()


def make_plots(config_df: pd.DataFrame, out_prefix: str = "eval_96_prev_corr"):
    """Simple factor-wise plots with error bars."""
    sns.set(style="whitegrid")

    # 1) RMSE vs p_corrupt
    if config_df["p_corrupt"].notna().any():
        plt.figure(figsize=(6, 4))
        sns.lineplot(
            data=config_df.sort_values("p_corrupt"),
            x="p_corrupt",
            y="mean_rmse",
            hue="mask_ratio",
            marker="o",
        )
        plt.ylabel("Mean RMSE")
        plt.title("Prev-corr RF: RMSE vs p_corrupt")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_rmse_vs_p_corrupt.png", dpi=150)
        plt.close()

    # 2) RMSE vs sigma
    if config_df["sigma"].notna().any():
        plt.figure(figsize=(6, 4))
        sns.lineplot(
            data=config_df.sort_values("sigma"),
            x="sigma",
            y="mean_rmse",
            hue="mask_ratio",
            marker="o",
        )
        plt.ylabel("Mean RMSE")
        plt.title("Prev-corr RF: RMSE vs sigma")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_rmse_vs_sigma.png", dpi=150)
        plt.close()

    # 3) Bar plot by mask_ratio
    if config_df["mask_ratio"].notna().any():
        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=config_df,
            x="mask_ratio",
            y="mean_rmse",
        )
        plt.ylabel("Mean RMSE")
        plt.title("Prev-corr RF: RMSE vs mask_ratio")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_rmse_vs_mask_ratio.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-csv",
        type=str,
        default="eval_96_prev_corr_runs.csv",
        help="Path for the raw runs CSV export.",
    )
    parser.add_argument(
        "--output-config-csv",
        type=str,
        default="eval_96_prev_corr_configs.csv",
        help="Path for the aggregated per-configuration CSV.",
    )
    args = parser.parse_args()

    print(f"Fetching runs from project: {PROJECT}")
    records = fetch_runs()
    df = to_dataframe(records)
    print(f"Fetched {len(df)} runs.")

    df.to_csv(args.output_csv, index=False)
    print(f"Wrote runs table to {args.output_csv}")

    config_df, _ = summarize_configs(df)
    config_df.to_csv(args.output_config_csv, index=False)
    print(f"Wrote per-configuration metrics to {args.output_config_csv}")

    make_plots(config_df, out_prefix="eval_96_prev_corr")
    print("Saved factor-wise RMSE plots (p_corrupt, sigma, mask_ratio).")


if __name__ == "__main__":
    main()

