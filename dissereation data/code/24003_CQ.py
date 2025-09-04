#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C–Q Regression (log(C) ~ log(Q)) Pipeline – for 24003 downstream station
- Robust to zero-variance groups
- Adds RMSE & MAE on log10-scale residuals
- Writes per-group figures, CSVs, and a markdown summary
"""

from __future__ import annotations
import re
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# --- Suppress statsmodels runtime warnings for near-zero variance ---
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)

# Try importing statsmodels; fallback to numpy.polyfit if unavailable
try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False
    warnings.warn("statsmodels not installed; falling back to numpy.polyfit (no p-values)")

# ===================== CONFIGURATION ===================== #
CONFIG = {
  
    "parquet_path": "24003_Flow–Concentration Merging.parquet",
    "columns": {
        "station": "sample.samplingPoint.notation",
        "datetime": "sample.sampleDateTime",
        "metal": "metal",
        "fraction": "fraction",
        "conc": "result",
        "flow": "flow"
    },

    # Filter settings
    "allowed_metals": ["zinc", "iron", "cadmium", "lead", "copper", "nickel", "aluminium"],
    "fractions_keep": None,  # e.g., ["total", "dissolved"]

    # Modelling parameters
    "min_points": 10,                 # Minimum required points per group
    "trim_quantiles": (0.01, 0.99),   # Trim extreme conc/flow values by quantile
    "chemostatic_eps": 0.10,          # |slope| ≤ eps → chemostatic
    "var_eps": 1e-12,                 # Variance threshold to detect zero-variance groups


    "out_dir": ""
}
# ========================================================== #

def _ensure_dirs(base: Path) -> Tuple[Path, Path]:
    out_base = base
    figs_dir = out_base / "figs"
    out_base.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    return out_base, figs_dir

def _build_renamer(cols_map: Dict[str, str], df_cols: List[str]) -> Dict[str, str]:
    return {v: k for k, v in cols_map.items() if v in df_cols and v is not None}

def _classify_trend(slope: float, eps: float) -> str:
    if pd.isna(slope): return "insufficient"
    if slope < -eps: return "dilution"
    if slope > eps: return "mobilisation"
    return "chemostatic"

def _safe_log10(series: pd.Series) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        arr = np.log10(series.to_numpy(dtype=float))
    return pd.Series(arr, index=series.index)

def _trim_extremes(g: pd.DataFrame, q: Tuple[float, float]) -> pd.DataFrame:
    ql, qh = q
    c_low, c_high = g['conc'].quantile([ql, qh])
    q_low, q_high = g['flow'].quantile([ql, qh])
    return g[g['conc'].between(c_low, c_high) & g['flow'].between(q_low, q_high)]

def _sanitize(name: str) -> str:
    name = re.sub(r"\s+", "_", str(name))
    return re.sub(r"[^A-Za-z0-9_.-]", "", name)

def _fit_ols_loglog(x: np.ndarray, y: np.ndarray, var_eps: float) -> Dict[str, float]:
    """
    Fit OLS for log10(C) ~ log10(Q)
    Returns: slope, intercept, r², p, n, RMSE, MAE
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 2 or np.std(x) < var_eps or np.std(y) < var_eps:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "p": np.nan, "n": n,
                "rmse": np.nan, "mae": np.nan}

    if HAS_SM:
        try:
            X = sm.add_constant(x)
            model = sm.OLS(y, X, missing='drop').fit()
            yhat = model.predict(X)
            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            mae  = float(np.mean(np.abs(y - yhat)))
            return {
                "slope": float(model.params[1]),
                "intercept": float(model.params[0]),
                "r2": float(model.rsquared),
                "p": float(model.pvalues[1]),
                "n": n,
                "rmse": rmse,
                "mae": mae
            }
        except Exception:
            return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "p": np.nan, "n": n,
                    "rmse": np.nan, "mae": np.nan}
    else:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        mae  = float(np.mean(np.abs(y - yhat)))
        return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "p": np.nan, "n": n,
                "rmse": rmse, "mae": mae}

def run_pipeline():
    cfg = CONFIG
    in_path = Path(cfg["parquet_path"]).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"Data file not found: {in_path}")

    out_dir, figs_dir = _ensure_dirs(Path(cfg["out_dir"]))
    df = pd.read_parquet(in_path)

    # Rename columns
    renamer = _build_renamer(cfg["columns"], list(df.columns))
    df = df.rename(columns=renamer)

    # Check required columns
    for col in ["station", "metal", "conc", "flow"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Fallback for missing fraction column
    if "fraction" not in df.columns or df["fraction"].isna().all():
        df["fraction"] = "All"

    # Apply filters
    if cfg["allowed_metals"]:
        df = df[df["metal"].isin(cfg["allowed_metals"])]
    if cfg["fractions_keep"]:
        df = df[df["fraction"].isin(cfg["fractions_keep"])]

    # Clean and log-transform
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df = df[(df["conc"] > 0) & (df["flow"] > 0)].copy()
    df["logC"] = _safe_log10(df["conc"])
    df["logQ"] = _safe_log10(df["flow"])

    rows = []
    qtrim = cfg["trim_quantiles"]
    min_n = cfg["min_points"]
    eps = cfg["chemostatic_eps"]
    var_eps = cfg["var_eps"]
    gcols = ["station", "metal", "fraction"]

    for keys, g in df.groupby(gcols, dropna=False):
        g = g.dropna(subset=["logC", "logQ"])
        if qtrim:
            g = _trim_extremes(g, qtrim)

        n_now = len(g)
        if n_now < min_n or np.std(g["logQ"]) < var_eps or np.std(g["logC"]) < var_eps:
            rows.append({
                **dict(zip(gcols, keys)),
                "slope": np.nan, "intercept": np.nan, "r2": np.nan, "p": np.nan,
                "n": n_now, "rmse": np.nan, "mae": np.nan,
                "trend": "insufficient"
            })
            continue

        stats = _fit_ols_loglog(g["logQ"].to_numpy(), g["logC"].to_numpy(), var_eps)
        rows.append({**dict(zip(gcols, keys)), **stats, "trend": _classify_trend(stats["slope"], eps)})

        # Plotting
        try:
            import matplotlib.pyplot as plt
            x, y = g["logQ"], g["logC"]
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = stats["slope"] * x_line + stats["intercept"] if not pd.isna(stats["slope"]) else None

            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, s=12, alpha=0.6)
            if y_line is not None:
                plt.plot(x_line, y_line, color="red", linewidth=2)
            title = (
                f"{keys[0]} | {keys[1]} | {keys[2]}\n"
                f"Slope={stats['slope']:.3f}, R²={stats['r2']:.2f}, n={stats['n']} "
                f"({_classify_trend(stats['slope'], eps)}); "
                f"RMSE={stats['rmse']:.3f}, MAE={stats['mae']:.3f}"
            )
            plt.title(title)
            plt.xlabel("log10(Q)")
            plt.ylabel("log10(C)")
            plt.tight_layout()
            fname = f"{_sanitize(keys[0])}__{_sanitize(keys[1])}__{_sanitize(keys[2])}_24003.png"
            plt.savefig(figs_dir / fname, dpi=200)
            plt.close()
        except Exception:
            pass

    # Save results table
    res_df = pd.DataFrame(rows)
    cols_order = ["station", "metal", "fraction",
                  "slope", "intercept", "r2", "p", "n", "rmse", "mae", "trend"]
    res_df = res_df.reindex(columns=cols_order)
    (Path(cfg["out_dir"]) / "cq_slopes_24003.csv").write_text(res_df.to_csv(index=False), encoding="utf-8")

    # Save pivot table (station × metal × fraction → slope)
    res_valid = res_df.dropna(subset=["slope"])
    if not res_valid.empty:
        pivot = res_valid.pivot_table(index=["station"], columns=["metal", "fraction"], values="slope")
        pivot.to_csv(Path(cfg["out_dir"]) / "cq_slopes_pivot_24003.csv")

    # Write markdown summary
    parts = []
    total = len(res_df)
    valid = res_df[res_df["n"] >= min_n]
    parts.append(f"Groups evaluated: {total}; valid groups (n≥{min_n}): {len(valid)}")

    if not valid.empty:
        parts.append(
            "Overall model fit (valid groups): "
            f"median R² = {valid['r2'].median():.2f}, "
            f"median RMSE = {valid['rmse'].median():.3f}, "
            f"median MAE = {valid['mae'].median():.3f}"
        )
        for (metal, fraction), sub in valid.groupby(["metal", "fraction"]):
            counts = sub["trend"].value_counts().to_dict()
            parts.append(
                f"- {metal} [{fraction}]: mobilisation {counts.get('mobilisation', 0)}, "
                f"dilution {counts.get('dilution', 0)}, chemostatic {counts.get('chemostatic', 0)}"
            )
        if HAS_SM and "p" in valid.columns and not valid["p"].isna().all():
            sig = valid[valid["p"] < 0.05]
            parts.append(f"\nGroups with p<0.05: {len(sig)}. Top 10 by smallest p-value:")
            show_cols = ["station", "metal", "fraction", "slope", "r2", "p", "n", "rmse", "mae", "trend"]
            parts.append(sig.sort_values("p").head(10)[show_cols].to_string(index=False))

    parts.append(
        f"\nNote: |slope| ≤ {eps} ⇒ chemostatic; slope < -{eps} ⇒ dilution; slope > {eps} ⇒ mobilisation."
    )
    parts.append("Fit metrics: RMSE and MAE are computed on log10-scale residuals.")
    (Path(cfg["out_dir"]) / "summary_24003.md").write_text("\n".join(parts), encoding="utf-8")

    print("✅ Done:")
    print(f"- Stats: {cfg['out_dir']}/cq_slopes_24003.csv")
    print(f"- Pivot: {cfg['out_dir']}/cq_slopes_pivot_24003.csv")
    print(f"- Summary: {cfg['out_dir']}/summary_24003.md")
    print(f"- Figures: {Path(cfg['out_dir']) / 'figs'}")

if __name__ == "__main__":
    run_pipeline()
