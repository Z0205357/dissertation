#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C–Q Regression (log(C) ~ log(Q)) Pipeline – for 24011 upstream station
"""

from __future__ import annotations
import os
import re
from pathlib import Path
import math
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False
    warnings.warn("statsmodels not installed; falling back to numpy.polyfit (no p-values)")

CONFIG = {
    "parquet_path": "Flow–Concentration Merging.parquet",
    "columns": {
        "station": "sample.samplingPoint.notation",
        "datetime": "sample.sampleDateTime",
        "metal": "metal",
        "fraction": "fraction",
        "conc": "result",
        "flow": "flow"
    },
    "allowed_metals": ["zinc", "iron", "cadmium", "lead", "copper", "nickel", "aluminium"],
    "fractions_keep": None,
    "min_points": 10,
    "trim_quantiles": (0.01, 0.99),
    "chemostatic_eps": 0.10,
    "out_dir": ""
}

def _ensure_dirs(base: Path) -> Tuple[Path, Path]:
    out_base = base
    figs_dir = out_base / "figs_cq_24011"
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

def _fit_ols_loglog(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 2 or np.all(y == y[0]) or np.std(y) == 0 or np.all(x == x[0]) or np.std(x) == 0:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "p": np.nan, "n": n}

    if HAS_SM:
        try:
            X = sm.add_constant(x)
            model = sm.OLS(y, X, missing='drop').fit()
            if len(model.params) < 2:
                raise ValueError("Model has no slope term.")
            return {
                "slope": float(model.params[1]),
                "intercept": float(model.params[0]),
                "r2": float(model.rsquared),
                "p": float(model.pvalues[1]),
                "n": n
            }
        except Exception:
            return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "p": np.nan, "n": n}
    else:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
        return {"slope": slope, "intercept": intercept, "r2": r2, "p": np.nan, "n": n}

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

def run_pipeline():
    cfg = CONFIG
    in_path = Path(cfg["parquet_path"]).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"Data file not found: {in_path}")

    out_dir, figs_dir = _ensure_dirs(Path(cfg["out_dir"]))
    df = pd.read_parquet(in_path)

    renamer = _build_renamer(cfg["columns"], list(df.columns))
    df = df.rename(columns=renamer)

    for col in ["station", "metal", "conc", "flow"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    if "fraction" not in df.columns or df["fraction"].isna().all():
        df["fraction"] = "All"

    if cfg["allowed_metals"]:
        df = df[df["metal"].isin(cfg["allowed_metals"])]
    if cfg["fractions_keep"] and "fraction" in df.columns:
        df = df[df["fraction"].isin(cfg["fractions_keep"])]

    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df = df[(df["conc"] > 0) & (df["flow"] > 0)].copy()
    df["logC"] = _safe_log10(df["conc"])
    df["logQ"] = _safe_log10(df["flow"])

    gcols = ["station", "metal", "fraction"]
    print("Loaded data shape:", df.shape)
    print("Metals included:", sorted(df['metal'].dropna().unique()))
    print("Fractions included:", sorted(df['fraction'].dropna().unique()))
    print("Grouping by:", gcols)

    rows = []
    qtrim = cfg["trim_quantiles"]
    min_n = cfg["min_points"]
    eps = cfg["chemostatic_eps"]

    for keys, g in df.groupby(gcols, dropna=False):
        g = g.dropna(subset=["logC", "logQ"])
        print(f"Checking group: {keys} → {len(g)} rows")

        if qtrim:
            g = _trim_extremes(g, qtrim)

        if len(g) < min_n:
            rows.append({**dict(zip(gcols, keys)),
                         **{k: np.nan for k in ["slope", "intercept", "r2", "p"]},
                         "n": len(g),
                         "trend": "insufficient"})
            continue

        stats = _fit_ols_loglog(g["logQ"].to_numpy(), g["logC"].to_numpy())
        rows.append({**dict(zip(gcols, keys)), **stats, "trend": _classify_trend(stats["slope"], eps)})

        try:
            import matplotlib.pyplot as plt
            x = g["logQ"]
            y = g["logC"]
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = stats["slope"] * x_line + stats["intercept"]

            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, s=12, alpha=0.6)
            if not pd.isna(stats["slope"]):
                plt.plot(x_line, y_line, color="red", linewidth=2)
            title = f"{keys[0]} | {keys[1]} | {keys[2]}\nSlope={stats['slope']:.3f}, R²={stats['r2']:.2f}, n={stats['n']} ({_classify_trend(stats['slope'], eps)})"
            plt.title(title)
            plt.xlabel("log10(Q)")
            plt.ylabel("log10(C)")
            plt.tight_layout()
            fname = f"{_sanitize(keys[0])}__{_sanitize(keys[1])}__{_sanitize(keys[2])}_24011.png"
            plt.savefig(figs_dir / fname, dpi=200)
            plt.close()
        except Exception:
            pass

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_dir / "cq_slopes_24011.csv", index=False)

    if "slope" in res_df.columns:
        res_valid = res_df.dropna(subset=["slope"])
        if not res_valid.empty:
            pivot = res_valid.pivot_table(index=["station"], columns=["metal", "fraction"], values="slope")
            pivot.to_csv(out_dir / "cq_slopes_pivot_24011.csv")

    parts = []
    total = len(res_df)
    if "n" in res_df.columns:
        valid = res_df[res_df["n"] >= min_n]
        parts.append(f"Groups evaluated: {total}; valid groups (n≥{min_n}): {len(valid)}")
        for (metal, fraction), sub in valid.groupby(["metal", "fraction"]):
            counts = sub["trend"].value_counts().to_dict()
            parts.append(f"- {metal} [{fraction}]: mobilisation {counts.get('mobilisation', 0)}, dilution {counts.get('dilution', 0)}, chemostatic {counts.get('chemostatic', 0)}")

        if HAS_SM and not valid["p"].isna().all():
            sig = valid[valid["p"] < 0.05]
            parts.append(f"\nGroups with p<0.05: {len(sig)}. Top 10:")
            parts.append(sig.sort_values("p").head(10).to_string(index=False))

    summary_text = "\n".join(parts) + f"\n\nNote: |slope| < {eps} ⇒ chemostatic; slope < -{eps} ⇒ dilution; slope > {eps} ⇒ mobilisation."
    (out_dir / "summary_24011.md").write_text(summary_text, encoding="utf-8")

    print("Done:")
    print(f"- Stats: {out_dir}/cq_slopes_24011.csv")
    print(f"- Pivot: {out_dir}/cq_slopes_pivot_24011.csv")
    print(f"- Summary: {out_dir}/summary_24011.md")
    print(f"- Figures: {figs_dir}")

if __name__ == "__main__":
    run_pipeline()

