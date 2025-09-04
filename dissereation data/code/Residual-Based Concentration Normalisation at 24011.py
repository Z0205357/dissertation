import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from pathlib import Path

# ========== [User Config] ==========
INPUT_PARQUET = Path("24011_Flow–Concentration Merging.parquet")
OUTPUT_DIR = Path("")
TARGET_METALS = ['lead', 'iron', 'cadmium', 'nickel', 'aluminium', 'copper', 'arsenic', 'zinc']

# ========== [Step 1: Load Data] ==========
df = pd.read_parquet(INPUT_PARQUET)
df = df.rename(columns={"result": "value"}) if "result" in df.columns else df
df = df.dropna(subset=["value", "flow", "metal", "sample_date"])
df = df[(df["value"] > 0) & (df["flow"] > 0)]

# ========== [Step 2: Log Transform] ==========
df["logC"] = np.log10(df["value"])
df["logQ"] = np.log10(df["flow"])
df["year"] = pd.to_datetime(df["sample_date"]).dt.year

# ========== [Step 3: Residual Normalisation per Metal] ==========
records = []
for metal in TARGET_METALS:
    subset = df[df["metal"] == metal].copy()
    if len(subset) < 30:
        continue
    model = ols("logC ~ logQ", data=subset).fit()
    subset["residual"] = model.resid
    subset["normalised_logC"] = subset["residual"] + model.params["Intercept"]
    yearly = subset.groupby("year")["normalised_logC"].mean().reset_index()
    yearly["metal"] = metal
    records.append(yearly)

# ========== [Step 4: Export & Plot] ==========
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result_df = pd.concat(records, ignore_index=True)
result_df.to_csv(OUTPUT_DIR / "24011_flow_normalised_residual_method.csv", index=False)

plt.figure(figsize=(10, 6))
for metal in TARGET_METALS:
    sub = result_df[result_df["metal"] == metal]
    plt.plot(sub["year"], sub["normalised_logC"], marker="o", label=metal)
plt.title("Flow-Normalised Metal Trends (Residual Method)")
plt.xlabel("Year")
plt.ylabel("Normalised log₁₀(Concentration)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "24011_flow_normalised_residual_method.png")
plt.close()
