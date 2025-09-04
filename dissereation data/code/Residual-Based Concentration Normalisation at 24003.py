import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from pathlib import Path

# ======= CONFIG =======
INPUT_PARQUET = Path("24003_Flow–Concentration Merging.parquet")
OUTPUT_DIR = Path("")
TARGET_METALS = ['lead', 'iron', 'cadmium', 'nickel', 'aluminium', 'copper', 'arsenic', 'zinc']

# ======= LOAD DATA =======
df = pd.read_parquet(INPUT_PARQUET)
df = df.rename(columns={"result": "value"})  
df = df.dropna(subset=["value", "flow", "metal", "sample_date"])
df = df[(df["value"] > 0) & (df["flow"] > 0)]
df["logC"] = np.log10(df["value"])
df["logQ"] = np.log10(df["flow"])
df["year"] = pd.to_datetime(df["sample_date"]).dt.year

# ======= RESIDUAL NORMALISATION =======
records = []
for metal in TARGET_METALS:
    subset = df[df["metal"] == metal].copy()
    if len(subset) < 30:
        continue
    model = ols("logC ~ logQ", data=subset).fit()
    subset["residual"] = model.resid
    subset["norm_logC"] = model.resid + model.params["Intercept"]
    yearly = subset.groupby("year")["norm_logC"].mean().reset_index()
    yearly["metal"] = metal
    records.append(yearly)

# ======= EXPORT =======
result = pd.concat(records, ignore_index=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_DIR / "flow_normalised_residual_24003.csv", index=False)

# ======= PLOT =======
plt.figure(figsize=(10, 6))
for metal in TARGET_METALS:
    sub = result[result["metal"] == metal]
    if not sub.empty:
        plt.plot(sub["year"], sub["norm_logC"], label=metal, marker="o")
plt.title("Flow-Normalised Metal Trends (Residual Method, 24003)")
plt.xlabel("Year")
plt.ylabel("Normalised log₁₀(Concentration)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "flow_normalised_residual_24003.png", dpi=300)
plt.close()

print("✅ 分析完成，结果保存在：", OUTPUT_DIR)
