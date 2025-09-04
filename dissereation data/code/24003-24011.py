import pandas as pd
from pathlib import Path

# ========== Path configuration ==========
BASE_DIR = Path("")
csv_24011 = BASE_DIR / "cq_slopes_24011.csv"
csv_24003 = BASE_DIR / "cq_slopes_24003.csv"
out_csv = BASE_DIR / "slope_comparison_24011_vs_24003.csv"

# ========== Load data ==========
df_up = pd.read_csv(csv_24011)
df_down = pd.read_csv(csv_24003)

# ========== Add location labels ==========
df_up["location"] = "upstream (24011)"
df_down["location"] = "downstream (24003)"

# ========== Merge for comparison ==========
cols_keep = ["station", "metal", "fraction", "slope", "r2", "trend", "location"]
df_all = pd.concat([df_up[cols_keep], df_down[cols_keep]], ignore_index=True)

df_wide = df_all.pivot_table(
    index=["metal", "fraction"],
    columns="location",
    values=["slope", "r2", "trend"],
    aggfunc="first"
)

df_wide.columns = ['_'.join(col).strip() for col in df_wide.columns.values]
df_wide = df_wide.reset_index()

# ========== Add trend change classification ==========
def classify_shift(row):
    t_up = row.get("trend_upstream (24011)")
    t_down = row.get("trend_downstream (24003)")
    if pd.isna(t_up) or pd.isna(t_down):
        return "insufficient"
    if t_up == t_down:
        return "same"
    return f"{t_up} → {t_down}"

df_wide["trend_change"] = df_wide.apply(classify_shift, axis=1)

# ========== Save output ==========
df_wide.to_csv(out_csv, index=False)
print("Comparison complete. Output saved to:", out_csv)
