from pathlib import Path
import pandas as pd

# ========= 1. Path configuration =========
BASE_DIR = Path("")
metal_path = BASE_DIR / "metal_24011.parquet"
flow_path = BASE_DIR / "24003_flow_cleaned.parquet" 
out_path = BASE_DIR / "24003_Flow–Concentration Merging.parquet"

# ========= 2. Load data =========
metal_df = pd.read_parquet(metal_path)
flow_df = pd.read_parquet(flow_path)

# ========= 3. Standardise date format for alignment =========
metal_df = metal_df.copy()
metal_df["sample_date"] = pd.to_datetime(metal_df["sample.sampleDateTime"], errors="coerce").dt.date

flow_df = flow_df.copy()
flow_df["sample_date"] = pd.to_datetime(flow_df["sample_datetime"], errors="coerce").dt.date

# ========= 4. Merge metal concentration with flow data =========
merged = pd.merge(metal_df, flow_df, on="sample_date", how="left")

# ========= 5. Save merged result =========
merged.to_parquet(out_path, index=False)
print("✅ Done. Merged shape:", merged.shape)
