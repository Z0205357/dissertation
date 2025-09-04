from pathlib import Path
import pandas as pd
import numpy as np
import re

# ========= CONFIGURATION =========
BASE_DIR = Path # download from https://environment.data.gov.uk/water-quality/view/sampling-point/NE-44100360
YEARS = list(range(2009, 2024))  # Year range: 2009–2024
SAMPLE_ROWS = 200000
OUT_DIR = BASE_DIR / "_clean"
VAL_DIR = BASE_DIR / "_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

# ========= REGEX PATTERNS =========
metal_pat = re.compile(r"\b(zinc|lead|copper|iron|manganese|aluminium|aluminum|nickel|cadmium|arsenic)\b", re.I)
diss_re = re.compile(r"(?:diss|dissolved|filter|filtered|filterable|0\.45\s*(?:um|\u00b5m|\u03bcm|micron))", re.I)
total_re = re.compile(r"(?:total|unfiltered|acid\s+soluble|total\s+recoverable|whole\s+water)", re.I)

# ========= SELECTED COLUMNS =========
KEEP_COLS = [
    "@id", "sample.sampleDateTime", "sample.samplingPoint.notation",
    "sample.samplingPoint.label",
    "determinand.label", "determinand.notation", "determinand.definition",
    "determinand.unit.label", "resultQualifier.notation", "result",
    "sample.sampledMaterialType.label", "sample.purpose.label",
    "sample.samplingPoint.easting", "sample.samplingPoint.northing"
]

# ========= STEP 1: Load and clean yearly CSV files =========
frames = []

def clean_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=KEEP_COLS, low_memory=False)
    df = df.dropna(subset=["result", "sample.sampleDateTime"])
    df["sample.sampleDateTime"] = pd.to_datetime(df["sample.sampleDateTime"], errors="coerce")
    df = df.dropna(subset=["sample.sampleDateTime"])
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df = df.dropna(subset=["result"])
    df["determinand.unit.label"] = df["determinand.unit.label"].astype(str).str.strip().str.lower()
    tok = df["determinand.label"].astype(str).str.extract(metal_pat, expand=False).str.lower()
    df["metal"] = tok.str.replace("aluminum", "aluminium", regex=False)
    df["year"] = int(path.stem.split(".")[0])
    return df

for y in YEARS:
    csv_path = BASE_DIR / f"{y}.csv"
    if csv_path.exists():
        frames.append(clean_one(csv_path))

full = pd.concat(frames, ignore_index=True)
print("✅ Data loaded. Total rows:", len(full))

# ========= STEP 2: Infer fraction type (dissolved / total) =========
map_path = VAL_DIR / "CODE_TO_FRACTION_suggested.csv"
code_map = pd.read_csv(map_path)
CODE_TO_FRACTION = dict(zip(code_map["notation"].astype(str), code_map["suggested_fraction"].astype(str)))

def infer_fraction_row(row) -> str:
    code = str(row.get("determinand.notation", ""))
    mapped = CODE_TO_FRACTION.get(code)
    if mapped in ("dissolved", "total"):
        return mapped
    text = f"{row.get('determinand.label','')} | {row.get('determinand.definition','')}".lower()
    if diss_re.search(text):
        return "dissolved"
    if total_re.search(text):
        return "total"
    return pd.NA

full["fraction"] = full.apply(infer_fraction_row, axis=1)

# ========= STEP 3: Filter metals with sufficient observations =========
recent_years = [2019, 2020, 2021, 2022, 2023, 2024]
metal_counts = (
    full[full["year"].isin(recent_years)]
    .dropna(subset=["metal"])
    .groupby(["metal", "year"])
    .size()
    .reset_index(name="count")
)

# Keep metals with ≥10 observations per year (in all selected years)
metals_valid = (
    metal_counts[metal_counts["count"] >= 10]["metal"]
    .drop_duplicates()
    .tolist()
)

print("🎯 Filtered metals with sufficient records:", metals_valid)

# ========= STEP 4: Export filtered datasets =========
is_target = full["metal"].isin(metals_valid)
has_water_type = full["sample.sampledMaterialType.label"].notna()
subset = full.loc[is_target & has_water_type].copy()

subset.to_parquet(OUT_DIR / "EA_2009_2024_target_metals.parquet", index=False)
subset[subset["fraction"] == "total"].to_parquet(OUT_DIR / "EA_2009_2024_strict_total.parquet", index=False)
subset[~subset["fraction"].eq("dissolved")].to_parquet(OUT_DIR / "EA_2009_2024_provisional_total.parquet", index=False)

print("✅ Export completed. Filtered rows:", len(subset))

# ========= STEP 5: Generate summary tables =========
RESULT_DIR = Path("/Users/meishaonvzhanshi/Desktop/Python/PythonProject/data_base/全新的文件夹/数据结果表输出")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# [1] Total observations per metal
metal_counts = subset["metal"].value_counts().reset_index()
metal_counts.columns = ["metal", "total_count"]
metal_counts.to_csv(RESULT_DIR / "summary_metal_total_counts.csv", index=False)

# [2] Observation count per metal × fraction type
metal_fraction_counts = (
    subset.groupby(["metal", "fraction"])
    .size()
    .reset_index(name="count")
    .sort_values(["metal", "fraction"])
)
metal_fraction_counts.to_csv(RESULT_DIR / "summary_metal_fraction_counts.csv", index=False)

# [3] Year × Metal pivot table
pivot_table = (
    subset.groupby(["year", "metal"])
    .size()
    .reset_index(name="count")
    .pivot(index="year", columns="metal", values="count")
    .fillna(0).astype(int)
)
pivot_table.to_csv(RESULT_DIR / "summary_year_metal_pivot.csv")
