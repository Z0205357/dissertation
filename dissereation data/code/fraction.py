from pathlib import Path
import pandas as pd
import re

# --- Configure base path ---
BASE_DIR = Path
YEARS = range(2009, 2024)  # 2009 to 2024

# --- Regular expressions for detecting 'dissolved' or 'total' keywords ---
diss_re = re.compile(
    r"(?:diss|dissolved|filter|filtered|filterable|0\\.45\\s*(?:um|\u00b5m|\u03bcm|micron))",
    re.IGNORECASE
)
total_re = re.compile(
    r"(?:total|unfiltered|acid\\s+soluble|total\\s+recoverable|whole\\s+water)",
    re.IGNORECASE
)
metal_pat = re.compile(
    r"\\b(zinc|lead|copper|iron|manganese|aluminium|aluminum|nickel|cadmium|arsenic)\\b",
    re.IGNORECASE
)

# --- Loop over each year and extract relevant information ---
rows = []
for y in YEARS:
    f = BASE_DIR / f"{y}.csv"
    if not f.exists():
        print(f"[WARN] missing: {f.name}")
        continue

    usecols = ["determinand.notation", "determinand.label", "determinand.definition"]
    df = pd.read_csv(f, usecols=usecols, low_memory=False)

    lab = df["determinand.label"].astype(str)
    defi = df["determinand.definition"].astype(str).fillna("")

    has_diss = lab.str.contains(diss_re, na=False) | defi.str.contains(diss_re, na=False)
    has_total = lab.str.contains(total_re, na=False) | defi.str.contains(total_re, na=False)

    metal_token = (
        lab.str.extract(metal_pat, expand=False)
           .str.lower()
           .str.replace("aluminum", "aluminium", regex=False)
    )

    tmp = pd.DataFrame({
        "year": y,
        "notation": df["determinand.notation"].astype(str),
        "metal": metal_token.fillna("na"),
        "has_diss": has_diss,
        "has_total": has_total,
    })
    rows.append(tmp)

# --- Stop if no valid data was found ---
if not rows:
    raise SystemExit("❌ No valid files found. Check BASE_DIR.")

lite = pd.concat(rows, ignore_index=True)

# --- Helper: get mode or empty string ---
def mode_or_empty(s: pd.Series):
    s = s.replace("na", pd.NA).dropna()
    return s.mode().iloc[:1].tolist()

# --- Generate summary table for each notation ---
summary = (
    lite.groupby("notation", as_index=False)
        .agg(
            n=("notation", "size"),
            years_present=("year", pd.Series.nunique),
            metal_mode=("metal", mode_or_empty),
            p_diss=("has_diss", "mean"),
            p_total=("has_total", "mean"),
        )
        .sort_values(["years_present", "n"], ascending=False)
)

# --- Infer the suggested fraction for each notation ---
def suggest_fraction(row):
    pdiss = float(row.get("p_diss", 0.0))
    ptotal = float(row.get("p_total", 0.0))
    if pdiss >= 0.95 and ptotal < 0.05:
        return "dissolved"
    if ptotal >= 0.95 and pdiss < 0.05:
        return "total"
    return "unspecified"

summary["suggested_fraction"] = summary.apply(suggest_fraction, axis=1)

# --- Save the suggested fraction mapping as CSV ---
out_dir = BASE_DIR / "_validation"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "CODE_TO_FRACTION_suggested.csv"
draft = summary[["notation", "years_present", "n", "p_diss", "p_total", "suggested_fraction"]]
draft.to_csv(out_path, index=False)
print("✅ Saved mapping to:", out_path)
