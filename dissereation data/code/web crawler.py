import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# ========== Configure paths ==========
PARQUET_PATH = Path("all_years_metal_raw.parquet")
OUTPUT_PATH = Path("sampling_locations_json.csv")

# ========== Load all unique sampling point IDs ==========
df_all = pd.read_parquet(PARQUET_PATH)
all_ids = df_all["sample.samplingPoint.notation"].dropna().unique().tolist()

# ========== Read previously saved results, if any ==========
if OUTPUT_PATH.exists():
    df_done = pd.read_csv(OUTPUT_PATH)
    done_ids = set(df_done["sampling_point_id"])
else:
    df_done = pd.DataFrame(columns=[
        "sampling_point_id", "url", "name", "lat", "lon", "easting", "northing", "status", "type", "area"
    ])
    done_ids = set()

# ========== Prepare the list of IDs to be fetched ==========
pending_ids = [pid for pid in all_ids if pid not in done_ids]
print(f"[INFO] Total: {len(all_ids)} | Done: {len(done_ids)} | Pending: {len(pending_ids)}")

# ========== Start web scraping from EA API ==========
for pid in tqdm(pending_ids, desc="Fetching JSON metadata"):
    url = f"https://environment.data.gov.uk/water-quality/id/sampling-point/{pid}.json"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"[⚠️] {pid} failed with status {r.status_code}")
            continue

        data = r.json()
        item = data["items"][0]

        df_done.loc[len(df_done)] = {
            "sampling_point_id": pid,
            "url": url,
            "name": item.get("label", ""),
            "lat": item.get("lat", ""),
            "lon": item.get("long", ""),
            "easting": item.get("easting", ""),
            "northing": item.get("northing", ""),
            "status": item.get("samplingPointStatus", {}).get("prefLabel", ""),
            "type": item.get("samplingPointType", {}).get("prefLabel", ""),
            "area": item.get("area", {}).get("prefLabel", "")
        }

        # Save progress after each successful fetch
        df_done.to_csv(OUTPUT_PATH, index=False)
        time.sleep(0.3)

    except Exception as e:
        print(f"[❌] {pid} error: {e}")
        continue

print(f"[✅] Finished. Saved to: {OUTPUT_PATH}")
