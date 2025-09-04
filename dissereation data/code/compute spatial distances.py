import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from pathlib import Path

# ===== File path configuration =====
INPUT_PATH = Path("sampling_locations_json.csv")
OUTPUT_PATH = Path("sampling_with_distance.csv")
DOWNSTREAM_PATH = Path("downstream_candidates.csv")
MAP_PATH = Path("downstream_map.png")

# ===== Load the sampling location table =====
df = pd.read_csv(INPUT_PATH)

# ===== Extract the coordinates of the pollution source =====
source_row = df[df["sampling_point_id"] == "NE-44100360"]
source_lat = source_row["lat"].values[0]
source_lon = source_row["lon"].values[0]

# ===== Vectorised Haversine distance calculation =====
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometres
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ===== Calculate distances from all points to the pollution source =====
df["distance_to_source_km"] = haversine_np(source_lat, source_lon, df["lat"].values, df["lon"].values)

# ===== Identify potential downstream points (simplified heuristic) =====
df_downstream = df[(df["lat"] < source_lat) & (df["distance_to_source_km"] > 0.5)]

# ===== Save outputs =====
df.to_csv(OUTPUT_PATH, index=False)
df_downstream.to_csv(DOWNSTREAM_PATH, index=False)

# ===== Optional: Visualise map =====
plt.figure(figsize=(8, 6))
plt.scatter(df["lon"], df["lat"], c="gray", label="All sampling points")
plt.scatter(source_lon, source_lat, color="red", label="Pollution source (NE-44100360)", s=100)
plt.scatter(df_downstream["lon"], df_downstream["lat"], color="blue", label="Downstream candidates")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Sampling Points and Downstream Candidates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("downstream_map.png")
plt.close()
print("[✅] Map saved to:downstream_map.png")
