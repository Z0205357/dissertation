import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# --- optional: try to use matplotlib_scalebar if available, else fallback to AnchoredSizeBar ---
USE_MATPLOTLIB_SCALEBAR = False
try:
    from matplotlib_scalebar.scalebar import ScaleBar
    USE_MATPLOTLIB_SCALEBAR = True
except Exception:
    pass

# -----------------------------
# helper: parse OS National Grid ref like "NY9836639040" -> (E, N) in EPSG:27700
# -----------------------------
def osgrid_to_en(gridref: str):
    s = gridref.strip().replace(" ", "").upper()
    l1, l2 = s[0], s[1]
    nums = s[2:]
    nlen = len(nums) // 2
    e_part = int(nums[:nlen])
    n_part = int(nums[nlen:])

    def li(ch):
        idx = ord(ch) - 65  # 'A'->0
        if idx > 7:        # skip 'I'
            idx -= 1
        return idx

    L1, L2 = li(l1), li(l2)
    # 100 km square indices (per Ordnance Survey scheme)
    e100 = ((L1 - 2) % 5) * 5 + (L2 % 5)
    n100 = (19 - (L1 // 5) * 5) - (L2 // 5)

    scale = 10 ** (5 - nlen)  # 5-digit per axis -> meter precision
    E = e100 * 100000 + e_part * scale
    N = n100 * 100000 + n_part * scale
    return E, N

# -----------------------------
# 1) Load data & set projection
# -----------------------------
df = pd.read_csv("/Users/meishaonvzhanshi/Desktop/Python/PythonProject/data_base/全新的文件夹/sampling_locations_json.csv")
assert {'lon','lat','sampling_point_id'}.issubset(df.columns), "Missing required columns: lon, lat, sampling_point_id"

# Build GeoDataFrame in WGS84 then project to Web Mercator (meters)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs="EPSG:4326").to_crs(epsg=3857)

# Pollution source (Park Level Discharge)
source = gdf[gdf['sampling_point_id'] == 'NE-44100360']
if source.empty:
    raise ValueError("Could not find sampling_point_id == 'NE-44100360' in the data.")

# -----------------------------
# 2) Flow gauges from OS grid refs -> GeoDataFrame (EPSG:3857)
# -----------------------------
flow_refs = [
    {"nrfa_id": "24003", "name": "Stanhope",            "grid": "NY9836639040"},
    {"nrfa_id": "24011", "name": "Burnhope Reservoir",  "grid": "NY8558439487"},
]
for r in flow_refs:
    E, N = osgrid_to_en(r["grid"])
    r["E"], r["N"] = E, N

flow_df = pd.DataFrame(flow_refs)
# create GeoDataFrame in British National Grid then project to Web Mercator
flow_gdf = gpd.GeoDataFrame(flow_df,
                            geometry=gpd.points_from_xy(flow_df["E"], flow_df["N"]),
                            crs="EPSG:27700").to_crs(epsg=3857)

# -----------------------------
# 3) Figure & base plotting
# -----------------------------
fig, ax = plt.subplots(figsize=(11, 7))  # landscape, easier to place in Word

# All sampling points (light grey)
gdf.plot(ax=ax, color='lightgrey', markersize=10, label='Sampling Points')

# Pollution source (red)
source.plot(ax=ax, color='red', markersize=60, label='Pollution Source (NE-44100360)')

# Flow gauges (blue triangles with black edge)
flow_gdf.plot(ax=ax, marker='^', color='tab:blue', edgecolor='black', markersize=90, label='Flow Gauges (NRFA)')

# Label each flow station
for _, r in flow_gdf.iterrows():
    ax.annotate(f"{r['nrfa_id']} {r['name']}",
                xy=(r.geometry.x, r.geometry.y),
                xytext=(6, 6), textcoords="offset points",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# ----------------------------------
# 4) Auto-zoom to include source + flow stations (with padding)
# ----------------------------------
layers = [source.geometry, flow_gdf.geometry]
bb = gpd.GeoSeries(pd.concat(layers), crs=gdf.crs).total_bounds  # (minx, miny, maxx, maxy)
pad = 2000  # meters padding
ax.set_xlim(bb[0]-pad, bb[2]+pad)
ax.set_ylim(bb[1]-pad, bb[3]+pad)

# -----------------------------
# 5) Basemap (OSM Mapnik)
# -----------------------------
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf.crs, reset_extent=False)

# -----------------------------
# 6) Scale bar (meters)
# -----------------------------
if USE_MATPLOTLIB_SCALEBAR:
    scalebar = ScaleBar(dx=1, units="m", location='lower left', box_alpha=0.5)
    ax.add_artist(scalebar)
else:
    scalebar_length_m = 2000  # 2 km
    scalebar = AnchoredSizeBar(ax.transData, scalebar_length_m, '2 km', 'lower left',
                               pad=0.4, color='black', frameon=True, size_vertical=5)
    ax.add_artist(scalebar)

# -----------------------------
# 7) North arrow (simple & clear)
# -----------------------------
ax.annotate('', xy=(0.95, 0.28), xytext=(0.95, 0.12),
            xycoords='axes fraction',
            arrowprops=dict(facecolor='black', width=3, headwidth=12, headlength=12))
ax.text(0.95, 0.30, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=12)

# -----------------------------
# 8) Grid, title, legend, attribution
# -----------------------------
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_title("Metal Sampling Points, Pollution Source, and NRFA Flow Gauges", fontsize=13)
ax.legend(loc='upper right')
ax.text(0.01, 0.01, "© OpenStreetMap contributors", transform=ax.transAxes, fontsize=8, alpha=0.7)

# -----------------------------
# 9) Save & show
# -----------------------------
plt.tight_layout()
outpath = "/Users/meishaonvzhanshi/Desktop/Python/PythonProject/data_base/全新的文件夹/picture/sampling_map_with_flow_gauges.png"
plt.savefig(outpath, dpi=300)
plt.show()
print(f"[Saved] {outpath}")
