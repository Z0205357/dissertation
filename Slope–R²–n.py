import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ========== [User Config] ==========
# Input paths for slope summary tables
file_24011 = "/Users/meishaonvzhanshi/Desktop/Python/PythonProject/data_base/全新的文件夹/cq回归结果/cq_slopes_24011.csv"
file_24003 = "/Users/meishaonvzhanshi/Desktop/Python/PythonProject/data_base/全新的文件夹/cq回归结果_24003/cq_slopes_24003.csv"

# Output directory and filename
output_path = "/Users/meishaonvzhanshi/Desktop/Python/PythonProject/data_base/全新的文件夹/picture图像/slope_r2_n_subplot.png"

# ========== [Step 1: Load and preprocess] ==========
df1 = pd.read_csv(file_24011)
df1["station"] = "24011"
df2 = pd.read_csv(file_24003)
df2["station"] = "24003"

# Combine both tables
df = pd.concat([df1, df2], ignore_index=True)

# Filter valid rows (n >= 10 and non-null slope)
df = df[df["n"] >= 10]
df = df.dropna(subset=["slope", "r2", "trend"])

# Define marker styles per trend type
shape_dict = {
    "mobilisation": "o",
    "dilution": "s",
    "chemostatic": "D"
}

# Define color palette per metal
metal_palette = sns.color_palette("tab10", n_colors=df["metal"].nunique())

# ========== [Step 2: Create subplots] ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
stations = ["24011", "24003"]

for ax, station in zip(axes, stations):
    sub = df[df["station"] == station]
    for trend, shape in shape_dict.items():
        trend_data = sub[sub["trend"] == trend]
        sns.scatterplot(
            data=trend_data,
            x="slope",
            y="r2",
            hue="metal",
            size="n",
            sizes=(40, 200),
            palette=metal_palette,
            marker=shape,
            ax=ax,
            legend=False  # suppress duplicate legends
        )
    ax.set_title(f"Station {station}")
    ax.set_xlabel("Slope (β₁)")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylabel("R²" if station == "24011" else "")
    ax.grid(True)

# ========== [Step 3: Final layout] ==========
fig.suptitle("Slope–R²–n Distribution per Station\n(Colour = Metal; Shape = Trend; Size = Sample Count)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space for suptitle

# Save and show
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()
print(f"✅ Plot saved to: {output_path}")
