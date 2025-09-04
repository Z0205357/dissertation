import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
path = "cq_slopes_24011.csv"
df = pd.read_csv(path)

# 2. Filter target metals and valid trends
df = df[df["metal"].isin(["zinc", "iron", "cadmium"])]
df_valid = df[df["trend"] != "insufficient"].copy()

# 3. Count number of stations per metal × trend
trend_counts = df_valid.groupby(["metal", "trend"]).size().reset_index(name="count")

# 4. Save trend count table to CSV
csv_path = ""
trend_counts.to_csv(csv_path, index=False)
print(f"📄 Trend count table saved to: {csv_path}")

# 5. Pivot for stacked bar chart
pivot_df = trend_counts.pivot(index="metal", columns="trend", values="count").fillna(0)

# 6. Set trend order and colour mapping
trend_order = ["mobilisation", "chemostatic", "dilution"]
pivot_df = pivot_df[trend_order]

colors = ["#E76F51", "#2A9D8F", "#457B9D"]  # reddish orange, teal green, blue-grey

# 7. Plot stacked bar chart
pivot_df.plot(kind="bar", stacked=True, figsize=(8, 6), color=colors)

plt.title("Stacked Trend Classification by Metal")
plt.xlabel("Metal")
plt.ylabel("Number of Stations")
plt.legend(title="Trend", loc="upper right")
plt.tight_layout()

# 8. Save figure
fig_path = ""
plt.savefig(fig_path, dpi=300)
print(f"📊 Stacked bar chart saved to: {fig_path}")
plt.show()
