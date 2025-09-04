import pandas as pd

# Step 1: Load the distance table
dist_df = pd.read_csv("sampling_with_distance.csv")

# Step 2: Identify downstream stations (those farther than the pollution source)
source_id = "NE-44100360"
source_dist = dist_df.loc[dist_df["sampling_point_id"] == source_id, "distance_to_source_km"].values[0]
downstream_ids = dist_df.loc[dist_df["distance_to_source_km"] > source_dist, "sampling_point_id"].tolist()

# Step 3: Filter the metal concentration dataset to keep only these downstream stations
metal_df = pd.read_parquet("EA_2009_2024_provisional_total.parquet")
filtered = metal_df[metal_df["sample.samplingPoint.notation"].isin(downstream_ids)]

# Output the filtering result
print("✅ Filtered downstream metal samples:", filtered.shape)

# Step 4: Save the filtered data to file
output_path = "downstream_metal_concentration.parquet"
filtered.to_parquet(output_path, index=False)
print(f"✅ Saved to: {output_path}")
