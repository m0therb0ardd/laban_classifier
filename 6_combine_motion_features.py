import os
import pandas as pd

input_root = "4_extracted_motion_features"
output_file = "6_motion_features.csv"

all_dfs = []

# Loop through each timestamp folder
for timestamp in os.listdir(input_root):
    subdir = os.path.join(input_root, timestamp)
    motion_csv = os.path.join(subdir, "motion_features.csv")

    if os.path.isfile(motion_csv):
        df = pd.read_csv(motion_csv)
        df["timestamp"] = timestamp  # add timestamp column
        all_dfs.append(df)
        print(f"✅ Added: {motion_csv}")
    else:
        print(f"Skipped (missing file): {motion_csv}")

# Combine and save
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f" Saved combined CSV to {output_file} with {len(combined_df)} rows")
else:
    print("No motion feature files found.")
