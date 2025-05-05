import os
import json
import numpy as np
import pandas as pd

# Parameters
dataset_root = "dance_dataset"
fps = 20
dt = 1 / fps
wrist_index = 16
min_visibility = 0.5

all_features = []

# Loop through labels (e.g., float, punch)
for label in os.listdir(dataset_root):
    label_path = os.path.join(dataset_root, label)
    if not os.path.isdir(label_path):
        continue

    # Loop through each _keypoints folder
    for item in os.listdir(label_path):
        if not item.endswith("_keypoints"):
            continue

        kp_dir = os.path.join(label_path, item)
        frame_files = sorted([f for f in os.listdir(kp_dir) if f.endswith(".json")])

        positions = []

        for f in frame_files:
            with open(os.path.join(kp_dir, f)) as jf:
                data = json.load(jf)
                kp = data[wrist_index]
                if kp["visibility"] > min_visibility:
                    positions.append([kp["x"], kp["y"], kp["z"]])
                else:
                    positions.append([np.nan, np.nan, np.nan])

        positions = np.array(positions)

        # Skip if not enough valid data
        if positions.shape[0] < 5:
            print(f"Skipping {kp_dir} (too few frames)")
            continue

        # Interpolate missing values
        for dim in range(3):
            series = positions[:, dim]
            mask = np.isnan(series)
            if not np.all(mask):
                series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series[~mask])
            positions[:, dim] = series

        # Compute motion features
        vel = np.gradient(positions, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        jerk = np.gradient(acc, dt, axis=0)

        vel_mag = np.linalg.norm(vel, axis=1)
        acc_mag = np.linalg.norm(acc, axis=1)
        jerk_mag = np.linalg.norm(jerk, axis=1)

        features = {
            "source" : item,
            "mean_velocity": np.mean(vel_mag),
            "max_velocity": np.max(vel_mag),
            "mean_acceleration": np.mean(acc_mag),
            "max_acceleration": np.max(acc_mag),
            "mean_jerk": np.mean(jerk_mag),
            "max_jerk": np.max(jerk_mag),
            "label": label
        }

        all_features.append(features)

# Save to CSV
df = pd.DataFrame(all_features)
df.to_csv("motion_features.csv", index=False)
print("✅ Saved motion_features.csv with", len(df), "entries.")
