import os
import json
import numpy as np
import pandas as pd

session_timestamp = "2025-05-18_21-51-32"  # change to match current session
output_root = "4_extracted_motion_features"
output_dir = os.path.join(output_root, session_timestamp)
os.makedirs(output_dir, exist_ok=True)

# Parameters
pose_root = os.path.join("3_skeleton_pose_data", session_timestamp)
fps = 20
dt = 1 / fps
min_visibility = 0.5

# Landmarks of interest
landmark_indices = {
    "right_wrist": 16,
    "left_ankle": 27,
    "right_ankle": 28
}

all_features = []

# Loop through clips  
for clip_name in os.listdir(pose_root):
    clip_path = os.path.join(pose_root, clip_name)
    if not os.path.isdir(clip_path):
        continue

    frame_files = sorted([f for f in os.listdir(clip_path) if f.endswith(".json")])
    positions = []

    for f in frame_files:
        with open(os.path.join(clip_path, f)) as jf:
            data = json.load(jf)
            pose_vector = []
            for i in range(len(data)):  # For all landmarks in the frame
                if data[i]["visibility"] > min_visibility:
                    pose_vector.extend([data[i]["x"], data[i]["y"], data[i]["z"]])
                else:
                    pose_vector.extend([np.nan, np.nan, np.nan])
            positions.append(pose_vector)

    positions = np.array(positions)

    # Debug print
    print(f"🔍 Clip: {clip_name}, Frames: {len(frame_files)}, Pose shape: {positions.shape}")

    # Defensive guard
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] == 0:
        print(f"⚠️ Skipping {clip_name}: no usable pose data or invalid structure")
        continue


    # Interpolate missing values
    for dim in range(positions.shape[1]):
        series = positions[:, dim]
        mask = np.isnan(series)
        if not np.all(mask):
            series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series[~mask])
        positions[:, dim] = series

    # Replace any remaining NaNs with zeros --> we need this so we dont get all NaN values when we do gradient for velocity adn acceleration later
    positions = np.nan_to_num(positions, nan=0.0)


    # # Final check before computing features
    # print(f"→ About to compute motion features for: {item}")
    # print("→ Any NaNs left in positions?", np.isnan(positions).any())
    # print("→ First few values:\n", positions[:3, :9])


    # Compute motion features
    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)

    vel_mag = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)
    jerk_mag = np.linalg.norm(jerk, axis=1)



    print("clip_name:", clip_name, "→", clip_name.split("_"))

    # === Get label
    label = clip_name.split("_")[2]  # e.g. "2025-05-18_21-51-32_punch_002" → "punch"

    # === Get landmark ranges
    range_features = {}
    for name, index in landmark_indices.items():
        x_vals = positions[:, index * 3 + 0]
        y_vals = positions[:, index * 3 + 1]
        range_features[f"range_x_{name}"] = np.max(x_vals) - np.min(x_vals)
        range_features[f"range_y_{name}"] = np.max(y_vals) - np.min(y_vals)


    features = {
        "source": clip_name,
        "label": label,  # ← use the properly extracted label!
        "mean_velocity": np.mean(vel_mag),
        "max_velocity": np.max(vel_mag),
        "std_velocity": np.std(vel_mag),
        "mean_acceleration": np.mean(acc_mag),
        "max_acceleration": np.max(acc_mag),
        "std_acceleration": np.std(acc_mag),
        "mean_jerk": np.mean(jerk_mag),
        "max_jerk": np.max(jerk_mag),
        "std_jerk": np.std(jerk_mag),
        **range_features  # ← unpacks the range_x/y_right_wrist etc.
    }


    all_features.append(features)

# Save to CSV
df = pd.DataFrame(all_features)

output_csv_path = os.path.join(output_dir, "motion_features.csv")
df.to_csv(output_csv_path, index=False)
print(f"Saved {len(df)} motion feature entries to {output_csv_path}")
