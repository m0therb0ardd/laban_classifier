import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# === CONFIG ===
session_timestamp = "2025-05-18_22-35-34"
pose_root = os.path.join("3_skeleton_pose_data", session_timestamp)
data_dict_path = "0_data_dictionary.csv"
output_root = "5_motion_output_graphs"
output_dir = os.path.join(output_root, session_timestamp)
os.makedirs(output_dir, exist_ok=True)

landmark_count = 33
min_visibility = 0.5

landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# === Load data dictionary
df = pd.read_csv(data_dict_path)
df = df[(df["timestamp"] == session_timestamp) & (df["gesture_captured"] == "yes")]

# === Group by gesture label
gesture_groups = defaultdict(list)
for _, row in df.iterrows():
    clip_name = row["clip_id"].replace(".mp4", "")
    label = row["label"]
    gesture_groups[label].append(clip_name)

# === Plot each gesture group
for label, clip_names in gesture_groups.items():
    print(f"Averaging landmarks for gesture: {label} ({len(clip_names)} clips)")
    aligned_clips = []

    for clip_name in clip_names:
        clip_path = os.path.join(pose_root, clip_name)
        if not os.path.isdir(clip_path):
            continue

        frame_files = sorted([f for f in os.listdir(clip_path) if f.endswith(".json")])
        clip_poses = []

        for f in frame_files:
            with open(os.path.join(clip_path, f)) as jf:
                data = json.load(jf)
                pose_vector = []
                for i in range(landmark_count):
                    if data[i]["visibility"] > min_visibility:
                        pose_vector.append(data[i]["x"])
                    else:
                        pose_vector.append(np.nan)
                clip_poses.append(pose_vector)

        if clip_poses:
            clip_poses = np.array(clip_poses)
            aligned_clips.append(clip_poses)

    if not aligned_clips:
        print(f"⚠️ No valid data for {label}")
        continue

    # Align length
    min_len = min(arr.shape[0] for arr in aligned_clips)
    aligned_clips = [arr[:min_len] for arr in aligned_clips]
    aligned_clips = np.array(aligned_clips)

    avg_trajectory = np.nanmean(aligned_clips, axis=0)
    mean_visibility = np.nanmean(~np.isnan(aligned_clips), axis=(0, 1))

    # Plot
    plt.figure(figsize=(12, 6))
    for i in range(landmark_count):
        if mean_visibility[i] > 0.5:
            plt.plot(avg_trajectory[:, i], label=f"{i}: {landmark_names[i]}")

    plt.title(f"Gesture: {label} — Avg X Trajectories (Visible Landmarks Only)")
    plt.xlabel("Frame")
    plt.ylabel("Normalized X Position")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="x-small")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"landmark_trajectory_{label}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved plot to {out_path}")
