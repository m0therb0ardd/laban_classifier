import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Folder containing your .json frames
kp_dir = "dance_dataset/float/float_2025-05-04_13-35-08_pose_overlay_keypoints"
frame_files = sorted([f for f in os.listdir(kp_dir) if f.endswith(".json")])
landmark_count = 33

# MediaPipe landmark names
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

# Store trajectories
x_vals = [[] for _ in range(landmark_count)]
vis_vals = [[] for _ in range(landmark_count)]

# Load data
for f in frame_files:
    with open(os.path.join(kp_dir, f)) as jf:
        data = json.load(jf)
        if len(data) == landmark_count:
            for i, lm in enumerate(data):
                x_vals[i].append(lm["x"])
                vis_vals[i].append(lm["visibility"])

# Calculate mean visibility
mean_visibility = [np.mean(v) for v in vis_vals]

# Only plot landmarks with good visibility
plt.figure(figsize=(12, 6))
for i in range(landmark_count):
    if mean_visibility[i] > 0.5:
        plt.plot(x_vals[i], label=f"{i}: {landmark_names[i]}")
plt.title("X-Coordinate Trajectories (Visible Landmarks Only)")
plt.xlabel("Frame")
plt.ylabel("Normalized X Position")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="x-small")
plt.tight_layout()
plt.show()
