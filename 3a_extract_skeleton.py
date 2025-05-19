import os
import cv2
import json
import pandas as pd
import mediapipe as mp

# === CONFIG ===
session_timestamp = "2025-05-18_22-35-34"
data_dict_path = "0_data_dictionary.csv"
clip_dir = os.path.join("2_extracted_clips", session_timestamp)
output_root = os.path.join("3_skeleton_pose_data", session_timestamp)
os.makedirs(output_root, exist_ok=True)

# === MediaPipe Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# === Load Data Dictionary ===
df = pd.read_csv(data_dict_path)
df = df[(df["timestamp"] == session_timestamp) & (df["gesture_captured"] == "yes")]

# === Process Each Valid Clip ===
for _, row in df.iterrows():
    clip_id = row["clip_id"]
    video_path = os.path.join(clip_dir, clip_id)
    output_dir = os.path.join(output_root, clip_id.replace(".mp4", ""))
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        continue

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = [{
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            } for lm in results.pose_landmarks.landmark]

            out_file = os.path.join(output_dir, f"frame_{frame_idx:04d}.json")
            with open(out_file, "w") as f:
                json.dump(keypoints, f)

        frame_idx += 1

    cap.release()
    print(f"✅ Saved {frame_idx} frames to {output_dir}")

pose.close()
