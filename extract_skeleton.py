# extract_skeletons.py
import cv2
import mediapipe as mp
import os
import numpy as np
import json

# Setup paths
input_video = "dance_dataset/float/float_2024-04-28_14-33-10.mp4"
output_dir = input_video.replace(".mp4", "_keypoints")
os.makedirs(output_dir, exist_ok=True)

# Init MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open video
cap = cv2.VideoCapture(input_video)
frame_idx = 0

print(f"Processing: {input_video}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        # Extract (x, y, z, visibility) for each of 33 keypoints
        keypoints = [{
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility
        } for lm in results.pose_landmarks.landmark]

        # Save to .json
        with open(f"{output_dir}/frame_{frame_idx:04d}.json", "w") as f:
            json.dump(keypoints, f)
    
    frame_idx += 1

cap.release()
pose.close()

print(f"Finished! Saved {frame_idx} frames of keypoints to {output_dir}/")
