# live_classify_sequence.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import os
import json
from datetime import datetime

# Load trained model
clf = joblib.load("random_forest_model.pkl")

# Constants
fps = 20
duration = 2  # seconds
n_frames = fps * duration
dt = 1 / fps
# wrist_index = 16
min_visibility = 0.5

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Setup output directory and video writer
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join("live_debug_logs", timestamp)
os.makedirs(log_dir, exist_ok = True)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_path = os.path.join(log_dir, "video.avi")
video_writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)


# Countdown
print("Recording starts in 3 seconds...")
time.sleep(3)

positions = []

frame_count = 0
start_time = time.time()

while frame_count < n_frames:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_vector = []
        for lm in results.pose_landmarks.landmark:
            if lm.visibility > min_visibility:
                pose_vector.extend([lm.x, lm.y, lm.z])
            else:
                pose_vector.extend([np.nan, np.nan, np.nan])
        positions.append(pose_vector)

    else:
        positions.append([np.nan] * (33 * 3))  # 33 landmarks * 3 coordinates

    # Add label to video 
    cv2.putText(frame, f"Capturing frame {frame_count+1}/{n_frames}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    frame_count += 1
    cv2.imshow("Capturing Movement...", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()

print("Finished capturing movement.")


# Save raw skeleton JSON
positions = np.array(positions)
for i, pose_frame in enumerate(positions):
    frame_dict = []
    for j in range(len(pose_frame) // 3):
        frame_dict.append({
            "x": pose_frame[j * 3 + 0],
            "y": pose_frame[j * 3 + 1],
            "z": pose_frame[j * 3 + 2],
            "visibility": 1.0
        })
    with open(os.path.join(log_dir, f"{i:03d}.json"), "w") as f:
        json.dump(frame_dict, f)

# Fill in missing values
for dim in range(positions.shape[1]):
    series = positions[:, dim]
    mask = np.isnan(series)
    if not np.all(mask):
        series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series[~mask])
    positions[:, dim] = series
    positions = np.nan_to_num(positions, nan=0.0)


# Compute features
vel = np.gradient(positions, dt, axis=0)
acc = np.gradient(vel, dt, axis=0)
jerk = np.gradient(acc, dt, axis=0)

vel_mag = np.linalg.norm(vel, axis=1)
acc_mag = np.linalg.norm(acc, axis=1)
jerk_mag = np.linalg.norm(jerk, axis=1)

# === Define landmark indices for right wrist, right ankle, left ankle
landmark_indices = {
    "right_wrist": 16,
    "right_ankle": 28,
    "left_ankle": 27
}

range_features = []
for name, index in landmark_indices.items():
    x_vals = positions[:, index * 3 + 0]
    y_vals = positions[:, index * 3 + 1]
    range_x = np.max(x_vals) - np.min(x_vals)
    range_y = np.max(y_vals) - np.min(y_vals)
    range_features.extend([range_x, range_y])

# Build final feature vector 
features = np.array([[
    np.mean(vel_mag), np.max(vel_mag), np.std(vel_mag),
    np.mean(acc_mag), np.max(acc_mag), np.std(acc_mag),
    np.mean(jerk_mag), np.max(jerk_mag), np.std(jerk_mag),
    *range_features
]])

# Predict
label = clf.predict(features)[0]
print(f"Predicted movement: **{label.upper()}**")

# Save metadata
debug_info = {
    "predicted_label": label,
    "features": features.tolist()
}
with open(os.path.join(log_dir, "prediction.json"), "w") as f:
    json.dump(debug_info, f, indent=2)

print(f"Debug logs saved to: {log_dir}")


