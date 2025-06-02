# # live_classify_sequence.py
# import cv2
# import numpy as np
# import mediapipe as mp
# import joblib
# import time
# import os
# import json
# from datetime import datetime

# # Load trained model
# clf = joblib.load("random_forest_model.pkl")

# # Constants
# fps = 20
# duration = 2  # seconds
# n_frames = fps * duration
# dt = 1 / fps
# # wrist_index = 16
# min_visibility = 0.5

# # Setup MediaPipe
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # Initialize webcam
# cap = cv2.VideoCapture(6)

# # Setup output directory and video writer
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# log_dir = os.path.join("live_debug_logs", timestamp)
# os.makedirs(log_dir, exist_ok = True)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# video_path = os.path.join(log_dir, "video.avi")
# video_writer = cv2.VideoWriter(
#     video_path,
#     cv2.VideoWriter_fourcc(*'XVID'),
#     fps,
#     (frame_width, frame_height)
# )


# # Countdown
# print("Recording starts in 6 seconds...")
# time.sleep(6)

# positions = []

# frame_count = 0
# start_time = time.time()

# while frame_count < n_frames:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(image_rgb)

#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#         pose_vector = []
#         for lm in results.pose_landmarks.landmark:
#             if lm.visibility > min_visibility:
#                 pose_vector.extend([lm.x, lm.y, lm.z])
#             else:
#                 pose_vector.extend([np.nan, np.nan, np.nan])
#         positions.append(pose_vector)

#     else:
#         positions.append([np.nan] * (33 * 3))  # 33 landmarks * 3 coordinates

#     # Add label to video 
#     cv2.putText(frame, f"Capturing frame {frame_count+1}/{n_frames}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
#     frame_count += 1
#     cv2.imshow("Capturing Movement...", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# pose.close()

# print("Finished capturing movement.")


# # Save raw skeleton JSON
# positions = np.array(positions)
# for i, pose_frame in enumerate(positions):
#     frame_dict = []
#     for j in range(len(pose_frame) // 3):
#         frame_dict.append({
#             "x": pose_frame[j * 3 + 0],
#             "y": pose_frame[j * 3 + 1],
#             "z": pose_frame[j * 3 + 2],
#             "visibility": 1.0
#         })
#     with open(os.path.join(log_dir, f"{i:03d}.json"), "w") as f:
#         json.dump(frame_dict, f)

# # Fill in missing values
# for dim in range(positions.shape[1]):
#     series = positions[:, dim]
#     mask = np.isnan(series)
#     if not np.all(mask):
#         series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series[~mask])
#     positions[:, dim] = series
#     positions = np.nan_to_num(positions, nan=0.0)

# # Normalize positions: center at midpoint of hips (left=23, right=24)
# # movement is measured relative to hips not camera
# # motion features now focus on how the limbs mooe not where the person is standing
# left_hip = positions[:, 23*3 : 23*3+3]
# right_hip = positions[:, 24*3 : 24*3+3]
# hip_center = (left_hip + right_hip) / 2.0
# positions -= np.repeat(hip_center, 33, axis=1)  # broadcast subtract




# # Compute features
# vel = np.gradient(positions, dt, axis=0)
# acc = np.gradient(vel, dt, axis=0)
# jerk = np.gradient(acc, dt, axis=0)

# vel_mag = np.linalg.norm(vel, axis=1)
# acc_mag = np.linalg.norm(acc, axis=1)
# jerk_mag = np.linalg.norm(jerk, axis=1)

# # === Define landmark indices for right wrist, right ankle, left ankle
# landmark_indices = {
#     "right_wrist": 16,
#     "right_ankle": 28,
#     "left_ankle": 27
# }

# range_features = []
# for name, index in landmark_indices.items():
#     x_vals = positions[:, index * 3 + 0]
#     y_vals = positions[:, index * 3 + 1]
#     range_x = np.max(x_vals) - np.min(x_vals)
#     range_y = np.max(y_vals) - np.min(y_vals)
#     range_features.extend([range_x, range_y])

# # Build final feature vector 
# features = np.array([[
#     np.mean(vel_mag), np.max(vel_mag), np.std(vel_mag),
#     np.mean(acc_mag), np.max(acc_mag), np.std(acc_mag),
#     np.mean(jerk_mag), np.max(jerk_mag), np.std(jerk_mag),
#     *range_features
# ]])

# # Predict
# label = clf.predict(features)[0]
# print(f"Predicted movement: **{label.upper()}**")

# # Prompt for true label
# true_label = input(f"Model predicted **{label.upper()}**. Enter correct label if wrong (or press Enter to confirm): ")
# if true_label.strip() == "":
#     true_label = label

# # Save prediction and features 
# debug_info = {
#     "predicted_label": label,
#     "true_label": true_label,
#     "features": features.tolist()
# }
# with open(os.path.join(log_dir, "prediction.json"), "w") as f:
#     json.dump(debug_info, f, indent=2)

# print(f"Debug logs saved to: {log_dir}")

import numpy as np
import mediapipe as mp
import joblib
import os
import time
import json
from datetime import datetime
import cv2
import csv


# === CONFIGURATION ===
participant_id = "Catherine"
camera_info = "RealSense D435i"
recording_location = "Fly Space"
frame_width = 640
frame_height = 480
frame_rate = 20
clip_duration = 2  # seconds
camera_index = 6
data_dict_path = "0_data_dictionary.csv"

# === TIMESTAMP + DIRECTORIES ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_folder = os.path.join("live_classify_logs", timestamp)
os.makedirs(session_folder, exist_ok=True)
raw_video_path = os.path.join(session_folder, f"{timestamp}_clip.mp4")

# === VIDEO SETUP ===
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, frame_rate)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(raw_video_path, fourcc, frame_rate, (frame_width, frame_height))

# === LOAD MODEL ===
clf = joblib.load("random_forest_model.pkl")
fps = frame_rate
dt = 1 / fps
min_visibility = 0.5
n_frames = int(fps * clip_duration)

# === MEDIAPIPE SETUP ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === COUNTDOWN ===
countdown_end = time.time() + 6
while time.time() < countdown_end:
    ret, frame = cap.read()
    if not ret:
        continue
    text = f"Recording in {int(countdown_end - time.time()) + 1}s"
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow("Live Classify", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === RECORDING ===
positions = []
frame_count = 0
while frame_count < n_frames:
    ret, frame = cap.read()
    if not ret:
        continue
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
        positions.append([np.nan] * 99)

    cv2.putText(frame, "Recording...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Live Classify", frame)
    cv2.waitKey(1)
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()

# === SAVE SKELETON DATA ===
positions = np.array(positions)
for dim in range(positions.shape[1]):
    series = positions[:, dim]
    mask = np.isnan(series)
    if not np.all(mask):
        series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series[~mask])
    positions[:, dim] = series
positions = np.nan_to_num(positions)


# === NORMALIZE: center all points around hip midpoint ===
left_hip = positions[:, 23 * 3 : 23 * 3 + 3]
right_hip = positions[:, 24 * 3 : 24 * 3 + 3]
hip_center = (left_hip + right_hip) / 2.0
positions -= np.repeat(hip_center, 33, axis=1)  # 33 landmarks * 3 dimensions


for i, pose in enumerate(positions):
    frame_dict = []
    for j in range(len(pose) // 3):
        frame_dict.append({
            "x": pose[j * 3 + 0],
            "y": pose[j * 3 + 1],
            "z": pose[j * 3 + 2],
            "visibility": 1.0
        })
    with open(os.path.join(session_folder, f"{i:03d}.json"), "w") as f:
        json.dump(frame_dict, f)

# === COMPUTE FEATURES ===
vel = np.gradient(positions, dt, axis=0)
acc = np.gradient(vel, dt, axis=0)
jerk = np.gradient(acc, dt, axis=0)
vel_mag = np.linalg.norm(vel, axis=1)
acc_mag = np.linalg.norm(acc, axis=1)
jerk_mag = np.linalg.norm(jerk, axis=1)

landmark_indices = {"right_wrist": 16, "right_ankle": 28, "left_ankle": 27}
range_features = []
for name, index in landmark_indices.items():
    x_vals = positions[:, index * 3 + 0]
    y_vals = positions[:, index * 3 + 1]
    range_features.extend([np.max(x_vals) - np.min(x_vals), np.max(y_vals) - np.min(y_vals)])

features = np.array([[
    np.mean(vel_mag), np.max(vel_mag), np.std(vel_mag),
    np.mean(acc_mag), np.max(acc_mag), np.std(acc_mag),
    np.mean(jerk_mag), np.max(jerk_mag), np.std(jerk_mag),
    *range_features
]])

# === PREDICT AND CONFIRM LABEL ===
label = clf.predict(features)[0]
true_label = input(f"Model predicted **{label.upper()}**. Enter correct label if wrong (or press Enter to confirm): ")
if true_label.strip() == "":
    true_label = label

# === SAVE DEBUG INFO ===
debug_info = {
    "predicted_label": label,
    "true_label": true_label,
    "features": features.tolist()
}
with open(os.path.join(session_folder, "prediction.json"), "w") as f:
    json.dump(debug_info, f, indent=2)

# === OPTIONAL: APPEND TO DATA DICTIONARY ===
save = input("Save this sample to data dictionary? (y/n): ").strip().lower()
if save == "y":
    file_exists = os.path.isfile(data_dict_path)
    with open(data_dict_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "timestamp", "clip_id", "label", "start_time", "end_time", "quality", "notes",
                "frame_rate", "resolution", "camera_info", "participant_id", "recording_location",
                "raw_video_path", "source_type"
            ])
        writer.writerow([
            timestamp,
            f"{timestamp}_{true_label}_live.mp4",
            true_label,
            "0.0",
            str(clip_duration),
            "unrated",
            "live classify sample",
            frame_rate,
            f"{frame_width}x{frame_height}",
            camera_info,
            participant_id,
            recording_location,
            raw_video_path,
            "live_classify"
        ])

print(f"\nSaved to {session_folder}")
