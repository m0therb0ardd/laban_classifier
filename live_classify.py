# live_classify_sequence.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time

# Load trained model
clf = joblib.load("random_forest_model.pkl")

# Constants
fps = 20
duration = 5  # seconds
n_frames = fps * duration
dt = 1 / fps
wrist_index = 16
min_visibility = 0.5

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

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
        kp = results.pose_landmarks.landmark[wrist_index]
        if kp.visibility > min_visibility:
            positions.append([kp.x, kp.y, kp.z])
        else:
            positions.append([np.nan, np.nan, np.nan])
    else:
        positions.append([np.nan, np.nan, np.nan])

    frame_count += 1
    cv2.imshow("Capturing Movement...", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()

print("Finished capturing movement.")

# Fill in missing values
positions = np.array(positions)
for dim in range(3):
    series = positions[:, dim]
    mask = np.isnan(series)
    if not np.all(mask):
        series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series[~mask])
    positions[:, dim] = series

# Compute features
vel = np.gradient(positions, dt, axis=0)
acc = np.gradient(vel, dt, axis=0)
jerk = np.gradient(acc, dt, axis=0)

vel_mag = np.linalg.norm(vel, axis=1)
acc_mag = np.linalg.norm(acc, axis=1)
jerk_mag = np.linalg.norm(jerk, axis=1)

features = np.array([[
    np.mean(vel_mag), np.max(vel_mag), np.std(vel_mag),
    np.mean(acc_mag), np.max(acc_mag), np.std(acc_mag),
    np.mean(jerk_mag), np.max(jerk_mag), np.std(jerk_mag)
]])


# Classify
label = clf.predict(features)[0]
print(f"Predicted movement: **{label.upper()}**")
