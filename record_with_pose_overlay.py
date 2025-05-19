# record_with_pose_overlay.py

import cv2
import mediapipe as mp
import os
from datetime import datetime

# --- Settings ---
label = input("Enter movement label (e.g., float, punch): ").strip().lower()
duration = 50  # seconds
fps = 20
frame_width = 640
frame_height = 480

# --- File paths ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"full_body_dance_dataset/{label}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/{label}_{timestamp}_pose_overlay.mp4"

# --- Setup video writer ---
out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
cap = cv2.VideoCapture(6)

# --- Setup MediaPipe Pose ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5)

# --- Start recording ---
frame_count = 0
max_frames = int(duration * fps)
print(f"Recording {label} for {duration} seconds with pose overlay...")

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Save frame with overlay
    out.write(frame)

    # Show live preview
    cv2.imshow("Pose Overlay Recording (Press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# --- Cleanup ---
cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()

print(f"Saved to {filename}")



# import cv2
# import mediapipe as mp

# # --- Settings ---
# camera_index =0  # You said RealSense RGB is at index 6
# frame_width = 640
# frame_height = 480

# # --- Setup webcam ---
# cap = cv2.VideoCapture(camera_index)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# if not cap.isOpened():
#     print(f"❌ Failed to open camera at index {camera_index}")
#     exit()

# # --- Setup MediaPipe Pose ---
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(min_detection_confidence=0.5)

# print("🎥 RealSense Camera with Pose Tracking. Press 'q' to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Failed to read frame from camera.")
#         continue

#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(image_rgb)

#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame,
#             results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS
#         )

#     cv2.imshow("🧍 RealSense Pose Tracker (Press Q to Quit)", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # --- Cleanup ---
# cap.release()
# cv2.destroyAllWindows()
# pose.close()
