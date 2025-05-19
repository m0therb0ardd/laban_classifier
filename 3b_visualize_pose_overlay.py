import os
import cv2
import json
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList

# === CONFIG ===
session_timestamp = "2025-05-18_21-51-32"  # change to your session
clip_name = "2025-05-18_21-51-32_punch_003"  # full clip name without extension

video_path = os.path.join("2_extracted_clips", session_timestamp, f"{clip_name}.mp4")
keypoints_dir = os.path.join("3_skeleton_pose_data", session_timestamp, clip_name)

# === SETUP ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# === Load video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    keypoint_file = os.path.join(keypoints_dir, f"frame_{frame_idx:04d}.json")
    if os.path.isfile(keypoint_file):
        with open(keypoint_file, "r") as f:
            keypoints = json.load(f)

        # Convert to normalized landmark format for drawing
        landmarks = []
        for kp in keypoints:
            landmark = NormalizedLandmark()
            landmark.x = kp["x"]
            landmark.y = kp["y"]
            landmark.z = kp["z"]
            landmark.visibility = kp["visibility"]
            landmarks.append(landmark)

        landmark_list = NormalizedLandmarkList(landmark=landmarks)

        # Draw pose
        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(
            annotated_frame,
            landmark_list,
            POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
    else:
        annotated_frame = frame

    cv2.imshow("Pose Overlay", annotated_frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
