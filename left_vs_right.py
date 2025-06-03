# import cv2
# import mediapipe as mp

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(image)

#     if results.pose_landmarks:
#         landmarks = results.pose_landmarks.landmark

#         # Get x-coordinates of wrists and hips
#         left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
#         right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
#         left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
#         right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x

#         # Compute horizontal distances between wrists and their corresponding hips
#         left_dist = abs(left_wrist_x - left_hip_x)
#         right_dist = abs(right_wrist_x - right_hip_x)

#         if left_dist > right_dist + 0.05:  # margin to avoid noise
#             print("Left hand extended")
#         elif right_dist > left_dist + 0.05:
#             print("Right hand extended")
#         else:
#             print("No clear hand extension")

#     cv2.imshow('Webcam', frame)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

gesture_history = []
GESTURE_THRESHOLD = 5
LAST_SENT = None

def get_gesture(landmarks):
    left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
    right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
    left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
    right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x

    left_dist = abs(left_wrist_x - left_hip_x)
    right_dist = abs(right_wrist_x - right_hip_x)

    if left_dist > right_dist + 0.05:
        return "LEFT"
    elif right_dist > left_dist + 0.05:
        return "RIGHT"
    else:
        return "NONE"

def write_command(command):
    with open("command.txt", "w") as f:
        f.write(command)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        current_gesture = get_gesture(landmarks)
        gesture_history.append(current_gesture)

        if len(gesture_history) > GESTURE_THRESHOLD:
            gesture_history.pop(0)

        # Check if all recent gestures are the same (and not NONE)
        if (
            len(gesture_history) == GESTURE_THRESHOLD
            and all(g == current_gesture for g in gesture_history)
            and current_gesture != "NONE"
            and current_gesture != LAST_SENT
        ):
            print(f"Sending command: {current_gesture}")
            write_command(current_gesture)
            LAST_SENT = current_gesture
    else:
        gesture_history = []

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
