import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get x-coordinates of wrists and hips
        left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
        right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
        left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
        right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x

        # Compute horizontal distances between wrists and their corresponding hips
        left_dist = abs(left_wrist_x - left_hip_x)
        right_dist = abs(right_wrist_x - right_hip_x)

        if left_dist > right_dist + 0.05:  # margin to avoid noise
            print("Left hand extended")
        elif right_dist > left_dist + 0.05:
            print("Right hand extended")
        else:
            print("No clear hand extension")

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
