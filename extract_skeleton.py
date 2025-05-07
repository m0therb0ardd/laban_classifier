import os
import cv2
import json
import mediapipe as mp

input_dir = "dance_dataset/punch"  # or punch, etc.
video_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mp4")])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    output_dir = video_path.replace(".mp4", "_keypoints")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f" Processing {video_file} ({frame_count} frames)...")

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
