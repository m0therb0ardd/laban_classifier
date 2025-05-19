import cv2
import csv
import os
import time
from datetime import datetime

# === CONFIGURATION ===
participant_id = "Catherine"
camera_info = "RealSense D435i"
recording_location = "Fly Space"
frame_width = 640
frame_height = 480
frame_rate = 30
camera_index = 4 #4 or 6 for realsense plug in
clip_duration = 2  # seconds
movements = ["float"] * 10 + ["punch"] * 10 # 50 float and 50 punch for now
data_dict_path = "0_data_dictionary.csv"

# === GENERATE TIMESTAMP ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_folder = os.path.join("1_recordings", timestamp)
os.makedirs(session_folder, exist_ok=True)
raw_video_path = os.path.join(session_folder, "raw_recording.mp4")

# === SETUP VIDEO CAPTURE AND WRITER ===
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, frame_rate)

if not cap.isOpened():
    print("Failed to open camera. Check camera_index")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(raw_video_path, fourcc, frame_rate, (frame_width, frame_height))

# === INITIALIZE OR APPEND TO DATA DICTIONARY ===
file_exists = os.path.isfile(data_dict_path)
with open(data_dict_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow([
            "timestamp", "clip_id", "label", "start_time", "end_time", "quality", "notes",
            "frame_rate", "resolution", "camera_info", "participant_id", "recording_location",
            "raw_video_path"
        ])

    print(f"Starting recording: {len(movements)} clips")
    print("Output directory:", session_folder)

    start_session_time = time.time()

    for i, label in enumerate(movements):
        clip_start = time.time()
        start_offset = clip_start - start_session_time
        # clip_id = f"{label}_{i+1:03}.mp4"
        clip_id = f"{timestamp}_{label}_{i+1:03}.mp4"


        print(f"\n▶️ {i+1}/{len(movements)} — {label.upper()} | Clip: {clip_id}")

        # Countdown before movement
        countdown_end = time.time() + 2
        while time.time() < countdown_end:
            ret, frame = cap.read()
            if not ret:
                continue
            text = f"Next: {label.upper()} in {int(countdown_end - time.time()) + 1}s"
            cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imshow("Recording", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Actual movement recording with pause resume and quit 
        frame_counter = 0
        paused = False

        while frame_counter < int(frame_rate * clip_duration):
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('p'):
                paused = True
                print("Paused. Press 'r' to resume.")

            while paused:
                ret, pause_frame = cap.read()
                if not ret:
                    continue
                cv2.putText(pause_frame, "PAUSED - Press 'r' to resume", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("Recording", pause_frame)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('r'):
                    paused = False
                    print("Resumed.")
                    break
                elif k == ord('q'):
                    exit()

            ret, frame = cap.read()
            if not ret:
                continue
            cv2.putText(frame, f"{label.upper()}...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("Recording", frame)
            out.write(frame)
            frame_counter += 1


        end_offset = time.time() - start_session_time
        writer.writerow([
            timestamp,
            clip_id,
            label,
            f"{start_offset:.2f}",
            f"{end_offset:.2f}",
            "unrated",
            "",
            frame_rate,
            f"{frame_width}x{frame_height}",
            camera_info,
            participant_id,
            recording_location,
            raw_video_path
        ])

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()
print("\nFinished session:", timestamp)
