# record_dance_video.py

import cv2
import os
from datetime import datetime

# Prompt for label
label = input("Enter movement label (e.g., float, punch): ").strip().lower()

# Create output folder
output_dir = f"dance_dataset/{label}"
os.makedirs(output_dir, exist_ok=True)

# Create timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{output_dir}/{label}_{timestamp}.mp4"

# Video capture settings
fps = 20.0
frame_width = 640
frame_height = 480
duration = 5  # seconds

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

print(f"Recording {label} at {timestamp}...")
frame_count = 0
max_frames = int(fps * duration)

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame)
    cv2.imshow(f'Recording {label} (press Q to quit early)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved to {filename}")
