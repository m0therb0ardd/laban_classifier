import os
import cv2
import pandas as pd
from config import session_timestamp

# === CONFIG ===
# session_timestamp = "2025-05-19_19-53-37"
clip_dir = os.path.join("2_extracted_clips", session_timestamp)
data_dict_path = "0_data_dictionary.csv"

# === Load data dictionary ===
df = pd.read_csv(data_dict_path)
if 'gesture_captured' not in df.columns:
    df['gesture_captured'] = ""

# === Review Loop ===
video_files = sorted([f for f in os.listdir(clip_dir) if f.endswith(".mp4")])

for video_file in video_files:
    clip_id = video_file
    current_entry = df[df["clip_id"] == clip_id]
    
    # Skip if already reviewed
    if not current_entry.empty and current_entry.iloc[0]["gesture_captured"] in ["yes", "no"]:
        print(f"Skipping reviewed: {clip_id}")
        continue

    cap = cv2.VideoCapture(os.path.join(clip_dir, video_file))
    if not cap.isOpened():
        print(f"⚠️ Couldn't open {video_file}")
        continue

    print(f"\nLooping: {clip_id} (Press: y=Yes, n=No, q=Quit)")
    decision = None
    
    while decision is None:  # Loop until decision made
        while True:  # Video playback loop
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to start
                continue
            
            # Add instructional text
            display_frame = frame.copy()
            cv2.putText(display_frame, "Review Loop - [y]Yes [n]No [q]Quit", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(display_frame, clip_id, 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            
            cv2.imshow("Gesture Review", display_frame)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                print(" Quitting program...")
                cap.release()
                cv2.destroyAllWindows()
                df.to_csv(data_dict_path, index=False)
                exit()
            elif key in [ord('y'), ord('n')]:
                decision = 'yes' if key == ord('y') else 'no'
                break

    cap.release()
    cv2.destroyAllWindows()
    
    df.loc[df["clip_id"] == clip_id, "gesture_captured"] = decision
    print(f"Marked '{clip_id}' as: {decision}")
    df.to_csv(data_dict_path, index=False)  # Save after each video

print("All clips reviewed for this session!")