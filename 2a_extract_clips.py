import cv2
import csv
import os

# === CONFIGURATION ===
data_dict_path = "0_data_dictionary.csv"
output_root = "2_extracted_clips"

# === HELPERS ===
def extract_clip(video_path, clip_start, clip_end, out_path, fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(frame_count / fps, 2)  # round to 2 decimal places

    # Validate timing
    if clip_start > duration:
        print(f"⚠️ Clip start beyond video length: {clip_start:.2f}s > {duration:.2f}s")
        return

    # Cap clip_end to the video length if it's too long
    if clip_end > duration:
        print(f"⚠️ Capping end time: {clip_end:.2f}s → {duration:.2f}s")
        clip_end = duration


    cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000)

    while cap.get(cv2.CAP_PROP_POS_MSEC) < clip_end * 1000:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

# === MAIN ===
with open(data_dict_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        timestamp = row["timestamp"]
        clip_id = row["clip_id"]
        label = row["label"]
        start_time = float(row["start_time"]) +2 
        end_time = float(row["end_time"])
        raw_video_path = row["raw_video_path"]
        fps = int(row["frame_rate"])



        out_dir = os.path.join(output_root, timestamp)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, clip_id)
        if os.path.isfile(out_path):
            print(f"Already exists: {clip_id}")
            continue

        print(f"✂️ Extracting {clip_id}...")
        extract_clip(raw_video_path, start_time, end_time, out_path, fps)

print("Done extracting all clips.")
