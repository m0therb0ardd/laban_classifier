import numpy as np
import mediapipe as mp
import joblib
import os
import time
import json
from datetime import datetime
import cv2
import csv
import tempfile
import shutil
import pandas as pd

# top of live_classify.py
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(THIS_DIR, "random_forest_model.pkl")
clf = joblib.load(MODEL_PATH)
print("Loaded model:", MODEL_PATH)
try:
    mtime = os.path.getmtime(MODEL_PATH)
    print("Model mtime:", datetime.fromtimestamp(mtime))
except Exception:
    pass


# === CONFIGURATION ===
participant_id = "Catherine"
camera_info = "RealSense D435i"
recording_location = "Fly Space"
frame_width = 640
frame_height = 480
frame_rate = 20
clip_duration = 2  # seconds
camera_index = 6
data_dict_path = "0_data_dictionary.csv"



# where the json file will live 
ROBOT_CONFIG_DEST = os.path.expanduser("~/coachbot_example/submission_repo/user/swarm_config.json")

# local file written first then copy to ROBOT_CONFIG_DEST
SWARM_CONFIG_PATH = "swarm_config.json"

# === WRITE AND COPY HELPERS === 
def atomic_write_json(path, obj):
    ddir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_swarm_config_", dir=ddir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(obj, sort_keys=True, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    except Exception:
        try:
            os.remove(tmp)
        except:
            pass
        raise

def atomic_copy_to(dest_path, src_path):
    ddir = os.path.dirname(os.path.abspath(dest_path)) or "."
    if not os.path.isdir(ddir):
        os.makedirs(ddir)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_swarm_config_", dir=ddir)
    try:
        os.close(fd)
        shutil.copy2(src_path, tmp)   # preserve mtime
        os.replace(tmp, dest_path)    # atomic on POSIX
    except Exception:
        try:
            os.remove(tmp)
        except:
            pass
        raise

# === TIMESTAMP + DIRECTORIES ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_folder = os.path.join("live_classify_logs", timestamp)
os.makedirs(session_folder, exist_ok=True)
raw_video_path = os.path.join(session_folder, f"{timestamp}_clip.mp4")

# === VIDEO SETUP ===
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, frame_rate)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(raw_video_path, fourcc, frame_rate, (frame_width, frame_height))

# === LOAD MODEL ===
# clf = joblib.load("random_forest_model.pkl")
fps = frame_rate
dt = 1 / fps
min_visibility = 0.5
n_frames = int(fps * clip_duration)

# === MEDIAPIPE SETUP ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === COUNTDOWN ===
countdown_end = time.time() + 6
while time.time() < countdown_end:
    ret, frame = cap.read()
    if not ret:
        continue
    text = f"Recording in {int(countdown_end - time.time()) + 1}s"
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow("Live Classify", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === RECORDING ===
positions = []
frame_count = 0
while frame_count < n_frames:
    ret, frame = cap.read()
    if not ret:
        continue
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_vector = []
        for lm in results.pose_landmarks.landmark:
            if lm.visibility > min_visibility:
                pose_vector.extend([lm.x, lm.y, lm.z])
            else:
                pose_vector.extend([np.nan, np.nan, np.nan])
        positions.append(pose_vector)
    else:
        positions.append([np.nan] * 99)

    cv2.putText(frame, "Recording...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Live Classify", frame)
    cv2.waitKey(1)
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()

# === SAVE SKELETON DATA ===
positions = np.array(positions)
for dim in range(positions.shape[1]):
    series = positions[:, dim]
    mask = np.isnan(series)
    if not np.all(mask):
        series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series[~mask])
    positions[:, dim] = series
positions = np.nan_to_num(positions)


# # === NORMALIZE: center all points around hip midpoint ===
# left_hip = positions[:, 23 * 3 : 23 * 3 + 3]
# right_hip = positions[:, 24 * 3 : 24 * 3 + 3]
# hip_center = (left_hip + right_hip) / 2.0
# positions -= np.repeat(hip_center, 33, axis=1)  # 33 landmarks * 3 dimensions
# === NORMALIZE (match training): center at hip midpoint AND scale by hip distance, per-frame
left_hip_index = 23
right_hip_index = 24
for i in range(positions.shape[0]):
    lhip = positions[i, left_hip_index*3:left_hip_index*3+3]
    rhip = positions[i, right_hip_index*3:right_hip_index*3+3]
    body_center = (lhip + rhip) / 2.0
    body_scale = np.linalg.norm(lhip - rhip)
    if body_scale == 0:
        body_scale = 1.0
    for j in range(33):
        s = j*3; e = s+3
        positions[i, s:e] = (positions[i, s:e] - body_center) / body_scale

# === PER FRAME SKELETON JSON ===
for i, pose in enumerate(positions):
    frame_dict = []
    for j in range(len(pose) // 3):
        frame_dict.append({
            "x": pose[j * 3 + 0],
            "y": pose[j * 3 + 1],
            "z": pose[j * 3 + 2],
            "visibility": 1.0
        })
    with open(os.path.join(session_folder, f"{i:03d}.json"), "w") as f:
        json.dump(frame_dict, f)


# --- after you finished NORMALIZE (positions is T x 99), BEFORE prediction ---

def compute_features(positions, dt):
    import numpy as np
    # indices
    I = {
        "left_shoulder":11, "right_shoulder":12,
        "left_wrist":15, "right_wrist":16,
        "left_hip":23, "right_hip":24,
        "left_ankle":27, "right_ankle":28
    }

    # helpers
    def joint_xy(A, idx):
        return A[:, idx*3+0], A[:, idx*3+1]

    def start_end_xy(A, idx):
        x, y = joint_xy(A, idx)
        return x[0], y[0], x[-1], y[-1]

    def path_len(A, idx):
        x, y = joint_xy(A, idx)
        return float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))

    def straightness(A, idx):
        x0, y0, x1, y1 = start_end_xy(A, idx)
        L = path_len(A, idx) + 1e-9
        return float(np.hypot(x1 - x0, y1 - y0) / L)

    # motion stats (must match training)
    vel  = np.gradient(positions, dt, axis=0)
    acc  = np.gradient(vel,       dt, axis=0)
    jerk = np.gradient(acc,       dt, axis=0)
    vel_mag  = np.linalg.norm(vel,  axis=1)
    acc_mag  = np.linalg.norm(acc,  axis=1)
    jerk_mag = np.linalg.norm(jerk, axis=1)

    feat = {
        "mean_velocity": float(np.mean(vel_mag)),
        "max_velocity":  float(np.max(vel_mag)),
        "std_velocity":  float(np.std(vel_mag)),
        "mean_acceleration": float(np.mean(acc_mag)),
        "max_acceleration":  float(np.max(acc_mag)),
        "std_acceleration":  float(np.std(acc_mag)),
        "mean_jerk": float(np.mean(jerk_mag)),
        "max_jerk":  float(np.max(jerk_mag)),
        "std_jerk":  float(np.std(jerk_mag)),
    }

    # ranges (must match training)
    for name, idx in {"left_wrist":15,"right_wrist":16,"left_ankle":27,"right_ankle":28}.items():
        x_vals = positions[:, idx*3+0]
        y_vals = positions[:, idx*3+1]
        feat[f"range_x_{name}"] = float(np.ptp(x_vals))
        feat[f"range_y_{name}"] = float(np.ptp(y_vals))

    # shoulder/hip reference levels
    LSh_y = float(np.mean(positions[:, I["left_shoulder"]*3+1]))
    RSh_y = float(np.mean(positions[:, I["right_shoulder"]*3+1]))
    shoulder_y = 0.5*(LSh_y + RSh_y)
    LH_y = float(np.mean(positions[:, I["left_hip"]*3+1]))
    RH_y = float(np.mean(positions[:, I["right_hip"]*3+1]))
    hip_y = 0.5*(LH_y + RH_y)
    def rel_levels(y): return float(y - shoulder_y), float(y - hip_y)

    # positional features for wrists/ankles
    for tag, jidx in [("lw", I["left_wrist"]), ("rw", I["right_wrist"]),
                      ("la", I["left_ankle"]), ("ra", I["right_ankle"])]:
        x0, y0, x1, y1 = start_end_xy(positions, jidx)
        dx, dy = (x1 - x0), (y1 - y0)
        y0_sh, y0_hip = rel_levels(y0)
        y1_sh, y1_hip = rel_levels(y1)
        L = path_len(positions, jidx)
        St = straightness(positions, jidx)
        feat[f"{tag}_x0"] = x0;  feat[f"{tag}_y0"] = y0
        feat[f"{tag}_x1"] = x1;  feat[f"{tag}_y1"] = y1
        feat[f"{tag}_dx"] = dx;  feat[f"{tag}_dy"] = dy
        feat[f"{tag}_y0_minus_sh"]  = y0_sh
        feat[f"{tag}_y0_minus_hip"] = y0_hip
        feat[f"{tag}_y1_minus_sh"]  = y1_sh
        feat[f"{tag}_y1_minus_hip"] = y1_hip
        feat[f"{tag}_path_len"]     = L
        feat[f"{tag}_straight"]     = St

    # symmetry cues (as in training)
    feat["wrist_y_diff_start"] = float(positions[0, I["right_wrist"]*3+1] - positions[0, I["left_wrist"]*3+1])
    feat["wrist_y_diff_end"]   = float(positions[-1, I["right_wrist"]*3+1] - positions[-1, I["left_wrist"]*3+1])

    return feat

# --- use it at inference ---
feat = compute_features(positions, dt)

# build X_infer in the model’s column order
cols = list(getattr(clf, "feature_names_in_", [])) or list(feat.keys())
for c in cols:
    if c not in feat:
        feat[c] = 0.0
X_infer = pd.DataFrame([[feat[c] for c in cols]], columns=cols)

# quick sanity peek
print({k: round(feat[k],3) for k in ["rw_dx","rw_dy","lw_dx","lw_dy","rw_y1_minus_sh","lw_y1_minus_sh"] if k in feat})

# then:
label = clf.predict(X_infer)[0]


# === OPTIONAL: show prediction probabilities ===
if hasattr(clf, "predict_proba"):
    print("\nModel confidence by class:")
    probs = clf.predict_proba(X_infer)[0]
    for c, p in zip(clf.classes_, probs):
        print(f"  {c:12s}: {p:.2f}")
    # Optional: show top class line too
    best_idx = int(np.argmax(probs))
    print(f"Top class: {clf.classes_[best_idx]} ({probs[best_idx]:.2f})\n")

# No confirmation prompt for now — auto-accept model label
true_label = str(label)

# === SAVE DEBUG INFO ===
debug_info = {
    "predicted_label": str(label),
    "true_label": true_label,
    "features": feat,
    "columns_used": list(X_infer.columns),
    "timestamp": timestamp,
    "session_folder": session_folder,
    "raw_video": raw_video_path
}
with open(os.path.join(session_folder, "prediction.json"), "w") as f:
    json.dump(debug_info, f, indent=2)


label_to_mode = {
    "float":      ("float",        {}),
    "glide":      ("glide",        {}),
    "handsup":    ("glitch",       {}),
    "lefthand":   ("directional",  {"direction": "left"}),
    "righthand":  ("directional",  {"direction": "right"}),
    "punch":      ("punch",        {}),
    "slash":      ("slash",        {}),
    "stillness":  ("encircling",   {}),
}

# === MAP LABEL -> MODE (MINIMAL CONFIG), WRITE JSON, COPY TO ROBOT REPO ===
label_lc = true_label.strip().lower()
mode, extras = label_to_mode.get(label_lc, ("encircling", {}))

cfg_obj = {
    "version": 1,
    "mode": mode,
    "timestamp": time.time(),
    "source": {
        "type": "live_classify",
        "session_folder": session_folder,
        "raw_video": raw_video_path,
        "label": true_label
    },
    "extras": extras if extras else {}
}

atomic_write_json(SWARM_CONFIG_PATH, cfg_obj)
print(f"Wrote {SWARM_CONFIG_PATH} with mode={mode}")
atomic_copy_to(ROBOT_CONFIG_DEST, SWARM_CONFIG_PATH)
print(f"Copied config to robot folder: {ROBOT_CONFIG_DEST}")

# === OPTIONAL: APPEND TO DATA DICTIONARY ===
save = input("Save this sample to data dictionary? (y/n): ").strip().lower()
if save == "y":
    file_exists = os.path.isfile(data_dict_path)
    with open(data_dict_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "timestamp", "clip_id", "label", "start_time", "end_time", "quality", "notes",
                "frame_rate", "resolution", "camera_info", "participant_id", "recording_location",
                "raw_video_path", "source_type"
            ])
        writer.writerow([
            timestamp,
            f"{timestamp}_{true_label}_live.mp4",
            true_label,
            "0.0",
            str(clip_duration),
            "unrated",
            "live classify sample",
            frame_rate,
            f"{frame_width}x{frame_height}",
            camera_info,
            participant_id,
            recording_location,
            raw_video_path,
            "live_classify"
        ])

print(f"\nSaved to {session_folder}")
