# import os
# import json
# import numpy as np
# import pandas as pd
# from config import session_timestamp


# session_timestamp = "2025-10-10_13-28-09"
# output_root = "4_extracted_motion_features"
# output_dir = os.path.join(output_root, session_timestamp)
# os.makedirs(output_dir, exist_ok=True)

# # Parameters
# pose_root = os.path.join("3_skeleton_pose_data", session_timestamp)
# fps = 20
# dt = 1 / fps
# min_visibility = 0.5

# # Landmarks of interest
# landmark_indices = {
#     "left_wrist": 15,
#     "right_wrist": 16,
#     "left_ankle": 27,
#     "right_ankle": 28
# }

# all_features = []

# # Loop through clips  
# for clip_name in os.listdir(pose_root):
#     clip_path = os.path.join(pose_root, clip_name)
#     if not os.path.isdir(clip_path):
#         continue

#     frame_files = sorted([f for f in os.listdir(clip_path) if f.endswith(".json")])
#     positions = []

#     for f in frame_files:
#         with open(os.path.join(clip_path, f)) as jf:
#             data = json.load(jf)
#             pose_vector = []
#             for i in range(len(data)):  # For all landmarks in the frame
#                 if data[i]["visibility"] > min_visibility:
#                     pose_vector.extend([data[i]["x"], data[i]["y"], data[i]["z"]])
#                 else:
#                     pose_vector.extend([np.nan, np.nan, np.nan])
#             positions.append(pose_vector)

#     positions = np.array(positions)

#     # Debug print
#     print(f"🔍 Clip: {clip_name}, Frames: {len(frame_files)}, Pose shape: {positions.shape}")

#     # Defensive guard
#     if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] == 0:
#         print(f"⚠️ Skipping {clip_name}: no usable pose data or invalid structure")
#         continue


#     # Interpolate missing values
#     for dim in range(positions.shape[1]):
#         series = positions[:, dim]
#         mask = np.isnan(series)
#         if not np.all(mask):
#             series[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), series[~mask])
#         positions[:, dim] = series

#     # Replace any remaining NaNs with zeros --> we need this so we dont get all NaN values when we do gradient for velocity adn acceleration later
#     positions = np.nan_to_num(positions, nan=0.0)

#     # === Normalize landmarks per frame: center and scale by hip distance
#     left_hip_index = 23
#     right_hip_index = 24

#     for i in range(positions.shape[0]):
#         # Get left and right hip (x, y, z)
#         lhip = positions[i, left_hip_index*3:left_hip_index*3+3]
#         rhip = positions[i, right_hip_index*3:right_hip_index*3+3]

#         # Compute body center and scale
#         body_center = (lhip + rhip) / 2
#         body_scale = np.linalg.norm(lhip - rhip)

#         if body_scale == 0:  # avoid divide-by-zero
#             body_scale = 1.0

#         # Normalize all landmarks
#         for j in range(33):  # 33 landmarks
#             start = j * 3
#             end = start + 3
#             positions[i, start:end] = (positions[i, start:end] - body_center) / body_scale

#     # Compute motion features
#     vel = np.gradient(positions, dt, axis=0)
#     acc = np.gradient(vel, dt, axis=0)
#     jerk = np.gradient(acc, dt, axis=0)

#     vel_mag = np.linalg.norm(vel, axis=1)
#     acc_mag = np.linalg.norm(acc, axis=1)
#     jerk_mag = np.linalg.norm(jerk, axis=1)



#     print("clip_name:", clip_name, "→", clip_name.split("_"))

#     # === 1. Pose indices (MediaPipe reference map) ===
#     I = {
#         "nose":0, "left_shoulder":11, "right_shoulder":12,
#         "left_elbow":13, "right_elbow":14, "left_wrist":15, "right_wrist":16,
#         "left_hip":23, "right_hip":24, "left_ankle":27, "right_ankle":28
#     }

#     # Convenience helpers
#     def joint_xy(idx):
#         x = positions[:, idx*3 + 0]
#         y = positions[:, idx*3 + 1]
#         return x, y

#     def start_end_xy(idx):
#         x, y = joint_xy(idx)
#         return x[0], y[0], x[-1], y[-1]

#     def path_len(idx):
#         x, y = joint_xy(idx)
#         return float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))

#     def straightness(idx):
#         x0, y0, x1, y1 = start_end_xy(idx)
#         net = np.hypot(x1 - x0, y1 - y0)
#         L = path_len(idx) + 1e-9
#         return float(net / L)  # 1.0 = perfectly straight line

#     # Shoulder/hip reference levels
#     LSh_y = np.mean(positions[:, I["left_shoulder"]*3 + 1])
#     RSh_y = np.mean(positions[:, I["right_shoulder"]*3 + 1])
#     shoulder_y = 0.5 * (LSh_y + RSh_y)

#     LH_y = np.mean(positions[:, I["left_hip"]*3 + 1])
#     RH_y = np.mean(positions[:, I["right_hip"]*3 + 1])
#     hip_y = 0.5 * (LH_y + RH_y)

#     def rel_to_levels(y):
#         return float(y - shoulder_y), float(y - hip_y)

#     # === 2. Wrist & ankle positional features ===
#     JOINTS = [
#         ("lw", I["left_wrist"]),
#         ("rw", I["right_wrist"]),
#         ("la", I["left_ankle"]),
#         ("ra", I["right_ankle"])
#     ]

#     pos_feats = {}
#     for tag, idx in JOINTS:
#         x0, y0, x1, y1 = start_end_xy(idx)
#         dx, dy = (x1 - x0), (y1 - y0)

#         y0_sh, y0_hip = rel_to_levels(y0)
#         y1_sh, y1_hip = rel_to_levels(y1)

#         L = path_len(idx)
#         St = straightness(idx)

#         pos_feats.update({
#             f"{tag}_x0": x0, f"{tag}_y0": y0,
#             f"{tag}_x1": x1, f"{tag}_y1": y1,
#             f"{tag}_dx": dx, f"{tag}_dy": dy,
#             f"{tag}_y0_minus_sh": y0_sh, f"{tag}_y0_minus_hip": y0_hip,
#             f"{tag}_y1_minus_sh": y1_sh, f"{tag}_y1_minus_hip": y1_hip,
#             f"{tag}_path_len": L, f"{tag}_straight": St
#         })

#     # Symmetry cues: which wrist is higher
#     pos_feats["wrist_y_diff_start"] = float(positions[0, I["right_wrist"]*3+1] - positions[0, I["left_wrist"]*3+1])
#     pos_feats["wrist_y_diff_end"]   = float(positions[-1, I["right_wrist"]*3+1] - positions[-1, I["left_wrist"]*3+1])

#     # === 3. Core motion & range features ===
#     label = clip_name.split("_")[2]
#     range_features = {}
#     for name, index in landmark_indices.items():
#         x_vals = positions[:, index * 3 + 0]
#         y_vals = positions[:, index * 3 + 1]
#         range_features[f"range_x_{name}"] = np.max(x_vals) - np.min(x_vals)
#         range_features[f"range_y_{name}"] = np.max(y_vals) - np.min(y_vals)

#     features = {
#         "source": clip_name,
#         "label": label,
#         "mean_velocity": np.mean(vel_mag),
#         "max_velocity": np.max(vel_mag),
#         "std_velocity": np.std(vel_mag),
#         "mean_acceleration": np.mean(acc_mag),
#         "max_acceleration": np.max(acc_mag),
#         "std_acceleration": np.std(acc_mag),
#         "mean_jerk": np.mean(jerk_mag),
#         "max_jerk": np.max(jerk_mag),
#         "std_jerk": np.std(jerk_mag),
#         **range_features,
#         **pos_feats  # 👈 new position-based descriptors
#     }

#     all_features.append(features)


# # Save to CSV
# df = pd.DataFrame(all_features)

# output_csv_path = os.path.join(output_dir, "motion_features.csv")
# df.to_csv(output_csv_path, index=False)
# print(f"Saved {len(df)} motion feature entries to {output_csv_path}")

# 4_extract_motion_features_all.py
import os, json
import numpy as np
import pandas as pd

POSE_ROOT = "3_skeleton_pose_data"            # input root (per-session folders live here)
FEATURE_ROOT = "4_extracted_motion_features"  # output root (mirrors session subfolders)

FPS = 20.0
DT = 1.0 / FPS
MIN_VIS = 0.5

# MediaPipe landmark indices we use
I = {
    "nose":0, "left_shoulder":11, "right_shoulder":12,
    "left_elbow":13, "right_elbow":14, "left_wrist":15, "right_wrist":16,
    "left_hip":23, "right_hip":24, "left_ankle":27, "right_ankle":28
}
RANGE_LMS = {"left_wrist":15, "right_wrist":16, "left_ankle":27, "right_ankle":28}

def load_clip_positions(clip_dir):
    """Load per-frame JSON -> (T, 99) array (33*3)."""
    frame_files = sorted(f for f in os.listdir(clip_dir) if f.endswith(".json"))
    if not frame_files:
        return None
    poses = []
    for f in frame_files:
        with open(os.path.join(clip_dir, f)) as jf:
            data = json.load(jf)
        row = []
        for lm in data:
            if lm.get("visibility", 1.0) > MIN_VIS:
                row.extend([lm["x"], lm["y"], lm["z"]])
            else:
                row.extend([np.nan, np.nan, np.nan])
        poses.append(row)
    A = np.array(poses, dtype=float)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] != 99:
        return None
    # interpolate NaNs per-dimension
    for d in range(A.shape[1]):
        s = A[:, d]
        m = np.isnan(s)
        if not np.all(m):
            s[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), s[~m])
        A[:, d] = s
    return np.nan_to_num(A, nan=0.0)

def normalize_per_frame(A):
    """Center at hip midpoint and scale by hip distance, frame-wise."""
    out = A.copy()
    for t in range(out.shape[0]):
        lhip = out[t, I["left_hip"]*3:I["left_hip"]*3+3]
        rhip = out[t, I["right_hip"]*3:I["right_hip"]*3+3]
        center = (lhip + rhip) / 2.0
        scale = np.linalg.norm(lhip - rhip) or 1.0
        for j in range(33):
            s = j*3; e = s+3
            out[t, s:e] = (out[t, s:e] - center) / scale
    return out

def motion_stats(A):
    vel  = np.gradient(A, DT, axis=0)
    acc  = np.gradient(vel, DT, axis=0)
    jerk = np.gradient(acc, DT, axis=0)
    vel_mag  = np.linalg.norm(vel,  axis=1)
    acc_mag  = np.linalg.norm(acc,  axis=1)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    return {
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

def ranges(A):
    feats = {}
    for name, idx in RANGE_LMS.items():
        x = A[:, idx*3+0]; y = A[:, idx*3+1]
        feats[f"range_x_{name}"] = float(np.ptp(x))
        feats[f"range_y_{name}"] = float(np.ptp(y))
    return feats

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
    net = float(np.hypot(x1 - x0, y1 - y0))
    L = path_len(A, idx) + 1e-9
    return float(net / L)

def positional_feats(A):
    # shoulder/hip reference levels
    LSh_y = float(np.mean(A[:, I["left_shoulder"]*3+1]))
    RSh_y = float(np.mean(A[:, I["right_shoulder"]*3+1]))
    shoulder_y = 0.5*(LSh_y + RSh_y)
    LH_y = float(np.mean(A[:, I["left_hip"]*3+1]))
    RH_y = float(np.mean(A[:, I["right_hip"]*3+1]))
    hip_y = 0.5*(LH_y + RH_y)
    def rel_levels(y): return float(y - shoulder_y), float(y - hip_y)

    joints = [("lw", I["left_wrist"]), ("rw", I["right_wrist"]),
              ("la", I["left_ankle"]), ("ra", I["right_ankle"])]
    feats = {}
    for tag, idx in joints:
        x0, y0, x1, y1 = start_end_xy(A, idx)
        dx, dy = (x1-x0), (y1-y0)
        y0_sh, y0_hip = rel_levels(y0)
        y1_sh, y1_hip = rel_levels(y1)
        L = path_len(A, idx)
        St = straightness(A, idx)
        feats.update({
            f"{tag}_x0": x0, f"{tag}_y0": y0,
            f"{tag}_x1": x1, f"{tag}_y1": y1,
            f"{tag}_dx": dx, f"{tag}_dy": dy,
            f"{tag}_y0_minus_sh": y0_sh, f"{tag}_y0_minus_hip": y0_hip,
            f"{tag}_y1_minus_sh": y1_sh, f"{tag}_y1_minus_hip": y1_hip,
            f"{tag}_path_len": L, f"{tag}_straight": St
        })
    # symmetry cues
    feats["wrist_y_diff_start"] = float(A[0, I["right_wrist"]*3+1] - A[0, I["left_wrist"]*3+1])
    feats["wrist_y_diff_end"]   = float(A[-1, I["right_wrist"]*3+1] - A[-1, I["left_wrist"]*3+1])
    return feats

def features_for_clip(session_ts, clip_name, clip_dir):
    A = load_clip_positions(clip_dir)
    if A is None: return None
    A = normalize_per_frame(A)

    parts = clip_name.split("_")
    label = parts[2] if len(parts) >= 3 else "unknown"

    feats = {
        "timestamp": session_ts,
        "source": clip_name,
        "label": label,
        **motion_stats(A),
        **ranges(A),
        **positional_feats(A),
    }
    return feats

def process_session(session_ts):
    session_pose_dir = os.path.join(POSE_ROOT, session_ts)
    if not os.path.isdir(session_pose_dir):
        return None, 0
    rows = []
    for clip_name in sorted(os.listdir(session_pose_dir)):
        clip_dir = os.path.join(session_pose_dir, clip_name)
        if not os.path.isdir(clip_dir): 
            continue
        feats = features_for_clip(session_ts, clip_name, clip_dir)
        if feats is None:
            print(f"  ↳ skip {clip_name}: no frames")
            continue
        rows.append(feats)
    if not rows:
        return None, 0
    out_dir = os.path.join(FEATURE_ROOT, session_ts)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "motion_features.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✅ {session_ts}: wrote {len(rows)} rows -> {out_csv}")
    return out_csv, len(rows)

def main():
    os.makedirs(FEATURE_ROOT, exist_ok=True)
    all_csvs = []
    total = 0
    for session_ts in sorted(os.listdir(POSE_ROOT)):
        session_dir = os.path.join(POSE_ROOT, session_ts)
        if not os.path.isdir(session_dir):
            continue
        print(f"\n=== Session: {session_ts} ===")
        csv_path, n = process_session(session_ts)
        if csv_path: 
            all_csvs.append(csv_path)
            total += n

    # Optional: merge into one master CSV for training
    if all_csvs:
        dfs = [pd.read_csv(p) for p in all_csvs]
        master = pd.concat(dfs, ignore_index=True)
        master_out = "6_motion_features.csv"
        master.to_csv(master_out, index=False)
        print(f"\n🧩 merged {len(all_csvs)} sessions / {len(master)} rows -> {master_out}")
    else:
        print("\n(no sessions found / no rows produced)")

if __name__ == "__main__":
    main()
