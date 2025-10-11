# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import defaultdict
# from config import session_timestamp


# # === CONFIG ===
# session_timestamp = "2025-10-10_13-28-09"
# pose_root = os.path.join("3_skeleton_pose_data", session_timestamp)
# data_dict_path = "0_data_dictionary.csv"
# output_root = "5_motion_output_graphs"
# output_dir = os.path.join(output_root, session_timestamp)
# os.makedirs(output_dir, exist_ok=True)

# landmark_count = 33
# min_visibility = 0.5

# landmark_names = [
#     "nose", "left_eye_inner", "left_eye", "left_eye_outer",
#     "right_eye_inner", "right_eye", "right_eye_outer",
#     "left_ear", "right_ear",
#     "mouth_left", "mouth_right",
#     "left_shoulder", "right_shoulder",
#     "left_elbow", "right_elbow",
#     "left_wrist", "right_wrist",
#     "left_pinky", "right_pinky",
#     "left_index", "right_index",
#     "left_thumb", "right_thumb",
#     "left_hip", "right_hip",
#     "left_knee", "right_knee",
#     "left_ankle", "right_ankle",
#     "left_heel", "right_heel",
#     "left_foot_index", "right_foot_index"
# ]

# # === Load data dictionary
# df = pd.read_csv(data_dict_path)
# df = df[(df["timestamp"] == session_timestamp) & (df["gesture_captured"] == "yes")]

# # === Group by gesture label
# gesture_groups = defaultdict(list)
# for _, row in df.iterrows():
#     clip_name = row["clip_id"].replace(".mp4", "")
#     label = row["label"]
#     gesture_groups[label].append(clip_name)

# # === Plot each gesture group
# for label, clip_names in gesture_groups.items():
#     print(f"Averaging landmarks for gesture: {label} ({len(clip_names)} clips)")
#     aligned_clips = []

#     for clip_name in clip_names:
#         clip_path = os.path.join(pose_root, clip_name)
#         if not os.path.isdir(clip_path):
#             continue

#         frame_files = sorted([f for f in os.listdir(clip_path) if f.endswith(".json")])
#         clip_poses = []

#         for f in frame_files:
#             with open(os.path.join(clip_path, f)) as jf:
#                 data = json.load(jf)
#                 pose_vector = []
#                 for i in range(landmark_count):
#                     if data[i]["visibility"] > min_visibility:
#                         pose_vector.append(data[i]["x"])
#                     else:
#                         pose_vector.append(np.nan)
#                 clip_poses.append(pose_vector)

#         if clip_poses:
#             clip_poses = np.array(clip_poses)
#             aligned_clips.append(clip_poses)

#     if not aligned_clips:
#         print(f"⚠️ No valid data for {label}")
#         continue

#     # Align length
#     min_len = min(arr.shape[0] for arr in aligned_clips)
#     aligned_clips = [arr[:min_len] for arr in aligned_clips]
#     aligned_clips = np.array(aligned_clips)

#     avg_trajectory = np.nanmean(aligned_clips, axis=0)
#     mean_visibility = np.nanmean(~np.isnan(aligned_clips), axis=(0, 1))

#     # Plot
#     plt.figure(figsize=(12, 6))
#     for i in range(landmark_count):
#         if mean_visibility[i] > 0.5:
#             plt.plot(avg_trajectory[:, i], label=f"{i}: {landmark_names[i]}")

#     plt.title(f"Gesture: {label} — Avg X Trajectories (Visible Landmarks Only)")
#     plt.xlabel("Frame")
#     plt.ylabel("Normalized X Position")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="x-small")
#     plt.tight_layout()

#     out_path = os.path.join(output_dir, f"landmark_trajectory_{label}.png")
#     plt.savefig(out_path)
#     plt.close()
#     print(f"✅ Saved plot to {out_path}")

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import session_timestamp

# === CONFIG ===
session_timestamp = "2025-10-10_13-28-09"
features_csv = os.path.join("4_extracted_motion_features", session_timestamp, "motion_features.csv")
outdir = os.path.join("5_motion_output_graphs", session_timestamp)
os.makedirs(outdir, exist_ok=True)
sns.set(style="whitegrid")

# === LOAD ===
df = pd.read_csv(features_csv)
assert "label" in df.columns, "Expected a 'label' column in features CSV."

# === PICK ONLY THE *NEW* POSITIONAL FEATURES ===
# They were created with prefixes: lw_, rw_, la_, ra_, wrist_y_diff_*
prefixes = ("lw_", "rw_", "la_", "ra_", "wrist_y_diff_")
pos_cols = [c for c in df.columns if c.startswith(prefixes)]

# Optional: keep only interpretable columns (drop raw x0/x1 if you want)
# keep_cols_regex = r"(dx|dy|y0_minus_sh|y0_minus_hip|y1_minus_sh|y1_minus_hip|path_len|straight|wrist_y_diff_)$"
# pos_cols = [c for c in pos_cols if re.search(keep_cols_regex, c)]

print("\nFound positional feature columns:")
for c in sorted(pos_cols):
    print("  -", c)

# Bail gently if none found
if not pos_cols:
    print("\nNo positional features found. Did you run the updated extractor?")
    raise SystemExit

# -----------------------------
# 1) Box/violin plots by gesture
# -----------------------------
def save_catplot(kind="box"):
    long_df = df.melt(id_vars=["label"], value_vars=pos_cols,
                      var_name="feature", value_name="value")
    long_df = long_df[np.isfinite(long_df["value"])]

    n = min(16, len(pos_cols))  # avoid too-wide grids; adjust as you like
    subset = pos_cols[:n]
    long_df_sub = long_df[long_df["feature"].isin(subset)]

    plt.figure(figsize=(16, 9))
    if kind == "box":
        sns.boxplot(x="feature", y="value", hue="label", data=long_df_sub)
        plt.title("Positional Features by Gesture — Boxplots")
    else:
        sns.violinplot(x="feature", y="value", hue="label", data=long_df_sub, cut=0, split=False)
        plt.title("Positional Features by Gesture — Violin Plots")

    plt.xticks(rotation=90)
    plt.tight_layout()
    fname = f"positional_{kind}_top{n}.png"
    plt.savefig(os.path.join(outdir, fname))
    plt.close()
    print(f"✅ saved {fname}")

save_catplot("box")
save_catplot("violin")

# -----------------------------------------
# 2) Pairplot of the most informative subset
#    (pick deltas, path_len, straightness)
# -----------------------------------------
candidate_order = [
    "lw_dx","lw_dy","rw_dx","rw_dy","la_dx","la_dy","ra_dx","ra_dy",
    "lw_path_len","rw_path_len","la_path_len","ra_path_len",
    "lw_straight","rw_straight","la_straight","ra_straight",
    "wrist_y_diff_start","wrist_y_diff_end",
    "lw_y0_minus_sh","lw_y1_minus_sh","rw_y0_minus_sh","rw_y1_minus_sh",
]
pair_cols = [c for c in candidate_order if c in pos_cols]
pair_cols = pair_cols[:6]  # keep it readable

if len(pair_cols) >= 2:
    g = sns.pairplot(df, vars=pair_cols, hue="label", corner=True, plot_kws=dict(alpha=0.7, s=35))
    g.fig.suptitle("Positional Features — Pairplot", y=1.02)
    g.savefig(os.path.join(outdir, "positional_pairplot.png"))
    plt.close()
    print("✅ saved positional_pairplot.png")
else:
    print("ℹ️ Not enough positional columns for a pairplot.")

# ------------------------------------------------
# 3) Path length vs straightness per joint (scatter)
# ------------------------------------------------
def joint_scatter(tag):
    pl = f"{tag}_path_len"
    st = f"{tag}_straight"
    if pl in df.columns and st in df.columns:
        plt.figure(figsize=(6,5))
        sns.scatterplot(data=df, x=pl, y=st, hue="label", s=60, alpha=0.85)
        plt.title(f"{tag.upper()}: Path Length vs Straightness")
        plt.tight_layout()
        fname = f"scatter_{tag}_pathlen_vs_straight.png"
        plt.savefig(os.path.join(outdir, fname))
        plt.close()
        print(f"✅ saved {fname}")

for tag in ("lw","rw","la","ra"):
    joint_scatter(tag)

# -------------------------------------------------------------
# 4) Δy vs shoulder/hip lines: start/end height relative levels
# -------------------------------------------------------------
def rel_height_grid(tag):
    cols_needed = [
        f"{tag}_y0_minus_sh", f"{tag}_y1_minus_sh",
        f"{tag}_y0_minus_hip", f"{tag}_y1_minus_hip",
        f"{tag}_dy"
    ]
    if not all(c in df.columns for c in cols_needed):
        return
    sub = df[["label"] + cols_needed].copy()
    sub = sub.rename(columns={
        f"{tag}_y0_minus_sh":"y0_sh", f"{tag}_y1_minus_sh":"y1_sh",
        f"{tag}_y0_minus_hip":"y0_hip", f"{tag}_y1_minus_hip":"y1_hip",
        f"{tag}_dy":"dy"
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.boxplot(data=sub, x="label", y="y0_sh", ax=axes[0]); axes[0].set_title(f"{tag}: y_start - shoulder")
    sns.boxplot(data=sub, x="label", y="y1_sh", ax=axes[1]); axes[1].set_title(f"{tag}: y_end - shoulder")
    sns.boxplot(data=sub, x="label", y="dy",    ax=axes[2]); axes[2].set_title(f"{tag}: Δy (end-start)")
    for ax in axes: 
        ax.set_xlabel(""); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    fname = f"{tag}_relative_heights_and_dy.png"
    plt.savefig(os.path.join(outdir, fname))
    plt.close()
    print(f"✅ saved {fname}")

for tag in ("lw","rw","la","ra"):
    rel_height_grid(tag)

# ---------------------------------------------------------
# 5) Heatmap of gesture means (z-scored per feature column)
# ---------------------------------------------------------
Z = df[pos_cols].copy()
Z = (Z - Z.mean()) / (Z.std(ddof=0) + 1e-9)
mean_by_label = Z.groupby(df["label"]).mean()  # gestures × features

plt.figure(figsize=(min(18, 1 + 0.3*len(pos_cols)), 6))
sns.heatmap(mean_by_label, cmap="vlag", center=0)
plt.title("Positional Features (z-score) — Gesture Means")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "positional_means_heatmap.png"))
plt.close()
print("✅ saved positional_means_heatmap.png")

print(f"\nAll figures saved to: {outdir}")

