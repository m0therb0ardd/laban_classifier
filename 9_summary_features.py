# # 9_summary_features.py
# import os
# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # === CONFIG ===
# MODEL_PATH = "random_forest_model.pkl"
# FEATURES_PATH = "6_motion_features.csv"
# OUTPUT_DIR = "8_feature_summary"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # === LOAD ===
# print("Loading model and features...")
# clf = joblib.load(MODEL_PATH)
# df = pd.read_csv(FEATURES_PATH)
# X = df.drop(columns=["label", "source", "timestamp"], errors="ignore")
# y = df["label"]

# print(f"✅ Loaded {len(df)} samples, {X.shape[1]} features, {y.nunique()} gesture classes.")

# # === 1. FEATURE IMPORTANCE ===
# feat_df = pd.DataFrame({
#     "feature": X.columns,
#     "importance": clf.feature_importances_
# }).sort_values("importance", ascending=False)

# top_feats = feat_df.head(12)
# print("\n💡 Top 12 most important features:\n", top_feats)

# plt.figure(figsize=(10,6))
# sns.barplot(y="feature", x="importance", data=top_feats, palette="viridis")
# plt.title("Top 12 Most Important Features")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "top_features.png"))
# plt.close()

# # === 2. FEATURE PROFILE PER GESTURE ===
# top_cols = top_feats["feature"].tolist()
# mean_df = df.groupby("label")[top_cols].mean()
# std_df  = df.groupby("label")[top_cols].std()

# plt.figure(figsize=(10,6))
# sns.heatmap(mean_df, annot=False, cmap="coolwarm", center=0)
# plt.title("Average Feature Profile per Gesture (Top Features)")
# plt.xlabel("Feature")
# plt.ylabel("Gesture Label")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "gesture_feature_heatmap.png"))
# plt.close()

# # === 3. PCA VISUALIZATION ===
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(X_scaled)

# plt.figure(figsize=(8,6))
# sns.scatterplot(
#     x=pca_result[:,0], y=pca_result[:,1],
#     hue=y, palette="Set2", s=60, alpha=0.8
# )
# plt.title("PCA Projection of Gesture Feature Space")
# plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
# plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "pca_gesture_space.png"))
# plt.close()

# # === 4. TRAJECTORY SHAPES (OPTIONAL EXAMPLE) ===
# # Show mean wrist path for each gesture (approximation using start→end deltas)
# if {"lw_x0","lw_x1","lw_y0","lw_y1","rw_x0","rw_x1","rw_y0","rw_y1"}.issubset(df.columns):
#     plt.figure(figsize=(6,6))
#     for lbl, sub in df.groupby("label"):
#         for side in ["lw", "rw"]:
#             x0, y0 = sub[f"{side}_x0"].mean(), sub[f"{side}_y0"].mean()
#             x1, y1 = sub[f"{side}_x1"].mean(), sub[f"{side}_y1"].mean()
#             plt.arrow(x0, y0, x1-x0, y1-y0, head_width=0.02, length_includes_head=True, alpha=0.6)
#             plt.text(x1, y1, lbl, fontsize=8, ha="center")
#     plt.title("Approx. Mean Wrist Movement Paths per Gesture")
#     plt.xlabel("X (normalized)")
#     plt.ylabel("Y (normalized)")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "gesture_mean_paths.png"))
#     plt.close()

# print(f"\n✨ Visualizations saved to '{OUTPUT_DIR}/'")
# print("   - top_features.png")
# print("   - gesture_feature_heatmap.png")
# print("   - pca_gesture_space.png")
# print("   - gesture_mean_paths.png (if wrist data available)")




# # visualize_average_paths.py
# import os, json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # === CONFIG ===
# POSE_ROOT = "3_skeleton_pose_data"
# session_dirs = [d for d in os.listdir(POSE_ROOT) if os.path.isdir(os.path.join(POSE_ROOT, d))]
# landmarks = {"left_wrist":15, "right_wrist":16, "left_ankle":27, "right_ankle":28}

# # === COLLECT PATHS PER LABEL ===
# def load_clip_positions(clip_dir):
#     frames = sorted(f for f in os.listdir(clip_dir) if f.endswith(".json"))
#     poses = []
#     for f in frames:
#         with open(os.path.join(clip_dir, f)) as jf:
#             data = json.load(jf)
#         row = [v for lm in data for v in (lm["x"], lm["y"], lm["z"])]
#         poses.append(row)
#     return np.array(poses)

# paths = {label: {lm: [] for lm in landmarks} for label in []}
# paths = {}

# for sess in session_dirs:
#     sess_dir = os.path.join(POSE_ROOT, sess)
#     for clip in os.listdir(sess_dir):
#         label = clip.split("_")[2] if "_" in clip else "unknown"
#         clip_dir = os.path.join(sess_dir, clip)
#         if not os.path.isdir(clip_dir): continue
#         A = load_clip_positions(clip_dir)
#         if A.shape[0] < 3: continue
#         if label not in paths: paths[label] = {lm: [] for lm in landmarks}
#         for name, idx in landmarks.items():
#             x = A[:, idx*3]; y = A[:, idx*3+1]
#             # normalize frame count (resample to 30 timesteps)
#             t = np.linspace(0, 1, len(x))
#             t_uniform = np.linspace(0, 1, 30)
#             x_u = np.interp(t_uniform, t, x)
#             y_u = np.interp(t_uniform, t, y)
#             paths[label][name].append(np.vstack([x_u, y_u]))

# # === PLOT AVERAGE PATHS ===
# out_dir = "gesture_path_plots"
# os.makedirs(out_dir, exist_ok=True)

# for label, parts in paths.items():
#     plt.figure(figsize=(6,6))
#     for name, seqs in parts.items():
#         arr = np.stack(seqs)   # (n_samples, 2, timesteps)
#         mean = arr.mean(0)
#         std = arr.std(0)
#         plt.plot(mean[0], mean[1], label=name)
#         plt.fill_between(mean[0], mean[1]-std[1], mean[1]+std[1], alpha=0.15)
#     plt.title(f"Average Path ({label})")
#     plt.xlabel("Horizontal movement (normalized x)")
#     plt.ylabel("Vertical movement (normalized y)")
#     plt.gca().invert_yaxis()
#     plt.legend()
#     plt.axis("equal")
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, f"{label}_paths.png"))
#     plt.close()

# print(f"✅ Saved average path plots to {out_dir}")


import pandas as pd
import numpy as np

df = pd.read_csv("6_motion_features.csv")

cols = [c for c in df.columns if c.startswith(("lw_", "rw_"))] + [
    "mean_velocity","max_velocity","mean_acceleration","max_acceleration"
]
print("Available wrist features:", [c for c in cols if c in df.columns])

# Compare signed displacement of left wrist (if present)
for k in ["lw_dx","lw_dy","rw_dx","rw_dy"]:
    if k in df.columns:
        print("\n", k)
        print(df.groupby("label")[k].mean().sort_values())

# Simple “direction” proxy: arctan(dy/dx) if you already have dx/dy
import math
def add_angle(row, tag):
    dx, dy = row.get(f"{tag}_dx", np.nan), row.get(f"{tag}_dy", np.nan)
    if pd.isna(dx) or pd.isna(dy): return np.nan
    return math.degrees(math.atan2(dy, dx))
for tag in ["lw","rw"]:
    ang = df.apply(lambda r: add_angle(r, tag), axis=1)
    df[f"{tag}_angle_deg"] = ang

for k in ["lw_angle_deg", "rw_angle_deg"]:
    if k in df.columns:
        print("\n", k)
        print(df.groupby("label")[k].mean().round(1))
