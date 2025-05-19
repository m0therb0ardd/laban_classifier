import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
session_timestamp = "2025-05-18_22-35-34"
features_csv = os.path.join("4_extracted_motion_features", session_timestamp, "motion_features.csv")
output_dir = os.path.join("5_motion_output_graphs", session_timestamp)
os.makedirs(output_dir, exist_ok=True)

# === Load motion features
df = pd.read_csv(features_csv)

# === Feature list
features_to_plot = [
    "mean_velocity", "max_velocity",
    "mean_acceleration", "max_acceleration",
    "mean_jerk", "max_jerk"
]
features_to_plot = [f for f in features_to_plot if f in df.columns and df[f].notna().sum() > 0]

# === Plot setup
n_cols = 2
n_rows = int((len(features_to_plot) + 1) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6), squeeze=False)
axes = axes.flatten()

# === Plot each feature
for idx, feat in enumerate(features_to_plot):
    ax = axes[idx]
    sns.boxplot(x="label", y=feat, data=df, ax=ax)
    ax.set_title(feat.replace("_", " ").title())
    ax.set_xlabel("Gesture")
    ax.set_ylabel("Value")

# Hide any unused axes
for i in range(len(features_to_plot), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
out_path = os.path.join(output_dir, "boxplot_motion_features.png")
plt.savefig(out_path)
plt.close()
print(f"✅ Saved boxplot to {out_path}")
