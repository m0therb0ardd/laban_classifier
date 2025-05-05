# plot_motion_features.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("motion_features.csv")

# Plot each feature with violin or box plots
features_to_plot = [
    "mean_velocity", "max_velocity",
    "mean_acceleration", "max_acceleration",
    "mean_jerk", "max_jerk"
]

plt.figure(figsize=(14, 10))
for i, feat in enumerate(features_to_plot, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x="label", y=feat, data=df)
    plt.title(feat.replace("_", " ").title())

plt.tight_layout()
plt.show()
