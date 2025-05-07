# plot_motion_features.py

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib
# matplotlib.use("TkAgg")  # or "QtAgg" if Tk isn't available


# df = pd.read_csv("motion_features.csv")

# print("Available columns:", df.columns.tolist())


# # Plot each feature with violin or box plots
# features_to_plot = [
#     "mean_velocity", "max_velocity",
#     "mean_acceleration", "max_acceleration",
#     "mean_jerk", "max_jerk"
# ]

# plt.figure(figsize=(14, 10))
# # Drop any features that aren't in the CSV or are all NaN
# features_to_plot = [f for f in features_to_plot if f in df.columns and df[f].notna().sum() > 0]

# for i, feat in enumerate(features_to_plot, 1):
#     plt.subplot(3, 2, i)
#     sns.boxplot(x="label", y=feat, data=df)
#     plt.title(feat.replace("_", " ").title())

# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("motion_features.csv")
print("Available columns:", df.columns.tolist())

features_to_plot = [
    "mean_velocity", "max_velocity", "std_velocity",
    "mean_acceleration", "max_acceleration", "std_acceleration",
    "mean_jerk", "max_jerk", "std_jerk"
]

# Clean list
features_to_plot = [f for f in features_to_plot if f in df.columns and df[f].notna().sum() > 0]

for feat in features_to_plot:
    plt.figure(figsize=(6, 4))
    try:
        sns.boxplot(x="label", y=feat, data=df)
        plt.title(feat.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(f"{feat}.png")

        plt.show()
    except Exception as e:
        print(f"⚠️ Skipping {feat} due to error: {e}")
