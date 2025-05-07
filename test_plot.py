import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("motion_features.csv")
print("Data sample:")
print(df.head())

plt.figure(figsize=(6, 4))
sns.boxplot(x="label", y="mean_velocity", data=df)
plt.title("Mean Velocity by Label")
plt.tight_layout()
plt.show()
