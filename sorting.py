import pandas as pd

df = pd.read_csv("motion_features.csv")

# Filter only 'float' examples
float_df = df[df["label"] == "float"]

# Sort by max acceleration, descending
top_float = float_df.sort_values(by="max_acceleration", ascending=False)

print(top_float[["source", "max_acceleration", "mean_acceleration"]].head(5))
