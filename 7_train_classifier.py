import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === LOAD ===
df = pd.read_csv("6_motion_features.csv")

# === CHECK ===
print("Label distribution in dataset:")
print(df["label"].value_counts(), "\n")

# === PREPARE ===
X = df.drop(columns=["label", "source", "timestamp"])  # drop non-feature columns
y = df["label"]

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === TRAIN ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# === PREDICT & EVAL ===
y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === FEATURE IMPORTANCE ===
importances = clf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# === SAVE ===
joblib.dump(clf, "random_forest_model.pkl")
print("Model saved to random_forest_model.pkl")
