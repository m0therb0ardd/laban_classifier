# run_all.py

import subprocess
from config import session_timestamp

print(f"Running full pipeline for session: {session_timestamp}")

# Extract clips
print("\n Extracting clips...")
subprocess.run(["python3", "2a_extract_clips.py"])

# Review extracted clips
print("\nReviewing clips...")
subprocess.run(["python3", "2b_review_extracted_clips.py"])

# Extract skeletons
print("\n Extracting skeletons...")
subprocess.run(["python3", "3a_extract_skeleton.py"])

# Extract motion features
print("\n Extracting motion features...")
subprocess.run(["python3", "4_extract_motion_features.py"])

# Plot motion features
print("\n Plotting motion features...")
subprocess.run(["python3", "5a_motion_features.py"])
subprocess.run(["python3", "5b_motion_features.py"])

# Combine all motion feature CSVs
print("\n Combining all motion_features.csv files...")
subprocess.run(["python3", "6_combine_motion_features.py"])

# Train classifier
print("\n Training classifier...")
subprocess.run(["python3", "7_train_classifier.py"])

print("\n All steps completed.")
