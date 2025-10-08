# # run_all.py

# import subprocess
# from config import session_timestamp

# print(f"Running full pipeline for session: {session_timestamp}")

# # # Extract clips
# # print("\n Extracting clips...")
# # subprocess.run(["python3", "2a_extract_clips.py"])

# # # Review extracted clips
# # print("\nReviewing clips...")
# # subprocess.run(["python3", "2b_review_extracted_clips.py"])

# # # Extract skeletons
# # print("\n Extracting skeletons...")
# # subprocess.run(["python3", "3a_extract_skeleton.py"])

# # Extract motion features
# print("\n Extracting motion features...")
# subprocess.run(["python3", "4_extract_motion_features.py"])

# # Plot motion features
# print("\n Plotting motion features...")
# subprocess.run(["python3", "5a_motion_features.py"])
# subprocess.run(["python3", "5b_motion_features.py"])

# # Combine all motion feature CSVs
# print("\n Combining all motion_features.csv files...")
# subprocess.run(["python3", "6_combine_motion_features.py"])

# # Train classifier
# print("\n Training classifier...")
# subprocess.run(["python3", "7_train_classifier.py"])

# print("\n All steps completed.")

#######

import subprocess
import os

all_sessions = [
    "2025-05-19_19-33-29",
    "2025-05-19_19-34-57",
    "2025-05-19_19-36-26",
    "2025-05-19_19-38-50",
    "2025-05-19_19-48-51",
    "2025-05-19_19-50-20",
    "2025-05-19_19-51-51",
    "2025-05-19_19-53-37",
    "2025-05-28_21-01-45",
    "2025-05-28_21-03-42",
    "2025-05-28_21-05-07",
    "2025-05-28_21-06-32",
    "2025-05-28_21-08-01",
    "2025-05-28_21-09-26", 
    "2025-05-28_21-10-52",
    "2025-05-28_21-12-18",
    "2025-05-28_22-06-27"
]

for session in all_sessions:
    print(f"\n Running full pipeline for session: {session}")
    env = os.environ.copy()
    env["SESSION_TIMESTAMP"] = session

    # subprocess.run(["python3", "2a_extract_clips.py"], env=env)
    # subprocess.run(["python3", "2b_review_extracted_clips.py"], env=env)
    subprocess.run(["python3", "3a_extract_skeleton.py"], env=env)
    subprocess.run(["python3", "4_extract_motion_features.py"], env=env)
    subprocess.run(["python3", "5a_plot_landmark_trajectories.py"], env=env)
    subprocess.run(["python3", "5b_motion_features.py"], env=env)

print("\n Done rerunning motion features for all sessions. Now combining and training...")

subprocess.run(["python3", "6_combine_motion_features.py"])
subprocess.run(["python3", "7_train_classifier.py"])

print("\n All steps completed for all sessions.")
