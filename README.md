# Laban Movement Classification and Live Response

## Getting Started

To resume work on this project:

```bash
cd ~/turtle_swarm
source env/bin/activate

```

### Project Summary 
This project aims to classify short [dance gestures](https://www.youtube.com/watch?v=rtnIfls5800) based on simplified Laban movement qualities (e.g., "float" vs. "punch") using pose data captured from MediaPipe. A trained machine learning model (Random Forest) predicts the movement quality of live-recorded gestures. The long-term goal is to use these classifications to control swarm robot behaviors in real-time, enabling interactive performances.

## Workflow to add to Dataset
1. Modify 1_record_with_timer.py to be the gestures names you want to record and the number of recordigns you want to record. Make sure full body is in frame when you record.
2. Run 2a_extract_clips.py
3. Run 2b_review_extracted_clip to make sure its good
4. Add the timestamp of the new recordign to config.py and then run 0_run_all. 0_run_all will do the rest if the correct time stamp is added. Read below for a longer explanation fo the remaining work flow. 

## Pipeline Overview

0. Data Management

    0_data_dictionary.csv: Metadata log of all recorded clips (ID, label, duration, etc.)

    config.py: Contains shared config values like participant ID, frame rate, etc.

1. Record New Gesture Clips

    1_record_with_timer.py
    Displays prompts and records labeled 5-second clips. Saves video and logs metadata into 0_data_dictionary.csv.

3. Extract and Visualize Pose Skeletons

    3a_extract_skeleton.py
    Uses MediaPipe to extract 33-point body landmarks for each frame.

    3b_visualize_pose_overlay.py
    Creates overlay videos showing the pose landmarks over the original clip for review.

4. Feature Extraction & Combination

    4_extract_motion_features.py
    Computes velocity, acceleration, jerk, and other features for key landmarks.

    5a_plot_landmark_trajectories.py
    Plots trajectories of selected landmarks for visual inspection.

    5b_motion_features.py
    Additional feature extraction (e.g., range of motion for wrists and ankles).

    6_combine_motion_features.py
    Merges all extracted features into one CSV (6_motion_features.csv).

7. Train the Classifier

    7_train_classifier.py
    Trains and evaluates a Random Forest classifier using the combined motion features.

    random_forest_model.pkl
    Saved model used for live classification.

8. Live Classification

    8_live_classify.py
    Captures a live 5-second clip, extracts features, and predicts the movement type. Also displays video and predicted label.

## General Resources 
1. [Performance Arc] (https://docs.google.com/document/d/1Sa50h2eDBSr1ljnjP7Vl_LQX1kHWB3tFU8qgbsUmAcs/edit?tab=t.0#heading=h.waxb0a52r4cj)
2. [Coachbot User Guide] (https://coachbotswarm.github.io/User-Guide/#available-robot-functions)
3. [Coachbot Swarm Repo](https://github.com/Coachbot-Swarm)
4, [Coachbot Simulator](https://github.com/michelleezhang/swarm_simulation/tree/master)