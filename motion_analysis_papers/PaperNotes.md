

### Link to Paper

- [Read the paper](https://research.dwi.ufl.edu/page/ai-driven-human-motion-classification-and-analysis-using-laban-movement-system/)
- [GitHub: HumanMotionAnalysis](https://github.com/guowenbin90/HumanMotionAnalysis)

https://www.youtube.com/watch?v=cafivJpiYUE

https://www.conductorsreference.com/laban



### Goal

To classify human movement using **Laban Movement Analysis (LMA)** into four expressive categories:
- **Effort**: light/strong
- **Shape**: opening/enclosing
- **Space**: side open/ side across
- **Body**: impulsive/ swing

<div style="display: flex; justify-content: center;">
  <img src="photos/labnan_hierarchy.png" alt="blocky" style="width: 75%; max-width: 1000px; height: auto; border-radius: 5px;">
</div>



The system uses **supervised machine learning** trained on video data to identify these qualities and enable expressive robotic responses.

---

### Dataset

- Videos sourced from:
  - University of Cyprus Dance Motion Capture Database
  - University of Florida Digital Worlds Institute (internal)
- **Four videos** were manually annotated with binary labels for each LMA dimension.

**TRYING TO GET LABELED DATASET**

---

### Method Overview

1. **Video Processing**
   - Frames extracted from video (every 10th frame).
   - Skeleton keypoints detected using a pretrained **Detectron2 model** (Keypoint R-CNN).
   - 17 COCO-style joints used per frame (e.g., nose, wrists, hips).

2. **Feature Extraction**
   - **In-frame features:** distances between joints (e.g., wrist to shoulder).
   - **Cross-frame features:** velocity, acceleration, and temporal stats (mean, std, etc.) across windows of 60 frames.
   - Categories mapped to LMA dimensions:
     - **Effort** → velocity & acceleration
     - **Shape** → body volume, torso height
     - **Space** → distance covered, area swept
     - **Body** → joint distances/posture

3. **Baseline Normalization**
   - Distances normalized by stable joint pairs (e.g., hip-to-knee) to remove scale effects.

4. **Machine Learning Models**
   - Trained separate binary classifiers per LMA dimension:
     - **Decision Tree**
     - **Random Forest** (best results)
     - **K-Nearest Neighbors (KNN)**
     - **Multi-layer Perceptron (Neural Net)**

---

### Evaluation

- Metrics used: **Precision, Recall, F1 Score**
- 70/30 train-test split
- **Random Forest** performed best overall, especially with cross-frame features

---

### Key Takeaways

- Combining **pose estimation + feature engineering** enables expressive motion classification
- Cross-frame features (like velocity/acceleration) were **critical** for capturing temporal qualities like "swing" or "impulsive"
- This method supports robotic systems that respond to *expression*, not just motion shape

---

