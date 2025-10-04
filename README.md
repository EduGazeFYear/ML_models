# EduGaze — ML Models

A collection of machine learning notebooks, pretrained models, and utilities used in EduGaze for face/pose/gaze/emotion analysis and engagement detection.

---

## Table of Contents

* [Overview](#overview)
* [Features & Capabilities](#features--capabilities)
* [Repository Structure](#repository-structure)
* [Dependencies](#dependencies)
* [Usage](#usage)

  * [Running Notebooks](#running-notebooks)
  * [Model Inference / APIs](#model-inference--apis)
* [Model Details](#model-details)
* [Data, Preprocessing & Encodings](#data-preprocessing--encodings)
* [Contributing](#contributing)
* [License & Citation](#license--citation)
* [Acknowledgments](#acknowledgments)

---

## Overview

This repository houses the core ML artifacts powering EduGaze’s engagement analytics. It includes:

* Jupyter notebooks exploring and training models (head pose, eye gaze, attention, emotion)
* Pretrained model weights
* Encodings, face-landmark models, and utilities
* Scripts or modules for inference pipelines

These models are integrated in the backend to analyze video frames, extract features, and generate engagement or emotion-based metrics.

---

## Features & Capabilities

* **Head pose estimation** (roll, pitch, yaw)
* **Eye gaze tracking / attention mapping**
* **Emotion recognition** using a MobileNetV3 variant
* **Face detection & matching** utilities
* **Pretrained weights** ready for inference
* **Encodings storage** for known faces (in `encodings.json`)
* Notebook demos for model training experiments

---

## Repository Structure

```
ML_models/
├── FaceDetection&Matching.ipynb
├── MODEL1(head_pose only).ipynb
├── MODEL2(head_pose_and_eye_gaze_tracking_w_attention).ipynb
├── Mobile_Net_V3.ipynb
├── emotion test.ipynb
├── encodings.json
├── mobilenetv3_emotion.pth
├── shape_predictor_68_face_landmarks.dat
└── README.md    ← you are here
```

* **Notebooks**: Exploratory/training scripts
* **encodings.json**: Serialized face embeddings for known participants
* **mobilenetv3_emotion.pth**: Pretrained emotion classification weights
* **shape_predictor_68_face_landmarks.dat**: dlib’s face landmark model
* **.ipynb_checkpoints/**: Notebook checkpoint folder

---

## Dependencies

To run the notebooks or inference scripts, you’ll need:

* Python 3.8+
* Core ML libraries:

  * `torch`
  * `torchvision`
  * `numpy`
  * `pandas`
  * `opencv-python`
  * `dlib`
  * `face_recognition`
  * `scikit-learn`
  * (Optional) `matplotlib`, `seaborn` for visualization

Install via:

```bash
pip install -r requirements.txt
```

*If a `requirements.txt` is not present, consider generating one from your environment.*

---

## Usage

### Running Notebooks

1. Clone the repo
2. Create a virtual environment and install dependencies
3. Launch Jupyter Lab or Notebook
4. Run relevant notebooks (e.g. `MODEL2(head_pose_and_eye_gaze_tracking_w_attention).ipynb`)
5. Inspect output metrics, visualizations, and exported model artifacts

### Model Inference / APIs

To use the pretrained models in your backend:

1. Load `mobilenetv3_emotion.pth` via PyTorch
2. Use `shape_predictor_68_face_landmarks.dat` with dlib or similar
3. For face matching, use `encodings.json` with `face_recognition` library
4. Pass video frames through:

   * Face detector → landmarks
   * Head pose & gaze model
   * Emotion model
   * Return feature vectors or class scores

You can wrap these in a service or endpoint (e.g. `predict`) that accepts image buffers and yields engagement/emotion predictions.

---

## Model Details

* **Head Pose / Gaze** 
  Includes angle estimation, attention scoring, gaze direction mapping.

* **Emotion Recognition** 
  Uses deepface VGG face backbone fine-tuned on emotion-labeled datasets.

* **Face Detection & Matching**
  Uses `face_recognition` library + dlib landmarks. The encodings are stored in `encodings.json`.

* **Attention Scoring**
  Combines gaze direction and head pose angles to yield an attention level metric.

---

## Data Preprocessing & Encodings

* **Data source**: raw face images, aligned and cropped
* **Preprocessing**: resize, normalization, landmark alignment
* **Encodings**: face embeddings stored in `encodings.json`
* **Usage**: At runtime, compare new face embeddings to known set (via cosine similarity)

---

## Contributing

1. Fork this repository
2. Create a feature branch (`feature/…` or `bugfix/…`)
3. Add notebooks, scripts, or model weights
4. Document changes in notebooks or `docs/`
5. Submit a pull request with a clear description

Guidelines:

* Maintain reproducibility
* Seed random for deterministic results
* Comment code and document hyperparameters
* Include model metrics (accuracy, loss curves, attention correlation)

---

## Acknowledgments

* Face recognition via the `face_recognition` / `dlib` libraries
* PyTorch and torchvision frameworks
* Inspiration from facial analytics and human-computer interaction research


