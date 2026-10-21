# Human Activity Recognition (HAR) — Deep Learning Model Evaluation & Interactive Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-blueviolet.svg)](https://scikit-learn.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3.0-38bdf8.svg)](https://tailwindcss.com/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAR%20(ID%20240)-green.svg)](https://doi.org/10.24432/C54S4K)

An end-to-end Machine Learning and Deep Learning system for **Human Activity Recognition (HAR)** using triaxial smartphone inertial sensors (accelerometer & gyroscope). This repository contains the complete preprocessing pipeline, model training experiments, checkpoint safety audits, live validation metrics, an interactive **Google Colab notebook**, and a standalone **full-width Web Dashboard** styled in a refined **Rust Orange, Steel Blue, and Slate Grey** design.

---

## Table of Contents
- [1. Project Overview & Motivation](#1-project-overview--motivation)
- [2. Repository Structure](#2-repository-structure)
- [3. Dataset & Preprocessing Pipeline](#3-dataset--preprocessing-pipeline)
- [4. Model Selection & Checkpoint Verification](#4-model-selection--checkpoint-verification)
- [5. Live Validation Performance](#5-live-validation-performance)
- [6. Interactive Web Dashboard](#6-interactive-web-dashboard)
- [7. Google Colab Dashboard Notebook](#7-google-colab-dashboard-notebook)
- [8. Academic Demonstration Notice](#8-academic-demonstration-notice)
- [9. Citation & References](#9-citation--references)

---

## 1. Project Overview & Motivation

Human Activity Recognition (HAR) enables intelligent health monitoring, fall detection, and context-aware pervasive computing. Using the benchmark **UCI HAR Dataset** (Anguita et al., 2013), this project addresses the classification of six distinct physical activities:
1. `WALKING`
2. `WALKING_UPSTAIRS`
3. `WALKING_DOWNSTAIRS`
4. `SITTING`
5. `STANDING`
6. `LAYING`

### The Core Machine Learning Challenge:
While dynamic locomotion activities (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`) produce distinct rhythmic acceleration oscillations and achieve near-perfect classification (>98.6% F1), differentiating static postures (`SITTING` vs. `STANDING`) is inherently difficult due to identical vertical gravitational alignment and low dynamic signal variance.

### The Solution:
By designing a **Targeted Class-Weighted 1D Convolutional Neural Network (CNN)**, the model penalizes sitting-standing confusion during loss optimization. This boosts the **SITTING F1-score from 0.4571 (unweighted baseline) to 0.7464 (+28.9 percentage points)** while preserving >98.6% F1 on dynamic gait, establishing it as the provisional best model.

---

## 2. Repository Structure

```
aiproject/
├── 01_Dataset_Preprocessing/
│   ├── Preprocessing_EDA.ipynb               # Exploratory data analysis & preprocessing notebook
│   ├── processed_data/
│   │   ├── har_processed_data.npz           # Normalized signals (X_train, X_test, y, subjects)
│   │   ├── normalization_stats.npz          # Channel-wise mean & std values
│   │   ├── activity_label_mapping.csv       # Activity label IDs (0 to 5)
│   │   └── data_quality_summary.csv         # Integrity checks (missing, infinite values)
│   ├── plots/                               # EDA distributions, waveforms & heatmaps
│   └── reports/                             # Markdown flowcharts & descriptive statistics
│
├── 02_Model_Training/
│   ├── 02_CNN_LSTM_Training.ipynb           # Training notebook for Baseline & Weighted CNN-LSTM
│   └── outputs/
│       ├── best_cnn_lstm.keras              # Baseline CNN-LSTM checkpoint
│       ├── class_weighted/                  # Weighted CNN-LSTM experiment outputs
│       ├── cnn_only/                        # Unweighted 1D-CNN experiment outputs
│       └── targeted_sitting_experiment/
│           └── best_targeted_cnn.keras      # Targeted class-weighted CNN checkpoint
│
├── 03_Final_Model/
│   ├── 03_HAR_Final_Dashboard.ipynb         # Interactive Colab Dashboard notebook (Final)
│   ├── HAR_Model_Comparison_Clean_Notebook.ipynb # Model comparison & evaluation notebook
│   ├── final_model/
│   │   ├── har_cnn_model_verified_candidate.keras # Active verified checkpoint
│   │   ├── model_metadata.json             # Architecture & evaluation specs
│   │   └── verified_candidate_metadata.json# Candidate validation metadata
│   ├── dashboard_outputs/
│   │   ├── dashboard_validation_metrics.csv # Live computed validation metrics
│   │   └── dashboard_validation_classification_report.csv
│   └── model_comparison/                    # Comparison tables, CSVs, and plots
│
├── dashboard/
│   ├── index.html                           # Standalone Web Dashboard (Rust Orange/Blue/Grey)
│   └── samples_data.js                      # Extracted validation sample bank
│
├── run_dashboard.py                         # One-click local HTTP server launcher
├── HAR_Final_Dashboard.ipynb                # Root-level Google Colab Dashboard notebook
├── .gitignore                               # Git ignore configuration
└── README.md                                # Project documentation
```

---

## 3. Dataset & Preprocessing Pipeline

- **Benchmark Source:** UCI Machine Learning Repository — Human Activity Recognition Using Smartphones (ID 240).
- **Participants:** 30 volunteers aged 19–48 performing activities wearing a smartphone on the waist.
- **Sensor Modalities:** Triaxial linear acceleration and triaxial angular velocity sampled at a constant rate of **50 Hz**.
- **Sliding Window:** Fixed windows of **128 timesteps (2.56 seconds)** with a 50% overlap (64 timesteps).
- **6 Inertial Channels:**
  1. `body_acc_x`: Lateral body acceleration (normalized $g$ units)
  2. `body_acc_y`: Vertical body acceleration (normalized $g$ units)
  3. `body_acc_z`: Forward/backward body acceleration (normalized $g$ units)
  4. `body_gyro_x`: Angular velocity around X-axis / pitch (normalized rad/s)
  5. `body_gyro_y`: Angular velocity around Y-axis / roll (normalized rad/s)
  6. `body_gyro_z`: Angular velocity around Z-axis / yaw (normalized rad/s)
- **Normalization:** Channel-wise z-score normalization computed strictly from training samples (`normalization_stats.npz`). Signals are used directly without secondary scaling.
- **Participant-Separated Validation Partition:**
  - Method: `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` grouped by `subject_train`.
  - **Fit Subset:** 5,551 windows across 16 participants (`[5, 6, 7, 8, 11, 14, 16, 17, 19, 21, 22, 23, 26, 28, 29, 30]`).
  - **Validation Subset:** 1,801 windows across 5 participants (`[1, 3, 15, 25, 27]`).
  - **Data Leakage:** **Strictly zero** (`fit_subjects.isdisjoint(val_subjects) == True`).

---

## 4. Model Selection & Checkpoint Verification

Before deploying the dashboard, an equivalence audit was conducted between the two candidate checkpoints:
- **Checkpoint A:** `02_Model_Training/outputs/targeted_sitting_experiment/best_targeted_cnn.keras`
- **Checkpoint B:** `03_Final_Model/final_model/har_cnn_model_verified_candidate.keras`

### Audit Results:
| Audit Test | Result / Finding | Status |
| :--- | :---: | :---: |
| **Discrete Prediction Match** | **100.0000%** (1,801 / 1,801 identical classes) | PASSED |
| **Max Probability Delta** | **0.00000000** | PASSED |
| **Weight Stream Byte Length** | **1,267,468 bytes** (Exact binary match) | PASSED |
| **Input Shape Compatibility** | `(None, 128, 6)` | PASSED |
| **Output Classes** | `(None, 6)` | PASSED |
| **Selected Active Checkpoint** | `03_Final_Model/final_model/har_cnn_model_verified_candidate.keras` | ACTIVE |

> [!IMPORTANT]
> **Checkpoint Safety:** No model files were retrained, overwritten, renamed, or deleted. Both files retain their original SHA256 hashes (`306EDBA...` and `7B87A87...`).

---

## 5. Live Validation Performance

All metrics below were computed live from predictions generated by `har_cnn_model_verified_candidate.keras` across all 1,801 participant-separated validation windows:

### Summary Performance:
| Metric | Value | Percentage |
| :--- | :---: | :---: |
| **Validation Accuracy** | `0.877846` | **87.78%** |
| **Macro Precision** | `0.883394` | **88.34%** |
| **Macro Recall** | `0.882282` | **88.23%** |
| **Macro F1-Score** | `0.881358` | **88.14%** |
| **Weighted F1-Score** | `0.876473` | **87.65%** |

### Per-Activity Classification Report:
| Class ID | Activity Name | Motion Type | Precision | Recall | F1-Score | Validation Windows |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `0` | **WALKING** | Dynamic Gait | 97.41% | 100.00% | **0.9869** | 338 |
| `1` | **WALKING_UPSTAIRS** | Dynamic Gait | 98.21% | 99.28% | **0.9874** | 276 |
| `2` | **WALKING_DOWNSTAIRS** | Dynamic Gait | 98.77% | 99.17% | **0.9897** | 242 |
| `3` | **SITTING** | Static Posture | 79.54% | 70.31% | **0.7464** | 293 |
| `4` | **STANDING** | Static Posture | 74.66% | 86.29% | **0.8006** | 321 |
| `5` | **LAYING** | Static Posture | 81.46% | 74.32% | **0.7773** | 331 |
| — | **Macro Average** | — | **88.34%** | **88.23%** | **0.8814** | **1,801** |
| — | **Weighted Average** | — | **87.82%** | **87.78%** | **0.8765** | **1,801** |

---

## 6. Interactive Web Dashboard

The web dashboard is styled in a **Rust Orange, Steel Blue, and Charcoal/Slate Grey** visual theme, providing full-width responsiveness without unneeded navigation bars.

### Dashboard Architecture:
- **Full-Width Header & Controls:** Activity category filter, validation sample selector, and interactive `Run Prediction` button.
- **Top Row (4 Metric Cards):** Validation Accuracy (`87.78%`), Macro F1 (`88.14%`), Macro Recall (`88.23%`), and Validation Windows (`1,801`).
- **Second Row (3 Interactive Cards):**
  - *Active Model Card:* Shows predicted activity, recorded label, match indicator, and glowing neural wave artwork.
  - *Confidence Gauge:* Semi-circular SVG gauge with animated Rust/Blue gradient arc displaying the Softmax probability score.
  - *Quality Index:* Dynamic vs. static F1 comparison with a circular ring score meter.
- **Third Row (Waveforms & Distribution):**
  - *Inertial Signal Overview:* Canvas-rendered spline wave area curves displaying Body Acceleration (Rust Orange) and Gyroscope (Steel Blue) over the 128 timesteps (2.56 s).
  - *Softmax Probability Distribution:* 6 vertical glowing bars with live percentage readouts and 4 mini-stat boxes.
- **Fourth Row (Full-Width Validation Table):**
  - Comprehensive per-activity breakdown detailing Activity Name, Motion Type, Validation Windows, Precision, Recall, and F1 completion progress bars across all 6 classes.

### How to Run the Web Dashboard:

#### Option 1: One-Click Local Server (Recommended)
Run the launcher script using Python:
```bash
python run_dashboard.py
```
*(Or with the Python launcher: `py -3 run_dashboard.py`)*

This starts a local HTTP server at `http://localhost:8000` and automatically opens your default web browser.

#### Option 2: Direct File Open
Double-click `dashboard/index.html` in your file explorer. Because the dataset samples and signal streams are self-contained within the file, it works immediately offline with zero server requirements.

---

## 7. Google Colab Dashboard Notebook

For notebook execution and code inspection in Google Colab:
- **Notebook File:** `03_Final_Model/03_HAR_Final_Dashboard.ipynb` *(also available at `HAR_Final_Dashboard.ipynb`)*
- **Execution Flow:**
  1. Mounts Google Drive (`/content/drive/MyDrive/Human_Activity_Recognition`) or resolves local paths automatically.
  2. Loads `har_processed_data.npz` and recreates the participant-separated validation split.
  3. Verifies checkpoint equivalence between `best_targeted_cnn.keras` and `har_cnn_model_verified_candidate.keras`.
  4. Computes live validation metrics, classification report, and confusion matrix heatmap.
  5. Renders an interactive **IPyWidgets** dashboard with dropdown selector, Run Prediction button, probability bar charts, and 6-channel sensor signal plots.
  6. Provides a programmatic `predict_sample(sample_index, show_signals=True)` API function.

---

## 8. Academic Demonstration Notice

> [!NOTE]
> This system performs **offline evaluation and inference demonstration** on pre-recorded benchmark sensor windows from the UCI Human Activity Recognition dataset. It does **not** perform live smartphone sensing, wireless hardware streaming, or real-time physical activity tracking. All reported metrics represent evaluated model performance on the participant-separated validation subset.

---

## 9. Citation & References

```bibtex
@article{anguita2013public,
  title={A Public Domain Dataset for Human Activity Recognition Using Smartphones},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge L},
  journal={ESANN 2013 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning},
  year={2013}
}
```

- **Dataset DOI:** [10.24432/C54S4K](https://doi.org/10.24432/C54S4K)
- **Repository Author:** Srishti & Team
- **Model Checkpoint:** `har_cnn_model_verified_candidate.keras`
