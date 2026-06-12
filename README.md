## 🎯 Problem Statement

In the rice processing industry, accurate identification of Sella rice varieties is critical for quality control, pricing, and supply chain management. Currently, variety classification is done manually by experts, which is:

- **Time-consuming** — manual inspection of large batches
- **Inconsistent** — subject to human error and fatigue
- **Unscalable** — cannot handle industrial volumes

RiceVision addresses this by building an automated two-stage computer vision pipeline that can **detect and classify Sella rice grain varieties** (1401_sella, 1509_sella, 1718_sella, Taaj_sella) from images with high accuracy — enabling faster, consistent, and scalable quality assessment in industrial settings.

# 🌾 RiceVision — Rice Grain Detection & Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A two-stage computer vision pipeline for rice grain detection and classification. RiceVision combines YOLO-based object detection with a deep learning classification system to identify and classify rice grain varieties from images.

---

## 📌 Overview

| Stage | Task | Approach | Accuracy |
|---|---|---|---|
| Stage 1 | Object Detection | YOLO fine-tuned | 96% |
| Stage 2 | Classification | Different Classification Model | XX% |

---

## 🗂️ Dataset

### Detection Dataset
- Custom annotated dataset for YOLO training
- YOLO format annotations (bounding boxes)
- Built and labeled from scratch

### Classification Dataset
- **Total Images:** 1,600 across 4 rice grain varieties
- **Built from scratch** — expanded iteratively from 800 to 1,600 images
- **Classes:** 1401_sella, 1509_sella, 1718_sella, Taaj_sella
- **Split:** 320 train / 80 test

```
dataset/
├── detection/
│   ├── images/
│   └── labels/          # YOLO format annotations
└── classification/
    ├── train/
    │   ├── 1401_sella/
    │   ├── 1509_sella/
    │   ├── 1718_sella/
    │   └── Taaj_sella/
    └── test/
        ├── 1401_sella/
        ├── 1509_sella/
        ├── 1718_sella/
        └── Taaj_sella/
```
---

## 🧪 Experiments

### Stage 1 — Detection
- Explored traditional OpenCV contour detection as baseline
- Fine-tuned YOLO on custom annotated dataset
- **YOLO achieved 96% detection accuracy**

### Stage 2 — Classification
- **150+ experiments** across multiple architectures and training strategies
- Strategies explored: Transfer Learning, Fine-tuning, Full fine-tuning
- Training techniques: K-Fold CV, Data Augmentation, Dropout, Weight Decay, Early Stopping
- **Best model upto now: EfficientNet-B3 with K-Fold + Fine-tuning with 83% val accuracy and 92%train accuracy.**

---

## ⚙️ Setup

```bash
git clone https://github.com/yourusername/RiceVision.git
cd RiceVision
pip install -r requirements.txt
```

### requirements.txt
torch>=2.0
torchvision>=0.15
ultralytics
numpy
scikit-learn
opencv-python
matplotlib
seaborn
requests
pandas
---

## 📈 Experiment Logging

All 150+ classification experiments auto-logged to Google Sheets tracking:
- Model architecture and parameter counts
- Train/Val accuracy and loss
- Hyperparameters — LR, batch size, epochs, patience
- K-Fold per-fold and mean accuracy
- Early stopping info

---

## 🔍 Key Findings

- YOLO significantly outperformed traditional contour-based detection
- Large classification models overfit severely on small datasets
- Fine-tuning last 2 blocks consistently outperformed full freeze or full unfreeze
- Data augmentation and regularization significantly reduced overfitting
- Classification Experiment still in progress

---

## 👩‍💻 Author

**Madhavi Chipade**


---

## 📄 License

MIT License
