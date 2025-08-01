# VL-SurgPT: Bridging Vision and Language for Robust Surgical Point Tracking

This repository contains the official implementation of the paper:

> **VL-SurgPT: Bridging Vision and Language for Robust Surgical Point Tracking**

VL-SurgPT is a multimodal framework for robust tracking of arbitrary points on tissues and instruments in endoscopic surgery videos. It combines vision-language modeling with surgical motion dynamics to improve point tracking and state prediction under realistic conditions.

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/VL-SurgPT.git
cd VL-SurgPT
```

### (Optional) Create a virtual environment
```bash
conda create -n vl-surgpt python==3.10
conda activate vl-surgpt
pip install -r requirements.txt
```

## 🚀 Inference

### 🔹 Instrument Point Tracking

Input: Surgical video and initial point prompts on instruments.
Output: Predicted 2D coordinates and motion status per point over time.

To track arbitrary points on surgical instruments and predict their motion status:
```bash
python VL-SurgT-code/code/vis_tracking_status_instrument_update.py
```

### 🔹 Tissue Point Tracking

To track arbitrary points on tissue surfaces and predict their deformation status:
```bash
python VL-SurgT-code/code/vis_tracking_status_tissue_update.py
```

## 🧠 Training
To train VL-SurgPT with text guidance supervision, run:
```bash
python VL-SurgT-code/code/train_text_guidance_V1.py

```


## 📂 Dataset Structure
Download address of the dataset:
```bash
https://tinyurl.com/3am5674h

```


```
dataset/
 ├── tissue_with_text/
 │   ├── 0/  # Deformation
 │       └── ...
 │   ├── 1/  # Instrument Blocking
 │       └── ...
 │   ├── 2/  # Jitter
 │       └── ...
 │   ├── 3/  # Reflection
 │       └── ...
 │   └── 4/  # Smoke
 │       └── left/
 │           ├── seq000/
 │           │   ├── frames/
 │           │   │   └── 00000000ms-00001234ms-visible.mp4
 │           │   └── annotation/
 │           │       ├── labels.json         # Point tracking labels
 │           │       └── texts.json          # Text annotations
 │           ├── seq001/
 │           │   └── ...
 │           └── seq...
 └── instrument_with_text/
     └── 0/
          └── left/
              ├── seq000/
              │   ├── frames/
              │   │   └── 00000000ms-00001234ms-visible.mp4
              │   └── annotation/
              │       ├── labels.json
              │       └── texts.json
              ├── seq001/
              │   └── ...
              └── seq...
```

## 🏷️ Label Structure

Each frame in the dataset contains two types of annotation:

- 2D coordinates of tracked points (`labels.json`)
- Semantic attributes for each point (`texts.json`)

### 📁`labels.json` File Structure

```
labels.json
├── "0"           # Frame index (as string)
│   ├── [x1, y1]  # Coordinates of point 0
│   ├── [x2, y2]  # Coordinates of point 1
│   ├── ...
│   └── [xN, yN]  # Coordinates of point N or null if obscured
├── "30"
│   └── [...]
└── ...
```