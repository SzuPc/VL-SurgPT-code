# Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark

This repository contains the official implementation of the paper:

> **Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark**

VL-SurgPT is a multimodal framework for robust tracking of arbitrary points on tissues and instruments in endoscopic surgery videos. It combines vision-language modeling with surgical motion dynamics to improve point tracking and state prediction under realistic conditions.

## Abstract

Accurate point tracking in surgical environments remains challenging due to complex visual conditions, including smoke occlusion, specular reflections, and tissue deformation. While existing surgical tracking datasets provide coordinate information, they lack the semantic context necessary to understand tracking failure mechanisms. We introduce VL-SurgPT, the first large-scale multimodal dataset that bridges visual tracking with textual descriptions of point status in surgical scenes. The dataset comprises 908 in vivo video clips, including 754 for tissue tracking (17,171 annotated points across five challenging scenarios) and 154 for instrument tracking (covering seven instrument types with detailed keypoint annotations). We establish comprehensive benchmarks using eight state-of-the-art tracking methods and propose TG-SurgPT, a text-guided tracking approach that leverages semantic descriptions to improve robustness in visually challenging conditions. Experimental results demonstrate that incorporating point status information significantly improves tracking accuracy and reliability, particularly in adverse visual scenarios where conventional vision-only methods struggle. By bridging visual and linguistic modalities, VL-SurgPT enables the development of context-aware tracking systems crucial for advancing computer-assisted surgery applications that can maintain performance even under challenging intraoperative conditions.

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
