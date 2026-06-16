# Real-Time Hand Gesture Detection — Design Spec

**Date:** 2026-06-15 (updated: 2026-06-15)
**Stack:** YOLOv8n · ONNX Runtime · OpenCV · Python · Google Colab

---

## Overview

A full end-to-end demo project that covers every stage of a real-world object detection pipeline:
dataset setup → model training → ONNX export → real-time inference on a desktop webcam.

The dataset comes **pre-labeled in YOLO format** from Kaggle — no manual annotation needed.
The goal is a learning/demo system — not production deployment.

---

## Dataset Source

**Kaggle:** [Hand Gesture Recognition Dataset for YOLOv8](https://www.kaggle.com/datasets/rohitagrawal2610/hand-gesture-recognition-dataset-for-yolov8)

- Pre-labeled in YOLO `.txt` format — labels are included, no LabelImg needed
- Contains a `data.yaml` that defines the class names and split paths — **use this file directly**
- After downloading, check the included `data.yaml` to find the exact class names and count
- Copy the dataset's `data.yaml` to the project root (replace the placeholder one)

> **Class names are determined by the Kaggle dataset's own `data.yaml`.**
> Update `CLASS_NAMES` in `src/detector.py` to match after downloading.

---

## Project Structure

```
demo/
├── dataset/                 ← extracted Kaggle dataset goes here
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/           ← YOLO .txt annotations (pre-labeled)
│       ├── val/
│       └── test/
├── data.yaml                ← copied from Kaggle dataset (defines class names)
├── notebooks/
│   └── train_and_export.ipynb   # Google Colab: train YOLOv8n + export ONNX
├── models/
│   └── gesture.onnx         # Exported ONNX model
├── src/
│   ├── __init__.py
│   ├── capture.py           # Webcam / video input wrapper
│   ├── detector.py          # ONNX inference + preprocessing
│   ├── visualizer.py        # Bounding box drawing + FPS overlay
│   └── main.py              # Entry point, CLI args, main loop
├── scripts/
│   ├── split_dataset.py     # Auto train/val/test split (use only if dataset has no split)
│   └── verify_dataset.py    # Validate label ↔ image pairing
└── requirements.txt
```

---

## Phase 1 — Dataset Setup

1. Download from Kaggle (requires free account) — click **Download** on the dataset page
2. Extract the ZIP — it contains `images/`, `labels/`, and `data.yaml`
3. Copy everything into `dataset/` so the structure matches above
4. Copy the Kaggle `data.yaml` to the project root
5. Run `scripts/verify_dataset.py` to confirm all image-label pairs are intact
6. If the Kaggle dataset does **not** have a pre-existing train/val/test split, run `scripts/split_dataset.py`

---

## Phase 2 — Model Training (Google Colab)

### Workflow
1. Zip `dataset/` + `data.yaml` locally, upload to Google Drive
2. Open `notebooks/train_and_export.ipynb` in Colab, select **T4 GPU** runtime
3. Mount Drive, unzip dataset
4. Install `ultralytics`
5. Train YOLOv8n with transfer learning from COCO pretrained weights
6. Export best weights to ONNX
7. Copy `gesture.onnx` back to Drive, download locally to `models/`

### Training Config
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='data.yaml',     # use the Kaggle data.yaml directly
    epochs=50,
    imgsz=640,
    batch=16,
    device='0',           # T4 GPU
)
```

### ONNX Export
```python
model = YOLO('runs/detect/train/weights/best.pt')
model.export(format='onnx', dynamic=True, simplify=True, imgsz=640)
```

---

## Phase 3 — Real-Time Inference System

### Architecture: Modular Python Package

```
capture.py → detector.py → visualizer.py → cv2.imshow (main.py)
   frame       Detection[]   annotated frame
```

### `capture.py` — VideoCapture wrapper

```python
class VideoCapture:
    def __init__(self, source: int | str = 0): ...
    def read(self) -> np.ndarray: ...
    def release(self): ...
```

### `detector.py` — ONNX Inference

```python
# CLASS_NAMES must match the Kaggle dataset's data.yaml names list
CLASS_NAMES = [...]   # update after downloading dataset

@dataclass
class Detection:
    box: tuple        # (x1, y1, x2, y2)
    class_id: int
    confidence: float

class GestureDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5): ...
    def detect(self, frame: np.ndarray) -> list[Detection]: ...
```

### `visualizer.py` — Drawing

```python
class Visualizer:
    def draw(self, frame, detections, fps) -> np.ndarray: ...
```

### `main.py` — Entry Point

```bash
python src/main.py --model models/gesture.onnx --source 0 --conf 0.5
```

---

## Dependencies

```txt
# requirements.txt  (local — inference only)
onnxruntime>=1.17.0
opencv-python>=4.8.0
numpy>=1.24.0
pytest>=7.0.0
```

```txt
# requirements-train.txt  (Colab only)
ultralytics>=8.0.0
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| YOLO version | YOLOv8n | Mature API, smallest model, fastest inference |
| Dataset source | Kaggle pre-labeled | Skips manual annotation, saves hours |
| Training platform | Google Colab (T4) | No local GPU required |
| Inference runtime | ONNX Runtime (CPU) | No PyTorch needed at runtime |
| Class names | From Kaggle data.yaml | Use dataset as-is, no remapping needed |
| Architecture | Modular package | Teachable, testable, each file has one job |

---

## Success Criteria

- [ ] Dataset downloaded, extracted, verified (all image-label pairs match)
- [ ] YOLOv8n trains to mAP50 > 0.80 on validation set
- [ ] ONNX model exported and runs locally without PyTorch
- [ ] Real-time webcam detection runs at ≥ 15 FPS on CPU
- [ ] Gestures detected correctly with ≥ 0.5 confidence in a live demo
