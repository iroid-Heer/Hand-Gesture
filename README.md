# Gesture Detection

Real-time hand gesture detection using YOLOv8 and ONNX inference.

## Gestures Supported

`c`, `down`, `fist`, `fist_moved`, `index`, `l`, `ok`, `palm`, `palm_moved`, `thumb`

## Project Structure

```
Demo/
├── src/
│   ├── main.py          # Entry point
│   ├── capture.py       # Webcam/video capture
│   ├── detector.py      # ONNX inference + NMS
│   └── visualizer.py    # Bounding boxes + FPS overlay
├── scripts/
│   ├── split_dataset.py # Split images into train/val/test
│   └── verify_dataset.py
├── tests/
├── models/              # Place gesture.onnx here
├── dataset/             # Training data (not committed to git)
└── data.yaml            # YOLO dataset config
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python onnxruntime pyyaml
```

## Run

```powershell
python -m src.main --model models/gesture.onnx --source 0 --conf 0.5
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to `.onnx` model file | required |
| `--source` | Webcam index (`0`, `1`) or video file path | `0` |
| `--conf` | Confidence threshold (0–1) | `0.5` |

Press **ESC** to quit.

## Training (Google Colab)

1. Upload `gesture-dataset.zip` to Colab
2. Extract and verify dataset structure:
   ```
   dataset/
   ├── images/train/   (18,000 images)
   ├── images/val/     (1,000 images)
   └── images/test/
   ```
3. Train with YOLOv8n:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   model.train(data='data.yaml', epochs=50, imgsz=416, batch=32, cache=True)
   ```
4. Export to ONNX:
   ```python
   model.export(format='onnx')
   ```

Training takes ~1.5–2 hours on a T4 GPU.
