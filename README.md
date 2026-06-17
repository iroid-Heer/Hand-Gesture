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
.venv\Scripts\python.exe -m pip install opencv-python==4.10.0.84 onnxruntime pyyaml
```

> **Note:** Use `opencv-python==4.10.0.84` specifically. Version 4.13+ has a broken
> `imshow` on Windows and the detection window will not open.

## Run

**Always use `.venv\Scripts\python.exe` directly** — do not use `python` or `py`,
as those may resolve to a different system installation that has the wrong OpenCV version.

### Live webcam

```powershell
.venv\Scripts\python.exe -m src.main --model models\gesture.onnx --source 0
```

If the webcam doesn't open, try `--source 1` or `--source 2` (each number is a
different camera plugged into your PC).

### Video file (demo / test without a webcam)

First generate the demo video from the training images (only need to do this once):

```powershell
.venv\Scripts\python.exe scripts\make_demo_video.py
```

Then run detection on it:

```powershell
.venv\Scripts\python.exe -m src.main --model models\gesture.onnx --source demo.mp4
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to `.onnx` model file | required |
| `--source` | Webcam index (`0`, `1`, `2`) or path to a `.mp4` / `.avi` file | `0` |
| `--conf` | Confidence threshold (0–1). Lower = more detections, higher = fewer false positives | `0.5` |
| `--min-hand` | Minimum hand size as % of frame. Raise if tiny background blobs are detected | `2` |
| `--max-hand` | Maximum hand size as % of frame. **Raise to 90 if hand is very close to camera** | `60` |

Press **ESC** to quit.

### Lighting requirements

The detector needs reasonable light. It will fail in a dark room.

| Condition | Works? |
|-----------|--------|
| Bright room or lamp on hand | Yes |
| Normal indoor lighting | Yes |
| Dim room | Borderline |
| Dark room | No — hand not visible to model |

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
