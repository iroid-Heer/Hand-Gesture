# Gesture Detection

Real-time hand gesture detection using YOLOv8 and ONNX inference.

## Gestures Supported

`c`, `down`, `fist`, `fist_moved`, `index`, `l`, `ok`, `palm`, `palm_moved`, `thumb`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install opencv-python==4.10.0.84 onnxruntime pyyaml
```

### Live webcam

```powershell
.venv\Scripts\python.exe -m src.main --model models\gesture.onnx --source 0
```

If the webcam does not open, try `--source 1` or `--source 2`.

### Video file

First generate the demo video from training images (only needed once):

```powershell
.venv\Scripts\python.exe scripts\make_demo_video.py
```

Then run detection on it:

```powershell
.venv\Scripts\python.exe -m src.main --model models\gesture.onnx --source demo.mp4
```



