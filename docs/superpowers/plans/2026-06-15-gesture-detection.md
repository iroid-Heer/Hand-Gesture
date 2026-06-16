# Real-Time Hand Gesture Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete end-to-end hand gesture detection demo — pre-labeled Kaggle dataset (YOLO format already included, no manual annotation needed) → YOLOv8n training on Google Colab → ONNX export → real-time webcam detection with a modular Python package.

**Architecture:** Gesture detector trained with transfer learning from COCO weights. Class names and count are determined by the downloaded Kaggle `data.yaml`. Local inference uses a modular package (`capture.py` → `detector.py` → `visualizer.py` → `main.py`) driven by ONNX Runtime — no PyTorch needed at runtime.

**Tech Stack:** Python 3.10+, YOLOv8n (Ultralytics), ONNX Runtime 1.17+, OpenCV 4.8+, Google Colab (T4 GPU), pytest

---

## File Map

| File | Purpose |
|------|---------|
| `requirements.txt` | Local inference deps only (onnxruntime, opencv, numpy) |
| `requirements-train.txt` | Colab-only (ultralytics) |
| `data.yaml` | YOLO dataset config — class names + split paths |
| `scripts/__init__.py` | Makes scripts a package for test imports |
| `scripts/split_dataset.py` | Splits raw labeled images into train/val/test (70/20/10) |
| `scripts/verify_dataset.py` | Asserts every image has a label, no orphans |
| `notebooks/train_and_export.ipynb` | Single Colab notebook — trains YOLOv8n then exports ONNX (combines spec's train.ipynb + export.ipynb for simplicity) |
| `src/__init__.py` | Empty — marks src as package |
| `src/capture.py` | `VideoCapture` wrapper around `cv2.VideoCapture` |
| `src/detector.py` | `GestureDetector` — ONNX inference, pre/post-processing. `Detection` dataclass |
| `src/visualizer.py` | `Visualizer` — draws boxes, labels, FPS onto frames |
| `src/main.py` | CLI entry point — wires all modules, runs the display loop |
| `tests/__init__.py` | Empty |
| `tests/test_split_dataset.py` | Unit tests for split logic and ratio correctness |
| `tests/test_verify_dataset.py` | Unit tests for missing-label and orphan detection |
| `tests/test_detector.py` | Unit tests for Detection dataclass, preprocessing, postprocessing |
| `tests/test_visualizer.py` | Unit tests for draw output shape and frame immutability |
| `tests/test_capture.py` | Unit tests for invalid source error and context manager |
| `conftest.py` | Adds project root to sys.path for test imports |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `requirements-train.txt`
- Create: `.gitignore`
- Create: `conftest.py`
- Create: `pytest.ini`
- Create: `src/__init__.py`
- Create: `scripts/__init__.py`
- Create: `tests/__init__.py`
- Create all directories

- [ ] **Step 1: Create the full directory tree**

Run from `C:\Users\iRoid\Documents\Demo`:

```powershell
mkdir dataset\raw\images, dataset\raw\labels
mkdir dataset\images\train, dataset\images\val, dataset\images\test
mkdir dataset\labels\train, dataset\labels\val, dataset\labels\test
mkdir models, notebooks, src, scripts, tests
```

- [ ] **Step 2: Write `requirements.txt`**

```
onnxruntime>=1.17.0
opencv-python>=4.8.0
numpy>=1.24.0
pytest>=7.0.0
```

- [ ] **Step 3: Write `requirements-train.txt`**

```
ultralytics>=8.0.0
```

- [ ] **Step 4: Write `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.pytest_cache/
models/*.onnx
dataset/raw/
.superpowers/
*.egg-info/
dist/
```

- [ ] **Step 5: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
```

- [ ] **Step 6: Write `conftest.py` at project root**

```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
```

- [ ] **Step 7: Create empty `__init__.py` files**

```powershell
New-Item src\__init__.py -ItemType File
New-Item scripts\__init__.py -ItemType File
New-Item tests\__init__.py -ItemType File
```

- [ ] **Step 8: Install local dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 9: Verify pytest discovers tests (empty run)**

```bash
pytest --collect-only
```

Expected output: `no tests ran` — confirms discovery works.

- [ ] **Step 10: Commit**

```bash
git init
git add .
git commit -m "chore: project scaffold — dirs, requirements, pytest config"
```

---

## Task 2: Download & Organize Pre-Labeled Dataset from Kaggle

> **Manual task — no code to write. The dataset already includes YOLO-format labels. No LabelImg needed.**

**Files:**
- Populate: `dataset/` with the extracted Kaggle dataset (images + labels already paired)
- Copy: `data.yaml` from the Kaggle ZIP to the project root

- [ ] **Step 1: Download the dataset from Kaggle**

Go to: https://www.kaggle.com/datasets/rohitagrawal2610/hand-gesture-recognition-dataset-for-yolov8?resource=download

Click **Download** (requires free Kaggle account). Extract the ZIP to a temporary folder.

- [ ] **Step 2: Inspect the extracted folder**

Open the extracted folder. It contains:
- `images/` — subfolders (likely `train/`, `val/`, `test/` or by class name) with `.jpg` files
- `labels/` — matching YOLO `.txt` annotation files (one per image)
- `data.yaml` — defines `nc` (class count) and `names` (class names list)

Open `data.yaml` now and note the exact class names and their order. **These are the names you will use in `src/detector.py` `CLASS_NAMES`.** Example format:
```yaml
nc: 6
names: [call, dislike, fist, like, peace, stop]
```

- [ ] **Step 3: Copy data.yaml to the project root**

Copy the Kaggle `data.yaml` to `C:\Users\iRoid\Documents\Demo\data.yaml` (replace the placeholder).

- [ ] **Step 4a: If Kaggle provides pre-split data (train/val/test folders already exist)**

Copy the dataset directly into `dataset/` so the structure becomes:

```
dataset/
  images/
    train/   ← copy from Kaggle images/train/
    val/     ← copy from Kaggle images/val/
    test/    ← copy from Kaggle images/test/
  labels/
    train/   ← copy from Kaggle labels/train/
    val/     ← copy from Kaggle labels/val/
    test/    ← copy from Kaggle labels/test/
```

Then skip to Step 6.

- [ ] **Step 4b: If Kaggle does NOT have a pre-split (images grouped by class, not by split)**

Copy all images into `dataset/raw/images/` and all labels into `dataset/raw/labels/`. Then run the split script:

```bash
python scripts/split_dataset.py --images dataset/raw/images --labels dataset/raw/labels --output dataset
```

- [ ] **Step 5: Update CLASS_NAMES in src/detector.py**

Open `src/detector.py`. Find the line:
```python
CLASS_NAMES = ["thumbs_up", "open_palm", "fist", "peace"]
```

Replace the list with the exact names from the Kaggle `data.yaml`, in the same order:
```python
CLASS_NAMES = ["call", "dislike", "fist", "like", "peace", "stop"]  # example — use YOUR data.yaml values
```

Also update `test_class_names_are_correct` and `test_detection_label_maps_class_id` in `tests/test_detector.py` to match.

- [ ] **Step 6: Verify dataset integrity**

```bash
python scripts/verify_dataset.py --dataset dataset
```

Expected: `Dataset verification passed — all image-label pairs match.`

---

## Task 4: split_dataset.py — with Tests

**Files:**
- Create: `scripts/split_dataset.py`
- Create: `tests/test_split_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_split_dataset.py`:

```python
import pytest
import shutil
from pathlib import Path
from scripts.split_dataset import split_dataset


def _make_raw_data(tmp_path, n=10):
    img_dir = tmp_path / "raw" / "images"
    lbl_dir = tmp_path / "raw" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(n):
        (img_dir / f"img_{i:03d}.jpg").write_bytes(b"fake")
        (lbl_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    return img_dir, lbl_dir


def test_split_creates_train_val_test_dirs(tmp_path):
    img_dir, lbl_dir = _make_raw_data(tmp_path)
    out = tmp_path / "dataset"
    split_dataset(str(img_dir), str(lbl_dir), str(out))
    for split in ["train", "val", "test"]:
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()


def test_split_70_20_10_ratio(tmp_path):
    img_dir, lbl_dir = _make_raw_data(tmp_path, n=10)
    out = tmp_path / "dataset"
    counts = split_dataset(str(img_dir), str(lbl_dir), str(out))
    assert counts["train"] == 7
    assert counts["val"] == 2
    assert counts["test"] == 1


def test_split_total_equals_input(tmp_path):
    img_dir, lbl_dir = _make_raw_data(tmp_path, n=20)
    out = tmp_path / "dataset"
    counts = split_dataset(str(img_dir), str(lbl_dir), str(out))
    assert sum(counts.values()) == 20


def test_split_each_image_has_matching_label(tmp_path):
    img_dir, lbl_dir = _make_raw_data(tmp_path, n=10)
    out = tmp_path / "dataset"
    split_dataset(str(img_dir), str(lbl_dir), str(out))
    for split in ["train", "val", "test"]:
        imgs = list((out / "images" / split).glob("*.jpg"))
        for img in imgs:
            assert (out / "labels" / split / img.stem).with_suffix(".txt").exists()


def test_split_raises_on_empty_dir(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    with pytest.raises(ValueError, match="No image-label pairs"):
        split_dataset(str(img_dir), str(lbl_dir), str(tmp_path / "out"))
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_split_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.split_dataset'`

- [ ] **Step 3: Write `scripts/split_dataset.py`**

```python
import random
import shutil
from pathlib import Path

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2


def split_dataset(images_dir: str, labels_dir: str, output_dir: str, seed: int = 42) -> dict:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    pairs = sorted([
        f for f in images_dir.glob("*.jpg")
        if (labels_dir / f.stem).with_suffix(".txt").exists()
    ])
    pairs += sorted([
        f for f in images_dir.glob("*.png")
        if (labels_dir / f.stem).with_suffix(".txt").exists()
    ])

    if not pairs:
        raise ValueError(f"No image-label pairs found in {images_dir} / {labels_dir}")

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split, files in splits.items():
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path in files:
            lbl_path = (labels_dir / img_path.stem).with_suffix(".txt")
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    return {split: len(files) for split, files in splits.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default="dataset")
    args = parser.parse_args()

    counts = split_dataset(args.images, args.labels, args.output)
    for split, count in counts.items():
        print(f"  {split}: {count} images")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_split_dataset.py -v
```

Expected:
```
test_split_creates_train_val_test_dirs PASSED
test_split_70_20_10_ratio PASSED
test_split_total_equals_input PASSED
test_split_each_image_has_matching_label PASSED
test_split_raises_on_empty_dir PASSED
5 passed
```

- [ ] **Step 5: Run the script on your real dataset**

```bash
python scripts/split_dataset.py --images dataset/raw/images --labels dataset/raw/labels --output dataset
```

Expected output:
```
  train: 224 images
  val: 64 images
  test: 32 images
```

- [ ] **Step 6: Commit**

```bash
git add scripts/split_dataset.py tests/test_split_dataset.py
git commit -m "feat: split_dataset script with 70/20/10 train/val/test split"
```

---

## Task 5: verify_dataset.py — with Tests

**Files:**
- Create: `scripts/verify_dataset.py`
- Create: `tests/test_verify_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verify_dataset.py`:

```python
from pathlib import Path
from scripts.verify_dataset import verify_dataset


def _make_valid_dataset(tmp_path):
    for split in ["train", "val", "test"]:
        img_dir = tmp_path / "images" / split
        lbl_dir = tmp_path / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        (img_dir / "img001.jpg").write_bytes(b"fake")
        (lbl_dir / "img001.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    return tmp_path


def test_valid_dataset_has_no_errors(tmp_path):
    dataset = _make_valid_dataset(tmp_path)
    assert verify_dataset(str(dataset)) == []


def test_missing_label_is_reported(tmp_path):
    dataset = _make_valid_dataset(tmp_path)
    (dataset / "images" / "train" / "img002.jpg").write_bytes(b"fake")
    errors = verify_dataset(str(dataset))
    assert any("img002" in e for e in errors)


def test_orphan_label_is_reported(tmp_path):
    dataset = _make_valid_dataset(tmp_path)
    (dataset / "labels" / "train" / "ghost.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    errors = verify_dataset(str(dataset))
    assert any("ghost" in e for e in errors)


def test_missing_split_dir_is_reported(tmp_path):
    dataset = _make_valid_dataset(tmp_path)
    import shutil
    shutil.rmtree(dataset / "images" / "test")
    errors = verify_dataset(str(dataset))
    assert any("test" in e for e in errors)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_verify_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.verify_dataset'`

- [ ] **Step 3: Write `scripts/verify_dataset.py`**

```python
import sys
from pathlib import Path


def verify_dataset(dataset_dir: str) -> list:
    dataset_dir = Path(dataset_dir)
    errors = []

    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            errors.append(f"[{split}] Missing directory: {img_dir}")
            continue
        if not lbl_dir.exists():
            errors.append(f"[{split}] Missing directory: {lbl_dir}")
            continue

        images = {f.stem for f in img_dir.glob("*.jpg")} | {f.stem for f in img_dir.glob("*.png")}
        labels = {f.stem for f in lbl_dir.glob("*.txt")}

        for stem in sorted(images - labels):
            errors.append(f"[{split}] Image has no label: {stem}")
        for stem in sorted(labels - images):
            errors.append(f"[{split}] Label has no image: {stem}")

    return errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()

    errors = verify_dataset(args.dataset)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)
    print("Dataset verification passed — all image-label pairs match.")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_verify_dataset.py -v
```

Expected:
```
test_valid_dataset_has_no_errors PASSED
test_missing_label_is_reported PASSED
test_orphan_label_is_reported PASSED
test_missing_split_dir_is_reported PASSED
4 passed
```

- [ ] **Step 5: Run on your real dataset**

```bash
python scripts/verify_dataset.py --dataset dataset
```

Expected: `Dataset verification passed — all image-label pairs match.`

If you see errors, fix the missing labels in LabelImg before continuing.

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_dataset.py tests/test_verify_dataset.py
git commit -m "feat: verify_dataset script to catch missing or orphan label files"
```

---

## Task 6: data.yaml

**Files:**
- Update: `data.yaml` (copy from Kaggle dataset in Task 2, or update path if Kaggle's yaml uses absolute paths)

> The Kaggle dataset ships with its own `data.yaml`. Copy it to the project root in Task 2, Step 3.
> If Kaggle's `data.yaml` uses absolute paths (e.g. `/kaggle/input/...`), update the `path:` field to `./dataset`.

- [ ] **Step 1: Confirm data.yaml has correct paths**

Open `data.yaml` and ensure these keys are present and point to the local dataset:

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test
nc: <number from Kaggle data.yaml>
names: [<exact names from Kaggle data.yaml, in order>]
```

- [ ] **Step 2: Commit**

```bash
git add data.yaml
git commit -m "chore: data.yaml from Kaggle dataset — nc and names match pre-labeled annotations"
```

---

## Task 7: Google Colab Training Notebook

**Files:**
- Create: `notebooks/train_and_export.ipynb`

> **This notebook runs entirely in Google Colab. Do not run it locally.**

- [ ] **Step 1: Prepare your dataset ZIP locally**

```powershell
Compress-Archive -Path dataset, data.yaml -DestinationPath gesture-dataset.zip
```

- [ ] **Step 2: Upload `gesture-dataset.zip` to Google Drive**

Go to https://drive.google.com → upload `gesture-dataset.zip` to **My Drive** (root level).

- [ ] **Step 3: Create `notebooks/train_and_export.ipynb` — copy each cell into a new Colab notebook**

Go to https://colab.research.google.com → New notebook → Change runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU).

**Cell 1 — Mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 — Install Ultralytics:**
```python
!pip install ultralytics -q
```

**Cell 3 — Unzip dataset:**
```python
import zipfile, os

zip_path = '/content/drive/MyDrive/gesture-dataset.zip'
extract_path = '/content/gesture'

with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_path)

print("Extracted files:")
for root, dirs, files in os.walk(extract_path):
    level = root.replace(extract_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
```

**Cell 4 — Verify dataset structure:**
```python
import os

dataset_path = '/content/gesture/dataset'
for split in ['train', 'val', 'test']:
    imgs = len(os.listdir(f'{dataset_path}/images/{split}'))
    lbls = len(os.listdir(f'{dataset_path}/labels/{split}'))
    print(f'{split}: {imgs} images, {lbls} labels')
```

Expected output:
```
train: 224 images, 224 labels
val: 64 images, 64 labels
test: 32 images, 32 labels
```

**Cell 5 — Train YOLOv8n:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # downloads pretrained COCO nano weights (~6MB)

results = model.train(
    data='/content/gesture/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='0',               # T4 GPU
    project='/content/runs',
    name='gesture_v1',
    patience=15,              # early stopping if no improvement for 15 epochs
    save=True,
    plots=True,
)

print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
```

Expected: training completes in ~15–30 min. mAP50 should exceed 0.80 for 4 distinct gesture classes.

**Cell 6 — Validate on test set:**
```python
best_model = YOLO('/content/runs/gesture_v1/weights/best.pt')
metrics = best_model.val(data='/content/gesture/data.yaml', split='test')
print(f"Test mAP50: {metrics.box.map50:.4f}")
```

**Cell 7 — Export to ONNX:**
```python
best_model.export(
    format='onnx',
    dynamic=True,     # variable batch size
    simplify=True,    # remove unused ONNX ops
    imgsz=640,
)
print("ONNX export complete: runs/gesture_v1/weights/best.onnx")
```

**Cell 8 — Save ONNX to Google Drive:**
```python
import shutil

src = '/content/runs/gesture_v1/weights/best.onnx'
dst = '/content/drive/MyDrive/gesture.onnx'
shutil.copy(src, dst)
print(f"Saved to Google Drive: {dst}")
```

- [ ] **Step 4: Run all cells top-to-bottom in Colab**

Runtime → Run all. Monitor the training loss in the output. Wait for all cells to finish (~25–35 min total).

- [ ] **Step 5: Download `gesture.onnx` to your local `models/` folder**

In Google Drive, right-click `gesture.onnx` → Download. Save to:
```
C:\Users\iRoid\Documents\Demo\models\gesture.onnx
```

- [ ] **Step 6: Verify the ONNX model locally**

```bash
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('models/gesture.onnx', providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
print('Input name:', inp.name)
print('Input shape:', inp.shape)
out = sess.get_outputs()[0]
print('Output name:', out.name)
print('Output shape:', out.shape)
"
```

Expected output:
```
Input name: images
Input shape: [1, 3, 640, 640]
Output name: output0
Output shape: [1, 8, 8400]
```

(Shape `[1, 8, 8400]` = batch × (4 box coords + 4 class scores) × 8400 anchors)

- [ ] **Step 7: Save the notebook to the repo and commit**

Download the Colab notebook: File → Download → Download .ipynb. Save to `notebooks/train_and_export.ipynb`.

```bash
git add notebooks/train_and_export.ipynb models/.gitkeep
git commit -m "feat: Colab training notebook + ONNX export — gesture.onnx not committed (see .gitignore)"
```

---

## Task 8: capture.py — with Tests

**Files:**
- Create: `src/capture.py`
- Create: `tests/test_capture.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_capture.py`:

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.capture import VideoCapture


def test_invalid_file_path_raises_runtime_error(tmp_path):
    fake = str(tmp_path / "nonexistent.mp4")
    with pytest.raises(RuntimeError, match="Cannot open video source"):
        VideoCapture(fake)


def test_context_manager_calls_release():
    mock_cv_cap = MagicMock()
    mock_cv_cap.isOpened.return_value = True

    with patch("cv2.VideoCapture", return_value=mock_cv_cap):
        with VideoCapture(0):
            pass

    mock_cv_cap.release.assert_called_once()


def test_read_raises_when_frame_unavailable():
    mock_cv_cap = MagicMock()
    mock_cv_cap.isOpened.return_value = True
    mock_cv_cap.read.return_value = (False, None)

    with patch("cv2.VideoCapture", return_value=mock_cv_cap):
        cap = VideoCapture(0)
        with pytest.raises(RuntimeError, match="Failed to read frame"):
            cap.read()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_capture.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.capture'`

- [ ] **Step 3: Write `src/capture.py`**

```python
import cv2
import numpy as np


class VideoCapture:
    def __init__(self, source: int | str = 0):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def read(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video source")
        return frame

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_capture.py -v
```

Expected:
```
test_invalid_file_path_raises_runtime_error PASSED
test_context_manager_calls_release PASSED
test_read_raises_when_frame_unavailable PASSED
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/capture.py tests/test_capture.py
git commit -m "feat: VideoCapture wrapper with context manager support"
```

---

## Task 9: detector.py — with Tests

**Files:**
- Create: `src/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detector.py`:

```python
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.detector import Detection, CLASS_NAMES, GestureDetector


def _make_detector():
    with patch("onnxruntime.InferenceSession"):
        return GestureDetector("fake.onnx", conf_threshold=0.5)


def test_class_names_are_correct():
    assert CLASS_NAMES == ["thumbs_up", "open_palm", "fist", "peace"]


def test_detection_label_maps_class_id():
    assert Detection(box=(0, 0, 10, 10), class_id=0, confidence=0.9).label == "thumbs_up"
    assert Detection(box=(0, 0, 10, 10), class_id=1, confidence=0.9).label == "open_palm"
    assert Detection(box=(0, 0, 10, 10), class_id=2, confidence=0.9).label == "fist"
    assert Detection(box=(0, 0, 10, 10), class_id=3, confidence=0.9).label == "peace"


def test_preprocess_output_shape():
    det = _make_detector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blob, scale, pad_w, pad_h = det._preprocess(frame)
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32


def test_preprocess_values_in_zero_one():
    det = _make_detector()
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    blob, _, _, _ = det._preprocess(frame)
    assert blob.min() >= 0.0
    assert blob.max() <= 1.0


def test_preprocess_white_frame_near_one():
    det = _make_detector()
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    blob, _, _, _ = det._preprocess(frame)
    assert blob.max() <= 1.0
    assert blob.mean() > 0.9


def test_postprocess_filters_low_confidence():
    det = _make_detector()
    # Shape (1, 8, 8400): 4 box coords + 4 class scores, all zeros = zero confidence
    raw_output = np.zeros((1, 8, 8400), dtype=np.float32)
    result = det._postprocess(raw_output, scale=1.0, pad_w=0, pad_h=0, orig_h=480, orig_w=640)
    assert result == []


def test_postprocess_returns_detection_above_threshold():
    det = _make_detector()
    # Create output with one high-confidence detection for class 2 (fist)
    raw_output = np.zeros((1, 8, 8400), dtype=np.float32)
    # Anchor 0: x=320, y=240, w=100, h=100, class 2 conf=0.9
    raw_output[0, 0, 0] = 320.0  # x_center
    raw_output[0, 1, 0] = 240.0  # y_center
    raw_output[0, 2, 0] = 100.0  # width
    raw_output[0, 3, 0] = 100.0  # height
    raw_output[0, 6, 0] = 0.9   # class 2 (fist) confidence

    result = det._postprocess(raw_output, scale=1.0, pad_w=0, pad_h=0, orig_h=480, orig_w=640)

    assert len(result) == 1
    assert result[0].class_id == 2
    assert result[0].label == "fist"
    assert result[0].confidence == pytest.approx(0.9, abs=0.01)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_detector.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.detector'`

- [ ] **Step 3: Write `src/detector.py`**

```python
import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass

CLASS_NAMES = ["thumbs_up", "open_palm", "fist", "peace"]
INPUT_SIZE = 640
IOU_THRESHOLD = 0.45


@dataclass
class Detection:
    box: tuple          # (x1, y1, x2, y2) in original frame pixel coords
    class_id: int
    confidence: float

    @property
    def label(self) -> str:
        return CLASS_NAMES[self.class_id]


class GestureDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold

    def _preprocess(self, frame: np.ndarray) -> tuple:
        """Letterbox-resize to 640x640, normalize to float32. Returns (blob, scale, pad_w, pad_h)."""
        h, w = frame.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        pad_h = (INPUT_SIZE - new_h) // 2
        pad_w = (INPUT_SIZE - new_w) // 2
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # BGR → RGB, normalize, HWC → CHW, add batch dim
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]
        return blob, scale, pad_w, pad_h

    def _postprocess(
        self,
        raw_output: np.ndarray,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_h: int,
        orig_w: int,
    ) -> list:
        """Parse YOLOv8 output (1, 8, 8400), apply NMS, return Detection list."""
        predictions = raw_output[0].T  # (8400, 8)

        boxes, scores, class_ids = [], [], []
        for pred in predictions:
            x_c, y_c, bw, bh = pred[:4]
            class_confs = pred[4:]
            class_id = int(np.argmax(class_confs))
            confidence = float(class_confs[class_id])
            if confidence < self.conf_threshold:
                continue

            x1 = max(0, int((x_c - bw / 2 - pad_w) / scale))
            y1 = max(0, int((y_c - bh / 2 - pad_h) / scale))
            x2 = min(orig_w, int((x_c + bw / 2 - pad_w) / scale))
            y2 = min(orig_h, int((y_c + bh / 2 - pad_h) / scale))

            boxes.append([x1, y1, x2 - x1, y2 - y1])   # x,y,w,h for NMS
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, IOU_THRESHOLD)
        if len(indices) == 0:
            return []
        # .flatten() handles both (N,) and (N,1) shapes across OpenCV versions
        return [
            Detection(
                box=(
                    boxes[i][0],
                    boxes[i][1],
                    boxes[i][0] + boxes[i][2],
                    boxes[i][1] + boxes[i][3],
                ),
                class_id=class_ids[i],
                confidence=scores[i],
            )
            for i in indices.flatten()
        ]

    def detect(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(frame)
        raw = self.session.run(None, {self.input_name: blob})[0]
        return self._postprocess(raw, scale, pad_w, pad_h, h, w)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_detector.py -v
```

Expected:
```
test_class_names_are_correct PASSED
test_detection_label_maps_class_id PASSED
test_preprocess_output_shape PASSED
test_preprocess_values_in_zero_one PASSED
test_preprocess_white_frame_near_one PASSED
test_postprocess_filters_low_confidence PASSED
test_postprocess_returns_detection_above_threshold PASSED
7 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/detector.py tests/test_detector.py
git commit -m "feat: GestureDetector with ONNX inference, letterbox preprocessing, NMS postprocessing"
```

---

## Task 10: visualizer.py — with Tests

**Files:**
- Create: `src/visualizer.py`
- Create: `tests/test_visualizer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_visualizer.py`:

```python
import numpy as np
import pytest
from src.detector import Detection
from src.visualizer import Visualizer


def _blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_returns_same_shape():
    out = Visualizer().draw(_blank(), [], fps=30.0)
    assert out.shape == (480, 640, 3)
    assert out.dtype == np.uint8


def test_draw_does_not_mutate_input():
    frame = _blank()
    original = frame.copy()
    Visualizer().draw(frame, [], fps=30.0)
    np.testing.assert_array_equal(frame, original)


def test_draw_with_detection_modifies_pixels():
    frame = _blank()
    det = Detection(box=(50, 50, 300, 300), class_id=0, confidence=0.92)
    out = Visualizer().draw(frame, [det], fps=25.0)
    assert not np.array_equal(out, frame)


def test_draw_accepts_all_four_class_ids():
    viz = Visualizer()
    frame = _blank()
    for class_id in range(4):
        det = Detection(box=(10, 10, 200, 200), class_id=class_id, confidence=0.8)
        out = viz.draw(frame, [det], fps=30.0)
        assert out.shape == frame.shape


def test_draw_fps_overlay_does_not_crash_on_zero():
    out = Visualizer().draw(_blank(), [], fps=0.0)
    assert out.shape == (480, 640, 3)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_visualizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.visualizer'`

- [ ] **Step 3: Write `src/visualizer.py`**

```python
import cv2
import numpy as np
from src.detector import Detection, CLASS_NAMES

# One distinct BGR color per class
COLORS = [
    (241, 102, 99),   # thumbs_up  — indigo
    (246, 92, 139),   # open_palm  — purple
    (153, 72, 236),   # fist       — pink
    (22, 115, 249),   # peace      — orange
]


class Visualizer:
    def __init__(self, class_names: list = CLASS_NAMES):
        self.class_names = class_names

    def draw(self, frame: np.ndarray, detections: list, fps: float) -> np.ndarray:
        out = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.box
            color = COLORS[det.class_id % len(COLORS)]
            label = f"{det.label} {det.confidence:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(y1, text_h + 8)
            cv2.rectangle(out, (x1, label_y - text_h - 8), (x1 + text_w, label_y), color, -1)
            cv2.putText(out, label, (x1, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(out, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return out
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_visualizer.py -v
```

Expected:
```
test_draw_returns_same_shape PASSED
test_draw_does_not_mutate_input PASSED
test_draw_with_detection_modifies_pixels PASSED
test_draw_accepts_all_four_class_ids PASSED
test_draw_fps_overlay_does_not_crash_on_zero PASSED
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/visualizer.py tests/test_visualizer.py
git commit -m "feat: Visualizer draws bounding boxes, class labels, confidence, and FPS overlay"
```

---

## Task 11: main.py — Entry Point

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Write `src/main.py`**

```python
import argparse
import time
import cv2
from src.capture import VideoCapture
from src.detector import GestureDetector
from src.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Real-time hand gesture detection")
    parser.add_argument("--model", required=True, help="Path to gesture.onnx")
    parser.add_argument("--source", default="0", help="Webcam index (0,1,2) or video file path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0–1)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Loading model: {args.model}")
    detector = GestureDetector(args.model, conf_threshold=args.conf)
    visualizer = Visualizer()
    print("Model loaded. Press ESC to quit.")

    with VideoCapture(source) as cap:
        prev_time = time.time()
        while True:
            frame = cap.read()
            detections = detector.detect(frame)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            annotated = visualizer.draw(frame, detections, fps)
            cv2.imshow("Gesture Detection — ESC to quit", annotated)

            if cv2.waitKey(1) & 0xFF == 27:   # ESC
                break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests to confirm nothing is broken**

```bash
pytest -v
```

Expected: all tests pass (no new failures).

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat: main.py entry point with CLI args — wires capture, detector, visualizer"
```

---

## Task 12: End-to-End Smoke Test

> **Prerequisites:** `models/gesture.onnx` must exist (downloaded from Colab in Task 7).

- [ ] **Step 1: Run with your webcam**

```bash
python src/main.py --model models/gesture.onnx --source 0 --conf 0.5
```

A window titled **"Gesture Detection — ESC to quit"** opens showing your webcam feed with:
- Green FPS counter (top-left) — target ≥ 15 FPS on CPU
- Colored bounding boxes around detected gestures
- Class label + confidence score above each box

- [ ] **Step 2: Test all 4 gestures**

Hold each gesture in front of your webcam and verify:

| Gesture | Expected label | Expected confidence |
|---------|---------------|---------------------|
| 👍 Thumb up | `thumbs_up` | ≥ 0.50 |
| 🖐 Open palm | `open_palm` | ≥ 0.50 |
| ✊ Fist | `fist` | ≥ 0.50 |
| ✌️ Peace | `peace` | ≥ 0.50 |

- [ ] **Step 3: Test with lower confidence threshold if detections are missed**

```bash
python src/main.py --model models/gesture.onnx --source 0 --conf 0.3
```

- [ ] **Step 4: Final test suite run**

```bash
pytest -v
```

Expected: all 19 tests pass.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete gesture detection demo — dataset, labeling, training, ONNX, real-time inference"
```

---

## Quick Reference

```bash
# Run full test suite
pytest -v

# Split raw labeled dataset
python scripts/split_dataset.py --images dataset/raw/images --labels dataset/raw/labels --output dataset

# Verify dataset integrity
python scripts/verify_dataset.py --dataset dataset

# Run real-time detection (webcam)
python src/main.py --model models/gesture.onnx --source 0 --conf 0.5

# Run with a video file instead of webcam
python src/main.py --model models/gesture.onnx --source path/to/video.mp4 --conf 0.5
```
