import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.detector import Detection, CLASS_NAMES, GestureDetector


def _make_detector():
    with patch("src.detector.ort.InferenceSession"):
        return GestureDetector("fake.onnx", conf_threshold=0.5)


def test_class_names_are_correct():
    assert CLASS_NAMES == ["c", "down", "fist", "fist_moved", "index", "l", "ok", "palm", "palm_moved", "thumb"]


def test_detection_label_maps_class_id():
    assert Detection(box=(0, 0, 10, 10), class_id=0, confidence=0.9).label == "c"
    assert Detection(box=(0, 0, 10, 10), class_id=1, confidence=0.9).label == "down"
    assert Detection(box=(0, 0, 10, 10), class_id=2, confidence=0.9).label == "fist"
    assert Detection(box=(0, 0, 10, 10), class_id=3, confidence=0.9).label == "fist_moved"
    assert Detection(box=(0, 0, 10, 10), class_id=6, confidence=0.9).label == "ok"
    assert Detection(box=(0, 0, 10, 10), class_id=9, confidence=0.9).label == "thumb"


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
    assert blob.dtype == np.float32
    assert blob.min() >= 0.0
    assert blob.max() <= 1.0


def test_postprocess_filters_low_confidence():
    det = _make_detector()
    # Shape (1, 14, 8400): 4 box coords + 10 class scores, all zeros = zero confidence
    raw_output = np.zeros((1, 14, 8400), dtype=np.float32)
    result = det._postprocess(raw_output, scale=1.0, pad_w=0, pad_h=0, orig_h=480, orig_w=640)
    assert result == []


def test_postprocess_returns_detection_above_threshold():
    det = _make_detector()
    # Shape (1, 14, 8400): 4 box coords + 10 class scores
    raw_output = np.zeros((1, 14, 8400), dtype=np.float32)
    raw_output[0, 0, 0] = 320.0  # x_center
    raw_output[0, 1, 0] = 240.0  # y_center
    raw_output[0, 2, 0] = 100.0  # width
    raw_output[0, 3, 0] = 100.0  # height
    raw_output[0, 6, 0] = 0.9   # class 2 (fist) confidence — index 4+2=6

    result = det._postprocess(raw_output, scale=1.0, pad_w=0, pad_h=0, orig_h=480, orig_w=640)

    assert len(result) == 1
    assert result[0].class_id == 2
    assert result[0].label == "fist"
    assert result[0].confidence == pytest.approx(0.9, abs=0.01)
