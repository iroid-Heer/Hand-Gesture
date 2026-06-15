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
