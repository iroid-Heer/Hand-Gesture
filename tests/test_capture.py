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

    with patch("src.capture.cv2.VideoCapture", return_value=mock_cv_cap):
        with VideoCapture(0):
            pass

    mock_cv_cap.release.assert_called_once()


def test_read_raises_when_frame_unavailable():
    mock_cv_cap = MagicMock()
    mock_cv_cap.isOpened.return_value = True
    mock_cv_cap.read.return_value = (False, None)

    with patch("src.capture.cv2.VideoCapture", return_value=mock_cv_cap):
        cap = VideoCapture(0)
        with pytest.raises(RuntimeError, match="Failed to read frame"):
            cap.read()
