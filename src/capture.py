import cv2
import numpy as np


class VideoCapture:
    def __init__(self, source: int | str = 0):
        self._cap = self._open(source)

    @staticmethod
    def _open(source: int | str) -> cv2.VideoCapture:
        # For video files, open directly
        if isinstance(source, str):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap
            raise RuntimeError(f"Cannot open video source: {source}")

        # For webcam indices, try platform backends in order of reliability on Windows
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                return cap
            cap.release()

        raise RuntimeError(
            f"Cannot open camera index {source}. "
            "Check that no other app (Teams, Zoom, browser) is using the camera."
        )

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
