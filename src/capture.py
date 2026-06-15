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
