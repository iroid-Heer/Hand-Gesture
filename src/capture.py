import cv2
import numpy as np


class VideoCapture:
    def __init__(self, source: int | str = 0):
        self._source = source
        self._is_file = isinstance(source, str)
        self._cap = self._open(source)

    @staticmethod
    def _open(source: int | str) -> cv2.VideoCapture:
        if isinstance(source, str):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # Verify the file actually has readable frames
                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    raise RuntimeError(
                        f"Video file opened but contains no readable frames: {source}\n"
                        "The file may be empty or use an unsupported codec.\n"
                        "Try running:  .venv\\Scripts\\python.exe scripts\\make_demo_video.py"
                    )
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # seek back to start
                return cap
            raise RuntimeError(f"Cannot open video source: {source}")

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
            if self._is_file:
                # Video ended — loop back to the beginning
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    raise RuntimeError(
                        f"Cannot read from video file: {self._source}"
                    )
            else:
                raise RuntimeError("Failed to read frame from video source")
        return frame

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
