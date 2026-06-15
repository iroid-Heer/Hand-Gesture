import cv2
import numpy as np
from src.detector import Detection, CLASS_NAMES

COLORS = [
    (241, 102, 99),   # thumbs_up
    (246, 92, 139),   # open_palm
    (153, 72, 236),   # fist
    (22, 115, 249),   # peace
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
