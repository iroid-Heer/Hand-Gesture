import cv2
import numpy as np
from src.detector import Detection, CLASS_NAMES

COLORS = [
    (241, 102,  99),   # c          — red
    ( 22, 115, 249),   # down       — blue
    (153,  72, 236),   # fist       — purple
    (246,  92, 139),   # fist_moved — pink
    ( 52, 199,  89),   # index      — green
    (255, 159,  10),   # l          — orange
    ( 48, 209, 255),   # ok         — cyan
    (255, 214,  10),   # palm       — yellow
    ( 94, 134, 255),   # palm_moved — indigo
    (255,  69,  58),   # thumb      — crimson
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

    def draw_debug(self, frame: np.ndarray, accepted: list, rejected: list, fps: float) -> np.ndarray:
        """Like draw() but also shows rejected detections in red with rejection reason."""
        out = self.draw(frame, accepted, fps)

        for det, reason in rejected:
            x1, y1, x2, y2 = det.box
            # Red dashed-style box for rejected detections
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 1)
            label = f"{det.label} {det.confidence:.2f} [{reason}]"
            cv2.putText(out, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)

        if rejected:
            msg = f"DEBUG: {len(accepted)} accepted  {len(rejected)} rejected"
            cv2.putText(out, msg, (10, out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        return out
