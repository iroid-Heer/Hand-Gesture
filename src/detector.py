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
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 255, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

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

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, IOU_THRESHOLD)
        if len(indices) == 0:
            return []
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
