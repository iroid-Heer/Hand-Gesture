import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass

CLASS_NAMES = ["c", "down", "fist", "fist_moved", "index", "l", "ok", "palm", "palm_moved", "thumb"]
INPUT_SIZE = 640
IOU_THRESHOLD = 0.45

# A real hand must occupy between 2% and 60% of the frame area.
# Detections outside this range are almost always background false positives.
MIN_BOX_AREA_RATIO = 0.02
MAX_BOX_AREA_RATIO = 0.60

# Minimum fraction of pixels inside a detection box that must be skin-toned.
# YCrCb skin range works across most skin tones regardless of brightness.
MIN_SKIN_RATIO = 0.12

# Skin tone range in YCrCb color space (Y=luma, Cr=red-diff, Cb=blue-diff)
_SKIN_LOWER = np.array([0,   133, 77],  dtype=np.uint8)
_SKIN_UPPER = np.array([255, 173, 127], dtype=np.uint8)


@dataclass
class Detection:
    box: tuple          # (x1, y1, x2, y2) in original frame pixel coords
    class_id: int
    confidence: float

    @property
    def label(self) -> str:
        return CLASS_NAMES[self.class_id]


class GestureDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.3,
        min_hand_pct: float = 2.0,
        max_hand_pct: float = 60.0,
    ):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.min_box_ratio = min_hand_pct / 100.0
        self.max_box_ratio = max_hand_pct / 100.0

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

            frame_area = orig_h * orig_w
            box_area = (x2 - x1) * (y2 - y1)
            if not (self.min_box_ratio * frame_area <= box_area <= self.max_box_ratio * frame_area):
                continue

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

    @staticmethod
    def _is_skin(frame: np.ndarray, box: tuple) -> bool:
        """Return True if the box region contains enough skin-toned pixels."""
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, _SKIN_LOWER, _SKIN_UPPER)
        skin_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
        return skin_ratio >= MIN_SKIN_RATIO

    def detect(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(frame)
        raw = self.session.run(None, {self.input_name: blob})[0]
        return self._postprocess(raw, scale, pad_w, pad_h, h, w)

    def detect_debug(self, frame: np.ndarray) -> tuple:
        """Returns (accepted, rejected) where rejected is list of (Detection, reason)."""
        h, w = frame.shape[:2]
        blob, scale, pad_w, pad_h = self._preprocess(frame)
        raw = self.session.run(None, {self.input_name: blob})[0]

        # Run postprocess at very low conf to catch everything the model sees
        saved_conf = self.conf_threshold
        self.conf_threshold = 0.1
        candidates = self._postprocess(raw, scale, pad_w, pad_h, h, w)
        self.conf_threshold = saved_conf

        accepted, rejected = [], []
        frame_area = h * w
        for det in candidates:
            box_area = (det.box[2] - det.box[0]) * (det.box[3] - det.box[1])
            area_pct = box_area / frame_area * 100

            if det.confidence < self.conf_threshold:
                rejected.append((det, f"conf {det.confidence:.2f}<{self.conf_threshold}"))
            elif box_area < self.min_box_ratio * frame_area:
                rejected.append((det, f"too small {area_pct:.0f}%"))
            elif box_area > self.max_box_ratio * frame_area:
                rejected.append((det, f"too large {area_pct:.0f}%"))
            else:
                accepted.append(det)

        return accepted, rejected
