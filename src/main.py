import argparse
import time
import cv2
from src.capture import VideoCapture
from src.detector import GestureDetector
from src.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Real-time hand gesture detection")
    parser.add_argument("--model", required=True, help="Path to gesture.onnx")
    parser.add_argument("--source", default="0", help="Webcam index (0,1,2) or video file path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0-1)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Loading model: {args.model}")
    detector = GestureDetector(args.model, conf_threshold=args.conf)
    visualizer = Visualizer()
    print("Model loaded. Press ESC to quit.")

    with VideoCapture(source) as cap:
        prev_time = time.time()
        while True:
            frame = cap.read()
            detections = detector.detect(frame)

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            annotated = visualizer.draw(frame, detections, fps)
            cv2.imshow("Gesture Detection — ESC to quit", annotated)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
