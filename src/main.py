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
    parser.add_argument("--min-hand", type=float, default=2.0,
                        help="Minimum hand size as %% of frame area (default 2). "
                             "Increase if far-away hands are detected as noise.")
    parser.add_argument("--max-hand", type=float, default=60.0,
                        help="Maximum hand size as %% of frame area (default 60). "
                             "Increase (up to 90) if hand is very close to camera.")
    parser.add_argument("--debug", action="store_true",
                        help="Show rejected detections in red with rejection reason.")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    print(f"Loading model: {args.model}")
    print(f"Hand size range: {args.min_hand:.0f}%–{args.max_hand:.0f}% of frame")
    detector = GestureDetector(
        args.model,
        conf_threshold=args.conf,
        min_hand_pct=args.min_hand,
        max_hand_pct=args.max_hand,
    )
    visualizer = Visualizer()
    print("Model loaded. Press ESC to quit.")

    with VideoCapture(source) as cap:
        prev_time = time.time()
        while True:
            frame = cap.read()

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            if args.debug:
                accepted, rejected = detector.detect_debug(frame)
                annotated = visualizer.draw_debug(frame, accepted, rejected, fps)
            else:
                detections = detector.detect(frame)
                annotated = visualizer.draw(frame, detections, fps)

            # Scale down for display if frame is wider than 960px
            disp = annotated
            h_d, w_d = disp.shape[:2]
            if w_d > 960:
                scale = 960 / w_d
                disp = cv2.resize(disp, (960, int(h_d * scale)))

            cv2.imshow("Gesture Detection — ESC to quit", disp)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
