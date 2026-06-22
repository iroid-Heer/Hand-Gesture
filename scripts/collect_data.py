"""
Capture real webcam images for each gesture for retraining.

Usage:
    .venv\Scripts\python.exe scripts\collect_data.py
    .venv\Scripts\python.exe scripts\collect_data.py --gesture fist   # one gesture only
    .venv\Scripts\python.exe scripts\collect_data.py --frames 150     # more images

Hold each gesture steady in front of the camera when prompted.
Images are saved to dataset/real/images/<gesture>/
"""
import cv2
import os
import time
import argparse

CLASS_NAMES = ["c", "down", "fist", "fist_moved", "index", "l", "ok", "palm", "palm_moved", "thumb"]

GESTURE_TIPS = {
    "c":          "Make a C-shape with your hand",
    "down":       "Point index finger DOWN",
    "fist":       "Make a closed fist",
    "fist_moved": "Make a fist, move it sideways",
    "index":      "Point index finger UP",
    "l":          "Make L-shape: thumb + index finger",
    "ok":         "Make OK sign (circle with fingers)",
    "palm":       "Open palm facing camera",
    "palm_moved": "Open palm, move it sideways",
    "thumb":      "Thumbs up",
}


def open_camera(source):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(source, backend)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError(f"Cannot open camera {source}")


def countdown(cap, gesture, tip, seconds=4):
    """Show a countdown screen so user can prepare the gesture."""
    end = time.time() + seconds
    while time.time() < end:
        ret, frame = cap.read()
        if not ret:
            break
        remaining = int(end - time.time()) + 1
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, f"NEXT: {gesture.upper()}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 120), 2)
        cv2.putText(frame, tip, (16, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Starting in {remaining}s  (ESC to skip)", (16, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        cv2.imshow("Collect Data", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            return False
    return True


def capture_gesture(cap, gesture, output_dir, n_frames):
    """Capture n_frames images for the gesture and save them."""
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    saved = 0

    while saved < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        idx = existing + saved + 1
        path = os.path.join(output_dir, f"{gesture}_{idx:04d}.jpg")
        cv2.imwrite(path, frame)
        saved += 1

        h, w = frame.shape[:2]
        progress = int(w * saved / n_frames)
        display = frame.copy()
        cv2.rectangle(display, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.rectangle(display, (0, h - 12), (w, h), (40, 40, 40), -1)
        cv2.rectangle(display, (0, h - 12), (progress, h), (0, 200, 100), -1)
        cv2.putText(display, f"CAPTURING  {gesture.upper()}  {saved}/{n_frames}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 120), 2)

        cv2.imshow("Collect Data", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index (default 0)")
    parser.add_argument("--gesture", help="Collect only this gesture (optional)")
    parser.add_argument("--frames", type=int, default=100,
                        help="Images to capture per gesture (default 100, recommend 200+)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    gestures = [args.gesture] if args.gesture and args.gesture in CLASS_NAMES else CLASS_NAMES

    print(f"Will collect {args.frames} images each for: {', '.join(gestures)}")
    print("Press ESC during countdown to skip a gesture. Press ESC during capture to stop early.")
    print()

    cap = open_camera(source)

    for gesture in gestures:
        out_dir = os.path.join("dataset", "real", "images", gesture)
        os.makedirs(out_dir, exist_ok=True)

        tip = GESTURE_TIPS.get(gesture, "")
        if not countdown(cap, gesture, tip):
            print(f"  Skipped {gesture}")
            continue

        saved = capture_gesture(cap, gesture, out_dir, args.frames)
        print(f"  {gesture}: saved {saved} images to {out_dir}")

    cap.release()
    cv2.destroyAllWindows()
    print()
    print("Collection complete. Now run:")
    print("  .venv\\Scripts\\python.exe scripts\\auto_label.py")


if __name__ == "__main__":
    main()
