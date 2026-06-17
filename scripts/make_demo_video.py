"""Build a demo video from training images — all 10 gestures, 3 seconds each."""
import cv2
import numpy as np
import argparse
import os

CLASS_NAMES = ["c", "down", "fist", "fist_moved", "index", "l", "ok", "palm", "palm_moved", "thumb"]
FPS = 15
SECONDS_PER_GESTURE = 3
FRAMES_PER_GESTURE = FPS * SECONDS_PER_GESTURE
OUTPUT_SIZE = (640, 360)   # width x height (16:9, good for display)


def load_images_for_gesture(gesture: str, count: int) -> list:
    imgs = []
    for i in range(1, count + 1):
        path = f"dataset/images/train/{gesture}_{i:03d}.png"
        if not os.path.exists(path):
            break
        img = cv2.imread(path)
        if img is not None:
            imgs.append(img)
        if len(imgs) >= count:
            break
    return imgs


def add_overlay(frame: np.ndarray, gesture: str, frame_idx: int, total: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Dark banner at top
    cv2.rectangle(out, (0, 0), (w, 48), (20, 20, 20), -1)

    # Gesture name
    cv2.putText(out, "Gesture: " + gesture.upper(), (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 120), 2)

    # Progress bar at bottom
    bar_y = h - 8
    cv2.rectangle(out, (0, bar_y), (w, h), (40, 40, 40), -1)
    fill = int(w * frame_idx / total)
    cv2.rectangle(out, (0, bar_y), (fill, h), (0, 200, 100), -1)

    return out


def main():
    parser = argparse.ArgumentParser(description="Build a demo video from training images")
    parser.add_argument("--out", default="demo.mp4", help="Output video path")
    args = parser.parse_args()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, FPS, OUTPUT_SIZE)

    total_frames = len(CLASS_NAMES) * FRAMES_PER_GESTURE

    print(f"Building {args.out}  ({len(CLASS_NAMES)} gestures x {SECONDS_PER_GESTURE}s @ {FPS}fps)")

    written = 0
    for gesture in CLASS_NAMES:
        imgs = load_images_for_gesture(gesture, FRAMES_PER_GESTURE)
        if not imgs:
            print(f"  WARNING: no images found for '{gesture}', skipping")
            continue

        print(f"  {gesture:<12} {len(imgs)} source images -> {FRAMES_PER_GESTURE} frames")

        for i in range(FRAMES_PER_GESTURE):
            src = imgs[i % len(imgs)]
            # Resize to output dimensions
            frame = cv2.resize(src, OUTPUT_SIZE)
            frame = add_overlay(frame, gesture, written, total_frames)
            writer.write(frame)
            written += 1

    writer.release()
    size_kb = os.path.getsize(args.out) // 1024
    print(f"\nSaved: {args.out}  ({size_kb} KB, {written} frames)")
    print()
    print("Run detection on this video:")
    print(f"  python -m src.main --model models\\gesture.onnx --source {args.out}")


if __name__ == "__main__":
    main()
