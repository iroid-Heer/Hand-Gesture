"""
Remap class IDs in a downloaded YOLO dataset to match our 10 gesture class names.

Usage:
    1. Edit EXTERNAL_TO_OURS dict below to map the downloaded dataset's class names
       to our class names.
    2. Run:
       .venv\Scripts\python.exe scripts\remap_classes.py \
           --labels path/to/downloaded/labels \
           --src-yaml path/to/downloaded/data.yaml \
           --out dataset/remapped

The script copies .txt label files, replacing class IDs, and skips any class
that has no mapping (removes those detections).
"""
import os
import yaml
import shutil
import argparse

OUR_CLASSES = ["c", "down", "fist", "fist_moved", "index", "l", "ok", "palm", "palm_moved", "thumb"]

# -----------------------------------------------------------------------
# EDIT THIS: map external class name → our class name
# Leave a class out (or map to None) to discard those detections entirely.
# Common mappings for popular Roboflow / Kaggle datasets:
# -----------------------------------------------------------------------
EXTERNAL_TO_OURS = {
    # --- HaGRID (Roboflow version: machinelearning/hagrid) ---
    "palm":             "palm",
    "stop":             "palm",        # open palm / stop sign
    "fist":             "fist",
    "rock":             "fist",        # rock = closed fist
    "like":             "thumb",       # thumbs up
    "dislike":          "down",        # thumbs down
    "ok":               "ok",
    "call":             "l",           # hang-loose = L-shape
    "one":              "index",       # one finger pointing up
    "mute":             "palm_moved",  # flat hand sideways
    "no_gesture":       None,          # background — discard
    "peace":            None,          # V-sign — no matching class
    "peace_inverted":   None,
    "stop_inverted":    None,
    "two_up":           None,
    "two_up_inverted":  None,
    "three":            None,
    "three2":           None,
    "four":             None,

    # --- Arnav dataset (Palm, Call, Fist, Like, Mute, OK, Peace) ---
    "Palm":             "palm",
    "Call":             "l",
    "Fist":             "fist",
    "Like":             "thumb",
    "Mute":             "palm_moved",
    "OK":               "ok",
    "Peace":            None,

    # --- Generic names found in other Roboflow datasets ---
    "thumbs_up":        "thumb",
    "thumbs_down":      "down",
    "open":             "palm",
    "open_palm":        "palm",
    "pointing":         "index",
    "point":            "index",
    "okay":             "ok",
    "closed_fist":      "fist",
    "l_shape":          "l",
    "c_shape":          "c",
    "five":             "palm",
}
# -----------------------------------------------------------------------


def load_src_classes(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return names


def build_id_map(src_classes):
    """Return dict: src_class_id -> our_class_id, skipping unmapped classes."""
    mapping = {}
    print("Class mapping:")
    for src_id, src_name in enumerate(src_classes):
        our_name = EXTERNAL_TO_OURS.get(src_name)
        if our_name is None:
            print(f"  [{src_id}] {src_name:<20} -> DISCARD")
        elif our_name not in OUR_CLASSES:
            print(f"  [{src_id}] {src_name:<20} -> '{our_name}' NOT IN OUR CLASSES — discarding")
        else:
            our_id = OUR_CLASSES.index(our_name)
            mapping[src_id] = our_id
            print(f"  [{src_id}] {src_name:<20} -> [{our_id}] {our_name}")
    return mapping


def remap_label_file(src_path, dst_path, id_map):
    lines_out = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            src_id = int(parts[0])
            if src_id not in id_map:
                continue                    # discard unmapped class
            our_id = id_map[src_id]
            lines_out.append(f"{our_id} " + " ".join(parts[1:]))
    if lines_out:
        with open(dst_path, "w") as f:
            f.write("\n".join(lines_out) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",   required=True, help="Path to downloaded labels folder")
    parser.add_argument("--images",   default=None,  help="Path to downloaded images folder (optional, copies images too)")
    parser.add_argument("--src-yaml", required=True, help="Path to downloaded data.yaml")
    parser.add_argument("--out",      default=os.path.join("dataset", "remapped"),
                        help="Output folder (default: dataset/remapped)")
    args = parser.parse_args()

    src_classes = load_src_classes(args.src_yaml)
    print(f"\nSource dataset has {len(src_classes)} classes: {src_classes}")
    print()

    id_map = build_id_map(src_classes)
    print(f"\n{len(id_map)} classes will be kept.\n")

    # Remap labels
    out_labels = os.path.join(args.out, "labels")
    os.makedirs(out_labels, exist_ok=True)

    converted = skipped = 0
    for root, _, files in os.walk(args.labels):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            src_path = os.path.join(root, fname)
            rel = os.path.relpath(root, args.labels)
            dst_dir = os.path.join(out_labels, rel)
            os.makedirs(dst_dir, exist_ok=True)
            dst_path = os.path.join(dst_dir, fname)
            remap_label_file(src_path, dst_path, id_map)
            converted += 1

    print(f"Remapped {converted} label files -> {out_labels}")

    # Copy images if provided
    if args.images:
        out_images = os.path.join(args.out, "images")
        print(f"Copying images -> {out_images}")
        shutil.copytree(args.images, out_images, dirs_exist_ok=True)

    # Write new data.yaml
    yaml_out = os.path.join(args.out, "data.yaml")
    with open(yaml_out, "w") as f:
        f.write(f"path: {args.out}\n")
        f.write("train: images\n")
        f.write("val:   images\n\n")
        f.write(f"nc: {len(OUR_CLASSES)}\n")
        f.write(f"names: {OUR_CLASSES}\n")
    print(f"Wrote {yaml_out}")

    print()
    print("Retrain on Colab:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolov8n.pt')")
    print(f"  model.train(data='{yaml_out}', epochs=50, imgsz=640, batch=16)")
    print("  model.export(format='onnx')")


if __name__ == "__main__":
    main()
