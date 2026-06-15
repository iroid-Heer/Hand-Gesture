import sys
from pathlib import Path


def verify_dataset(dataset_dir: str) -> list:
    dataset_dir = Path(dataset_dir)
    errors = []

    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            errors.append(f"[{split}] Missing directory: {img_dir}")
            continue
        if not lbl_dir.exists():
            errors.append(f"[{split}] Missing directory: {lbl_dir}")
            continue

        images = {f.stem for f in img_dir.glob("*.jpg")} | {f.stem for f in img_dir.glob("*.png")}
        labels = {f.stem for f in lbl_dir.glob("*.txt")}

        for stem in sorted(images - labels):
            errors.append(f"[{split}] Image has no label: {stem}")
        for stem in sorted(labels - images):
            errors.append(f"[{split}] Label has no image: {stem}")

    return errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()

    errors = verify_dataset(args.dataset)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        sys.exit(1)
    print("Dataset verification passed — all image-label pairs match.")
