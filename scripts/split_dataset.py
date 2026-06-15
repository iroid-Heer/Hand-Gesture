import random
import shutil
from pathlib import Path

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2


def split_dataset(images_dir: str, labels_dir: str, output_dir: str, seed: int = 42) -> dict:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    pairs = sorted([
        f for f in images_dir.glob("*.jpg")
        if (labels_dir / f.stem).with_suffix(".txt").exists()
    ])
    pairs += sorted([
        f for f in images_dir.glob("*.png")
        if (labels_dir / f.stem).with_suffix(".txt").exists()
    ])

    if not pairs:
        raise ValueError(f"No image-label pairs found in {images_dir} / {labels_dir}")

    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split, files in splits.items():
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path in files:
            lbl_path = (labels_dir / img_path.stem).with_suffix(".txt")
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    return {split: len(files) for split, files in splits.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default="dataset")
    args = parser.parse_args()

    counts = split_dataset(args.images, args.labels, args.output)
    for split, count in counts.items():
        print(f"  {split}: {count} images")
