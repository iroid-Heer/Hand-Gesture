import pytest
import shutil
from pathlib import Path
from scripts.split_dataset import split_dataset


def _make_raw_data(tmp_path, n=10):
    img_dir = tmp_path / "raw" / "images"
    lbl_dir = tmp_path / "raw" / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(n):
        (img_dir / f"img_{i:03d}.jpg").write_bytes(b"fake")
        (lbl_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    return img_dir, lbl_dir


def test_split_creates_train_val_test_dirs(tmp_path):
    img_dir, lbl_dir = _make_raw_data(tmp_path)
    out = tmp_path / "dataset"
    split_dataset(str(img_dir), str(lbl_dir), str(out))
    for split in ["train", "val", "test"]:
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()


def test_split_70_20_10_ratio(tmp_path):
    img_dir, lbl_dir = _make_raw_data(tmp_path, n=10)
    out = tmp_path / "dataset"
    counts = split_dataset(str(img_dir), str(lbl_dir), str(out))
    assert counts["train"] == 7
    assert counts["val"] == 2
    assert counts["test"] == 1


def test_split_total_equals_input(tmp_path):
    img_dir, lbl_dir = _make_raw_data(tmp_path, n=20)
    out = tmp_path / "dataset"
    counts = split_dataset(str(img_dir), str(lbl_dir), str(out))
    assert sum(counts.values()) == 20


def test_split_each_image_has_matching_label(tmp_path):
    img_dir, lbl_dir = _make_raw_data(tmp_path, n=10)
    out = tmp_path / "dataset"
    split_dataset(str(img_dir), str(lbl_dir), str(out))
    for split in ["train", "val", "test"]:
        imgs = list((out / "images" / split).glob("*.jpg"))
        for img in imgs:
            assert (out / "labels" / split / img.stem).with_suffix(".txt").exists()


def test_split_raises_on_empty_dir(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    with pytest.raises(ValueError, match="No image-label pairs"):
        split_dataset(str(img_dir), str(lbl_dir), str(tmp_path / "out"))
