from pathlib import Path
from scripts.verify_dataset import verify_dataset


def _make_valid_dataset(tmp_path):
    for split in ["train", "val", "test"]:
        img_dir = tmp_path / "images" / split
        lbl_dir = tmp_path / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        (img_dir / "img001.jpg").write_bytes(b"fake")
        (lbl_dir / "img001.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    return tmp_path


def test_valid_dataset_has_no_errors(tmp_path):
    dataset = _make_valid_dataset(tmp_path)
    assert verify_dataset(str(dataset)) == []


def test_missing_label_is_reported(tmp_path):
    dataset = _make_valid_dataset(tmp_path)
    (dataset / "images" / "train" / "img002.jpg").write_bytes(b"fake")
    errors = verify_dataset(str(dataset))
    assert any("img002" in e for e in errors)


def test_orphan_label_is_reported(tmp_path):
    dataset = _make_valid_dataset(tmp_path)
    (dataset / "labels" / "train" / "ghost.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    errors = verify_dataset(str(dataset))
    assert any("ghost" in e for e in errors)


def test_missing_split_dir_is_reported(tmp_path):
    dataset = _make_valid_dataset(tmp_path)
    import shutil
    shutil.rmtree(dataset / "images" / "test")
    errors = verify_dataset(str(dataset))
    assert any("test" in e for e in errors)
