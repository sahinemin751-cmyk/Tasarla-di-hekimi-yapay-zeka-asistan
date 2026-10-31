from pathlib import Path

import pytest

from dental_asistan.dataset import (
    assign_splits,
    find_class_names,
    find_samples,
    is_yolo_label,
    max_class_id,
    write_yolo_dataset,
)


def _touch(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _img(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff")  # içerik önemli değil, kopyalanıyor
    return path


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    """Zenodo arşivine benzeyen yapı: aynı görüntüler birden çok format klasöründe."""
    root = tmp_path / "raw"
    for split in ("train", "val"):
        for i in range(3):
            stem = f"{split}_{i}"
            _img(root / "YOLO" / split / "images" / f"{stem}.jpg")
            _touch(root / "YOLO" / split / "labels" / f"{stem}.txt", "0 0.5 0.5 0.2 0.2\n")
            _img(root / "COCO" / split / f"{stem}.jpg")  # kopya, etiketsiz
            _touch(root / "VOC" / split / f"{stem}.xml", "<annotation/>")
    _img(root / "YOLO" / "train" / "images" / "unlabeled.jpg")
    _touch(root / "YOLO" / "notes.txt", "bu bir not, etiket değil")
    _touch(root / "YOLO" / "classes.txt", "caries\n")
    return root


def test_is_yolo_label(tmp_path):
    assert is_yolo_label(_touch(tmp_path / "a.txt", "0 0.1 0.2 0.3 0.4\n"))
    assert is_yolo_label(_touch(tmp_path / "empty.txt", ""))
    assert not is_yolo_label(_touch(tmp_path / "b.txt", "merhaba dünya"))
    assert not is_yolo_label(_touch(tmp_path / "c.txt", "0 120 40 30 30\n"))  # piksel, normalize değil
    assert not is_yolo_label(_touch(tmp_path / "d.txt", "0.5 0.1 0.2 0.3 0.4\n"))


def test_find_samples_dedupes_and_pairs(raw):
    samples, warnings = find_samples(raw)
    assert warnings == []
    by_stem = {s.image.stem: s for s in samples}
    assert len(by_stem) == 7
    s = by_stem["train_0"]
    assert s.label is not None and s.label.name == "train_0.txt"
    assert "YOLO" in s.image.parts  # etiketin yanındaki kopya seçildi
    assert s.split == "train"
    assert by_stem["val_2"].split == "val"
    assert by_stem["unlabeled"].label is None


def test_class_names_and_max_id(raw):
    assert find_class_names(raw) == ["caries"]
    samples, _ = find_samples(raw)
    assert max_class_id(samples) == 0


def test_assign_splits_keeps_folder_split_and_splits_rest(tmp_path):
    root = tmp_path / "flat"
    for i in range(10):
        _img(root / f"img{i}.jpg")
        _touch(root / f"img{i}.txt", "0 0.5 0.5 0.1 0.1\n")
    _img(root / "test" / "t.jpg")
    samples, _ = find_samples(root)
    splits = assign_splits(samples, seed=1)
    assert {s.image.stem for s in splits["test"]} >= {"t"}
    assert sum(len(v) for v in splits.values()) == 11
    assert len(splits["train"]) == 8
    # aynı tohum aynı bölmeyi verir
    again = assign_splits(samples, seed=1)
    assert [s.image for s in again["train"]] == [s.image for s in splits["train"]]


def test_write_yolo_dataset(raw, tmp_path):
    samples, _ = find_samples(raw)
    splits = assign_splits(samples)
    out = tmp_path / "out"
    yaml_path = write_yolo_dataset(splits, out, ["caries"])
    assert (out / "images" / "train" / "train_0.jpg").exists()
    assert (out / "labels" / "train" / "train_0.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert (out / "labels" / "train" / "unlabeled.txt").read_text() == ""
    text = yaml_path.read_text(encoding="utf-8")
    assert 'names:\n  0: "caries"' in text
    assert "train: images/train" in text
