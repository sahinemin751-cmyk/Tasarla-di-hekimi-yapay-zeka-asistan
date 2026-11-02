import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from dental_asistan.coco import coco_to_yolo

REPO = Path(__file__).resolve().parents[1]


def _jpg(path: Path, w: int = 200, h: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


def _benchmark_like(root: Path) -> Path:
    """Zenodo 'Benchmarking Dataset' benzeri: split/images, split/coco, split/yolo."""
    base = root / "Benchmarking Dataset"
    for split, names in (("train", ["a", "b", "c"]), ("valid", ["d"]), ("test", ["e", "f"])):
        images, anns = [], []
        for i, n in enumerate(names):
            _jpg(base / split / "images" / f"{n}.jpg")
            img = {"id": i, "file_name": f"{n}.jpg", "width": 200, "height": 100}
            if n == "f":
                img.pop("width")  # boyut eksikse görüntüden okunmalı
            images.append(img)
        # a: iki kutu (iki sınıf), b: bulgusuz, c: bir kutu; d: bir kutu; e: bulgusuz; f: bir kutu
        boxes = {"a": [(1, [20, 10, 40, 20]), (2, [100, 50, 20, 20])], "c": [(1, [0, 0, 200, 100])],
                 "d": [(1, [10, 10, 10, 10])], "f": [(2, [190, 90, 20, 20])]}
        for i, n in enumerate(names):
            for cat, bbox in boxes.get(n, []):
                anns.append({"id": len(anns), "image_id": i, "category_id": cat, "bbox": bbox})
        coco = {
            "images": images,
            "annotations": anns,
            "categories": [
                {"id": 0, "name": "dental-caries", "supercategory": "none"},  # Roboflow'un boş üst kategorisi
                {"id": 1, "name": "Caries", "supercategory": "dental-caries"},
                {"id": 2, "name": "Secondary Caries", "supercategory": "dental-caries"},
            ],
        }
        (base / split / "coco").mkdir(parents=True)
        (base / split / "coco" / "_annotations.coco.json").write_text(json.dumps(coco))
        # Bozuk/eksik YOLO klasörü de dursun; COCO tercih edilmeli
        (base / split / "yolo").mkdir()
        (base / split / "yolo" / f"{names[0]}.txt").write_text("0 120 40 30 30\n")
    return base


def test_coco_to_yolo(tmp_path):
    _benchmark_like(tmp_path)
    res = coco_to_yolo(tmp_path, tmp_path / "labels")
    assert res is not None
    assert res.class_names == ["Caries", "Secondary Caries"]
    assert (res.with_boxes, res.without_boxes) == (4, 2)
    assert res.boxes_per_class == {"Caries": 3, "Secondary Caries": 2}
    by_stem = {s.image.stem: s for s in res.samples}
    assert {k: v.split for k, v in by_stem.items()} == {
        "a": "train", "b": "train", "c": "train", "d": "val", "e": "test", "f": "test"}
    assert by_stem["a"].label.read_text() == (
        "0 0.200000 0.200000 0.200000 0.200000\n1 0.550000 0.600000 0.100000 0.200000\n")
    assert by_stem["b"].label.read_text() == ""
    # f: genişlik görüntüden okundu, taşan kutu kırpıldı
    assert by_stem["f"].label.read_text() == "1 0.975000 0.950000 0.050000 0.100000\n"


def test_no_negatives(tmp_path):
    _benchmark_like(tmp_path)
    res = coco_to_yolo(tmp_path, tmp_path / "labels", include_negatives=False)
    assert sorted(s.image.stem for s in res.samples) == ["a", "c", "d", "f"]


def test_no_coco_returns_none(tmp_path):
    _jpg(tmp_path / "x.jpg")
    assert coco_to_yolo(tmp_path, tmp_path / "labels") is None


def test_prepare_script_end_to_end(tmp_path):
    _benchmark_like(tmp_path / "raw")
    out = tmp_path / "data" / "caries"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_caries_dataset.py"),
         "--source", str(tmp_path / "raw"), "--out", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert "Etiket kaynağı: COCO" in proc.stdout
    assert "0: Caries (3 kutu)" in proc.stdout
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert '0: "Caries"' in yaml_text and '1: "Secondary Caries"' in yaml_text
    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == ["a.jpg", "b.jpg", "c.jpg"]
    assert sorted(p.name for p in (out / "labels" / "test").iterdir()) == ["e.txt", "f.txt"]
    assert (out / "labels" / "val" / "d.txt").read_text().startswith("0 ")
