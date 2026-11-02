"""COCO JSON etiketlerini YOLO etiketlerine çevirir.

Zenodo 14827784'ün YOLO klasörlerinde yalnızca bulgu içeren görüntülerin
etiketi var, sınıf adları yok ve bazı dosyalar piksel koordinatlı. COCO
dosyası ise her görüntüyü (bulgusuz olanlar dahil), kutuları ve sınıf
adlarını birlikte taşıyor. Bu yüzden COCO varsa onu tercih ediyoruz.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dental_asistan.dataset import (
    IMAGE_EXTS,
    Sample,
    _shared_prefix_len,
    _split_from_path,
)


@dataclass
class CocoResult:
    samples: list[Sample]
    class_names: list[str]
    warnings: list[str] = field(default_factory=list)
    with_boxes: int = 0
    without_boxes: int = 0
    boxes_per_class: dict[str, int] = field(default_factory=dict)


def _load_coco(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict) and all(k in data for k in ("images", "annotations", "categories")):
        return data
    return None


def find_coco_files(root: Path) -> list[tuple[Path, dict]]:
    return [(p, d) for p in sorted(root.rglob("*.json")) if (d := _load_coco(p)) is not None]


def _image_size(path: Path) -> tuple[int, int] | None:
    import cv2

    img = cv2.imread(str(path))
    return None if img is None else (img.shape[1], img.shape[0])


def coco_to_yolo(root: Path, label_dir: Path, include_negatives: bool = True) -> CocoResult | None:
    """Kök altındaki tüm COCO dosyalarını okuyup YOLO etiketlerini label_dir'e yazar.

    Sınıflar, en az bir kutusu olan COCO kategorilerinden oluşur (Roboflow
    dışa aktarımlarındaki boş üst kategori böylece elenir) ve 0'dan
    numaralanır. COCO dosyası yoksa None döner.
    """
    files = find_coco_files(root)
    if not files:
        return None

    warnings: list[str] = []
    images_by_name: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images_by_name[p.name.lower()].append(p)

    # Kategori adları dosyalar arasında birleşir; numaralar dosyaya özel olabilir.
    used_names: set[str] = set()
    for _, data in files:
        cat_name = {c["id"]: str(c["name"]) for c in data["categories"]}
        for ann in data["annotations"]:
            if ann.get("category_id") in cat_name:
                used_names.add(cat_name[ann["category_id"]])
    class_names = sorted(used_names)
    class_index = {n: i for i, n in enumerate(class_names)}

    # (görüntü yolu) -> (ayrım, YOLO satırları)
    per_image: dict[Path, tuple[str | None, list[str]]] = {}
    missing = 0
    boxes_per_class: dict[str, int] = defaultdict(int)
    for json_path, data in files:
        cat_name = {c["id"]: str(c["name"]) for c in data["categories"]}
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        for img in data["images"]:
            name = Path(str(img["file_name"]).replace("\\", "/")).name.lower()
            candidates = images_by_name.get(name, [])
            if not candidates:
                missing += 1
                continue
            image_path = max(candidates, key=lambda c: _shared_prefix_len(c, json_path))
            width, height = img.get("width"), img.get("height")
            if not width or not height:
                size = _image_size(image_path)
                if size is None:
                    missing += 1
                    continue
                width, height = size

            split, lines = per_image.get(image_path, (_split_from_path(json_path) or _split_from_path(image_path), []))
            for ann in anns_by_image.get(img["id"], []):
                bbox = ann.get("bbox") or []
                cname = cat_name.get(ann.get("category_id"))
                if len(bbox) != 4 or cname is None:
                    continue
                x, y, w, h = (float(v) for v in bbox)
                # Görüntü dışına taşan kutuyu kenarlarından kırp
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(width), x + w), min(float(height), y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                vals = [(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height]
                lines.append(" ".join([str(class_index[cname])] + [f"{v:.6f}" for v in vals]))
                boxes_per_class[cname] += 1
            per_image[image_path] = (split, lines)

    if missing:
        warnings.append(f"COCO'da adı geçen {missing} görüntü arşivde bulunamadı veya okunamadı")

    samples: list[Sample] = []
    with_boxes = without_boxes = 0
    used_stems: set[str] = set()
    for image_path, (split, lines) in sorted(per_image.items()):
        if not lines:
            without_boxes += 1
            if not include_negatives:
                continue
        else:
            with_boxes += 1
        stem = image_path.stem
        if stem.lower() in used_stems:
            warnings.append(f"aynı adlı görüntü birden çok kez var, ikincisi atlandı: {image_path.name}")
            continue
        used_stems.add(stem.lower())
        label = label_dir / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        samples.append(Sample(image_path, label, split))

    return CocoResult(samples, class_names, warnings, with_boxes, without_boxes, dict(boxes_per_class))
