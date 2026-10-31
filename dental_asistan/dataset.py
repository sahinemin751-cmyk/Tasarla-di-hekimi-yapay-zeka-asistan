"""Açık veri setlerini Ultralytics YOLO formatına getiren yardımcılar.

Zenodo 14827784 ("Annotated intraoral image dataset for dental caries
detection") aynı görüntüleri YOLO, COCO, Pascal VOC ve LabelMe formatlarında
veriyor. Arşivin tam klasör yapısı bilinmediği için bu modül yapıya
güvenmez: tüm görüntüleri bulur, her birini aynı adlı YOLO etiket dosyasıyla
eşler ve klasör adlarında train/val/test varsa o ayrımı korur.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None
    split: str | None  # klasör adından bulunan ayrım, yoksa None


def is_yolo_label(path: Path) -> bool:
    """Dosya YOLO formatında mı: her satır 'sınıf cx cy w h' (0-1 arası)."""
    try:
        lines = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except (UnicodeDecodeError, OSError):
        return False
    for parts in lines:
        if len(parts) < 5:
            return False
        try:
            cls = float(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            return False
        if cls < 0 or cls != int(cls) or any(c < 0 or c > 1 for c in coords):
            return False
    return True  # boş dosya = arka plan görüntüsü, geçerli


def _split_from_path(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        alias = SPLIT_ALIASES.get(part.lower())
        if alias:
            return alias
    return None


def _shared_prefix_len(a: Path, b: Path) -> int:
    n = 0
    for x, y in zip(a.parts, b.parts):
        if x != y:
            break
        n += 1
    return n


def find_samples(root: Path) -> tuple[list[Sample], list[str]]:
    """Kök altındaki görüntüleri bulur ve YOLO etiketleriyle eşler.

    Aynı ada sahip görüntünün birden çok kopyası varsa (her format klasörü
    kendi kopyasını taşıyabilir) yalnızca biri tutulur. Uyarılar ikinci
    değer olarak döner.
    """
    warnings: list[str] = []
    labels_by_stem: dict[str, list[Path]] = defaultdict(list)
    images_by_stem: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            images_by_stem[path.stem].append(path)
        elif suffix == ".txt" and path.name.lower() != "classes.txt" and is_yolo_label(path):
            labels_by_stem[path.stem].append(path)

    samples: list[Sample] = []
    for stem, images in sorted(images_by_stem.items()):
        candidates = labels_by_stem.get(stem, [])
        # Etiketi olan kopyayı, etiket dosyasına klasör olarak en yakın olanı seç.
        best_image, best_label, best_score = images[0], None, -1
        for img in images:
            for lbl in candidates:
                score = _shared_prefix_len(img, lbl)
                if score > best_score:
                    best_image, best_label, best_score = img, lbl, score
        if len(images) > 1:
            splits = {_split_from_path(i) for i in images}
            if len(splits) > 1:
                warnings.append(f"'{stem}' farklı ayrımlarda birden çok kez var: {sorted(map(str, splits))}")
        samples.append(Sample(best_image, best_label, _split_from_path(best_image)))

    return samples, warnings


def find_class_names(root: Path) -> list[str] | None:
    """classes.txt veya YOLO data.yaml içindeki sınıf adlarını bulur."""
    for path in sorted(root.rglob("classes.txt")):
        names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if names:
            return names
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return None
    for path in sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    return None


def max_class_id(samples: list[Sample]) -> int:
    top = -1
    for s in samples:
        if s.label is None:
            continue
        for ln in s.label.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                top = max(top, int(float(ln.split()[0])))
    return top


def assign_splits(
    samples: list[Sample],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """Klasör adından gelen ayrımı korur; ayrımı olmayanları rastgele böler."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("oranların toplamı 1 olmalı")
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    unassigned = []
    for s in samples:
        if s.split:
            out[s.split].append(s)
        else:
            unassigned.append(s)
    rng = random.Random(seed)
    rng.shuffle(unassigned)
    n = len(unassigned)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    out["train"] += unassigned[:n_train]
    out["val"] += unassigned[n_train : n_train + n_val]
    out["test"] += unassigned[n_train + n_val :]
    return out


def write_yolo_dataset(
    splits: dict[str, list[Sample]],
    out_dir: Path,
    class_names: list[str],
) -> Path:
    """Ultralytics'in beklediği images/ labels/ yapısını ve data.yaml'ı yazar."""
    for split, items in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for s in items:
            shutil.copy2(s.image, img_dir / s.image.name)
            target = lbl_dir / f"{s.image.stem}.txt"
            if s.label is not None:
                shutil.copy2(s.label, target)
            else:
                target.write_text("", encoding="utf-8")

    yaml_path = out_dir / "data.yaml"
    lines = [f"path: {out_dir.resolve().as_posix()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    lines += [f"  {i}: {json.dumps(name, ensure_ascii=False)}" for i, name in enumerate(class_names)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path
