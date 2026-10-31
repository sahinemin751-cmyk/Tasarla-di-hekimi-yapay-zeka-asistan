"""Zenodo çürük veri setini (14827784) YOLO eğitimine hazırlar.

Kullanım:
    1. https://zenodo.org/records/14827784 adresinden "Dataset.zip" dosyasını indir.
    2. python scripts/prepare_caries_dataset.py --source indirilenler/Dataset.zip --out data/caries

--source bir zip dosyası ya da zaten açılmış bir klasör olabilir.
Lisans: CC BY 4.0. Yayında ve üründe kaynağa atıf yapılmalı (README'ye bak).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dental_asistan.dataset import (  # noqa: E402
    assign_splits,
    find_class_names,
    find_samples,
    max_class_id,
    write_yolo_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Dataset.zip veya açılmış klasör")
    parser.add_argument("--out", type=Path, default=Path("data/caries"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-unlabeled", action="store_true", help="Etiketsiz görüntüleri arka plan olarak tut")
    args = parser.parse_args()

    source = args.source
    if source.suffix.lower() == ".zip":
        extract_dir = args.out.parent / f"{source.stem}_raw"
        if not extract_dir.exists():
            print(f"Açılıyor: {source} -> {extract_dir}")
            with zipfile.ZipFile(source) as zf:
                zf.extractall(extract_dir)
        source = extract_dir

    samples, warnings = find_samples(source)
    for w in warnings[:20]:
        print("UYARI:", w)
    if len(warnings) > 20:
        print(f"UYARI: {len(warnings) - 20} uyarı daha var")

    labeled = [s for s in samples if s.label is not None]
    unlabeled = len(samples) - len(labeled)
    print(f"Bulunan görüntü: {len(samples)}, YOLO etiketli: {len(labeled)}, etiketsiz: {unlabeled}")
    if not labeled:
        print("HATA: hiç YOLO etiketi bulunamadı. Arşivin içinde YOLO klasörü var mı kontrol et.")
        return 1
    if not args.keep_unlabeled:
        samples = labeled

    names = find_class_names(source)
    top = max_class_id(samples)
    if names is None or len(names) <= top:
        names = ["caries"] if top <= 0 else [f"class_{i}" for i in range(top + 1)]
        print(f"Sınıf adları bulunamadı, varsayılan kullanılıyor: {names}")
    else:
        print(f"Sınıflar: {names}")

    splits = assign_splits(samples, seed=args.seed)
    for split, items in splits.items():
        print(f"  {split}: {len(items)}")

    yaml_path = write_yolo_dataset(splits, args.out, names)
    print(f"Hazır: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
