"""Çürük tespit modelini (Ultralytics YOLO) eğitir ve test ayrımında ölçer.

Kullanım:
    python scripts/train_caries.py --data data/caries/data.yaml --epochs 100

NVIDIA ekran kartı varsa otomatik kullanılır (--device 0). Kart yoksa
--device cpu ile çalışır ama çok yavaştır; o durumda Google Colab gibi
GPU'lu bir ortam önerilir.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/caries/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="Başlangıç ağırlıkları (yolo11n/s/m.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="0, 'cpu' vb. Boş bırakılırsa otomatik")
    parser.add_argument("--name", default="caries")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name=args.name,
        patience=20,
        seed=42,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"En iyi ağırlıklar: {best}")

    metrics = YOLO(str(best)).val(data=str(args.data), split="test", imgsz=args.imgsz, device=args.device)
    print("Test sonuçları:")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall (duyarlılık): {metrics.box.mr:.3f}")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
