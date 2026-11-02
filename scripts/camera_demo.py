"""Canlı kamera demosu: kalite kapısı + çürük modeli + zamansal oylama.

Kullanım:
    python scripts/camera_demo.py --weights runs/caries/weights/best.pt --camera 0

Tuşlar:  s = kareyi ve tespitleri captures/ klasörüne kaydet,  q = çık

Herhangi bir USB/UVC kamera (intraoral kamera, loup kamerası, web kamerası)
çalışır. ARAŞTIRMA PROTOTİPİDİR, KLİNİK KARAR İÇİN DEĞİLDİR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dental_asistan.quality import QualityThresholds, assess_frame  # noqa: E402
from dental_asistan.temporal import Detection, TemporalVoter  # noqa: E402

# OpenCV yazı tipi Türkçe karakterleri çizemez; ekrandaki yazılar ASCII.
BANNER = "ARASTIRMA PROTOTIPI - KLINIK KARAR ICIN DEGILDIR"
STATUS_ASCII = {"ok": "OK", "bulanık": "BULANIK", "çok karanlık": "KARANLIK",
                "çok parlak": "PARLAK", "parlama": "PARLAMA"}


def open_camera(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--min-sharpness", type=float, default=100.0)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--captures", type=Path, default=Path("captures"))
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names = model.names
    thresholds = QualityThresholds(min_sharpness=args.min_sharpness)
    voter = TemporalVoter(window=args.window, min_hits=args.min_hits)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        print(f"Kamera {args.camera} açılamadı.")
        return 1

    confirmed: list[Detection] = []
    last = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kameradan kare alınamadı.")
            break

        quality = assess_frame(frame, thresholds)
        if quality.usable:
            result = model.predict(frame, conf=args.conf, verbose=False)[0]
            dets = [
                Detection(*map(float, box.xyxy[0].tolist()), int(box.cls[0]), float(box.conf[0]))
                for box in result.boxes
            ]
            confirmed = voter.update(dets)
        # Bulanık karede eski teyitli kutular ekranda kalır, oylamaya girmez.

        view = frame.copy()
        for d in confirmed:
            p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
            cv2.rectangle(view, p1, p2, (0, 165, 255), 2)
            cv2.putText(view, f"{names[d.cls]} {d.conf:.2f}", (p1[0], max(p1[1] - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        now = time.perf_counter()
        fps, last = 1.0 / max(now - last, 1e-6), now
        status = STATUS_ASCII[quality.reason]
        cv2.putText(view, f"{status}  netlik={quality.sharpness:.0f}  fps={fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, BANNER, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Dental asistan", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            args.captures.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(str(args.captures / f"{stamp}.jpg"), frame)
            (args.captures / f"{stamp}.json").write_text(json.dumps({
                "sharpness": quality.sharpness,
                "detections": [{"cls": names[d.cls], "conf": d.conf, "xyxy": [d.x1, d.y1, d.x2, d.y2]}
                               for d in confirmed],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Kaydedildi: {stamp}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
