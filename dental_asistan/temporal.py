"""Zamansal oylama: tek karede görülen tespit uyarı üretmez.

Bir bulgu, son `window` net karenin en az `min_hits` tanesinde aynı yerde
(IoU >= `iou_threshold`) ve aynı sınıfta görülürse "teyitli" sayılır.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: float


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


@dataclass
class TemporalVoter:
    window: int = 8
    min_hits: int = 3
    iou_threshold: float = 0.3
    _history: deque = field(init=False)

    def __post_init__(self) -> None:
        if not 1 <= self.min_hits <= self.window:
            raise ValueError("min_hits 1 ile window arasında olmalı")
        self._history = deque(maxlen=self.window)

    def reset(self) -> None:
        self._history.clear()

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Yeni karenin tespitlerini ekler, teyitli tespitleri döndürür.

        Dönen her tespit, bu karedeki tespittir (güncel konum); güveni,
        geçmişte eşleştiği karelerdeki güvenlerin ortalamasıdır.
        """
        self._history.append(list(detections))
        confirmed: list[Detection] = []
        for det in detections:
            confs = [det.conf]
            for past in list(self._history)[:-1]:
                match = max(
                    (p for p in past if p.cls == det.cls),
                    key=lambda p: iou(p, det),
                    default=None,
                )
                if match is not None and iou(match, det) >= self.iou_threshold:
                    confs.append(match.conf)
            if len(confs) >= self.min_hits:
                confirmed.append(
                    Detection(det.x1, det.y1, det.x2, det.y2, det.cls, sum(confs) / len(confs))
                )
        return confirmed
