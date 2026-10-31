"""Kare kalite kapısı: bulanık, çok karanlık veya parlamalı kareleri eler.

Loup kamerası hekimin başıyla hareket ettiği için karelerin çoğu bulanıktır.
Yalnızca net kareleri modele vermek yanlış alarmları ciddi ölçüde azaltır.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_sharpness: float = 100.0  # Laplacian varyansı; kameraya göre ayarlanmalı
    min_brightness: float = 40.0  # 0-255 ortalama parlaklık
    max_brightness: float = 220.0
    max_glare_ratio: float = 0.08  # doymuş (>=250) piksellerin oranı


@dataclass(frozen=True)
class QualityResult:
    usable: bool
    sharpness: float
    brightness: float
    glare_ratio: float
    reason: str


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def assess_frame(frame: np.ndarray, thresholds: QualityThresholds | None = None) -> QualityResult:
    """Kareyi değerlendirir ve kullanılabilir olup olmadığını döndürür."""
    t = thresholds or QualityThresholds()
    gray = _to_gray(frame)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    glare_ratio = float((gray >= 250).mean())

    if brightness < t.min_brightness:
        reason = "çok karanlık"
    elif brightness > t.max_brightness:
        reason = "çok parlak"
    elif glare_ratio > t.max_glare_ratio:
        reason = "parlama"
    elif sharpness < t.min_sharpness:
        reason = "bulanık"
    else:
        reason = "ok"

    return QualityResult(
        usable=reason == "ok",
        sharpness=sharpness,
        brightness=brightness,
        glare_ratio=glare_ratio,
        reason=reason,
    )
