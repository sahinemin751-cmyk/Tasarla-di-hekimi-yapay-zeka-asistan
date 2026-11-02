import cv2
import numpy as np

from dental_asistan.quality import assess_frame


def _textured(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, size=(240, 320, 3), dtype=np.uint8)
    return img


def test_sharp_frame_is_usable():
    r = assess_frame(_textured())
    assert r.usable, r
    assert r.reason == "ok"


def test_blurred_frame_is_rejected():
    blurred = cv2.GaussianBlur(_textured(), (31, 31), 10)
    r = assess_frame(blurred)
    assert not r.usable
    assert r.reason == "bulanık"


def test_dark_frame_is_rejected():
    r = assess_frame(np.full((100, 100, 3), 10, dtype=np.uint8))
    assert r.reason == "çok karanlık"


def test_glare_is_rejected():
    img = _textured()
    img[:60, :, :] = 255  # karenin %25'i doymuş
    r = assess_frame(img)
    assert r.reason == "parlama"


def test_grayscale_input_supported():
    gray = cv2.cvtColor(_textured(), cv2.COLOR_BGR2GRAY)
    assert assess_frame(gray).usable
