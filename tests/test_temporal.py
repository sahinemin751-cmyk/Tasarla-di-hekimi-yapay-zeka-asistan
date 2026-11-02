import pytest

from dental_asistan.temporal import Detection, TemporalVoter, iou


def d(x, cls=0, conf=0.5):
    return Detection(x, 10, x + 20, 30, cls, conf)


def test_iou_identical_and_disjoint():
    assert iou(d(0), d(0)) == pytest.approx(1.0)
    assert iou(d(0), d(100)) == 0.0


def test_single_frame_does_not_confirm():
    v = TemporalVoter(window=5, min_hits=3)
    assert v.update([d(0)]) == []


def test_consistent_detection_confirms_after_min_hits():
    v = TemporalVoter(window=5, min_hits=3)
    v.update([d(0, conf=0.4)])
    v.update([d(1, conf=0.6)])
    out = v.update([d(2, conf=0.8)])
    assert len(out) == 1
    assert out[0].x1 == 2  # güncel konum
    assert out[0].conf == pytest.approx(0.6)


def test_flicker_does_not_confirm():
    v = TemporalVoter(window=5, min_hits=3)
    v.update([d(0)])
    v.update([])
    v.update([])
    assert v.update([d(0)]) == []


def test_different_class_does_not_count():
    v = TemporalVoter(window=5, min_hits=2)
    v.update([d(0, cls=1)])
    assert v.update([d(0, cls=0)]) == []


def test_old_hits_fall_out_of_window():
    v = TemporalVoter(window=3, min_hits=3)
    v.update([d(0)])
    v.update([d(0)])
    v.update([])
    v.update([])
    assert v.update([d(0)]) == []


def test_invalid_config():
    with pytest.raises(ValueError):
        TemporalVoter(window=2, min_hits=3)
