import numpy as np
import pytest

from sarvision.config import Config
from sarvision.metrics import Accumulator, HitRule, iou
from sarvision.tracking import Detection, Tracker


def det(x, y, conf=0.6):
    return Detection(x - 5, y - 5, 10, 10, 10.0, conf)


def test_persistent_coherent_track_is_promoted():
    cfg = Config(min_persistence=3)
    tr = Tracker(cfg)
    for f in range(5):
        tr.update(f, [det(100 + 0.5 * f, 50 + 0.3 * f)])
        ranked = tr.ranked(f)
        if f < 2:
            assert ranked == []
        else:
            assert len(ranked) == 1 and ranked[0].hits == f + 1
    assert ranked[0].coherence > 0.8


def test_transient_detections_never_promoted():
    cfg = Config(min_persistence=3)
    tr = Tracker(cfg)
    rng = np.random.default_rng(0)
    for f in range(20):
        dets = [det(*rng.uniform(0, 1000, 2)) for _ in range(3)]       # whitecaps: random places
        tr.update(f, dets)
        assert tr.ranked(f) == []


def test_jumpy_track_rejected_by_coherence():
    cfg = Config(min_persistence=3, match_radius=60, drift_sigma=4, min_coherence=0.25)
    tr = Tracker(cfg)
    pts = [(100, 100), (140, 100), (100, 100), (140, 100), (100, 100), (140, 100)]
    for f, p in enumerate(pts):
        tr.update(f, [det(*p)])
    assert tr.ranked(len(pts) - 1) == []


def test_gap_tolerance():
    cfg = Config(min_persistence=3, max_gap=2)
    tr = Tracker(cfg)
    for f in range(4):
        tr.update(f, [det(200, 200)])
    tr.update(4, [])                                               # hidden under a wave
    tr.update(5, [])
    r = tr.ranked(5)
    assert len(r) == 1 and r[0].missed == 2
    tr.update(6, [det(200, 201)])
    assert tr.ranked(6)[0].hits == 5
    for f in range(7, 11):
        tr.update(f, [])
    assert tr.tracks == []


def test_camera_shift_keeps_identity():
    cfg = Config(min_persistence=3, match_radius=10)
    tr = Tracker(cfg)
    for f in range(5):
        tr.update(f, [det(100 + 30 * f, 100)], shift=(30.0, 0.0) if f else (0.0, 0.0))
    r = tr.ranked(4)
    assert len(r) == 1 and r[0].hits == 5


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_hit_rules():
    gt = (100, 100, 12, 12)
    assert HitRule("centre").hit((102, 103, 10, 10), gt)            # centres 2.2 px apart
    assert not HitRule("centre").hit((104, 104, 30, 30), gt)        # 18.4 px apart > radius 12
    assert HitRule("centre", centre_factor=2.0).hit((104, 104, 30, 30), gt)
    assert not HitRule("centre", min_radius=8).hit((140, 140, 10, 10), gt)
    assert HitRule("iou", iou_thr=0.5).hit((101, 101, 12, 12), gt)
    assert not HitRule("iou", iou_thr=0.5).hit((106, 106, 12, 12), gt)


def test_accumulator_recall_and_false_alarms():
    acc = Accumulator(kmax=3)
    gts = [(100, 100, 10, 10), (300, 300, 10, 10)]
    # frame 1: rank1 false, rank2 hits gt0, rank3 false
    acc.add([(500, 500, 10, 10), (101, 101, 10, 10), (700, 50, 10, 10)], gts, others=[(700, 50, 10, 10)])
    # frame 2: no predictions
    acc.add([], [(10, 10, 10, 10)])
    assert acc.recall_at(1) == 0.0
    assert acc.recall_at(2) == pytest.approx(1 / 3)
    assert acc.recall_at(3) == pytest.approx(1 / 3)
    assert acc.fa_per_frame(1) == pytest.approx(0.5)
    assert acc.fa_per_frame(3) == pytest.approx(1.0)
    assert acc.fa_per_frame(3, exclude_other_objects=True) == pytest.approx(0.5)
    assert acc.precision_at(3) == pytest.approx(1 / 3)
    s = acc.summary(2)
    assert s["persons"] == 3 and s["frames"] == 2
    assert s["by_size"]["<16 px"]["n"] == 3
