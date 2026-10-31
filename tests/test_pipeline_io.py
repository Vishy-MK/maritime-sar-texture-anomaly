import json

import cv2
import numpy as np
import pytest

from sarvision.config import Config
from sarvision.datasets import load_coco, load_yolo
from sarvision.evaluate import evaluate, to_markdown
from sarvision.io_utils import imwrite
from sarvision.metrics import HitRule
from sarvision.pipeline import Pipeline, estimate_shift
from sarvision.synthetic import SceneSpec, SeaRenderer, generate_dataset
from sarvision.viz import STAGE_NAMES, render_stage


def test_config_validation():
    with pytest.raises(ValueError):
        Config(tile=30, stride=16).validate()
    with pytest.raises(ValueError):
        Config(reference="nope").validate()
    with pytest.raises(ValueError):
        Config(min_area=100, max_area=10).validate()
    c = Config.from_dict({**Config().to_dict(), "top_k": 7, "unknown_key": 1})
    assert c.top_k == 7


def test_area_limits_from_altitude():
    low = Config(altitude_m=50).area_limits(1280)
    high = Config(altitude_m=150).area_limits(1280)
    assert low[0] > high[0] and low[1] > high[1]
    assert Config().area_limits(1280) == (Config().min_area, Config().max_area)


def test_phase_correlation_shift_sign():
    rng = np.random.default_rng(0)
    a = cv2.GaussianBlur((rng.random((240, 320)) * 255).astype(np.uint8), (5, 5), 0)
    M = np.float32([[1, 0, 12], [0, 1, -6]])                      # content moves +12 x, -6 y
    b = cv2.warpAffine(a, M, (320, 240), borderMode=cv2.BORDER_REFLECT)
    dx, dy = estimate_shift(a, b)
    assert abs(dx - 12) < 1.5 and abs(dy + 6) < 1.5


def _centre_hit(c, boxes):
    return any(HitRule().hit((c.det.x, c.det.y, c.det.w, c.det.h), b) for b in boxes)


def test_pipeline_finds_person_and_temporal_removes_whitecaps():
    spec = SceneSpec(width=640, height=360, frames=12, altitude=60, n_people=1, whitecap_rate=1.5,
                     seed=11, colours=["orange"])
    r = SeaRenderer(spec)
    pipe = Pipeline(Config(top_k=3))
    hits = 0
    for t in range(spec.frames):
        img, boxes = r.render(t)
        res = pipe.process(img, keep_stages=(t == spec.frames - 1))
        if t >= 4 and boxes and any(_centre_hit(c, boxes) for c in res.shortlist):
            hits += 1
    assert hits >= 6
    # every stage renders at processing resolution or original size
    for name in STAGE_NAMES:
        out = render_stage(name, res, img, pipe.cfg)
        assert out.ndim == 3 and out.dtype == np.uint8


def test_downscaling_maps_boxes_to_original_coordinates():
    spec = SceneSpec(width=960, height=540, frames=1, altitude=40, n_people=1, seed=4, colours=["orange"],
                     whitecap_rate=0, glint=0)
    img, boxes = SeaRenderer(spec).render(0)
    res = Pipeline(Config(max_width=480, temporal_enabled=False, min_area=4)).process(img)
    assert res.scale == pytest.approx(0.5)
    assert any(_centre_hit(c, boxes) for c in res.shortlist)


def test_synthetic_dataset_round_trip_and_evaluate(tmp_path):
    specs = [SceneSpec(width=480, height=320, frames=6, n_people=1, seed=21, altitude=60, colours=["red"])]
    ann = generate_dataset(tmp_path, specs, write_video=False)
    seqs = load_coco(ann)
    assert len(seqs) == 1 and len(seqs[0]) == 6
    assert seqs[0].frames[0].altitude == 60
    assert [f.index for f in seqs[0].frames] == list(range(6))
    res = evaluate(seqs, Config(), k=3, kmax=5, ablations=["full", "no_temporal"])
    full, nt = res["runs"]["full"], res["runs"]["no_temporal"]
    assert full["frames"] == 6 and nt["persons"] == full["persons"]
    assert 0 <= full["recall@3"] <= 1
    assert "| full |" in to_markdown(res)


def test_coco_loader_groups_and_categories(tmp_path):
    for i in range(3):
        imwrite(tmp_path / f"v_{i}.jpg", np.zeros((10, 10, 3), np.uint8))
    data = {
        "images": [{"id": 10 + i, "file_name": f"v_{i}.jpg", "video_id": 7, "frame_index": 2 - i,
                    "meta": {"altitude": 55.0, "gimbal_pitch": -60}} for i in range(3)],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [1, 1, 3, 3]},
            {"id": 2, "image_id": 10, "category_id": 2, "bbox": [5, 5, 3, 3]},
            {"id": 3, "image_id": 11, "category_id": 3, "bbox": [5, 5, 3, 3]},
        ],
        "categories": [{"id": 1, "name": "swimmer"}, {"id": 2, "name": "boat"}, {"id": 3, "name": "ignored"}],
    }
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(data))
    seqs = load_coco(p)
    assert len(seqs) == 1
    names = [f.path.name for f in seqs[0].frames]
    assert names == ["v_2.jpg", "v_1.jpg", "v_0.jpg"]               # ordered by frame_index
    f0 = seqs[0].frames[2]
    assert f0.persons == [(1.0, 1.0, 3.0, 3.0)] and f0.others == [(5.0, 5.0, 3.0, 3.0)]
    assert seqs[0].frames[1].others == []                           # 'ignored' dropped
    assert f0.altitude == 55.0 and f0.angle == -60


def test_yolo_loader(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    imwrite(tmp_path / "images" / "a.jpg", np.zeros((100, 200, 3), np.uint8))
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.2\n2 0.1 0.1 0.05 0.05\n")
    seqs = load_yolo(tmp_path / "images")
    assert len(seqs) == 1
    f = seqs[0].frames[0]
    x, y, w, h = f.persons[0]
    assert (x, y, w, h) == pytest.approx((90, 40, 20, 20))
    assert len(f.others) == 1
