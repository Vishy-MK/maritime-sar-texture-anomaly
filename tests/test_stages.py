import cv2
import numpy as np
import pytest

from sarvision.anomaly import chi_square, local_reference, robust_z, score_anomalies, tiles_to_pixels
from sarvision.config import Config
from sarvision.glint import chroma_distance, glint_mask, suppress_glint
from sarvision.morphology import eccentricity, refine, solidity
from sarvision.normalise import normalise, piecewise_linear
from sarvision.texture import compute_texture


# ------------------------------------------------------------------ stage 1
def test_piecewise_linear_knees():
    x = np.arange(256, dtype=np.uint8).reshape(16, 16)
    y = piecewise_linear(x, 50, 10, 200, 245)
    assert y.flat[0] == 0 and y.flat[255] == 255
    assert y.flat[50] == 10 and y.flat[200] == 245
    assert np.all(np.diff(y.ravel().astype(int)) >= 0)              # monotonic


def test_normalise_flattens_gradient():
    h, w = 120, 160
    grad = np.tile(np.linspace(60, 200, w, dtype=np.float32), (h, 1))
    rng = np.random.default_rng(0)
    gray = np.clip(grad + rng.normal(0, 4, (h, w)), 0, 255).astype(np.uint8)
    bgr = cv2.merge([gray, gray, gray])
    _, g = normalise(bgr, Config())
    left, right = g[:, :40].mean(), g[:, -40:].mean()
    assert abs(left - right) < abs(gray[:, :40].mean() - gray[:, -40:].mean())


# ------------------------------------------------------------------ stage 3
def test_chi_square_properties():
    a = np.array([0.2, 0.3, 0.5])
    assert chi_square(a, a) == pytest.approx(0.0)
    b = np.array([0.5, 0.3, 0.2])
    assert chi_square(a, b) == pytest.approx(chi_square(b, a))
    assert chi_square(np.array([1.0, 0]), np.array([0, 1.0])) == pytest.approx(1.0)


def test_robust_z_ignores_outliers():
    x = np.concatenate([np.random.default_rng(0).normal(0, 1, 1000), [100.0]])
    z = robust_z(x)
    assert z[-1] > 50 and abs(np.median(z)) < 0.1


def test_local_reference_constant():
    d = np.full((20, 30, 4), 0.25)
    np.testing.assert_allclose(local_reference(d, 9), d)


def test_local_reference_follows_gradient():
    ny, nx = 30, 30
    d = np.repeat(np.linspace(1, 3, ny)[:, None, None], nx, axis=1)
    ref = local_reference(d, 9)
    # exact (linear) between the first and last window centres, clamped outside them
    assert np.abs(ref - d)[4:26].max() < 1e-9
    assert np.abs(ref - d).max() <= 4 * (2.0 / 29) + 1e-9


def test_tiles_to_pixels_max_cover():
    cfg = Config(tile=32, stride=16)
    tiles = np.zeros((3, 3), np.float32)
    tiles[1, 1] = 5.0                                               # covers pixels 16..47
    dense = tiles_to_pixels(tiles, cfg, (64, 64), (64, 64))
    assert dense[20, 20] == 5.0 and dense[47, 47] == 5.0
    assert dense[10, 10] == 0.0 and dense[50, 50] == 0.0


def test_anomaly_flags_inserted_object():
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:256, 0:384].astype(np.float32)
    sea = 110 + 25 * np.sin(2 * np.pi * (0.8 * x + 0.6 * y) / 12) + rng.normal(0, 3, x.shape)
    bgr = np.dstack([sea + 40, sea + 10, sea * 0.4]).clip(0, 255).astype(np.uint8)
    cv2.circle(bgr, (200, 128), 6, (0, 110, 245), -1)               # orange object
    cfg = Config()
    bgr_n, g = normalise(bgr, cfg)
    an = score_anomalies(compute_texture(g, cfg), bgr, cfg)
    assert an.zmap[128, 200] > cfg.anomaly_z
    assert an.candidates[128, 200] == 255
    assert an.region.mean() < 0.15                                  # most of the sea is normal


# ------------------------------------------------------------------ stage 4
def test_chroma_and_glint():
    px = np.array([[[250, 250, 250], [0, 110, 245], [40, 40, 40], [255, 245, 238]]], np.uint8)
    cd = chroma_distance(px)[0]
    assert cd[0] < 0.01 and cd[1] > 0.3
    gm = glint_mask(px, Config())[0]
    assert gm.tolist() == [True, False, False, True]                # dark grey is not glint


def test_suppress_glint_drops_white_component_keeps_orange():
    cfg = Config()
    img = np.full((60, 60, 3), (90, 70, 30), np.uint8)
    cv2.circle(img, (15, 15), 4, (252, 252, 252), -1)
    cv2.circle(img, (45, 45), 4, (0, 110, 245), -1)
    cand = np.zeros((60, 60), np.uint8)
    cv2.circle(cand, (15, 15), 5, 255, -1)
    cv2.circle(cand, (45, 45), 5, 255, -1)
    out, _ = suppress_glint(cand, img, cfg)
    assert out[15, 15] == 0 and out[45, 45] == 255
    off, _ = suppress_glint(cand, img, Config(glint_enabled=False))
    assert off[15, 15] == 255


# ------------------------------------------------------------------ stage 5
def test_eccentricity_and_solidity():
    disk = np.zeros((41, 41), np.uint8)
    cv2.circle(disk, (20, 20), 15, 1, -1)
    line = np.zeros((41, 41), np.uint8)
    line[20, 2:39] = 1
    assert eccentricity(disk) < 0.2
    assert eccentricity(line) > 0.99
    assert solidity(disk, int(disk.sum())) > 0.9
    ring = disk.copy()
    cv2.circle(ring, (20, 20), 11, 0, -1)
    assert solidity(ring, int(ring.sum())) < 0.6


def test_refine_rejects_streak_keeps_blob():
    cfg = Config()
    m = np.zeros((100, 100), np.uint8)
    cv2.circle(m, (30, 30), 6, 255, -1)                             # person-like blob
    cv2.line(m, (50, 80), (95, 70), 255, 2)                         # foam streak
    m[5, 90] = 255                                                  # speck
    cleaned, blobs = refine(m, cfg, (cfg.min_area, cfg.max_area))
    assert cleaned[30, 30] == 255
    assert cleaned[75, 72] == 0
    assert cleaned[5, 90] == 0
    assert any(b.rejected == "eccentricity" for b in blobs)
