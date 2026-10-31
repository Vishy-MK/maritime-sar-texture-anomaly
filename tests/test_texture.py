import math

import numpy as np
import pytest

from sarvision.config import Config
from sarvision.texture import (UNIFORM_LUT, build_gabor_bank, cell_sums, compute_texture, estimate_wave,
                               gabor_energy, lbp_codes, pyramid_level, tile_sums, uniform_lbp, WaveEstimate)


def stripes(h, w, wavelength, theta_deg):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    t = math.radians(theta_deg)
    return (127 + 60 * np.sin(2 * math.pi * (x * math.cos(t) + y * math.sin(t)) / wavelength)).astype(np.uint8)


def test_uniform_lut_has_58_uniform_codes():
    assert len(set(UNIFORM_LUT.tolist())) == 59
    assert (UNIFORM_LUT < 58).sum() == 58
    assert UNIFORM_LUT[0] != 58 and UNIFORM_LUT[255] != 58      # all-zeros / all-ones are uniform
    assert UNIFORM_LUT[0b01010101] == 58                          # 8 transitions -> catch-all


def test_lbp_constant_image_is_all_ones():
    g = np.full((10, 10), 77, np.uint8)
    assert (lbp_codes(g) == 255).all()


def test_lbp_invariant_to_brightness_offset():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 200, (32, 32)).astype(np.uint8)
    assert np.array_equal(uniform_lbp(g), uniform_lbp(g + 40))


def test_lbp_single_bright_pixel():
    g = np.zeros((3, 3), np.uint8)
    g[1, 1] = 9
    assert lbp_codes(g)[1, 1] == 0                                 # every neighbour darker


@pytest.mark.parametrize("wl,theta", [(10, 0), (12, 30), (8, 90), (16, 135)])
def test_fft_recovers_wavelength_and_orientation(wl, theta):
    est = estimate_wave(stripes(256, 256, wl, theta), 4, 40)
    assert abs(est.wavelength - wl) / wl < 0.12
    diff = abs((math.degrees(est.theta) - theta + 90) % 180 - 90)
    assert diff < 6


def test_gabor_prefers_matching_orientation():
    img = stripes(128, 128, 10, 30)
    cfg = Config(n_orientations=4)
    bank = build_gabor_bank([WaveEstimate(10, math.radians(30), 1.0)], cfg)
    energies = [float(gabor_energy(img.astype(np.float32), f)[32:96, 32:96].mean()) for f in bank]
    assert int(np.argmax(energies)) == 0                           # first filter aligned with the waves
    assert energies[0] > 3 * energies[2]                           # orthogonal filter responds weakly


def test_pyramid_level():
    assert pyramid_level(6, 16) == 0
    assert pyramid_level(8, 16) == 1
    assert pyramid_level(20, 16) == 2
    assert pyramid_level(20, 2) == 1                               # limited by the stride


def test_tile_sums_match_brute_force():
    rng = np.random.default_rng(1)
    a = rng.random((64, 96)).astype(np.float32)
    s, m = 8, 4
    tiles = tile_sums(cell_sums(a, s), m)[..., 0]
    for i in (0, 3, tiles.shape[0] - 1):
        for j in (0, 5, tiles.shape[1] - 1):
            assert tiles[i, j] == pytest.approx(a[i * s:i * s + m * s, j * s:j * s + m * s].sum(), rel=1e-6)


def test_compute_texture_shapes_and_normalisation():
    cfg = Config(tile=32, stride=16)
    g = stripes(200, 300, 12, 45)                                 # not multiples of the stride
    f = compute_texture(g, cfg)
    H, W = f.pad_shape
    assert H % 16 == 0 and W % 16 == 0 and H >= 200 and W >= 300
    ny, nx = H // 16 - 1, W // 16 - 1
    assert f.gabor.shape[:2] == (ny, nx) and f.lbp.shape == (ny, nx, 59)
    np.testing.assert_allclose(f.lbp.sum(-1), 1.0, rtol=1e-9)
    assert f.energy_map.shape == (200, 300)
