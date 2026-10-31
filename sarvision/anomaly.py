"""Stage 3: anomaly scoring.

The sea fills most of the frame, so the "normal water" reference is estimated
from the frame itself: the median tile descriptor, taken either over the whole
frame or over a horizontal band of tile rows (so near- and far-field water each
get their own reference). The median, not the mean, keeps the reference clean
when a boat or coastline covers part of the frame.

Each tile is scored by its chi-square distance from the reference, separately
for the Gabor and LBP parts. The two distances are put on a common scale with a
robust z-score (median and MAD over all tiles) and fused. Tiles above
``anomaly_z`` form the anomalous region; inside it, pixels that differ strongly
from the local sea colour are kept as the candidate mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .texture import TextureFeatures

EPS = 1e-9
MAD_TO_SIGMA = 1.4826


def chi_square(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """0.5 * sum((a - b)^2 / (a + b)) over the last axis."""
    return 0.5 * np.sum((a - b) ** 2 / (a + b + EPS), axis=-1)


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * MAD_TO_SIGMA
    if mad < EPS:
        mad = x.std() + EPS
    return (x - med) / mad


def _window_starts(n: int, win: int) -> list[int]:
    if n <= win:
        return [0]
    step = max(1, win // 2)
    starts = list(range(0, n - win + 1, step))
    if starts[-1] != n - win:
        starts.append(n - win)
    return starts


def _interp_axis(grid: np.ndarray, centres: np.ndarray, n: int, axis: int) -> np.ndarray:
    """Linearly interpolate ``grid`` sampled at ``centres`` along ``axis`` onto 0..n-1."""
    if len(centres) == 1:
        return np.repeat(grid, n, axis=axis)
    pos = np.clip(np.arange(n, dtype=np.float64), centres[0], centres[-1])
    hi = np.clip(np.searchsorted(centres, pos, side="right"), 1, len(centres) - 1)
    lo = hi - 1
    t = (pos - centres[lo]) / (centres[hi] - centres[lo])
    shape = [1] * grid.ndim
    shape[axis] = n
    t = t.reshape(shape)
    return np.take(grid, lo, axis=axis) * (1 - t) + np.take(grid, hi, axis=axis) * t


def local_reference(desc: np.ndarray, win: int) -> np.ndarray:
    """Median descriptor over overlapping win x win tile windows, interpolated
    back to every tile. Adapts to near/far field and to illumination drift."""
    ny, nx, f = desc.shape
    ys, xs = _window_starts(ny, win), _window_starts(nx, win)
    wy, wx = min(win, ny), min(win, nx)
    grid = np.empty((len(ys), len(xs), f), dtype=np.float64)
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            grid[i, j] = np.median(desc[y0:y0 + wy, x0:x0 + wx].reshape(-1, f), axis=0)
    cy = np.array(ys, dtype=np.float64) + (wy - 1) / 2.0
    cx = np.array(xs, dtype=np.float64) + (wx - 1) / 2.0
    return _interp_axis(_interp_axis(grid, cy, ny, 0), cx, nx, 1)


def reference(desc: np.ndarray, cfg: Config) -> np.ndarray:
    """Median "normal sea" descriptor for every tile; same shape as ``desc``.

    global: one median over the whole frame.
    band:   median over a band of +-reference_window//2 tile rows (near vs far field).
    local:  median over overlapping square windows of reference_window tiles.
    """
    ny = desc.shape[0]
    win = max(3, int(cfg.reference_window))
    if cfg.reference == "global":
        med = np.median(desc.reshape(-1, desc.shape[-1]), axis=0)
        return np.broadcast_to(med, desc.shape)
    if cfg.reference == "local":
        return local_reference(desc, win)
    b = win // 2
    ref = np.empty_like(desc)
    for r in range(ny):
        lo, hi = max(0, r - b), min(ny, r + b + 1)
        ref[r] = np.median(desc[lo:hi].reshape(-1, desc.shape[-1]), axis=0)
    return ref


def tile_scores(feat: TextureFeatures, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fused z, Gabor chi-square, LBP chi-square), each (ny, nx)."""
    g_ref = reference(feat.gabor, cfg)
    g_rel = feat.gabor / (g_ref + EPS)              # reference becomes all-ones
    d_gabor = chi_square(g_rel, np.ones_like(g_rel)) * 2.0 / g_rel.shape[-1]

    l_ref = reference(feat.lbp, cfg)
    l_ref = l_ref / (l_ref.sum(axis=-1, keepdims=True) + EPS)
    d_lbp = chi_square(feat.lbp, l_ref)

    w = cfg.lbp_weight
    z_lbp = robust_z(d_lbp)
    fused = (1.0 - w) * robust_z(d_gabor) + w * z_lbp
    # Near the frame edge the Gabor kernels overlap the padding and respond to
    # mirrored waves; there the score falls back to the (3x3, border-safe) LBP term.
    edge = border_tiles(fused.shape, cfg, feat.shape, feat.gabor_margin)
    if edge.any() and not edge.all():
        fused = np.where(edge, z_lbp, fused)
    return fused, d_gabor, d_lbp


def border_tiles(grid: tuple, cfg: Config, shape: tuple, margin: int) -> np.ndarray:
    """Tiles that come within ``margin`` px of the frame edge (or lie in the padding)."""
    ny, nx = grid
    h, w = shape
    y0 = np.arange(ny) * cfg.stride
    x0 = np.arange(nx) * cfg.stride
    ok_y = (y0 >= margin) & (y0 + cfg.tile <= h - margin)
    ok_x = (x0 >= margin) & (x0 + cfg.tile <= w - margin)
    return ~(ok_y[:, None] & ok_x[None, :])


def tiles_to_pixels(tile_vals: np.ndarray, cfg: Config, pad_shape: tuple, out_shape: tuple) -> np.ndarray:
    """Dense map: each pixel takes the maximum over the tiles that cover it."""
    m = cfg.tile // cfg.stride
    ny, nx = tile_vals.shape
    cy, cx = ny + m - 1, nx + m - 1
    cells = np.full((cy, cx), -np.inf, dtype=np.float32)
    for dy in range(m):
        for dx in range(m):
            np.maximum(cells[dy:dy + ny, dx:dx + nx], tile_vals, out=cells[dy:dy + ny, dx:dx + nx])
    s = cfg.stride
    dense = np.repeat(np.repeat(cells, s, axis=0), s, axis=1)
    dense = dense[:pad_shape[0], :pad_shape[1]]
    return dense[:out_shape[0], :out_shape[1]]


@dataclass
class AnomalyResult:
    tile_z: np.ndarray          # (ny, nx) fused robust z-score per tile
    d_gabor: np.ndarray
    d_lbp: np.ndarray
    zmap: np.ndarray            # (H, W) dense anomaly map (max covering tile z)
    region: np.ndarray          # (H, W) bool: inside an anomalous tile
    deviation: np.ndarray       # (H, W) float: colour distance from local sea
    pixel_thresh: float
    candidates: np.ndarray      # (H, W) uint8 0/255 candidate mask


def colour_deviation(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Distance in Lab between each pixel and the median-filtered local sea colour.

    Computed on the un-stretched frame: the local median already removes the
    illumination gradient, and contrast stretching would inflate wave contrast
    relative to the target.
    """
    h, w = bgr.shape[:2]
    # the background is smooth, so the median is taken at half resolution
    small = cv2.resize(bgr, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    k = max(3, (cfg.background_ksize // 2) | 1)
    background = cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    return cv2.magnitude(cv2.magnitude(lab[..., 0] - lab_bg[..., 0], lab[..., 1] - lab_bg[..., 1]),
                         lab[..., 2] - lab_bg[..., 2])


def score_anomalies(feat: TextureFeatures, bgr: np.ndarray, cfg: Config) -> AnomalyResult:
    """``bgr`` is the frame at processing resolution, before normalisation."""
    h, w = bgr.shape[:2]
    fused, d_g, d_l = tile_scores(feat, cfg)
    zmap = tiles_to_pixels(fused.astype(np.float32), cfg, feat.pad_shape, (h, w))
    region = zmap > cfg.anomaly_z

    dev = colour_deviation(bgr, cfg)
    med = float(np.median(dev))
    mad = float(np.median(np.abs(dev - med))) * MAD_TO_SIGMA + EPS
    thr = med + cfg.pixel_z * mad
    cand = ((dev > thr) & region).astype(np.uint8) * 255
    return AnomalyResult(fused, d_g, d_l, zmap, region, dev, thr, cand)
