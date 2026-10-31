"""Stage 2: texture characterisation.

The frame is cut into overlapping tiles (side ``tile``, step ``stride``). Each
tile is described by

* the mean energy of an oriented Gabor filter bank whose wavelength and
  orientations are tuned to the dominant sea-wave spectrum, estimated with a
  2-D Fourier transform, and
* a 59-bin uniform Local Binary Pattern histogram.

Tile statistics are computed from non-overlapping ``stride x stride`` cells and
summed over ``m x m`` cells (``m = tile / stride``), so overlapping tiles cost no
more than non-overlapping ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config

N_LBP_BINS = 59
MAX_FFT_SIDE = 512
MIN_PYRAMID_WAVELENGTH = 4.0   # a filter runs on the coarsest pyramid level keeping lambda >= this


# --------------------------------------------------------------------------- LBP

def _uniform_lut() -> np.ndarray:
    """Map each 8-bit LBP code to one of 59 bins (58 uniform codes + 1 catch-all)."""
    lut = np.full(256, 58, dtype=np.uint8)
    nxt = 0
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            lut[code] = nxt
            nxt += 1
    assert nxt == 58
    return lut


UNIFORM_LUT = _uniform_lut()

# neighbour offsets (dy, dx) in circular order starting at the right-hand pixel
_NEIGHBOURS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def lbp_codes(gray: np.ndarray) -> np.ndarray:
    """Basic 8-neighbour LBP: bit i is 1 if neighbour i >= centre."""
    g = gray.astype(np.int16)
    p = np.pad(g, 1, mode="edge")
    h, w = g.shape
    code = np.zeros((h, w), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(_NEIGHBOURS):
        nb = p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        code |= ((nb >= g).astype(np.uint8) << bit)
    return code


def uniform_lbp(gray: np.ndarray) -> np.ndarray:
    """Uniform LBP bin index (0-58) per pixel."""
    return UNIFORM_LUT[lbp_codes(gray)]


# ------------------------------------------------------------ wave estimation

@dataclass
class WaveEstimate:
    wavelength: float      # px
    theta: float           # radians, direction of the wave vector (normal to the crests)
    strength: float        # spectral peak over mean power in the search annulus


def estimate_wave(gray: np.ndarray, min_wl: float, max_wl: float) -> WaveEstimate:
    """Dominant wavelength and orientation from the peak of the 2-D power spectrum."""
    h, w = gray.shape
    if w > MAX_FFT_SIDE:                      # a central crop is enough to find the peak
        x0 = (w - MAX_FFT_SIDE) // 2
        gray, w = gray[:, x0:x0 + MAX_FFT_SIDE], MAX_FFT_SIDE
    if h > MAX_FFT_SIDE:
        y0 = (h - MAX_FFT_SIDE) // 2
        gray, h = gray[y0:y0 + MAX_FFT_SIDE], MAX_FFT_SIDE
    f = gray.astype(np.float32)
    f = f - f.mean()
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    dft = cv2.dft(f * win, flags=cv2.DFT_COMPLEX_OUTPUT)
    spec = np.fft.fftshift(dft[..., 0] ** 2 + dft[..., 1] ** 2)
    spec = cv2.GaussianBlur(spec.astype(np.float32), (3, 3), 0)

    fy = (np.arange(h) - h // 2) / float(h)       # cycles / px
    fx = (np.arange(w) - w // 2) / float(w)
    FX, FY = np.meshgrid(fx, fy)
    rad = np.hypot(FX, FY)
    annulus = (rad >= 1.0 / max_wl) & (rad <= 1.0 / min_wl)
    if not annulus.any():
        return WaveEstimate((min_wl + max_wl) / 2.0, 0.0, 0.0)
    masked = np.where(annulus, spec, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    peak_f = rad[iy, ix]
    wavelength = float(np.clip(1.0 / max(peak_f, 1e-6), min_wl, max_wl))
    theta = float(math.atan2(FY[iy, ix], FX[iy, ix]) % math.pi)
    strength = float(spec[iy, ix] / (spec[annulus].mean() + 1e-12))
    return WaveEstimate(wavelength, theta, strength)


def estimate_waves(gray: np.ndarray, cfg: Config) -> list[WaveEstimate]:
    """One estimate per horizontal band (near field at the bottom, far field at the top)."""
    h = gray.shape[0]
    bands = max(1, int(cfg.wave_bands))
    edges = np.linspace(0, h, bands + 1).astype(int)
    out = []
    for i in range(bands):
        part = gray[edges[i]:edges[i + 1]]
        if part.shape[0] < 16 or part.shape[1] < 16:
            continue
        out.append(estimate_wave(part, cfg.min_wavelength, cfg.max_wavelength))
    if not out:
        out.append(estimate_wave(gray, cfg.min_wavelength, cfg.max_wavelength))
    return out


# --------------------------------------------------------------- Gabor bank

@dataclass
class GaborFilter:
    wavelength: float      # px at full processing resolution
    theta: float
    even: np.ndarray       # kernels at pyramid level ``level``
    odd: np.ndarray
    level: int = 0         # the filter runs on the image downsampled by 2**level


def pyramid_level(wavelength: float, stride: int, max_level: int = 2) -> int:
    """Coarsest level at which the wave is still sampled with >= 4 px per cycle
    and the cell grid stays integral. Energy is averaged over whole tiles, so
    filtering a long wave on a downsampled image loses nothing and is up to
    16x cheaper."""
    level = 0
    while (level < max_level and wavelength / 2 ** (level + 1) >= MIN_PYRAMID_WAVELENGTH
           and stride % 2 ** (level + 1) == 0):
        level += 1
    return level


def _merge_wavelengths(wls: list[float], tol: float = 0.25) -> list[float]:
    merged: list[float] = []
    for wl in sorted(wls):
        if merged and wl <= merged[-1] * (1.0 + tol):
            merged[-1] = (merged[-1] + wl) / 2.0
        else:
            merged.append(wl)
    return merged


def build_gabor_bank(waves: list[WaveEstimate], cfg: Config) -> list[GaborFilter]:
    """Filters at the estimated wavelengths and ``n_orientations`` evenly spaced
    orientations, the first aligned with the dominant (strongest) wave direction."""
    dominant = max(waves, key=lambda w: w.strength)
    wavelengths = _merge_wavelengths([w.wavelength for w in waves])
    bank: list[GaborFilter] = []
    n = max(1, int(cfg.n_orientations))
    for wl in wavelengths:
        level = pyramid_level(wl, cfg.stride)
        wl_l = wl / 2 ** level
        sigma = cfg.gabor_sigma_ratio * wl_l
        half = int(math.ceil(3.0 * sigma))
        ksize = (2 * half + 1, 2 * half + 1)
        for k in range(n):
            theta = (dominant.theta + k * math.pi / n) % math.pi
            even = cv2.getGaborKernel(ksize, sigma, theta, wl_l, cfg.gabor_gamma, 0.0, ktype=cv2.CV_32F)
            odd = cv2.getGaborKernel(ksize, sigma, theta, wl_l, cfg.gabor_gamma, math.pi / 2, ktype=cv2.CV_32F)
            even -= even.mean()                    # zero DC: insensitive to absolute brightness
            norm = float(np.abs(even).sum()) + 1e-9
            bank.append(GaborFilter(wl, theta, even / norm, odd / norm, level))
    return bank


def gabor_energy(gray: np.ndarray, filt: GaborFilter) -> np.ndarray:
    """Local energy sqrt(even^2 + odd^2): phase-invariant response magnitude."""
    f = gray.astype(np.float32)
    e = cv2.filter2D(f, cv2.CV_32F, filt.even, borderType=cv2.BORDER_REFLECT)
    o = cv2.filter2D(f, cv2.CV_32F, filt.odd, borderType=cv2.BORDER_REFLECT)
    return cv2.magnitude(e, o)


# ---------------------------------------------------------- tile aggregation

def cell_sums(arr: np.ndarray, stride: int) -> np.ndarray:
    """Sum ``arr`` (H, W[, C]) over non-overlapping stride x stride cells.
    H and W must be multiples of ``stride``."""
    h, w = arr.shape[:2]
    c = arr.shape[2] if arr.ndim == 3 else 1
    a = arr.reshape(h // stride, stride, w // stride, stride, c)
    return a.sum(axis=(1, 3), dtype=np.float64)


def tile_sums(cells: np.ndarray, m: int) -> np.ndarray:
    """Sum of every m x m block of cells (a box filter on the cell grid, valid region)."""
    cy, cx, c = cells.shape
    S = np.zeros((cy + 1, cx + 1, c), dtype=np.float64)
    S[1:, 1:] = cells.cumsum(0).cumsum(1)
    return S[m:, m:] - S[:-m, m:] - S[m:, :-m] + S[:-m, :-m]


def lbp_cell_hist(codes: np.ndarray, stride: int) -> np.ndarray:
    h, w = codes.shape
    cy, cx = h // stride, w // stride
    cell_id = (np.arange(h)[:, None] // stride) * cx + (np.arange(w)[None, :] // stride)
    idx = cell_id.ravel() * N_LBP_BINS + codes.ravel().astype(np.int64)
    counts = np.bincount(idx, minlength=cy * cx * N_LBP_BINS)
    return counts.reshape(cy, cx, N_LBP_BINS).astype(np.float64)


# ------------------------------------------------------------------- driver

@dataclass
class TextureFeatures:
    gabor: np.ndarray        # (ny, nx, F) mean Gabor energy per tile
    lbp: np.ndarray          # (ny, nx, 59) normalised LBP histogram per tile
    waves: list              # WaveEstimate per band
    bank: list               # GaborFilter list
    energy_map: np.ndarray   # (H, W) summed Gabor energy, for display
    lbp_map: np.ndarray      # (H, W) uniform LBP codes, for display
    pad_shape: tuple         # padded (H, W) the tile grid refers to
    shape: tuple = (0, 0)    # original (H, W)
    gabor_margin: int = 0    # px from the frame edge within which Gabor responses see padding


def pad_to_grid(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Reflect-pad so that both sides are multiples of ``stride`` and at least ``tile``."""
    h, w = img.shape[:2]
    s = cfg.stride
    H = max(cfg.tile, int(math.ceil(h / s)) * s)
    W = max(cfg.tile, int(math.ceil(w / s)) * s)
    if (H, W) == (h, w):
        return img
    return cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_REFLECT)


def compute_texture(gray: np.ndarray, cfg: Config, keep_maps: bool = True) -> TextureFeatures:
    h, w = gray.shape
    waves = estimate_waves(gray, cfg)
    bank = build_gabor_bank(waves, cfg)
    g = pad_to_grid(gray, cfg)
    H, W = g.shape
    s, m = cfg.stride, cfg.tile // cfg.stride

    pyramid = {0: g.astype(np.float32)}
    for filt in bank:
        lv = filt.level
        if lv not in pyramid:
            pyramid[lv] = cv2.resize(pyramid[0], (W >> lv, H >> lv), interpolation=cv2.INTER_AREA)

    gabor_tiles = []
    energy_total = np.zeros((H, W), dtype=np.float32) if keep_maps else None
    for filt in bank:
        lv = filt.level
        e = gabor_energy(pyramid[lv], filt)
        s_l = s >> lv
        gabor_tiles.append(tile_sums(cell_sums(e, s_l), m)[..., 0] / float((m * s_l) ** 2))
        if keep_maps:
            energy_total += e if lv == 0 else cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR)
    gabor = np.stack(gabor_tiles, axis=-1)

    codes = uniform_lbp(g)
    lbp = tile_sums(lbp_cell_hist(codes, s), m)
    lbp /= lbp.sum(axis=-1, keepdims=True)

    margin = int(math.ceil(2.0 * cfg.gabor_sigma_ratio * max(f.wavelength for f in bank)))
    return TextureFeatures(gabor=gabor, lbp=lbp, waves=waves, bank=bank,
                           energy_map=energy_total[:h, :w] if keep_maps else None,
                           lbp_map=codes[:h, :w], pad_shape=g.shape, shape=(h, w), gabor_margin=margin)
