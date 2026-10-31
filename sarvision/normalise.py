"""Stage 1: illumination normalisation.

A wide-angle aerial frame is brighter toward the sun and hazier in the far
field. Classical intensity operations flatten this so that texture statistics
in different parts of the frame are comparable:

0. Flat-fielding: the lightness channel is divided by a heavily smoothed copy
   of itself (the illumination estimate), which removes the smooth brightness
   gradient and equalises wave contrast between bright and dark regions.
1. Contrast-limited adaptive histogram equalisation (CLAHE) on the lightness
   channel, which equalises each region of the frame with its own histogram.
2. A piecewise-linear contrast stretch whose knees are placed at low/high
   percentiles of the equalised image.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def piecewise_linear(img: np.ndarray, r1: float, s1: float, r2: float, s2: float) -> np.ndarray:
    """Three-segment piecewise-linear intensity transform on a uint8 image.

    (0,0) -> (r1,s1) -> (r2,s2) -> (255,255). Implemented as a lookup table.
    """
    r1 = float(np.clip(r1, 0, 254))
    r2 = float(np.clip(r2, r1 + 1, 255))
    s1 = float(np.clip(s1, 0, 255))
    s2 = float(np.clip(s2, s1, 255))
    x = np.arange(256, dtype=np.float32)
    lut = np.interp(x, [0.0, r1, r2, 255.0], [0.0, s1, s2, 255.0])
    return cv2.LUT(img, np.clip(lut, 0, 255).astype(np.uint8))


def contrast_stretch(gray: np.ndarray, cfg: Config) -> np.ndarray:
    lo, hi = np.percentile(gray, [cfg.stretch_low_pct, cfg.stretch_high_pct])
    if hi - lo < 1:
        return gray.copy()
    return piecewise_linear(gray, lo, cfg.stretch_out_low, hi, cfg.stretch_out_high)


def flat_field(gray: np.ndarray, fraction: float) -> np.ndarray:
    """Divide by a Gaussian-smoothed illumination estimate (sigma = fraction * width).

    The blur is computed on an 8x reduced copy, which is equivalent for such a
    wide kernel and much cheaper.
    """
    if fraction <= 0:
        return gray
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA).astype(np.float32)
    sigma = max(1.0, fraction * small.shape[1])
    illum = cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma, borderType=cv2.BORDER_REFLECT),
                       (w, h), interpolation=cv2.INTER_LINEAR)
    flat = gray.astype(np.float32) * (float(illum.mean()) / np.maximum(illum, 1.0))
    return np.clip(flat, 0, 255).astype(np.uint8)


def normalise(bgr: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (normalised BGR frame, normalised grey frame), both uint8."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_flat = flat_field(l_chan, cfg.flatten_sigma)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    l_eq = clahe.apply(l_flat)
    l_st = contrast_stretch(l_eq, cfg)
    bgr_n = cv2.cvtColor(cv2.merge([l_st, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    return bgr_n, l_st
