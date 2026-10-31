"""Stage 4: sun-glint suppression by chromaticity.

Specular reflection of the sun saturates all three channels together, so glint
is bright and nearly achromatic: its normalised chromaticity
(r, g, b) = (R, G, B) / (R + G + B) sits close to (1/3, 1/3, 1/3). Skin,
clothing and flotation equipment keep a clear colour cast. A pixel is called
glint when it is both very bright and close to grey; candidate components made
mostly of glint are dropped, and glint pixels are cleared from the rest.
Requiring high brightness as well as low chroma keeps dark, colourless targets
(a person in a black wetsuit) from being removed.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def chromaticity(bgr: np.ndarray) -> np.ndarray:
    """Normalised (r, g, b) per pixel, shape (H, W, 3), order R, G, B."""
    f = bgr.astype(np.float32)
    s = f.sum(axis=-1, keepdims=True) + 1e-6
    return (f / s)[..., ::-1]


def chroma_distance(bgr: np.ndarray) -> np.ndarray:
    """Euclidean distance of the chromaticity from the achromatic point."""
    c = chromaticity(bgr)
    return np.sqrt(((c - 1.0 / 3.0) ** 2).sum(axis=-1))


def glint_mask(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean mask of bright, achromatic pixels."""
    total = bgr.astype(np.uint16).sum(axis=-1)
    bright = total >= 3.0 * 255.0 * cfg.glint_value
    out = np.zeros(bright.shape, dtype=bool)
    if bright.any():
        px = bgr[bright]
        out[bright] = chroma_distance(px[None])[0] <= cfg.glint_chroma
    return out


def suppress_glint(candidates: np.ndarray, bgr: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return (filtered candidate mask uint8 0/255, glint mask bool)."""
    glint = glint_mask(bgr, cfg)
    if not cfg.glint_enabled:
        return candidates.copy(), glint
    out = candidates.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((candidates > 0).astype(np.uint8), connectivity=8)
    if n > 1:
        glint_counts = np.bincount(labels.ravel(), weights=glint.ravel().astype(np.float64), minlength=n)
        frac = glint_counts / np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
        drop = frac > cfg.glint_component_frac
        drop[0] = False
        out[drop[labels]] = 0
    # clear remaining glint pixels and their one-pixel halo
    halo = cv2.dilate(glint.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out[halo] = 0
    return out, glint
