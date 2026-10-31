"""Stage 5: morphological refinement and shape filtering.

Opening removes isolated specks, closing joins the fragments of a partly
submerged body. Connected-component analysis then measures each blob, and blobs
whose geometry does not fit a floating person are dropped:

* area outside the expected range (noise, or far too large for a person),
* eccentricity close to 1 (long, thin foam streaks),
* low solidity (ragged, scattered foam).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config


@dataclass
class Blob:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float
    eccentricity: float
    solidity: float
    label: int = 0
    rejected: str = ""        # reason, empty if kept


def eccentricity(mask: np.ndarray) -> float:
    """Eccentricity of the ellipse with the same second moments (0 circle, ->1 line)."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return 0.0
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    common = np.sqrt(4.0 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1 = (mu20 + mu02 + common) / 2.0
    l2 = (mu20 + mu02 - common) / 2.0
    if l1 <= 1e-12:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - max(l2, 0.0) / l1)))


def solidity(mask: np.ndarray, area: int) -> float:
    """Pixel area over convex-hull area (hull taken around pixel squares)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    # use all four corners of every pixel so that small blobs get a fair hull
    pts = np.concatenate([
        np.stack([xs, ys], 1), np.stack([xs + 1, ys], 1),
        np.stack([xs, ys + 1], 1), np.stack([xs + 1, ys + 1], 1),
    ]).astype(np.int32)
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 1.0
    return float(min(1.0, area / hull_area))


def refine(mask: np.ndarray, cfg: Config, area_limits: tuple[int, int]) -> tuple[np.ndarray, list[Blob]]:
    """Return (cleaned mask uint8 0/255 with only accepted blobs, all blobs incl. rejected)."""
    work = (mask > 0).astype(np.uint8)
    if cfg.morph_enabled:
        if cfg.open_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_ksize, cfg.open_ksize))
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, k)
        if cfg.close_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_ksize, cfg.close_ksize))
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(work, connectivity=8)
    lo, hi = area_limits
    keep = np.zeros(n, dtype=bool)
    blobs: list[Blob] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        sub = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        ecc = eccentricity(sub)
        sol = solidity(sub, area)
        b = Blob(x, y, w, h, area, float(cents[i][0]), float(cents[i][1]), ecc, sol, i)
        if cfg.morph_enabled:
            if area < lo:
                b.rejected = "area<min"
            elif area > hi:
                b.rejected = "area>max"
            elif ecc > cfg.max_eccentricity:
                b.rejected = "eccentricity"
            elif sol < cfg.min_solidity:
                b.rejected = "solidity"
        elif area < 2:
            b.rejected = "area<min"
        keep[i] = not b.rejected
        blobs.append(b)
    cleaned = (keep[labels].astype(np.uint8) * 255)
    return cleaned, blobs
