"""The six-stage pipeline: one frame in, a ranked shortlist of candidate boxes out."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .anomaly import score_anomalies
from .config import Config
from .glint import suppress_glint
from .morphology import refine
from .normalise import normalise
from .texture import compute_texture
from .tracking import Detection, RankedCandidate, Tracker

MAX_DETECTIONS_PER_FRAME = 50


@dataclass
class FrameResult:
    index: int
    scale: float                                  # processing size / original size
    shortlist: list[RankedCandidate]              # top-k, boxes in ORIGINAL pixel coordinates
    detections: list[Detection]                   # all per-frame detections, ORIGINAL coordinates
    timings: dict = field(default_factory=dict)   # seconds per stage
    stages: dict = field(default_factory=dict)    # intermediate images (processing resolution)
    info: dict = field(default_factory=dict)      # wave estimates, thresholds, blob table ...


def estimate_shift(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """Camera translation from ``prev`` to ``cur`` by phase correlation (px).

    A point at p in ``prev`` appears at p + shift in ``cur``.
    """
    a = cv2.resize(prev, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(cur, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    if response < 0.05:
        return 0.0, 0.0
    return 2.0 * dx, 2.0 * dy


def _to_original(d: Detection, inv: float) -> Detection:
    return replace(d, x=d.x * inv, y=d.y * inv, w=d.w * inv, h=d.h * inv)


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = (cfg or Config()).validate()
        self.tracker = Tracker(self.cfg)
        self.reset()

    def reset(self) -> None:
        self.tracker = Tracker(self.cfg)
        self.frame_idx = -1
        self._prev_gray = None

    def set_config(self, cfg: Config) -> None:
        """Swap parameters; temporal state is reset because tracks depend on them."""
        self.cfg = cfg.validate()
        self.reset()

    # ------------------------------------------------------------------ core
    def detect(self, bgr: np.ndarray, keep_stages: bool = False) -> tuple[list[Detection], dict, dict, dict]:
        """Stages 1-5 on one processing-resolution frame. Returns
        (detections, stages, info, timings) in processing coordinates."""
        cfg = self.cfg.resolved(bgr.shape[1])
        t = {}
        t0 = time.perf_counter()
        bgr_n, gray_n = normalise(bgr, cfg)
        t["normalise"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        feat = compute_texture(gray_n, cfg, keep_maps=keep_stages)
        t["texture"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        an = score_anomalies(feat, bgr, cfg)
        t["anomaly"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        after_glint, glint = suppress_glint(an.candidates, bgr, cfg)
        t["glint"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        limits = cfg.area_limits(bgr.shape[1])
        cleaned, blobs = refine(after_glint, cfg, limits)
        dets = []
        for b in blobs:
            if b.rejected:
                continue
            sl = (slice(b.y, b.y + b.h), slice(b.x, b.x + b.w))
            inside = cleaned[sl] > 0
            if not inside.any():
                continue
            z_peak = float(an.zmap[sl][inside].max())
            contrast = float(an.deviation[sl][inside].mean() / max(an.pixel_thresh, 1e-6))
            score = max(z_peak, 0.0) * contrast
            conf = score / (score + cfg.score_scale)
            dets.append(Detection(b.x, b.y, b.w, b.h, score, conf, b.area, b.eccentricity, b.solidity))
        dets.sort(key=lambda d: -d.conf)
        dets = dets[:MAX_DETECTIONS_PER_FRAME]
        t["morphology"] = time.perf_counter() - t0

        info = {
            "waves": [(w.wavelength, math.degrees(w.theta), w.strength) for w in feat.waves],
            "n_filters": len(feat.bank),
            "tile": (cfg.tile, cfg.stride),
            "pixel_thresh": an.pixel_thresh,
            "area_limits": limits,
            "n_anomalous_tiles": int((an.tile_z > cfg.anomaly_z).sum()),
            "n_tiles": int(an.tile_z.size),
            "blobs": blobs,
        }
        stages = {}
        if keep_stages:
            stages = {
                "normalised": bgr_n,
                "gabor_energy": feat.energy_map,
                "lbp": feat.lbp_map,
                "anomaly_map": an.zmap,
                "anomaly_region": an.region,
                "deviation": an.deviation,
                "candidates": an.candidates,
                "glint": glint,
                "after_glint": after_glint,
                "morphology": cleaned,
            }
        return dets, stages, info, t

    def process(self, frame: np.ndarray, keep_stages: bool = False) -> FrameResult:
        cfg = self.cfg
        self.frame_idx += 1
        h, w = frame.shape[:2]
        scale = min(1.0, cfg.max_width / float(w))
        bgr = frame if scale == 1.0 else cv2.resize(frame, (round(w * scale), round(h * scale)),
                                                    interpolation=cv2.INTER_AREA)
        dets, stages, info, timings = self.detect(bgr, keep_stages)

        t0 = time.perf_counter()
        shift = (0.0, 0.0)
        if cfg.temporal_enabled and cfg.motion_compensation:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
                shift = estimate_shift(self._prev_gray, gray)
            self._prev_gray = gray
        info["camera_shift"] = shift

        if cfg.temporal_enabled:
            self.tracker.update(self.frame_idx, dets, shift)
            ranked = self.tracker.ranked(self.frame_idx)
        else:
            ranked = [RankedCandidate(d, d.conf) for d in dets]
        timings["temporal"] = time.perf_counter() - t0
        timings["total"] = sum(timings.values())

        inv = 1.0 / scale
        shortlist = [replace(c, det=_to_original(c.det, inv)) for c in ranked[:cfg.top_k]]
        detections = [_to_original(d, inv) for d in dets]
        if keep_stages:
            stages["input"] = bgr
        return FrameResult(self.frame_idx, scale, shortlist, detections, timings, stages, info)
