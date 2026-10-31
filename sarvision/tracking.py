"""Stage 6: temporal association.

Whitecaps and glint flashes decay within a second or two; a person stays and
drifts slowly and smoothly with the current. Candidates are linked across
frames by proximity (greedy nearest neighbour around a constant-velocity
prediction). A track is promoted once it has been seen ``min_persistence``
times and its frame-to-frame velocity is consistent ("coherent drift"). Short
gaps (a person hidden by a wave) are tolerated for ``max_gap`` frames.

The shortlist is ranked by a weighted sum of the anomaly confidence and a
persistence term (hits, saturating at ``persistence_horizon``, times coherence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .config import Config

HISTORY_LEN = 30


@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float          # raw anomaly score
    conf: float           # anomaly confidence in [0, 1]
    area: int = 0
    eccentricity: float = 0.0
    solidity: float = 0.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class Track:
    tid: int
    det: Detection                      # most recent detection (box moved with camera shift)
    history: list = field(default_factory=list)   # [(frame, cx, cy)]
    hits: int = 1
    last_frame: int = 0
    conf_ema: float = 0.0

    def velocity(self) -> np.ndarray:
        v = self.step_velocities()
        return v.mean(axis=0) if len(v) else np.zeros(2)

    def step_velocities(self) -> np.ndarray:
        h = self.history
        out = []
        for (f0, x0, y0), (f1, x1, y1) in zip(h[:-1], h[1:]):
            dt = max(1, f1 - f0)
            out.append(((x1 - x0) / dt, (y1 - y0) / dt))
        return np.array(out, dtype=np.float64).reshape(-1, 2)

    def coherence(self, cfg: Config) -> float:
        v = self.step_velocities()
        if len(v) < 2:
            return 1.0
        spread = float(np.sqrt(((v - v.mean(axis=0)) ** 2).sum(axis=1).mean()))
        return math.exp(-spread / max(cfg.drift_sigma, 1e-6))

    def predicted(self, frame: int) -> tuple[float, float]:
        if not self.history:
            return self.det.cx, self.det.cy
        f, x, y = self.history[-1]
        v = self.velocity()
        dt = frame - f
        return x + v[0] * dt, y + v[1] * dt


@dataclass
class RankedCandidate:
    det: Detection
    conf: float
    track_id: int = -1
    hits: int = 1
    coherence: float = 1.0
    missed: int = 0


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.tracks: list[Track] = []
        self._next_id = 1

    def _shift(self, dx: float, dy: float) -> None:
        if dx == 0 and dy == 0:
            return
        for t in self.tracks:
            t.history = [(f, x + dx, y + dy) for f, x, y in t.history]
            t.det = replace(t.det, x=t.det.x + dx, y=t.det.y + dy)

    def update(self, frame: int, dets: list[Detection], shift: tuple[float, float] = (0.0, 0.0)) -> None:
        cfg = self.cfg
        self._shift(*shift)
        pairs = []
        for ti, t in enumerate(self.tracks):
            px, py = t.predicted(frame)
            for di, d in enumerate(dets):
                dist = math.hypot(d.cx - px, d.cy - py)
                if dist <= cfg.match_radius:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            t.det = d
            t.history.append((frame, d.cx, d.cy))
            t.history = t.history[-HISTORY_LEN:]
            t.hits += 1
            t.last_frame = frame
            t.conf_ema = 0.5 * t.conf_ema + 0.5 * d.conf
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            self.tracks.append(Track(self._next_id, d, [(frame, d.cx, d.cy)], 1, frame, d.conf))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if frame - t.last_frame <= cfg.max_gap]

    def ranked(self, frame: int) -> list[RankedCandidate]:
        """Promoted tracks ranked by the combined anomaly/persistence confidence."""
        cfg = self.cfg
        out = []
        for t in self.tracks:
            coh = t.coherence(cfg)
            if t.hits < cfg.min_persistence or coh < cfg.min_coherence:
                continue
            missed = frame - t.last_frame
            persist = min(t.hits, cfg.persistence_horizon) / float(cfg.persistence_horizon) * coh
            conf = cfg.anomaly_weight * t.conf_ema + (1.0 - cfg.anomaly_weight) * persist
            conf *= 0.8 ** missed
            out.append(RankedCandidate(t.det, float(conf), t.tid, t.hits, coh, missed))
        out.sort(key=lambda c: -c.conf)
        return out
