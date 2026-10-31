"""Evaluation metrics: recall@k, false alarms per frame, stratification.

Hit rule. IoU is very harsh on 10-30 px boxes (a 3 px offset on a 12 px box
already drops IoU below 0.5), so the default is a centre-distance rule: a
predicted box hits a ground-truth person when their centres are within
``max(min_radius, centre_factor * max(gt_w, gt_h))`` pixels. ``criterion="iou"``
switches to the usual IoU >= ``iou_thr`` rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SIZE_BINS = [(0, 16, "<16 px"), (16, 32, "16-32 px"), (32, 64, "32-64 px"), (64, math.inf, ">=64 px")]
ALT_BINS = [(0, 60, "<60 m"), (60, 100, "60-100 m"), (100, 140, "100-140 m"), (140, math.inf, ">=140 m")]


def iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class HitRule:
    criterion: str = "centre"     # "centre" or "iou"
    iou_thr: float = 0.5
    centre_factor: float = 1.0
    min_radius: float = 8.0

    def hit(self, pred, gt) -> bool:
        if self.criterion == "iou":
            return iou(pred, gt) >= self.iou_thr
        px, py = pred[0] + pred[2] / 2, pred[1] + pred[3] / 2
        gx, gy = gt[0] + gt[2] / 2, gt[1] + gt[3] / 2
        radius = max(self.min_radius, self.centre_factor * max(gt[2], gt[3]))
        return math.hypot(px - gx, py - gy) <= radius


def _bin(value, bins) -> str | None:
    if value is None:
        return None
    for lo, hi, name in bins:
        if lo <= value < hi:
            return name
    return None


@dataclass
class Accumulator:
    """Collects ranked predictions (up to ``kmax``) per frame against ground truth."""
    kmax: int = 10
    rule: HitRule = field(default_factory=HitRule)
    gts: list = field(default_factory=list)       # per GT: (size, altitude, first_hit_rank or None, warmup)
    frames: list = field(default_factory=list)    # per frame: (fa_by_k, other_by_k, n_gt, n_pred, warmup)
    seconds: float = 0.0

    def add(self, preds: list, persons: list, others: list | None = None,
            altitude: float | None = None, warmup: bool = False) -> None:
        """``preds``: ranked [(x, y, w, h)], best first. ``warmup`` marks frames
        before the temporal stage can have promoted anything; they count in the
        overall numbers and are left out of the steady-state ones."""
        preds = preds[:self.kmax]
        others = others or []
        first_rank = [None] * len(persons)
        pred_is_person = [False] * len(preds)
        pred_is_other = [False] * len(preds)
        for r, p in enumerate(preds):
            for gi, g in enumerate(persons):
                if self.rule.hit(p, g):
                    pred_is_person[r] = True
                    if first_rank[gi] is None:
                        first_rank[gi] = r + 1
            if not pred_is_person[r]:
                pred_is_other[r] = any(self.rule.hit(p, o) for o in others)
        fa_by_k, other_by_k = [], []
        fa = oth = 0
        for r in range(self.kmax):
            if r < len(preds):
                fa += 0 if pred_is_person[r] else 1
                oth += 1 if pred_is_other[r] else 0
            fa_by_k.append(fa)
            other_by_k.append(oth)
        self.frames.append((fa_by_k, other_by_k, len(persons), len(preds), warmup))
        for g, fr in zip(persons, first_rank):
            self.gts.append((math.sqrt(g[2] * g[3]), altitude, fr, warmup))

    # ------------------------------------------------------------------ summary
    def recall_at(self, k: int, gts=None) -> float | None:
        gts = self.gts if gts is None else gts
        if not gts:
            return None
        return sum(1 for g in gts if g[2] is not None and g[2] <= k) / len(gts)

    def fa_per_frame(self, k: int, exclude_other_objects: bool = False, frames=None) -> float | None:
        frames = self.frames if frames is None else frames
        if not frames:
            return None
        i = min(k, self.kmax) - 1
        tot = sum(f[0][i] - (f[1][i] if exclude_other_objects else 0) for f in frames)
        return tot / len(frames)

    def precision_at(self, k: int) -> float | None:
        """Fraction of shown boxes (top-k, all frames) that hit a person."""
        i = min(k, self.kmax) - 1
        shown = sum(min(k, f[3]) for f in self.frames)
        if shown == 0:
            return None
        fa = sum(f[0][i] for f in self.frames)
        return (shown - fa) / shown

    def summary(self, k: int) -> dict:
        k = min(k, self.kmax)
        out = {
            "frames": len(self.frames),
            "persons": len(self.gts),
            f"recall@{k}": self.recall_at(k),
            "fa_per_frame": self.fa_per_frame(k),
            "fa_per_frame_excl_other_objects": self.fa_per_frame(k, True),
            f"precision@{k}": self.precision_at(k),
            "warmup_frames": sum(1 for f in self.frames if f[4]),
            f"steady_recall@{k}": self.recall_at(k, [g for g in self.gts if not g[3]]),
            "steady_fa_per_frame": self.fa_per_frame(k, frames=[f for f in self.frames if not f[4]]),
            "recall_curve": {kk: self.recall_at(kk) for kk in range(1, self.kmax + 1)},
            "fa_curve": {kk: self.fa_per_frame(kk) for kk in range(1, self.kmax + 1)},
            "by_size": {},
            "by_altitude": {},
            "fps": (len(self.frames) / self.seconds) if self.seconds > 0 else None,
        }
        for _, _, name in SIZE_BINS:
            sub = [g for g in self.gts if _bin(g[0], SIZE_BINS) == name]
            if sub:
                out["by_size"][name] = {"n": len(sub), f"recall@{k}": self.recall_at(k, sub)}
        for _, _, name in ALT_BINS:
            sub = [g for g in self.gts if _bin(g[1], ALT_BINS) == name]
            if sub:
                out["by_altitude"][name] = {"n": len(sub), f"recall@{k}": self.recall_at(k, sub)}
        return out
