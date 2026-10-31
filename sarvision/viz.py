"""Rendering of intermediate stages and of the operator overlay."""

from __future__ import annotations

import cv2
import numpy as np

from .pipeline import FrameResult

STAGE_NAMES = [
    "final",
    "input",
    "normalised",
    "gabor_energy",
    "lbp",
    "anomaly_map",
    "candidates",
    "glint",
    "after_glint",
    "morphology",
]

# rank colours (BGR): rank 1 red, then orange, yellow, cyan, magenta ...
RANK_COLOURS = [(0, 0, 255), (0, 140, 255), (0, 230, 255), (255, 255, 0), (255, 0, 255),
                (0, 255, 0), (255, 128, 0), (128, 0, 255), (200, 200, 200), (0, 128, 128)]


def _colormap(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    n = np.clip((x.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def _mask_overlay(base: np.ndarray, mask: np.ndarray, colour: tuple) -> np.ndarray:
    out = (base.astype(np.float32) * 0.45).astype(np.uint8)
    out[mask > 0] = colour
    return out


def render_stage(name: str, res: FrameResult, original: np.ndarray, cfg, frame_no: int | None = None) -> np.ndarray:
    """BGR uint8 image for ``name``; stage images are at processing resolution."""
    s = res.stages
    if name == "final" or not s:
        return draw_overlay(original, res, frame_no=frame_no)
    base = s["input"]
    if name == "input":
        return base.copy()
    if name == "normalised":
        return s["normalised"].copy()
    if name == "gabor_energy":
        e = s["gabor_energy"]
        return _colormap(e, 0, float(np.percentile(e, 99.5)))
    if name == "lbp":
        return cv2.cvtColor((s["lbp"].astype(np.float32) * (255.0 / 58.0)).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if name == "anomaly_map":
        img = _colormap(s["anomaly_map"], 0, 2.0 * cfg.anomaly_z)
        contours, _ = cv2.findContours(s["anomaly_region"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
        return img
    if name == "candidates":
        return _mask_overlay(base, s["candidates"], (0, 255, 255))
    if name == "glint":
        out = _mask_overlay(base, s["glint"].astype(np.uint8), (255, 255, 255))
        out[(s["candidates"] > 0) & s["glint"]] = (0, 0, 255)
        return out
    if name == "after_glint":
        return _mask_overlay(base, s["after_glint"], (0, 255, 255))
    if name == "morphology":
        out = _mask_overlay(base, s["morphology"], (0, 255, 0))
        for b in res.info.get("blobs", []):
            colour = (0, 255, 0) if not b.rejected else (0, 0, 200)
            cv2.rectangle(out, (b.x - 1, b.y - 1), (b.x + b.w, b.y + b.h), colour, 1)
        return out
    raise KeyError(name)


def draw_overlay(frame: np.ndarray, res: FrameResult, min_box: int = 24,
                 gt_boxes: list | None = None, frame_no: int | None = None) -> np.ndarray:
    """Draw the ranked shortlist on the original frame.

    Tiny boxes are padded to ``min_box`` pixels so the operator can see them.
    Optional ground-truth boxes (x, y, w, h) are drawn as thin green rectangles.
    ``frame_no`` is the 1-based source frame number shown in the HUD (default:
    the pipeline's own frame counter + 1).
    """
    out = frame.copy()
    th = max(1, round(frame.shape[1] / 900))
    if gt_boxes:
        for (x, y, w, h) in gt_boxes:
            cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0), th)
    for rank, c in enumerate(res.shortlist, 1):
        d = c.det
        cx, cy = d.cx, d.cy
        bw, bh = max(d.w, min_box), max(d.h, min_box)
        p1 = (int(round(cx - bw / 2)), int(round(cy - bh / 2)))
        p2 = (int(round(cx + bw / 2)), int(round(cy + bh / 2)))
        colour = RANK_COLOURS[(rank - 1) % len(RANK_COLOURS)]
        cv2.rectangle(out, p1, p2, colour, th + 1)
        label = f"#{rank} {c.conf:.2f}"
        fs = 0.4 * th + 0.1
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = p1[1] - 4 if p1[1] - tht - 6 > 0 else p2[1] + tht + 4
        cv2.rectangle(out, (p1[0], ty - tht - 2), (p1[0] + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (p1[0] + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, colour, th, cv2.LINE_AA)
    hud = f"frame {res.index + 1 if frame_no is None else frame_no}  shortlist {len(res.shortlist)}  {1.0 / max(res.timings.get('total', 1e-3), 1e-3):.1f} fps"
    (tw, tht), base = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, th)
    cv2.rectangle(out, (4, 22 * th - tht - 4), (12 + tw, 22 * th + base + 2), (0, 0, 0), -1)
    cv2.putText(out, hud, (8, 22 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, (255, 255, 255), th, cv2.LINE_AA)
    return out
