"""Synthetic aerial sea sequences with ground truth.

Used for self-checks, unit tests and the demo when the public datasets are not
on disk. Each sequence renders an animated directional wave field seen from an
oblique camera (finer waves in the far field), an illumination gradient, sun
glitter on wave facets that face the sun, short-lived whitecaps, elongated
floating debris, and one or more people drifting with the current: a skin-tone
head and a coloured or dark torso/life jacket that bob in and out of the water.

Annotations are written in COCO format with the same fields the dataset
loaders read for SeaDronesSee / MOBDrone (``video_id``, ``frame_index`` and
``meta.altitude``), so the whole evaluation path is exercised.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PERSON_COLOURS = {                       # BGR torso / life-jacket colours
    "orange": (0, 110, 245),
    "red": (30, 30, 200),
    "yellow": (20, 210, 235),
    "dark": (45, 40, 35),
    "green": (60, 160, 60),
}
SKIN = [(120, 160, 215), (90, 125, 180), (60, 85, 130)]


@dataclass
class SceneSpec:
    width: int = 1280
    height: int = 720
    frames: int = 60
    altitude: float = 80.0            # metres; sets person size
    sea_state: float = 1.0            # wave amplitude multiplier (0.3 calm .. 2 rough)
    whitecap_rate: float = 0.6        # expected new whitecaps per frame
    glint: float = 1.0                # glitter intensity multiplier (0 = overcast)
    n_people: int = 2
    n_debris: int = 1
    pan: tuple = (0.0, 0.0)           # camera drift px/frame
    perspective: float = 0.55
    seed: int = 0
    colours: list = field(default_factory=lambda: ["orange", "red", "yellow", "dark"])


@dataclass
class _Person:
    x: float
    y: float
    vx: float
    vy: float
    head_r: float
    body: tuple
    skin: tuple
    phase: float
    hidden_until: int = -1


def person_head_radius(altitude: float, width: int) -> float:
    """Head radius in px: about 4 px at 80 m in a 1280 px wide frame."""
    return max(1.6, 4.0 * (80.0 / altitude) * (width / 1280.0))


class SeaRenderer:
    def __init__(self, spec: SceneSpec):
        self.s = spec
        self.rng = np.random.default_rng(spec.seed)
        h, w = spec.height, spec.width
        y = np.arange(h, dtype=np.float32)[:, None]
        x = np.arange(w, dtype=np.float32)[None, :]
        p = spec.perspective
        depth = 1.0 + p * (1.0 - y / h)                     # 1 near (bottom) .. 1+p far (top)
        self.Xw = (x - w / 2.0) * depth
        self.Yw = y * (1.0 + p) - p * y * y / h
        self.Xw = np.broadcast_to(self.Xw, (h, w)).astype(np.float32)
        self.Yw = np.broadcast_to(self.Yw, (h, w)).astype(np.float32)
        self.depth = np.broadcast_to(depth, (h, w)).astype(np.float32)

        main_dir = self.rng.uniform(0, math.pi)
        self.waves = []
        for i in range(6):
            wl = self.rng.uniform(10, 22) if i < 3 else self.rng.uniform(5, 10)
            d = main_dir + self.rng.normal(0, 0.25 if i < 3 else 0.6)
            amp = (1.0 if i < 3 else 0.45) * self.rng.uniform(0.7, 1.0)
            speed = self.rng.uniform(0.15, 0.4)
            self.waves.append((2 * math.pi / wl, math.cos(d), math.sin(d), amp, speed, self.rng.uniform(0, 2 * math.pi)))

        self.base = np.array(self.rng.choice([(95, 70, 25), (105, 85, 35), (90, 80, 45)]), np.float32)
        gx = self.rng.uniform(-0.25, 0.25)
        gy = self.rng.uniform(-0.15, 0.25)
        self.illum = (1.0 + gx * (x / w - 0.5) + gy * (0.5 - y / h)).astype(np.float32)
        self.sun = np.array([self.rng.uniform(0.2, 0.8) * w, self.rng.uniform(0.0, 0.5) * h], np.float32)
        self.sun_falloff = np.exp(-(((x - self.sun[0]) / (0.35 * w)) ** 2
                                    + ((y - self.sun[1]) / (0.4 * h)) ** 2)).astype(np.float32)
        self.whitecaps: list[list] = []
        self.debris = []
        for _ in range(spec.n_debris):
            self.debris.append([self.rng.uniform(0.1, 0.9) * w, self.rng.uniform(0.1, 0.9) * h,
                                self.rng.uniform(0, 180), self.rng.uniform(14, 28), self.rng.uniform(1.5, 2.5)])
        self.people: list[_Person] = []
        for i in range(spec.n_people):
            r = person_head_radius(spec.altitude, w)
            colour = spec.colours[i % len(spec.colours)]
            ang = self.rng.uniform(0, 2 * math.pi)
            sp = self.rng.uniform(0.15, 0.6)
            self.people.append(_Person(
                self.rng.uniform(0.15, 0.85) * w, self.rng.uniform(0.2, 0.85) * h,
                sp * math.cos(ang), sp * math.sin(ang), r,
                PERSON_COLOURS[colour], SKIN[int(self.rng.integers(len(SKIN)))],
                self.rng.uniform(0, 2 * math.pi)))

    # ------------------------------------------------------------------
    def _height_and_slope(self, t: int, ox: float, oy: float):
        X = self.Xw + ox * self.depth
        Y = self.Yw + oy
        hgt = np.zeros_like(X)
        sx = np.zeros_like(X)
        sy = np.zeros_like(X)
        for k, cx, cy, amp, speed, ph in self.waves:
            arg = k * (cx * X + cy * Y) - speed * t + ph
            a = amp * self.s.sea_state
            hgt += a * np.sin(arg)
            c = a * k * np.cos(arg)
            sx += c * cx
            sy += c * cy
        return hgt, sx / self.depth, sy / self.depth

    def render(self, t: int) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
        s, rng = self.s, self.rng
        h, w = s.height, s.width
        ox, oy = s.pan[0] * t, s.pan[1] * t
        _, sx, sy = self._height_and_slope(t, ox, oy)

        shade = 0.82 + 0.35 * np.tanh(0.9 * (0.6 * sx - 0.8 * sy))
        img = self.base[None, None, :] * shade[..., None] * self.illum[..., None]
        img += rng.normal(0, 3.0, (h, w, 1)).astype(np.float32)

        # sun glitter: facets tilted towards the sun, concentrated around the sun's reflection
        if s.glint > 0:
            facet = (0.7 * sx - 0.7 * sy)
            spec = np.clip((facet - 0.9) * 4.0, 0, 1) * self.sun_falloff * s.glint
            spec *= (rng.random((h, w)) < 0.35)
            np.clip(spec, 0, 1, out=spec)
            img = img * (1 - spec[..., None]) + 252.0 * spec[..., None]

        # whitecaps: spawn, live a few frames, fade (drawn in a local patch)
        for _ in range(rng.poisson(s.whitecap_rate * s.sea_state)):
            self.whitecaps.append([rng.uniform(0, w), rng.uniform(0, h), rng.uniform(4, 14), rng.uniform(1.5, 4),
                                   rng.uniform(0, 180), int(rng.integers(3, 9)), 0])
        alive = []
        for wc in self.whitecaps:
            x, y, a, b, ang, life, age = wc
            if age >= life:
                continue
            strength = 0.85 * (1.0 - age / life)
            R = int(a) + 6
            cx, cy = int(x - s.pan[0] * age), int(y - s.pan[1] * age)
            x0, y0, x1, y1 = max(0, cx - R), max(0, cy - R), min(w, cx + R + 1), min(h, cy + R + 1)
            if x1 > x0 and y1 > y0:
                m = np.zeros((y1 - y0, x1 - x0), np.float32)
                cv2.ellipse(m, (cx - x0, cy - y0), (int(a), int(b)), ang, 0, 360, 1.0, -1)
                m = cv2.GaussianBlur(m, (5, 5), 0) * (rng.random(m.shape) < 0.75) * strength
                patch = img[y0:y1, x0:x1]
                img[y0:y1, x0:x1] = patch * (1 - m[..., None]) + 235.0 * m[..., None]
            wc[6] += 1
            alive.append(wc)
        self.whitecaps = alive

        # debris: thin elongated dark-brown floating sticks
        for d in self.debris:
            x, y, ang, L, th = d
            cv2.ellipse(img, (int(x - ox), int(y - oy)), (int(L / 2), max(1, int(th / 2))), ang, 0, 360,
                        (40, 70, 95), -1, cv2.LINE_AA)
            d[0] += 0.2
            d[1] += 0.1

        boxes = []
        for p in self.people:
            boxes.append(self._draw_person(img, p, t, ox, oy))
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        return img, [b for b in boxes if b is not None]

    def _draw_person(self, img, p: _Person, t, ox, oy):
        h, w = img.shape[:2]
        p.x += p.vx + self.rng.normal(0, 0.15)
        p.y += p.vy + self.rng.normal(0, 0.15)
        cx, cy = p.x - ox, p.y - oy
        if not (8 < cx < w - 8 and 8 < cy < h - 8):
            return None
        if p.hidden_until < t and self.rng.random() < 0.02:
            p.hidden_until = t + int(self.rng.integers(1, 3))
        if t <= p.hidden_until:
            return None
        bob = 0.75 + 0.25 * math.sin(0.25 * t + p.phase)        # fraction above water
        r = p.head_r
        R = int(math.ceil(r * 4)) + 3
        px0, py0 = int(round(cx)) - R, int(round(cy)) - R
        size = 2 * R + 1
        layer = np.zeros((size, size, 3), np.float32)
        alpha = np.zeros((size, size), np.float32)
        lx, ly = cx - px0, cy - py0
        body_ax = (max(1, int(round(r * 1.9))), max(1, int(round(r * 1.2 * bob))))
        body_c = (int(round(lx)), int(round(ly + r * 1.3)))
        ang = math.degrees(math.atan2(p.vy, p.vx))
        cv2.ellipse(layer, body_c, body_ax, ang, 0, 360, p.body, -1, cv2.LINE_AA)
        cv2.ellipse(alpha, body_c, body_ax, ang, 0, 360, 0.9 * bob, -1, cv2.LINE_AA)
        head_c = (int(round(lx)), int(round(ly)))
        cv2.circle(layer, head_c, max(1, int(round(r))), p.skin, -1, cv2.LINE_AA)
        cv2.circle(alpha, head_c, max(1, int(round(r))), 1.0, -1, cv2.LINE_AA)
        hair = (25, 25, 30)
        cv2.ellipse(layer, (head_c[0], head_c[1] - max(1, int(r * 0.4))), (max(1, int(r)), max(1, int(r * 0.6))),
                    0, 180, 360, hair, -1, cv2.LINE_AA)
        # clip the patch to the frame
        x0, y0 = max(0, px0), max(0, py0)
        x1, y1 = min(w, px0 + size), min(h, py0 + size)
        a = alpha[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
        lay = layer[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a[..., None]) + lay * a[..., None]
        ys, xs = np.nonzero(a > 0.3)
        if len(xs) == 0:
            return None
        bx0, by0 = xs.min() + x0, ys.min() + y0
        bx1, by1 = xs.max() + 1 + x0, ys.max() + 1 + y0
        return float(bx0), float(by0), float(bx1 - bx0), float(by1 - by0)


def generate_dataset(out_dir: str | Path, specs: list[SceneSpec], write_video: bool = True) -> Path:
    """Render every scene to ``out_dir/images/seqNN/*.jpg`` (+ mp4) and write
    ``out_dir/annotations.json`` in COCO format. Returns the annotation path."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "swimmer"}]}
    img_id = ann_id = 1
    for vid, spec in enumerate(specs, 1):
        seq = f"seq{vid:02d}"
        (out / "images" / seq).mkdir(parents=True, exist_ok=True)
        r = SeaRenderer(spec)
        writer = None
        if write_video:
            writer = cv2.VideoWriter(str(out / f"{seq}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                                     (spec.width, spec.height))
        for t in range(spec.frames):
            img, boxes = r.render(t)
            name = f"images/{seq}/{t:05d}.jpg"
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if writer is not None:
                writer.write(img)
            coco["images"].append({"id": img_id, "file_name": name, "width": spec.width, "height": spec.height,
                                   "video_id": vid, "frame_index": t,
                                   "meta": {"altitude": spec.altitude, "sea_state": spec.sea_state}})
            for (x, y, w, h) in boxes:
                coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
                ann_id += 1
            img_id += 1
        if writer is not None:
            writer.release()
    path = out / "annotations.json"
    path.write_text(json.dumps(coco, indent=1))
    return path


def default_specs(frames: int = 60) -> list[SceneSpec]:
    """A small benchmark spanning altitude, sea state, glint and camera motion."""
    return [
        SceneSpec(frames=frames, altitude=50, sea_state=0.8, seed=1),
        SceneSpec(frames=frames, altitude=80, sea_state=1.0, seed=2, n_people=3),
        SceneSpec(frames=frames, altitude=110, sea_state=1.2, whitecap_rate=1.2, seed=3),
        SceneSpec(frames=frames, altitude=150, sea_state=0.6, glint=1.6, seed=4),
        SceneSpec(frames=frames, altitude=80, sea_state=1.6, whitecap_rate=2.0, glint=1.3, seed=5,
                  colours=["dark", "orange"]),
        SceneSpec(frames=frames, altitude=80, sea_state=1.0, pan=(1.5, -0.8), seed=6),
    ]
