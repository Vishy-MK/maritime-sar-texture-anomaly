"""Loaders for annotated aerial maritime data.

* COCO-style JSON: SeaDronesSee (object detection and multi-object tracking
  splits) and MOBDrone both ship COCO annotations. Frames are grouped into
  sequences by ``video_id`` when the JSON provides it; per-image altitude and
  gimbal angle are read from ``meta`` (SeaDronesSee) when present.
* YOLO txt labels: AFO ships one ``class cx cy w h`` (normalised) file per image.
  Class 0 is "human" in AFO.

The method is unsupervised: annotations are used only to measure performance.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .io_utils import IMAGE_EXTS, imread

DEFAULT_PERSON_REGEX = r"swimmer|person|human|floater|people|overboard|^man$"


@dataclass
class GTFrame:
    path: Path
    persons: list = field(default_factory=list)   # [(x, y, w, h)] in original pixels
    others: list = field(default_factory=list)    # other annotated objects (boats, buoys ...)
    altitude: float | None = None
    angle: float | None = None
    index: int = 0

    def load(self):
        return imread(self.path)


@dataclass
class Sequence:
    name: str
    frames: list[GTFrame]

    def __len__(self) -> int:
        return len(self.frames)


def _meta_value(img: dict, keys: tuple[str, ...]) -> float | None:
    for src in (img.get("meta") or {}, img):
        if not isinstance(src, dict):
            continue
        for k in keys:
            v = src.get(k)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    pass
    return None


def load_coco(ann_path: str | Path, image_root: str | Path | None = None,
              person_regex: str = DEFAULT_PERSON_REGEX, as_sequence: bool = False,
              max_frames: int | None = None) -> list[Sequence]:
    """Read a COCO JSON. ``image_root`` defaults to the JSON's directory.

    With ``as_sequence`` all images are treated as one sequence ordered by file
    name (useful for frame folders exported from one video without video ids).
    """
    ann_path = Path(ann_path)
    root = Path(image_root) if image_root else ann_path.parent
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    pat = re.compile(person_regex, re.IGNORECASE)
    person_cats = {c["id"] for c in data.get("categories", []) if pat.search(str(c.get("name", "")))}
    if not person_cats:
        names = [c.get("name") for c in data.get("categories", [])]
        raise ValueError(f"no category matches /{person_regex}/; categories are {names}")
    ignore_cats = {c["id"] for c in data.get("categories", []) if str(c.get("name", "")).lower() == "ignored"}

    frames: dict[int, GTFrame] = {}
    order: dict[int, tuple] = {}
    groups: dict[object, list[int]] = defaultdict(list)
    for img in data["images"]:
        fname = img["file_name"]
        p = root / fname
        if not p.exists() and (root / Path(fname).name).exists():
            p = root / Path(fname).name
        fr = GTFrame(p, altitude=_meta_value(img, ("altitude", "alt")),
                     angle=_meta_value(img, ("gimbal_pitch", "camera_pitch", "pitch", "angle")))
        frames[img["id"]] = fr
        idx = img.get("frame_index", img.get("frame_id", img.get("frame")))
        order[img["id"]] = (idx if isinstance(idx, (int, float)) else 0, fname)
        if as_sequence:
            key = "all"
        else:
            key = img.get("video_id", img.get("video", img.get("sequence_id")))
            if key is None:
                key = ("img", img["id"])
        groups[key].append(img["id"])

    for a in data.get("annotations", []):
        fr = frames.get(a["image_id"])
        if fr is None or a.get("category_id") in ignore_cats:
            continue
        box = tuple(float(v) for v in a["bbox"][:4])
        if box[2] <= 0 or box[3] <= 0:
            continue
        (fr.persons if a["category_id"] in person_cats else fr.others).append(box)

    seqs = []
    for key, ids in groups.items():
        ids.sort(key=lambda i: order[i])
        fl = [frames[i] for i in ids]
        for j, f in enumerate(fl):
            f.index = j
        name = f"video_{key}" if not isinstance(key, tuple) else Path(fl[0].path).stem
        seqs.append(Sequence(str(name), fl))
    seqs.sort(key=lambda s: s.name)
    return _limit(seqs, max_frames)


def load_yolo(images_dir: str | Path, labels_dir: str | Path | None = None,
              person_classes: tuple[int, ...] = (0,), as_sequence: bool = False,
              max_frames: int | None = None) -> list[Sequence]:
    """Read YOLO labels (AFO). ``labels_dir`` defaults to the sibling ``labels``
    folder of ``images_dir``."""
    images_dir = Path(images_dir)
    if labels_dir is None:
        labels_dir = images_dir.parent / "labels" if images_dir.name == "images" else images_dir
    labels_dir = Path(labels_dir)
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    frames = []
    for p in imgs:
        fr = GTFrame(p)
        lab = labels_dir / (p.stem + ".txt")
        if lab.exists():
            text = lab.read_text().strip()
            if text:
                h, w = imread(p).shape[:2]
                for line in text.splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    c = int(float(parts[0]))
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    box = ((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h)
                    (fr.persons if c in person_classes else fr.others).append(box)
        frames.append(fr)
    if as_sequence:
        for j, f in enumerate(frames):
            f.index = j
        return _limit([Sequence(images_dir.name, frames)], max_frames)
    return _limit([Sequence(f.path.stem, [f]) for f in frames], max_frames)


def _limit(seqs: list[Sequence], max_frames: int | None) -> list[Sequence]:
    if not max_frames:
        return seqs
    out, n = [], 0
    for s in seqs:
        if n >= max_frames:
            break
        take = s.frames[:max_frames - n]
        out.append(Sequence(s.name, take))
        n += len(take)
    return out
