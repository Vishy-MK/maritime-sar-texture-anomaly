"""Frame input/output that also works with non-ASCII Windows paths."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}


def imread(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray, quality: int = 92) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def source_kind(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        return "folder"
    if p.suffix.lower() in VIDEO_EXTS:
        return "video"
    if p.suffix.lower() in IMAGE_EXTS:
        return "image"
    raise ValueError(f"unsupported input: {path}")


def iter_frames(path: str | Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, BGR frame) from a video file, an image folder or one image."""
    kind = source_kind(path)
    if kind == "image":
        yield Path(path).name, imread(path)
    elif kind == "folder":
        for p in list_images(path):
            yield p.name, imread(p)
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield f"frame_{i:06d}", frame
                i += 1
        finally:
            cap.release()


def count_frames(path: str | Path) -> int:
    kind = source_kind(path)
    if kind == "image":
        return 1
    if kind == "folder":
        return len(list_images(path))
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def video_fps(path: str | Path, default: float = 10.0) -> float:
    if source_kind(path) != "video":
        return default
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default
