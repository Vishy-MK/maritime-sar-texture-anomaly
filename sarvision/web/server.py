"""Web operator console: a small stdlib HTTP server around the pipeline.

When a sequence is opened the whole sequence is analysed in a background
thread (stages 1-5 run in parallel on a thread pool, stage 6 sequentially), so
the browser can play the result back at the source frame rate while the
timeline fills in. Intermediate stage images are rendered on demand for the
frame the operator is looking at. Changing a parameter restarts the analysis.

Endpoints (all JSON unless noted):
    GET  /                      the console (static/index.html)
    GET  /api/sources           videos / folders found under data/ and results/
    POST /api/open   {path}     open a source and start analysing it
    GET  /api/state             source, config, parameter schema
    GET  /api/scan?since=N      analysis records from frame N on, progress
    GET  /api/frame/N           frame N as JPEG
    GET  /api/stages/N          every intermediate stage of frame N (data URLs)
    POST /api/config {..}       change parameters (restarts the analysis)
    POST /api/config/reset      default parameters
"""

from __future__ import annotations

import base64
import json
import math
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ..config import Config
from ..datasets import load_coco
from ..gui import FrameSource
from ..io_utils import IMAGE_EXTS, VIDEO_EXTS
from ..metrics import HitRule
from ..pipeline import Pipeline, estimate_shift, _to_original
from ..tracking import RankedCandidate
from ..viz import render_stage

STATIC = Path(__file__).with_name("static")
ROOT = Path(__file__).resolve().parents[2]

# (field, label, group, min, max, step)
SCHEMA = [
    ("top_k", "Shortlist length k", "Output", 1, 20, 1),
    ("max_width", "Processing width", "Output", 320, 3840, 32),
    ("lbp_weight", "LBP weight vs Gabor", "Texture", 0.0, 1.0, 0.05),
    ("reference_window", "Reference window (tiles)", "Anomaly", 3, 21, 2),
    ("anomaly_z", "Tile anomaly threshold z", "Anomaly", 1.0, 8.0, 0.1),
    ("pixel_z", "Pixel deviation threshold z", "Anomaly", 1.0, 10.0, 0.1),
    ("glint_value", "Glint min brightness", "Glint", 0.5, 1.0, 0.01),
    ("glint_chroma", "Glint max chroma", "Glint", 0.0, 0.2, 0.005),
    ("min_area", "Min blob area (px)", "Morphology", 1, 200, 1),
    ("max_area", "Max blob area (px)", "Morphology", 50, 10000, 10),
    ("max_eccentricity", "Max eccentricity", "Morphology", 0.5, 1.0, 0.01),
    ("min_solidity", "Min solidity", "Morphology", 0.0, 1.0, 0.01),
    ("match_radius", "Match radius (px)", "Temporal", 5, 150, 1),
    ("min_persistence", "Min persistence (frames)", "Temporal", 1, 15, 1),
    ("max_gap", "Max gap (frames)", "Temporal", 0, 10, 1),
    ("min_coherence", "Min drift coherence", "Temporal", 0.0, 1.0, 0.01),
]
TOGGLES = [
    ("glint_enabled", "Glint suppression"),
    ("morph_enabled", "Morphology filter"),
    ("temporal_enabled", "Temporal association"),
    ("motion_compensation", "Camera-motion comp."),
]
STAGES = [
    ("input", "Input"),
    ("normalised", "Normalise"),
    ("gabor_energy", "Gabor energy"),
    ("lbp", "Uniform LBP"),
    ("anomaly_map", "Anomaly z-map"),
    ("candidates", "Candidates"),
    ("glint", "Glint"),
    ("after_glint", "De-glinted"),
    ("morphology", "Morphology"),
]


# --------------------------------------------------------------------- data

def find_sources() -> list[dict]:
    out = []
    seen = set()
    for base in (ROOT / "data", ROOT / "results"):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS and "annotated" not in p.stem:
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append({"path": key, "name": p.stem, "kind": "video", "group": p.parent.name})
    return out


class GroundTruth:
    """Person boxes per frame for the synthetic benchmark (matched by sequence name)."""

    def __init__(self):
        self.by_seq: dict[str, list] = {}
        self.meta: dict[str, dict] = {}
        ann = ROOT / "data" / "synthetic" / "annotations.json"
        if not ann.exists():
            return
        try:
            for s in load_coco(ann):
                name = Path(s.frames[0].path).parent.name
                self.by_seq[name] = [f.persons for f in s.frames]
            raw = json.loads(ann.read_text(encoding="utf-8"))
            for img in raw["images"]:
                name = Path(img["file_name"]).parent.name
                self.meta.setdefault(name, img.get("meta") or {})
        except Exception:  # GT is optional
            self.by_seq.clear()

    def get(self, path: str):
        return self.by_seq.get(Path(path).stem), self.meta.get(Path(path).stem, {})


def _jpeg(img: np.ndarray, q: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise IOError("jpeg encode failed")
    return buf.tobytes()


def _data_url(img: np.ndarray, q: int = 85) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg(img, q)).decode()


def _f(x, nd=2):
    return round(float(x), nd)


# ------------------------------------------------------------------ session

class Session:
    def __init__(self):
        self.lock = threading.RLock()
        self.cfg = Config()
        self.path: str | None = None
        self.n = 0
        self.fps = 10.0
        self.size = (0, 0)
        self.gt = None
        self.meta: dict = {}
        self.frames: dict[int, bytes] = {}
        self.records: list[dict | None] = []
        self.gen = 0
        self.scan_started = 0.0
        self.scan_seconds = 0.0
        self._stage_cache: tuple | None = None
        self._stage_lock = threading.Lock()
        self._frame_src: FrameSource | None = None
        self._gt = GroundTruth()

    # ---------------------------------------------------------------- open
    def open(self, path: str) -> None:
        src = FrameSource(path)
        first = src.get(0)
        with self.lock:
            if self._frame_src is not None:
                self._frame_src.close()
            self._frame_src = src
            self.path = path
            self.n = src.n
            self.fps = float(src.fps or 10.0)
            self.size = (first.shape[1], first.shape[0])
            self.gt, self.meta = self._gt.get(path)
            self.frames = {0: _jpeg(first)}
            self._stage_cache = None
        self.restart_scan()

    def frame_jpeg(self, i: int) -> bytes:
        with self.lock:
            if i in self.frames:
                return self.frames[i]
            img = self._frame_src.get(i)
            b = _jpeg(img)
            self.frames[i] = b
            return b

    # ---------------------------------------------------------------- scan
    def restart_scan(self) -> None:
        with self.lock:
            self.gen += 1
            gen = self.gen
            self.records = [None] * self.n
            self.scan_started = time.perf_counter()
            self.scan_seconds = 0.0
            self._stage_cache = None
            cfg = self._effective_cfg()
            path = self.path
        threading.Thread(target=self._scan, args=(gen, path, cfg), daemon=True).start()

    def _effective_cfg(self) -> Config:
        if self.n < self.cfg.min_persistence and self.cfg.temporal_enabled:
            return replace(self.cfg, temporal_enabled=False)
        return self.cfg

    def _scan(self, gen: int, path: str, cfg: Config) -> None:
        src = FrameSource(path)
        pipe = Pipeline(cfg)
        rule = HitRule()
        prev_gray = None

        def load(i):
            frame = src.get(i)
            h, w = frame.shape[:2]
            scale = min(1.0, cfg.max_width / float(w))
            bgr = frame if scale == 1.0 else cv2.resize(frame, (round(w * scale), round(h * scale)),
                                                        interpolation=cv2.INTER_AREA)
            return frame, bgr, scale

        def detect(item):
            frame, bgr, scale = item
            if gen != self.gen:
                return None
            return frame, bgr, scale, pipe.detect(bgr, keep_stages=False)

        try:
            # frames must be read sequentially (video), detection runs on a pool
            with ThreadPoolExecutor(max_workers=4) as pool:
                window = 6
                futs = {}
                nxt = 0
                for i in range(self.n):
                    while nxt < self.n and nxt < i + window:
                        futs[nxt] = pool.submit(detect, load(nxt))
                        nxt += 1
                    out = futs.pop(i).result()
                    if out is None or gen != self.gen:
                        for f in futs.values():
                            f.cancel()
                        return
                    frame, bgr, scale, (dets, _stages, info, timings) = out
                    t0 = time.perf_counter()
                    shift = (0.0, 0.0)
                    if cfg.temporal_enabled and cfg.motion_compensation:
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        if prev_gray is not None and prev_gray.shape == gray.shape:
                            shift = estimate_shift(prev_gray, gray)
                        prev_gray = gray
                    if cfg.temporal_enabled:
                        pipe.tracker.update(i, dets, shift)
                        ranked = pipe.tracker.ranked(i)
                    else:
                        ranked = [RankedCandidate(d, d.conf) for d in dets]
                    timings["temporal"] = time.perf_counter() - t0
                    timings["total"] = sum(timings.values())
                    inv = 1.0 / scale
                    shortlist = [replace(c, det=_to_original(c.det, inv)) for c in ranked[:cfg.top_k]]
                    rec = self._record(i, shortlist, [_to_original(d, inv) for d in dets], info, timings,
                                       shift, scale, rule)
                    with self.lock:
                        if gen != self.gen:
                            return
                        if i not in self.frames:
                            self.frames[i] = _jpeg(frame)
                        self.records[i] = rec
                        self.scan_seconds = time.perf_counter() - self.scan_started
        finally:
            src.close()

    def _record(self, i, shortlist, dets, info, timings, shift, scale, rule) -> dict:
        gts = (self.gt[i] if self.gt and i < len(self.gt) else None)
        sl = []
        for rank, c in enumerate(shortlist, 1):
            d = c.det
            sl.append({"rank": rank, "x": _f(d.x, 1), "y": _f(d.y, 1), "w": _f(d.w, 1), "h": _f(d.h, 1),
                       "conf": _f(c.conf, 3), "track": c.track_id, "hits": c.hits,
                       "coh": _f(c.coherence, 2), "missed": c.missed, "gt": False})
        gt_out = None
        if gts is not None:
            gt_out = []
            for g in gts:
                hit_rank = None
                for s in sl:
                    if rule.hit((s["x"], s["y"], s["w"], s["h"]), g):
                        hit_rank = s["rank"] if hit_rank is None else min(hit_rank, s["rank"])
                        s["gt"] = True
                gt_out.append({"x": g[0], "y": g[1], "w": g[2], "h": g[3], "hit": hit_rank})
        rej: dict[str, int] = {}
        for b in info.get("blobs", []):
            if b.rejected:
                rej[b.rejected] = rej.get(b.rejected, 0) + 1
        return {
            "i": i,
            "shortlist": sl,
            "gt": gt_out,
            "n_dets": len(dets),
            "dets": [{"x": _f(d.x, 1), "y": _f(d.y, 1), "w": _f(d.w, 1), "h": _f(d.h, 1), "conf": _f(d.conf, 3)}
                     for d in dets[:30]],
            "timings": {k: _f(1000 * v, 1) for k, v in timings.items()},
            "waves": [[_f(w, 1), _f(a, 0), _f(s, 0)] for (w, a, s) in info.get("waves", [])],
            "tile": list(info.get("tile", (0, 0))),
            "n_filters": info.get("n_filters"),
            "anom_tiles": info.get("n_anomalous_tiles"),
            "n_tiles": info.get("n_tiles"),
            "pixel_thresh": _f(info.get("pixel_thresh", 0), 1),
            "blobs": len(info.get("blobs", [])),
            "rejected": rej,
            "shift": [_f(shift[0], 1), _f(shift[1], 1)],
            "scale": _f(scale, 3),
        }

    def scan_state(self, since: int) -> dict:
        with self.lock:
            done = sum(r is not None for r in self.records)
            return {
                "gen": self.gen,
                "done": done,
                "total": self.n,
                "seconds": _f(self.scan_seconds, 1),
                "records": [r for r in self.records[since:] if r is not None]
                if since < len(self.records) else [],
            }

    # -------------------------------------------------------------- stages
    def stages(self, i: int) -> dict:
        with self._stage_lock:
            with self.lock:
                gen, cfg, path = self.gen, self._effective_cfg(), self.path
                cached = self._stage_cache
            if cached and cached[0] == gen and cached[1] == i:
                return cached[2]
            src = FrameSource(path)
            try:
                # stages 0-5 are per-frame, so no temporal warm-up is needed here
                pipe = Pipeline(replace(cfg, temporal_enabled=False))
                frame = src.get(i)
                res = pipe.process(frame, keep_stages=True)
            finally:
                src.close()
            out = {"i": i, "gen": gen,
                   "stages": {name: _data_url(render_stage(name, res, frame, pipe.cfg)) for name, _ in STAGES}}
            with self.lock:
                if gen == self.gen:
                    self._stage_cache = (gen, i, out)
            return out

    # -------------------------------------------------------------- config
    def state(self) -> dict:
        with self.lock:
            cfg = self.cfg
            src = None
            if self.path:
                src = {"path": self.path, "name": Path(self.path).stem, "n": self.n, "fps": self.fps,
                       "width": self.size[0], "height": self.size[1], "has_gt": self.gt is not None,
                       "meta": self.meta}
            return {
                "source": src,
                "gen": self.gen,
                "config": cfg.to_dict(),
                "schema": [{"key": k, "label": l, "group": g, "min": lo, "max": hi, "step": st,
                            "int": isinstance(getattr(Config(), k), int)} for k, l, g, lo, hi, st in SCHEMA],
                "toggles": [{"key": k, "label": l} for k, l in TOGGLES],
                "stages": [{"key": k, "label": l} for k, l in STAGES],
                "references": ["local", "band", "global"],
            }

    def set_config(self, patch: dict) -> None:
        names = {f.name for f in fields(Config)}
        with self.lock:
            d = self.cfg.to_dict()
            for k, v in patch.items():
                if k in names:
                    cur = getattr(Config(), k)
                    if isinstance(cur, bool):
                        v = bool(v)
                    elif isinstance(cur, int) and v is not None:
                        v = int(round(float(v)))
                    elif isinstance(cur, float) and v is not None:
                        v = float(v)
                    d[k] = v
            if d["min_area"] > d["max_area"]:
                d["max_area"] = d["min_area"]
            self.cfg = Config.from_dict(d)
        if self.path:
            self.restart_scan()

    def reset_config(self) -> None:
        with self.lock:
            self.cfg = Config()
        if self.path:
            self.restart_scan()


# ------------------------------------------------------------------ handler

SESSION = Session()


class Handler(BaseHTTPRequestHandler):
    server_version = "sarvision/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path
        try:
            if p in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if p == "/api/sources":
                return self._json(find_sources())
            if p == "/api/state":
                return self._json(SESSION.state())
            if p == "/api/scan":
                return self._json(SESSION.scan_state(int(q.get("since", ["0"])[0])))
            if p.startswith("/api/frame/"):
                i = int(p.rsplit("/", 1)[1].split(".")[0])
                return self._send(200, SESSION.frame_jpeg(i), "image/jpeg")
            if p.startswith("/api/stages/"):
                return self._json(SESSION.stages(int(p.rsplit("/", 1)[1])))
            self._send(404, b"not found", "text/plain")
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            if p == "/api/open":
                path = self._body().get("path", "")
                pp = Path(path)
                if not pp.exists() or not (pp.is_dir() or pp.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS):
                    return self._json({"error": f"cannot open {path}"}, 400)
                SESSION.open(str(pp.resolve()))
                return self._json(SESSION.state())
            if p == "/api/config":
                SESSION.set_config(self._body())
                return self._json(SESSION.state())
            if p == "/api/config/reset":
                SESSION.reset_config()
                return self._json(SESSION.state())
            self._send(404, b"not found", "text/plain")
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:
            self._json({"error": str(e)}, 500)


def main(initial: str | None = None, port: int = 8765, open_browser: bool = True) -> None:
    if initial is None:
        srcs = find_sources()
        pick = next((s for s in srcs if s["name"] == "seq02"), srcs[0] if srcs else None)
        initial = pick["path"] if pick else None
    if initial:
        SESSION.open(str(Path(initial).resolve()))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"SARVision console on {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
