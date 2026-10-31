"""Command-line interface.

    python -m sarvision gui
    python -m sarvision run      --input VIDEO|FOLDER|IMAGE [--output out.mp4] [--csv shortlist.csv]
    python -m sarvision evaluate --coco ann.json [--images ROOT] | --yolo IMAGES_DIR [--labels DIR]
    python -m sarvision synth    --out data/synthetic
    python -m sarvision config   --dump config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

from .config import Config
from .io_utils import count_frames, imwrite, iter_frames, source_kind, video_fps
from .pipeline import Pipeline
from .viz import draw_overlay


def load_config(path: str | None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path:
        cfg = Config.from_dict(json.loads(Path(path).read_text()))
    for item in overrides or []:
        key, _, val = item.partition("=")
        if not hasattr(cfg, key):
            raise SystemExit(f"unknown parameter {key!r}")
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            v = val.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            v = int(val)
        elif isinstance(cur, float) or cur is None:
            v = None if val.lower() == "none" else float(val)
        else:
            v = val
        setattr(cfg, key, v)
    return cfg.validate()


def _progress(done: int, total: int, prefix: str = "") -> None:
    if total <= 0:
        return
    pct = 100.0 * done / total
    sys.stdout.write(f"\r{prefix}{done}/{total} ({pct:5.1f}%)")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


# ---------------------------------------------------------------- commands

def cmd_run(args) -> int:
    cfg = load_config(args.config, args.set)
    if args.k:
        cfg.top_k = args.k
    kind = source_kind(args.input)
    total = count_frames(args.input)
    if total < cfg.min_persistence and cfg.temporal_enabled:
        print(f"note: {total} frame(s) - temporal stage disabled (needs >= {cfg.min_persistence})")
        cfg = replace(cfg, temporal_enabled=False)
    pipe = Pipeline(cfg)
    writer = None
    out = Path(args.output) if args.output else None
    rows = []
    t_start = time.perf_counter()
    for i, (name, frame) in enumerate(iter_frames(args.input)):
        res = pipe.process(frame)
        overlay = draw_overlay(frame, res)
        for rank, c in enumerate(res.shortlist, 1):
            d = c.det
            rows.append([i, name, rank, round(d.x, 1), round(d.y, 1), round(d.w, 1), round(d.h, 1),
                         round(c.conf, 4), c.track_id, c.hits])
        if out is not None:
            if out.suffix.lower() in (".mp4", ".avi"):
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out.suffix.lower() == ".mp4" else "XVID"))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out), fourcc, video_fps(args.input), (frame.shape[1], frame.shape[0]))
                writer.write(overlay)
            else:
                target = out / (Path(name).stem + ".jpg") if kind != "image" or out.suffix == "" else out
                imwrite(target, overlay)
        if args.show:
            cv2.imshow("sarvision", overlay)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        _progress(i + 1, total, "processing ")
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    n = len({r[0] for r in rows}) if rows else 0
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "name", "rank", "x", "y", "w", "h", "confidence", "track_id", "hits"])
            w.writerows(rows)
    print(f"done in {elapsed:.1f}s; {len(rows)} shortlist entries over {n} frame(s) with candidates")
    return 0


def cmd_evaluate(args) -> int:
    from .datasets import load_coco, load_yolo
    from .evaluate import evaluate, save_results, to_markdown
    from .metrics import HitRule

    cfg = load_config(args.config, args.set)
    if args.use_altitude:
        cfg.use_altitude_metadata = True
    if args.coco:
        seqs = load_coco(args.coco, args.images, args.person_regex, args.as_sequence, args.max_frames)
    elif args.yolo:
        classes = tuple(int(c) for c in args.person_classes.split(","))
        seqs = load_yolo(args.yolo, args.labels, classes, args.as_sequence, args.max_frames)
    else:
        raise SystemExit("give --coco or --yolo")
    n_frames = sum(len(s) for s in seqs)
    n_persons = sum(len(f.persons) for s in seqs for f in s.frames)
    print(f"{len(seqs)} sequence(s), {n_frames} frame(s), {n_persons} annotated person(s)")
    rule = HitRule(args.criterion, args.iou, args.centre_factor, args.min_radius)
    ablations = [a.strip() for a in args.ablations.split(",") if a.strip()]
    res = evaluate(seqs, cfg, args.k, args.kmax, ablations, rule,
                   progress=lambda name, d, t: _progress(d, t, f"{name:>14}: "))
    paths = save_results(res, args.out)
    print(to_markdown(res))
    print(f"saved {paths['json']}, {paths['markdown']} and plots in {args.out}")
    return 0


def cmd_synth(args) -> int:
    from .synthetic import default_specs, generate_dataset
    path = generate_dataset(args.out, default_specs(args.frames), write_video=not args.no_video)
    print(f"wrote {path}")
    return 0


def cmd_config(args) -> int:
    Path(args.dump).write_text(json.dumps(Config().to_dict(), indent=2))
    print(f"wrote default parameters to {args.dump}")
    return 0


def cmd_gui(args) -> int:
    from .gui import main as gui_main
    gui_main(args.input)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sarvision", description="Texture-anomaly survivor localisation for maritime SAR")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="JSON parameter file (see `config --dump`)")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override one parameter (repeatable)")

    g = sub.add_parser("gui", help="open the operator / tuning GUI")
    g.add_argument("input", nargs="?", help="optional video, image folder or image to open")
    g.set_defaults(func=cmd_gui)

    r = sub.add_parser("run", help="process a video / folder / image and write the annotated result")
    r.add_argument("--input", required=True)
    r.add_argument("--output", help="output .mp4/.avi, or a folder for annotated images")
    r.add_argument("--csv", help="write the per-frame shortlist to this CSV")
    r.add_argument("--k", type=int, help="shortlist length")
    r.add_argument("--show", action="store_true", help="display while processing")
    common(r)
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="recall@k / false alarms / stratification / ablation")
    e.add_argument("--coco", help="COCO annotation JSON (SeaDronesSee, MOBDrone, synthetic)")
    e.add_argument("--images", help="image root for --coco (default: JSON folder)")
    e.add_argument("--yolo", help="image folder with YOLO labels (AFO)")
    e.add_argument("--labels", help="label folder for --yolo (default: sibling 'labels')")
    e.add_argument("--person-classes", default="0", help="YOLO class ids that are persons (default 0)")
    e.add_argument("--person-regex", default=r"swimmer|person|human|floater|people|overboard|^man$",
                   help="COCO category names counted as persons")
    e.add_argument("--as-sequence", action="store_true", help="treat all images as one ordered sequence")
    e.add_argument("--max-frames", type=int, help="evaluate at most this many frames")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--kmax", type=int, default=10)
    e.add_argument("--ablations", default="full,no_temporal",
                   help="comma list from: full,no_temporal,no_glint,no_morphology")
    e.add_argument("--criterion", choices=["centre", "iou"], default="centre")
    e.add_argument("--iou", type=float, default=0.5)
    e.add_argument("--centre-factor", type=float, default=1.0)
    e.add_argument("--min-radius", type=float, default=8.0)
    e.add_argument("--use-altitude", action="store_true", help="derive blob area limits from altitude metadata")
    e.add_argument("--out", default="results")
    common(e)
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("synth", help="render the synthetic benchmark with ground truth")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--frames", type=int, default=60)
    s.add_argument("--no-video", action="store_true")
    s.set_defaults(func=cmd_synth)

    c = sub.add_parser("config", help="write the default parameters to JSON")
    c.add_argument("--dump", default="config.json")
    c.set_defaults(func=cmd_config)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
