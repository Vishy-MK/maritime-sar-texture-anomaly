"""Evaluation harness: recall@k, false alarms per frame, size/altitude
stratification and stage ablations."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .config import Config
from .datasets import Sequence
from .metrics import ALT_BINS, SIZE_BINS, Accumulator, HitRule
from .pipeline import Pipeline

ABLATIONS = {
    "full": {},
    "no_temporal": {"temporal_enabled": False},
    "no_glint": {"glint_enabled": False},
    "no_morphology": {"morph_enabled": False},
}


def run(seqs: list[Sequence], cfg: Config, kmax: int = 10, rule: HitRule | None = None,
        progress: Callable[[int, int], None] | None = None) -> tuple[Accumulator, dict]:
    """Run the pipeline over every sequence and accumulate metrics.

    Sequences shorter than ``min_persistence`` frames (single images, as in
    AFO) cannot confirm a track, so they are processed with the temporal stage
    off; the number of such frames is reported.
    """
    acc = Accumulator(kmax=kmax, rule=rule or HitRule())
    total = sum(len(s) for s in seqs)
    done = 0
    notes = {"frames_without_temporal": 0}
    eval_cfg = replace(cfg, top_k=kmax)
    for seq in seqs:
        seq_cfg = replace(eval_cfg)
        if cfg.temporal_enabled and len(seq) < cfg.min_persistence:
            seq_cfg.temporal_enabled = False
            notes["frames_without_temporal"] += len(seq)
        pipe = Pipeline(seq_cfg)
        is_video = len(seq) >= cfg.min_persistence
        for pos, fr in enumerate(seq.frames):
            img = fr.load()
            if cfg.use_altitude_metadata and fr.altitude is not None:
                pipe.cfg.altitude_m = fr.altitude     # area limits follow the per-frame altitude
            t0 = time.perf_counter()
            res = pipe.process(img)
            acc.seconds += time.perf_counter() - t0
            preds = [(c.det.x, c.det.y, c.det.w, c.det.h) for c in res.shortlist]
            # same frames are flagged with and without the temporal stage, so ablations compare like for like
            warmup = is_video and pos < cfg.min_persistence - 1
            acc.add(preds, fr.persons, fr.others, fr.altitude, warmup)
            done += 1
            if progress:
                progress(done, total)
    return acc, notes


def evaluate(seqs: list[Sequence], cfg: Config, k: int = 5, kmax: int = 10,
             ablations: list[str] | None = None, rule: HitRule | None = None,
             progress: Callable[[str, int, int], None] | None = None) -> dict:
    ablations = ablations or ["full", "no_temporal"]
    results = {"k": k, "kmax": kmax, "hit_rule": vars(rule or HitRule()), "config": cfg.to_dict(), "runs": {}}
    for name in ablations:
        if name not in ABLATIONS:
            raise ValueError(f"unknown ablation {name!r}; choose from {list(ABLATIONS)}")
        run_cfg = replace(cfg, **ABLATIONS[name]).validate()
        cb = (lambda d, t, n=name: progress(n, d, t)) if progress else None
        acc, notes = run(seqs, run_cfg, kmax, rule, cb)
        summ = acc.summary(k)
        summ.update(notes)
        results["runs"][name] = summ
    return results


def _fmt(v, pct=False) -> str:
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v:.2f}"


def to_markdown(results: dict) -> str:
    k = results["k"]
    lines = [f"## Results (k = {k}, hit rule: {results['hit_rule']['criterion']})", "",
             f"| run | frames | persons | recall@{k} | precision@{k} | false alarms / frame | "
             f"FA / frame (excl. other objects) | steady-state recall@{k} | steady-state FA / frame | fps |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results["runs"].items():
        lines.append(f"| {name} | {r['frames']} | {r['persons']} | {_fmt(r[f'recall@{k}'], True)} | "
                     f"{_fmt(r[f'precision@{k}'], True)} | {_fmt(r['fa_per_frame'])} | "
                     f"{_fmt(r['fa_per_frame_excl_other_objects'])} | {_fmt(r[f'steady_recall@{k}'], True)} | "
                     f"{_fmt(r['steady_fa_per_frame'])} | {_fmt(r['fps'])} |")
    lines += ["", "Steady-state numbers leave out the first min_persistence - 1 frames of each "
              "sequence, during which the temporal stage cannot yet have promoted a track."]
    for key, title, bins in (("by_size", "target size (sqrt of box area)", SIZE_BINS),
                             ("by_altitude", "altitude", ALT_BINS)):
        present = {b for r in results["runs"].values() for b in r[key]}
        names = [n for _, _, n in bins if n in present]
        if not names:
            continue
        counts = {n: max(r[key].get(n, {}).get("n", 0) for r in results["runs"].values()) for n in names}
        lines += ["", f"### Recall@{k} by {title}", "",
                  "| run | " + " | ".join(f"{n} (n={counts[n]})" for n in names) + " |",
                  "|---|" + "---|" * len(names)]
        for name, r in results["runs"].items():
            lines.append(f"| {name} | " + " | ".join(_fmt(r[key].get(n, {}).get(f"recall@{k}"), True) for n in names) + " |")
    lines += ["", "### Recall vs shortlist length", "",
              "| run | " + " | ".join(f"k={i}" for i in range(1, results["kmax"] + 1)) + " |",
              "|---|" + "---|" * results["kmax"]]
    for name, r in results["runs"].items():
        lines.append(f"| {name} | " + " | ".join(_fmt(r["recall_curve"][i], True) for i in range(1, results["kmax"] + 1)) + " |")
    return "\n".join(lines) + "\n"


def save_plots(results: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ks = list(range(1, results["kmax"] + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, r in results["runs"].items():
        axes[0].plot(ks, [r["recall_curve"][k] or 0 for k in ks], marker="o", label=name)
        axes[1].plot(ks, [r["fa_curve"][k] or 0 for k in ks], marker="o", label=name)
    axes[0].set(xlabel="shortlist length k", ylabel="recall@k", ylim=(0, 1.02), title="Recall vs k")
    axes[1].set(xlabel="shortlist length k", ylabel="false alarms per frame", title="False alarms vs k")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    p1 = out_dir / "recall_fa_vs_k.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    # recall vs false alarms trade-off
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, r in results["runs"].items():
        ax.plot([r["fa_curve"][k] or 0 for k in ks], [r["recall_curve"][k] or 0 for k in ks], marker="o", label=name)
    ax.set(xlabel="false alarms per frame", ylabel="recall", ylim=(0, 1.02), title="Recall / false-alarm trade-off (k = 1..kmax)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "recall_vs_fa.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    return [p1, p2]


def save_results(results: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    md = to_markdown(results)
    (out / "results.md").write_text(md, encoding="utf-8")
    plots = save_plots(results, out)
    return {"json": out / "results.json", "markdown": out / "results.md", "plots": plots}
