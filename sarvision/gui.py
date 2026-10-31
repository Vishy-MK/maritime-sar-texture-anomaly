"""Operator / tuning GUI (Tkinter).

Load a video, an image folder or a single image; play or step through it;
choose which pipeline stage to display; tune the parameters with live
re-processing; read the ranked shortlist; export the annotated sequence.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .config import Config
from .io_utils import IMAGE_EXTS, VIDEO_EXTS, imread, list_images, source_kind, video_fps
from .pipeline import Pipeline
from .viz import STAGE_NAMES, draw_overlay, render_stage

STAGE_LABELS = {
    "final": "6. Final shortlist (overlay)",
    "input": "0. Input",
    "normalised": "1. Normalised (CLAHE + stretch)",
    "gabor_energy": "2a. Gabor energy",
    "lbp": "2b. Uniform LBP codes",
    "anomaly_map": "3a. Anomaly map (chi-square z)",
    "candidates": "3b. Candidate pixels",
    "glint": "4a. Glint pixels",
    "after_glint": "4b. After glint suppression",
    "morphology": "5. Morphology + shape filter",
}

# (field, label, from, to, resolution)
SLIDERS = [
    ("anomaly_z", "Tile anomaly threshold (z)", 1.0, 8.0, 0.1),
    ("pixel_z", "Pixel deviation threshold (z)", 1.0, 10.0, 0.1),
    ("lbp_weight", "LBP weight (vs Gabor)", 0.0, 1.0, 0.05),
    ("reference_window", "Reference window (tiles)", 3, 21, 2),
    ("glint_value", "Glint min brightness", 0.5, 1.0, 0.01),
    ("glint_chroma", "Glint max chroma", 0.0, 0.2, 0.005),
    ("min_area", "Min blob area (px)", 1, 200, 1),
    ("max_area", "Max blob area (px)", 50, 10000, 10),
    ("max_eccentricity", "Max eccentricity", 0.5, 1.0, 0.01),
    ("min_solidity", "Min solidity", 0.0, 1.0, 0.01),
    ("match_radius", "Match radius (px)", 5, 150, 1),
    ("min_persistence", "Min persistence (frames)", 1, 15, 1),
    ("max_gap", "Max gap (frames)", 0, 10, 1),
    ("min_coherence", "Min drift coherence", 0.0, 1.0, 0.01),
    ("top_k", "Shortlist length k", 1, 20, 1),
    ("max_width", "Processing width (px)", 320, 3840, 32),
]
CHECKS = [
    ("glint_enabled", "Glint suppression"),
    ("morph_enabled", "Morphology / shape filter"),
    ("temporal_enabled", "Temporal association"),
    ("motion_compensation", "Camera-motion compensation"),
]
TILE_CHOICES = ["auto", "16/8", "24/12", "32/16", "48/24", "64/32"]


class FrameSource:
    """Random access over a video, an image folder or a single image."""

    def __init__(self, path: str):
        self.path = path
        self.kind = source_kind(path)
        self.cap = None
        self.files: list[Path] = []
        if self.kind == "video":
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise IOError(f"cannot open {path}")
            self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._pos = 0
        elif self.kind == "folder":
            self.files = list_images(path)
            self.n = len(self.files)
        else:
            self.files = [Path(path)]
            self.n = 1
        if self.n <= 0:
            raise IOError(f"no frames in {path}")
        self.fps = video_fps(path)

    def get(self, i: int) -> np.ndarray:
        i = int(np.clip(i, 0, self.n - 1))
        if self.cap is None:
            return imread(self.files[i])
        if i != self._pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        if not ok:
            raise IOError(f"cannot read frame {i}")
        self._pos = i + 1
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class App:
    def __init__(self, root: tk.Tk, initial: str | None = None):
        self.root = root
        root.title("SAR survivor localisation - texture anomaly pipeline")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1600, int(sw * 0.92))}x{min(950, int(sh * 0.85))}+10+10")
        self.cfg = Config()
        self.pipe = Pipeline(self.cfg)
        self.src: FrameSource | None = None
        self.idx = 0
        self.frame = None
        self.result = None
        self.playing = False
        self._pending = None
        self._photo = None
        self._export_q: queue.Queue = queue.Queue()
        self.vars: dict[str, tk.Variable] = {}
        self._build()
        if initial:
            self.open_path(initial)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # left: scrollable controls
        left_outer = ttk.Frame(main, width=330)
        left_outer.pack(side="left", fill="y")
        canvas = tk.Canvas(left_outer, width=320, highlightthickness=0)
        sb = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
        left = ttk.Frame(canvas)
        left.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=left, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="y")
        sb.pack(side="left", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        files = ttk.LabelFrame(left, text="Input")
        files.pack(fill="x", padx=6, pady=4)
        for txt, cmd in (("Open video...", self.open_video), ("Open image folder...", self.open_folder),
                         ("Open image...", self.open_image)):
            ttk.Button(files, text=txt, command=cmd).pack(fill="x", padx=4, pady=2)

        play = ttk.LabelFrame(left, text="Playback")
        play.pack(fill="x", padx=6, pady=4)
        row = ttk.Frame(play)
        row.pack(fill="x")
        self.play_btn = ttk.Button(row, text="Play", command=self.toggle_play)
        self.play_btn.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        ttk.Button(row, text="Step", command=self.step).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(row, text="Restart", command=self.restart).pack(side="left", expand=True, fill="x", padx=2)
        self.pos_var = tk.IntVar(value=0)
        self.pos_scale = ttk.Scale(play, from_=0, to=0, orient="horizontal", variable=self.pos_var)
        self.pos_scale.pack(fill="x", padx=4)
        self.pos_scale.bind("<ButtonRelease-1>", lambda e: self.seek(int(self.pos_var.get())))
        self.pos_label = ttk.Label(play, text="no input")
        self.pos_label.pack(anchor="w", padx=4)

        view = ttk.LabelFrame(left, text="Display stage")
        view.pack(fill="x", padx=6, pady=4)
        self.stage_var = tk.StringVar(value=STAGE_LABELS["final"])
        cb = ttk.Combobox(view, textvariable=self.stage_var, state="readonly",
                          values=[STAGE_LABELS[s] for s in STAGE_NAMES])
        cb.pack(fill="x", padx=4, pady=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        params = ttk.LabelFrame(left, text="Parameters")
        params.pack(fill="x", padx=6, pady=4)
        ttk.Label(params, text="Tile / stride (px)").pack(anchor="w", padx=4)
        self.tile_var = tk.StringVar(value="auto" if self.cfg.auto_tile else f"{self.cfg.tile}/{self.cfg.stride}")
        tcb = ttk.Combobox(params, textvariable=self.tile_var, values=TILE_CHOICES, state="readonly")
        tcb.pack(fill="x", padx=4)
        tcb.bind("<<ComboboxSelected>>", lambda e: self.param_changed())
        ttk.Label(params, text="Reference").pack(anchor="w", padx=4)
        self.ref_var = tk.StringVar(value=self.cfg.reference)
        rcb = ttk.Combobox(params, textvariable=self.ref_var, values=["local", "band", "global"], state="readonly")
        rcb.pack(fill="x", padx=4)
        rcb.bind("<<ComboboxSelected>>", lambda e: self.param_changed())
        for name, label in CHECKS:
            v = tk.BooleanVar(value=getattr(self.cfg, name))
            self.vars[name] = v
            ttk.Checkbutton(params, text=label, variable=v, command=self.param_changed).pack(anchor="w", padx=4)
        self.slider_labels = {}
        for name, label, lo, hi, res in SLIDERS:
            fr = ttk.Frame(params)
            fr.pack(fill="x", padx=4, pady=1)
            lab = ttk.Label(fr, text="")
            lab.pack(anchor="w")
            is_int = isinstance(getattr(self.cfg, name), int)
            v = tk.DoubleVar(value=getattr(self.cfg, name))
            self.vars[name] = v
            self.slider_labels[name] = (lab, label, is_int)
            s = tk.Scale(fr, from_=lo, to=hi, resolution=res, orient="horizontal", variable=v,
                         showvalue=False, length=270, command=lambda _v, n=name: self._slider_moved(n))
            s.pack(fill="x")
            self._slider_label(name)
        row = ttk.Frame(params)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="Defaults", command=self.load_defaults).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(row, text="Save...", command=self.save_config).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(row, text="Load...", command=self.load_config).pack(side="left", expand=True, fill="x", padx=2)

        exp = ttk.LabelFrame(left, text="Export")
        exp.pack(fill="x", padx=6, pady=4)
        ttk.Button(exp, text="Export annotated video...", command=self.export).pack(fill="x", padx=4, pady=2)
        self.export_label = ttk.Label(exp, text="")
        self.export_label.pack(anchor="w", padx=4)

        # right: shortlist + info (packed before the image so it is never squeezed out)
        right = ttk.Frame(main, width=330)
        right.pack(side="right", fill="y")

        # centre: image
        centre = ttk.Frame(main)
        centre.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(centre, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        ttk.Label(right, text="Shortlist (look here first)", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=6, pady=(6, 2))
        cols = ("rank", "conf", "x", "y", "w", "h", "track", "hits")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=14)
        for c, wdt in zip(cols, (38, 50, 46, 46, 34, 34, 42, 36)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=wdt, anchor="center")
        self.tree.pack(fill="x", padx=6)
        ttk.Label(right, text="Frame diagnostics", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=6, pady=(10, 2))
        self.info = tk.Text(right, width=44, height=26, font=("Consolas", 9), wrap="word")
        self.info.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Right>", lambda e: self.step())

    # ----------------------------------------------------------------- params
    def _slider_label(self, name: str) -> None:
        lab, label, is_int = self.slider_labels[name]
        v = self.vars[name].get()
        lab.configure(text=f"{label}: {int(round(v)) if is_int else round(v, 3)}")

    def _slider_moved(self, name: str) -> None:
        self._slider_label(name)
        self.param_changed()

    def config_from_widgets(self) -> Config:
        kw = {}
        for name, _ in CHECKS:
            kw[name] = bool(self.vars[name].get())
        for name, *_ in SLIDERS:
            v = self.vars[name].get()
            kw[name] = int(round(v)) if self.slider_labels[name][2] else float(v)
        if self.tile_var.get() == "auto":
            kw["auto_tile"] = True
        else:
            kw["auto_tile"] = False
            kw["tile"], kw["stride"] = (int(x) for x in self.tile_var.get().split("/"))
        kw["reference"] = self.ref_var.get()
        if kw["min_area"] > kw["max_area"]:
            kw["max_area"] = kw["min_area"]
        return replace(self.cfg, **kw).validate()

    def widgets_from_config(self, cfg: Config) -> None:
        for name, _ in CHECKS:
            self.vars[name].set(getattr(cfg, name))
        for name, *_ in SLIDERS:
            self.vars[name].set(getattr(cfg, name))
            self._slider_label(name)
        self.tile_var.set("auto" if cfg.auto_tile else f"{cfg.tile}/{cfg.stride}")
        self.ref_var.set(cfg.reference)

    def param_changed(self) -> None:
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(350, self._apply_params)

    def _apply_params(self) -> None:
        self._pending = None
        try:
            self.cfg = self.config_from_widgets()
        except ValueError as e:
            messagebox.showerror("Invalid parameters", str(e))
            return
        if self.src is not None:
            self.reprocess_current()

    def load_defaults(self) -> None:
        self.cfg = Config()
        self.widgets_from_config(self.cfg)
        self.param_changed()

    def save_config(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            Path(p).write_text(json.dumps(self.config_from_widgets().to_dict(), indent=2))

    def load_config(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            self.cfg = Config.from_dict(json.loads(Path(p).read_text()))
        except (ValueError, TypeError) as e:
            messagebox.showerror("Invalid config", str(e))
            return
        self.widgets_from_config(self.cfg)
        self.param_changed()

    # ------------------------------------------------------------------ input
    def open_video(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Video", " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))), ("All", "*.*")])
        if p:
            self.open_path(p)

    def open_folder(self) -> None:
        p = filedialog.askdirectory()
        if p:
            self.open_path(p)

    def open_image(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Images", " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))), ("All", "*.*")])
        if p:
            self.open_path(p)

    def open_path(self, path: str) -> None:
        self.playing = False
        self.play_btn.configure(text="Play")
        try:
            src = FrameSource(path)
        except (IOError, ValueError) as e:
            messagebox.showerror("Cannot open", str(e))
            return
        if self.src is not None:
            self.src.close()
        self.src = src
        self.pos_scale.configure(to=max(0, src.n - 1))
        self.root.title(f"SAR survivor localisation - {Path(path).name}")
        self.seek(0)

    def _effective_cfg(self) -> Config:
        if self.src is not None and self.src.n < self.cfg.min_persistence and self.cfg.temporal_enabled:
            return replace(self.cfg, temporal_enabled=False)
        return self.cfg

    # -------------------------------------------------------------- processing
    def seek(self, i: int) -> None:
        if self.src is None:
            return
        self.idx = int(np.clip(i, 0, self.src.n - 1))
        self.reprocess_current()

    def reprocess_current(self) -> None:
        """Rebuild temporal state by replaying a short warm-up before the current frame."""
        if self.src is None:
            return
        cfg = self._effective_cfg()
        self.pipe.set_config(cfg)
        warm = (cfg.min_persistence + cfg.max_gap) if cfg.temporal_enabled else 0
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            for j in range(max(0, self.idx - warm), self.idx):
                self.pipe.process(self.src.get(j))
            self.frame = self.src.get(self.idx)
            self.result = self.pipe.process(self.frame, keep_stages=True)
        finally:
            self.root.configure(cursor="")
        self.show()

    def step(self) -> None:
        if self.src is None:
            return
        if self.idx >= self.src.n - 1:
            self.playing = False
            self.play_btn.configure(text="Play")
            return
        self.idx += 1
        self.frame = self.src.get(self.idx)
        self.result = self.pipe.process(self.frame, keep_stages=True)
        self.show()

    def restart(self) -> None:
        self.seek(0)

    def toggle_play(self) -> None:
        if self.src is None:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        self.step()
        if self.playing:
            self.root.after(1, self._tick)

    # ---------------------------------------------------------------- display
    def show(self) -> None:
        self.pos_var.set(self.idx)
        self.pos_label.configure(text=f"frame {self.idx + 1} / {self.src.n}")
        self.redraw()
        self.update_table()
        self.update_info()

    def current_stage(self) -> str:
        label = self.stage_var.get()
        for k, v in STAGE_LABELS.items():
            if v == label:
                return k
        return "final"

    def redraw(self) -> None:
        if self.result is None or self.frame is None:
            return
        img = render_stage(self.current_stage(), self.result, self.frame, self.pipe.cfg, frame_no=self.idx + 1)
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        h, w = img.shape[:2]
        s = min(cw / w, ch / h)
        disp = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")

    def update_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for r, c in enumerate(self.result.shortlist, 1):
            d = c.det
            self.tree.insert("", "end", values=(r, f"{c.conf:.2f}", int(d.x), int(d.y), int(d.w), int(d.h),
                                                c.track_id if c.track_id >= 0 else "-", c.hits))

    def update_info(self) -> None:
        res, cfg = self.result, self.pipe.cfg
        info = res.info
        lines = [f"processing scale: {res.scale:.3f}"]
        for i, (wl, ang, st) in enumerate(info.get("waves", [])):
            lines.append(f"wave band {i}: lambda={wl:.1f}px  theta={ang:.0f} deg  peak/mean={st:.0f}")
        lines.append(f"Gabor filters: {info.get('n_filters')}   tile/stride: {info.get('tile')}")
        lines.append(f"anomalous tiles: {info.get('n_anomalous_tiles')} / {info.get('n_tiles')}")
        lines.append(f"pixel threshold: {info.get('pixel_thresh', 0):.1f} (Lab distance)")
        lines.append(f"area limits: {info.get('area_limits')}")
        lines.append(f"camera shift: ({info.get('camera_shift', (0, 0))[0]:.1f}, {info.get('camera_shift', (0, 0))[1]:.1f})")
        lines.append(f"temporal: {'on' if cfg.temporal_enabled else 'off'}")
        blobs = info.get("blobs", [])
        rej: dict[str, int] = {}
        for b in blobs:
            if b.rejected:
                rej[b.rejected] = rej.get(b.rejected, 0) + 1
        lines.append(f"blobs: {len(blobs)}  kept: {len(res.detections)}  rejected: {rej or 0}")
        lines.append("")
        lines.append("timings (ms):")
        for k, v in res.timings.items():
            lines.append(f"  {k:<11}{1000 * v:7.1f}")
        self.info.delete("1.0", "end")
        self.info.insert("end", "\n".join(lines))

    # ----------------------------------------------------------------- export
    def export(self) -> None:
        if self.src is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if not p:
            return
        cfg = self._effective_cfg()
        src_path = self.src.path

        def work():
            try:
                src = FrameSource(src_path)
                pipe = Pipeline(replace(cfg))
                first = src.get(0)
                h, w = first.shape[:2]
                vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h))
                for i in range(src.n):
                    frame = first if i == 0 else src.get(i)
                    vw.write(draw_overlay(frame, pipe.process(frame)))
                    self._export_q.put(("progress", i + 1, src.n))
                vw.release()
                src.close()
                self._export_q.put(("done", p, None))
            except Exception as e:  # reported in the GUI
                self._export_q.put(("error", str(e), None))

        threading.Thread(target=work, daemon=True).start()
        self._poll_export()

    def _poll_export(self) -> None:
        try:
            while True:
                kind, a, b = self._export_q.get_nowait()
                if kind == "progress":
                    self.export_label.configure(text=f"exporting {a}/{b}")
                elif kind == "done":
                    self.export_label.configure(text=f"saved {Path(a).name}")
                    return
                else:
                    self.export_label.configure(text="export failed")
                    messagebox.showerror("Export failed", a)
                    return
        except queue.Empty:
            pass
        self.root.after(200, self._poll_export)


def main(initial: str | None = None) -> None:
    try:                                   # crisp rendering and correct sizes on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    App(root, initial)
    root.mainloop()


if __name__ == "__main__":
    main()
