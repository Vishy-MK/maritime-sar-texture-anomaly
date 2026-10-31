"""Pipeline parameters.

Every tunable number in the system lives here so that the GUI, the CLI and the
evaluation harness all drive the same object. Defaults are chosen for frames of
roughly 1280-2048 px width in which a person spans 15-40 px.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, fields


@dataclass
class Config:
    # ---- general -------------------------------------------------------------
    max_width: int = 1920          # frames wider than this are downscaled for processing
    top_k: int = 5                 # length of the shortlist shown to the operator

    # ---- stage 1: normalisation ---------------------------------------------
    flatten_sigma: float = 0.125   # flat-field blur sigma as a fraction of frame width (0 = off)
    clahe_clip: float = 2.0        # CLAHE contrast limit
    clahe_grid: int = 8            # CLAHE tiles per side
    stretch_low_pct: float = 1.0   # piecewise-linear stretch: input percentile mapped to out_low
    stretch_high_pct: float = 99.0 # input percentile mapped to out_high
    stretch_out_low: int = 10      # output level for the low knee
    stretch_out_high: int = 245    # output level for the high knee

    # ---- stage 2: texture characterisation ----------------------------------
    auto_tile: bool = True         # pick tile/stride from frame width (or altitude) instead of the two below
    tile: int = 32                 # tile side (px, processing resolution)
    stride: int = 16               # tile step; tile must be a multiple of stride
    n_orientations: int = 4        # Gabor orientations, offsets from the dominant wave direction
    wave_bands: int = 3            # horizontal bands in which the wave spectrum is estimated
    min_wavelength: float = 4.0    # px, clamp for the estimated wave wavelength
    max_wavelength: float = 32.0   # px
    gabor_gamma: float = 0.5       # spatial aspect ratio
    gabor_sigma_ratio: float = 0.56  # sigma = ratio * lambda (about one octave bandwidth)
    lbp_weight: float = 0.5        # weight of the LBP distance in the fused score (Gabor gets 1 - w)

    # ---- stage 3: anomaly scoring -------------------------------------------
    reference: str = "local"       # "global", "band" or "local": where the normal-sea reference comes from
    reference_window: int = 9      # window side (tiles) of the band / local reference
    anomaly_z: float = 3.0         # tile threshold, robust z-score (median + z * 1.4826 MAD)
    pixel_z: float = 4.0           # pixel deviation threshold inside anomalous tiles
    background_ksize: int = 31     # median-blur size used to estimate the local sea colour
    score_scale: float = 50.0      # anomaly score s maps to confidence s / (s + score_scale)

    # ---- stage 4: glint suppression -----------------------------------------
    glint_enabled: bool = True
    glint_value: float = 0.80      # mean brightness (0-1) above which a pixel can be glint
    glint_chroma: float = 0.06     # rgb-chromaticity distance from grey below which it is glint
    glint_component_frac: float = 0.5  # component dropped if this fraction of it is glint

    # ---- stage 5: morphological refinement ----------------------------------
    morph_enabled: bool = True
    open_ksize: int = 3
    close_ksize: int = 5
    min_area: int = 12             # px at processing resolution
    max_area: int = 2500
    max_eccentricity: float = 0.97
    min_solidity: float = 0.45

    # ---- stage 6: temporal association --------------------------------------
    temporal_enabled: bool = True
    match_radius: float = 40.0     # px at processing resolution
    min_persistence: int = 5       # hits needed before a track is promoted (~0.5 s at 10 fps;
                                   # raise it with frame rate so it outlasts a whitecap)
    max_gap: int = 3               # frames a track may go unseen before it is dropped
    drift_sigma: float = 6.0       # px; step-to-step velocity spread tolerated for "coherent drift"
    min_coherence: float = 0.25
    persistence_horizon: int = 10  # hits at which the persistence term saturates
    anomaly_weight: float = 0.4    # weight of anomaly vs persistence in the ranking
    motion_compensation: bool = False  # estimate camera translation by phase correlation

    # ---- optional geometry ---------------------------------------------------
    altitude_m: float | None = None  # if set, area limits are derived from it
    hfov_deg: float = 70.0           # camera horizontal field of view used with altitude
    use_altitude_metadata: bool = False  # evaluation: take altitude_m from dataset metadata per frame

    def validate(self) -> "Config":
        if self.stride <= 0 or self.tile <= 0:
            raise ValueError("tile and stride must be positive")
        if self.tile % self.stride:
            raise ValueError(f"tile ({self.tile}) must be a multiple of stride ({self.stride})")
        if not 0.0 <= self.lbp_weight <= 1.0:
            raise ValueError("lbp_weight must be in [0, 1]")
        if not 0.0 <= self.anomaly_weight <= 1.0:
            raise ValueError("anomaly_weight must be in [0, 1]")
        if self.reference not in ("global", "band", "local"):
            raise ValueError("reference must be 'global', 'band' or 'local'")
        if self.min_area > self.max_area:
            raise ValueError("min_area must not exceed max_area")
        if self.score_scale <= 0:
            raise ValueError("score_scale must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.n_orientations < 1:
            raise ValueError("n_orientations must be at least 1")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names}).validate()

    def gsd(self, image_width: int) -> float | None:
        """Ground sampling distance (metres per pixel) from altitude and field of view."""
        if self.altitude_m is None or self.altitude_m <= 0:
            return None
        ground_width = 2.0 * self.altitude_m * math.tan(math.radians(self.hfov_deg) / 2.0)
        return ground_width / float(image_width)

    def resolved(self, image_width: int) -> "Config":
        """Copy with tile/stride fixed for a frame of this processing width.

        The tile should be about twice the expected target size: much larger
        and the person barely changes the tile histogram, much smaller and the
        texture statistics are unreliable. With altitude the visible part of a
        person (~0.5 m) is converted to pixels; without it the tile scales with
        the frame width (32 px at 1920 px). Tiles are multiples of 8 in [16, 64]
        and the stride is half the tile.
        """
        if not self.auto_tile:
            return self
        g = self.gsd(image_width)
        target = 2.5 * 0.5 / g if g else 32.0 * image_width / 1920.0
        tile = int(min(64, max(16, 8 * round(target / 8.0))))
        c = Config.from_dict(self.to_dict())
        c.tile, c.stride = tile, tile // 2
        return c.validate()

    def area_limits(self, image_width: int) -> tuple[int, int]:
        """Blob area limits in processing pixels.

        Without altitude the configured limits are used. With altitude the ground
        sampling distance (metres per pixel) gives the expected size of the
        visible part of a person (about 0.3-0.6 m across), and the limits are set
        a factor of two either side of that.
        """
        gsd = self.gsd(image_width)
        if gsd is None:
            return self.min_area, self.max_area
        lo_px = 0.3 / gsd
        hi_px = 0.6 / gsd
        lo = max(4, int((lo_px / 2.0) ** 2))
        hi = max(lo + 1, int((hi_px * 2.0) ** 2))
        return lo, hi
