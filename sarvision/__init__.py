"""Vision-assisted survivor localisation in maritime search-and-rescue
using texture-anomaly modelling (classical OpenCV pipeline, no training)."""

from .config import Config
from .pipeline import FrameResult, Pipeline

__all__ = ["Config", "Pipeline", "FrameResult"]
__version__ = "1.0.0"
