"""Apparent cloud/image motion estimation (not atmospheric wind/AMV)."""

from .diagnostics import image_space_diagnostics, robust_statistics
from .optical_flow import FarnebackConfig, MotionField, estimate_motion
from .visualization import plot_geographic_motion_overlay

__all__ = [
    "FarnebackConfig",
    "MotionField",
    "estimate_motion",
    "image_space_diagnostics",
    "robust_statistics",
    "plot_geographic_motion_overlay",
]
