"""Image-space kinematic proxies and robust summaries."""

from __future__ import annotations

import numpy as np

from .optical_flow import MotionField


def image_space_diagnostics(field: MotionField) -> dict[str, np.ndarray]:
    """Compute derivatives per image pixel from displacement per frame.

    These are image-space proxies, not geographic divergence or vorticity.
    """
    u = np.asarray(field.dx, dtype=np.float32)
    v = np.asarray(field.dy, dtype=np.float32)
    # Gradient order is (row/y, column/x).
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    valid = field.quality_mask
    return {
        "magnitude_px": np.where(valid, np.hypot(u, v), np.nan).astype(np.float32),
        "divergence_proxy": np.where(valid, du_dx + dv_dy, np.nan).astype(np.float32),
        "vorticity_proxy": np.where(valid, dv_dx - du_dy, np.nan).astype(np.float32),
    }


def robust_statistics(field: MotionField) -> dict[str, float]:
    valid = field.quality_mask & np.isfinite(field.speed_px)
    values = field.speed_px[valid]
    stats = {
        "valid_fraction": float(np.mean(valid)),
        "valid_vectors": int(np.sum(valid)),
        "total_pixels": int(valid.size),
    }
    if values.size:
        stats.update(
            speed_median_px=float(np.median(values)),
            speed_p05_px=float(np.percentile(values, 5)),
            speed_p95_px=float(np.percentile(values, 95)),
            dx_median_px=float(np.nanmedian(field.dx)),
            dy_median_px=float(np.nanmedian(field.dy)),
        )
    return stats

