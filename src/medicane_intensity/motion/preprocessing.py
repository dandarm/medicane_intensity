"""Radiometrically consistent preprocessing for optical flow."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0
    gaussian_sigma: float = 0.8


def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}")
    return array


def preprocess_pair(
    image_t: np.ndarray,
    image_t1: np.ndarray,
    valid_mask: np.ndarray | None = None,
    config: PreprocessConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a pair with one shared radiometric transform.

    Invalid values are filled only for OpenCV. The returned mask keeps them out
    of the final field. Using joint percentiles avoids independently changing
    the contrast of consecutive physical-channel images.
    """
    cfg = config or PreprocessConfig()
    first = _validate_image(image_t, "image_t")
    second = _validate_image(image_t1, "image_t1")
    if first.shape != second.shape:
        raise ValueError(f"Image shapes differ: {first.shape} vs {second.shape}")
    if not 0 <= cfg.lower_percentile < cfg.upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    valid = np.isfinite(first) & np.isfinite(second)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != first.shape:
            raise ValueError("valid_mask must have the same shape as the images")
        valid &= supplied
    if not np.any(valid):
        raise ValueError("No jointly valid pixels in the image pair")

    combined = np.concatenate((first[valid], second[valid]))
    low, high = np.percentile(combined, [cfg.lower_percentile, cfg.upper_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(combined))
        high = float(np.max(combined))
    if high <= low:
        raise ValueError("The image pair has no radiometric contrast")

    fill = float(np.median(combined))

    def transform(array: np.ndarray) -> np.ndarray:
        # Do not let explicitly masked pixels influence neighbouring flow.
        work = np.where(valid, array, fill)
        work = np.clip(work, low, high)
        work = ((work - low) * (255.0 / (high - low))).astype(np.float32)
        if cfg.gaussian_sigma > 0:
            work = cv2.GaussianBlur(work, (0, 0), cfg.gaussian_sigma)
        return work

    return transform(first), transform(second), valid


def local_texture(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """Return local standard deviation in normalized-image units."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    image = np.asarray(image, dtype=np.float32)
    mean = cv2.boxFilter(image, -1, (window_size, window_size), normalize=True)
    mean_sq = cv2.boxFilter(image * image, -1, (window_size, window_size), normalize=True)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)
