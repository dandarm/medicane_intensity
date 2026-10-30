"""Dense optical flow for co-registered SEVIRI channel images."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import PreprocessConfig, local_texture, preprocess_pair


@dataclass(frozen=True)
class FarnebackConfig:
    pyr_scale: float = 0.5
    levels: int = 4
    winsize: int = 31
    iterations: int = 5
    poly_n: int = 7
    poly_sigma: float = 1.5
    flags: int = 0
    max_displacement_px: float = 30.0
    texture_window: int = 15
    min_texture_std: float = 2.0
    forward_backward: bool = True
    fb_max_error_px: float = 1.5


@dataclass
class MotionField:
    """Apparent displacement from ``timestamp_t`` to ``timestamp_t1``.

    ``dx`` is positive toward increasing image column (right); ``dy`` is
    positive toward increasing image row (down). Values are pixels per frame,
    not geophysical wind. Direction is ``atan2(dy, dx)`` in degrees.
    """

    dx: np.ndarray
    dy: np.ndarray
    speed_px: np.ndarray
    direction_deg: np.ndarray
    dt_seconds: float
    quality_mask: np.ndarray
    confidence: np.ndarray
    channel: str
    timestamp_t: datetime
    timestamp_t1: datetime
    forward_backward_error: np.ndarray | None = None

    @property
    def direction(self) -> np.ndarray:
        """Alias requested by the public API; values are degrees in image space."""
        return self.direction_deg

    def save_npz(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dx": self.dx,
            "dy": self.dy,
            "speed_px": self.speed_px,
            "direction_deg": self.direction_deg,
            "direction": self.direction_deg,
            "dt_seconds": np.float64(self.dt_seconds),
            "quality_mask": self.quality_mask,
            "confidence": self.confidence,
            "channel": np.array(self.channel),
            "timestamp_t": np.array(self.timestamp_t.isoformat()),
            "timestamp_t1": np.array(self.timestamp_t1.isoformat()),
        }
        if self.forward_backward_error is not None:
            payload["forward_backward_error"] = self.forward_backward_error
        np.savez_compressed(destination, **payload)
        return destination


def _farneback(first: np.ndarray, second: np.ndarray, cfg: FarnebackConfig) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        first,
        second,
        None,
        cfg.pyr_scale,
        cfg.levels,
        cfg.winsize,
        cfg.iterations,
        cfg.poly_n,
        cfg.poly_sigma,
        cfg.flags,
    )


def _sample_flow(flow: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    height, width = dx.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x + dx
    map_y = grid_y + dy
    sampled_x = cv2.remap(
        flow[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    sampled_y = cv2.remap(
        flow[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    return np.stack((sampled_x, sampled_y), axis=-1)


def estimate_motion(
    image_t: np.ndarray,
    image_t1: np.ndarray,
    timestamp_t: datetime,
    timestamp_t1: datetime,
    *,
    channel: str = "IR_108",
    valid_mask: np.ndarray | None = None,
    config: FarnebackConfig | None = None,
    preprocess_config: PreprocessConfig | None = None,
) -> MotionField:
    """Estimate dense apparent image displacement with Farneback flow."""
    cfg = config or FarnebackConfig()
    dt_seconds = (timestamp_t1 - timestamp_t).total_seconds()
    if dt_seconds <= 0:
        raise ValueError("timestamp_t1 must be later than timestamp_t")

    first, second, jointly_valid = preprocess_pair(
        image_t, image_t1, valid_mask=valid_mask, config=preprocess_config
    )
    forward = _farneback(first, second, cfg)
    dx = forward[..., 0].astype(np.float32)
    dy = forward[..., 1].astype(np.float32)
    speed = np.hypot(dx, dy).astype(np.float32)
    direction = np.degrees(np.arctan2(dy, dx)).astype(np.float32)

    texture = local_texture(first, cfg.texture_window)
    texture_ok = texture >= cfg.min_texture_std
    finite = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(speed)
    within_range = speed <= cfg.max_displacement_px
    quality = jointly_valid & texture_ok & finite & within_range
    texture_confidence = np.clip(texture / max(cfg.min_texture_std * 2.0, 1e-6), 0, 1)
    range_confidence = np.clip(1.0 - speed / max(cfg.max_displacement_px, 1e-6), 0, 1)
    confidence = texture_confidence * range_confidence

    fb_error = None
    if cfg.forward_backward:
        backward = _farneback(second, first, cfg)
        sampled_backward = _sample_flow(backward, dx, dy)
        fb_error = np.hypot(
            dx + sampled_backward[..., 0], dy + sampled_backward[..., 1]
        ).astype(np.float32)
        fb_ok = np.isfinite(fb_error) & (fb_error <= cfg.fb_max_error_px)
        quality &= fb_ok
        fb_confidence = np.exp(-np.square(fb_error / max(cfg.fb_max_error_px, 1e-6)))
        confidence *= np.nan_to_num(fb_confidence, nan=0.0)

    confidence = np.where(quality, confidence, 0.0).astype(np.float32)
    dx = np.where(quality, dx, np.nan).astype(np.float32)
    dy = np.where(quality, dy, np.nan).astype(np.float32)
    speed = np.where(quality, speed, np.nan).astype(np.float32)
    direction = np.where(quality, direction, np.nan).astype(np.float32)

    return MotionField(
        dx=dx,
        dy=dy,
        speed_px=speed,
        direction_deg=direction,
        dt_seconds=float(dt_seconds),
        quality_mask=quality,
        confidence=confidence,
        channel=channel,
        timestamp_t=timestamp_t,
        timestamp_t1=timestamp_t1,
        forward_backward_error=fb_error,
    )
