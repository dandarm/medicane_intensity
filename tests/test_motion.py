from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medicane_intensity.motion import FarnebackConfig, estimate_motion  # noqa: E402


class MotionSyntheticTest(unittest.TestCase):
    def test_known_translation_and_sign_convention(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.normal(0, 1, (192, 192)).astype(np.float32)
        image = cv2.GaussianBlur(image, (0, 0), 2.0)
        dx_true, dy_true = 3.0, -2.0
        translated = cv2.warpAffine(
            image,
            np.float32([[1, 0, dx_true], [0, 1, dy_true]]),
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        timestamp = datetime(2026, 1, 1)
        field = estimate_motion(
            image,
            translated,
            timestamp,
            timestamp + timedelta(minutes=5),
            config=FarnebackConfig(
                min_texture_std=1.0,
                max_displacement_px=10,
                fb_max_error_px=0.75,
            ),
        )
        interior = np.zeros_like(image, dtype=bool)
        interior[24:-24, 24:-24] = True
        valid = interior & field.quality_mask
        self.assertGreater(float(np.mean(valid)), 0.45)
        self.assertAlmostEqual(float(np.nanmedian(field.dx[valid])), dx_true, delta=0.25)
        self.assertAlmostEqual(float(np.nanmedian(field.dy[valid])), dy_true, delta=0.25)
        self.assertEqual(field.dt_seconds, 300.0)

    def test_rejects_non_positive_time_step(self) -> None:
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        timestamp = datetime(2026, 1, 1)
        with self.assertRaises(ValueError):
            estimate_motion(image, image, timestamp, timestamp)


if __name__ == "__main__":
    unittest.main()

