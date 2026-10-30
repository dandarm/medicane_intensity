#!/usr/bin/env python3
"""Estimate and visualize motion from two consecutive IR_108 NPZ files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medicane_intensity.motion import estimate_motion, robust_statistics  # noqa: E402
from medicane_intensity.motion.visualization import (  # noqa: E402
    plot_geographic_motion_overlay,
    plot_motion_qa,
)
from medicane_intensity.seviri import load_ir108_npz  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--coastlines", action="store_true")
    parser.add_argument("--center-lat", type=float)
    parser.add_argument("--center-lon", type=float)
    parser.add_argument("--zoom-radius-x", type=int)
    parser.add_argument("--zoom-radius-y", type=int)
    parser.add_argument("--quiver-step", type=int, default=15)
    parser.add_argument("--arrow-exaggeration", type=float, default=4.0)
    args = parser.parse_args()
    first, timestamp_t, metadata_t = load_ir108_npz(args.first)
    second, timestamp_t1, metadata_t1 = load_ir108_npz(args.second)
    if metadata_t["channel"] != metadata_t1["channel"]:
        raise SystemExit("I due frame appartengono a canali differenti")

    field = estimate_motion(
        first,
        second,
        timestamp_t,
        timestamp_t1,
        channel=metadata_t["channel"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"motion_{timestamp_t:%Y%m%d_%H%M}_{timestamp_t1:%Y%m%d_%H%M}"
    npz_path = field.save_npz(args.output_dir / f"{stem}.npz")
    center = None
    if (args.center_lat is None) != (args.center_lon is None):
        raise SystemExit("--center-lat e --center-lon devono essere forniti insieme")
    if args.center_lat is not None:
        center = (args.center_lat, args.center_lon)
    zoom = None
    if (args.zoom_radius_x is None) != (args.zoom_radius_y is None):
        raise SystemExit("--zoom-radius-x e --zoom-radius-y devono essere forniti insieme")
    if args.zoom_radius_x is not None:
        zoom = (args.zoom_radius_x, args.zoom_radius_y)
    if args.coastlines:
        figure_path = args.output_dir / f"{stem}_coasts_overlay.png"
        figure = plot_geographic_motion_overlay(
            first,
            field,
            figure_path,
            quiver_step=args.quiver_step,
            arrow_exaggeration=args.arrow_exaggeration,
            cyclone_center_latlon=center,
            zoom_radius_px=zoom,
        )
    else:
        figure_path = args.output_dir / f"{stem}_qa.png"
        figure = plot_motion_qa(first, field, figure_path)
    figure.clear()
    stats = robust_statistics(field)
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Campo: {npz_path}")
    print(f"QA: {figure_path}")


if __name__ == "__main__":
    main()
