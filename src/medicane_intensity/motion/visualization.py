"""QA figures and animations for apparent image motion."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import image_space_diagnostics
from .optical_flow import MotionField


@lru_cache(maxsize=8)
def _coastline_polylines(
    image_width: int,
    image_height: int,
    demetra_root: str = "/media/isacDisk1/Demetra",
) -> tuple[np.ndarray, ...]:
    """Return Demetra's geostationary coastlines in image-pixel coordinates."""
    utils_dir = Path(demetra_root) / "moduli" / "videomae" / "medicane_utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from geo_const import create_basemap_obj  # type: ignore

    basemap = create_basemap_obj()
    if not getattr(basemap, "coastsegs", None):
        basemap.drawcoastlines()
    x_span = float(basemap.xmax - basemap.xmin)
    y_span = float(basemap.ymax - basemap.ymin)
    lines = []
    split_jump_px = max(80.0, 0.2 * float(max(image_width, image_height)))
    for segment in basemap.coastsegs:
        points = []
        for x_map, y_map in segment:
            x_pixel = (float(x_map) - float(basemap.xmin)) / x_span * (image_width - 1)
            y_pixel = (float(basemap.ymax) - float(y_map)) / y_span * (image_height - 1)
            if 0 <= x_pixel < image_width and 0 <= y_pixel < image_height:
                points.append((x_pixel, y_pixel))
        if len(points) < 2:
            continue
        current = [points[0]]
        for previous, point in zip(points, points[1:]):
            jump = float(np.hypot(point[0] - previous[0], point[1] - previous[1]))
            if jump > split_jump_px:
                if len(current) >= 2:
                    lines.append(np.asarray(current, dtype=np.float32))
                current = [point]
            else:
                current.append(point)
        if len(current) >= 2:
            lines.append(np.asarray(current, dtype=np.float32))
    return tuple(lines)


def _lonlat_to_pixel(
    longitude: float,
    latitude: float,
    image_width: int,
    image_height: int,
    demetra_root: str,
) -> tuple[float, float]:
    utils_dir = Path(demetra_root) / "moduli" / "videomae" / "medicane_utils"
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from geo_const import create_basemap_obj  # type: ignore

    basemap = create_basemap_obj()
    x_map, y_map = basemap(longitude, latitude)
    x_pixel = (float(x_map) - float(basemap.xmin)) / float(basemap.xmax - basemap.xmin)
    y_pixel = (float(basemap.ymax) - float(y_map)) / float(basemap.ymax - basemap.ymin)
    return x_pixel * (image_width - 1), y_pixel * (image_height - 1)


def plot_geographic_motion_overlay(
    image: np.ndarray,
    field: MotionField,
    output_path: str | Path | None = None,
    *,
    quiver_step: int = 15,
    arrow_exaggeration: float = 4.0,
    cyclone_center_latlon: tuple[float, float] | None = None,
    zoom_radius_px: tuple[int, int] | None = None,
    demetra_root: str = "/media/isacDisk1/Demetra",
):
    """Plot IR imagery, coastlines and apparent-motion vectors.

    ``cyclone_center_latlon`` uses ``(latitude, longitude)``. Arrow lengths may
    be visually exaggerated, while their colors retain the measured magnitude.
    """
    if quiver_step < 1:
        raise ValueError("quiver_step must be >= 1")
    if arrow_exaggeration <= 0:
        raise ValueError("arrow_exaggeration must be positive")
    image = np.asarray(image)
    if image.ndim != 2 or image.shape != field.dx.shape:
        raise ValueError("image and motion arrays must share the same 2-D shape")
    height, width = image.shape
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    shown = axis.imshow(image, cmap="gray_r", origin="upper")
    temperature_bar = fig.colorbar(shown, ax=axis, pad=0.015, shrink=0.82)
    temperature_bar.set_label("Brightness temperature IR 10.8 [K]")

    for line in _coastline_polylines(width, height, demetra_root):
        axis.plot(line[:, 0], line[:, 1], color="white", linewidth=0.75, alpha=0.9, zorder=3)

    yy, xx = np.mgrid[0:height:quiver_step, 0:width:quiver_step]
    uu = field.dx[::quiver_step, ::quiver_step]
    vv = field.dy[::quiver_step, ::quiver_step]
    magnitude = field.speed_px[::quiver_step, ::quiver_step]
    valid = field.quality_mask[::quiver_step, ::quiver_step] & np.isfinite(magnitude)
    vectors = axis.quiver(
        xx[valid],
        yy[valid],
        uu[valid],
        vv[valid],
        magnitude[valid],
        cmap="turbo",
        clim=(0, max(1.0, float(np.nanpercentile(field.speed_px, 98)))),
        angles="xy",
        scale_units="xy",
        scale=1.0 / arrow_exaggeration,
        width=0.0022,
        headwidth=3.5,
        headlength=4.5,
        zorder=4,
    )
    vector_bar = fig.colorbar(vectors, ax=axis, pad=0.055, shrink=0.82)
    vector_bar.set_label("Modulo del moto apparente [pixel/5 min]")
    axis.quiverkey(
        vectors,
        0.82,
        1.035,
        2.0,
        "2 pixel/5 min",
        coordinates="axes",
        labelpos="E",
    )

    center_pixel = None
    if cyclone_center_latlon is not None:
        latitude, longitude = cyclone_center_latlon
        center_pixel = _lonlat_to_pixel(
            longitude, latitude, width, height, demetra_root
        )
        axis.scatter(
            [center_pixel[0]], [center_pixel[1]], marker="x", s=140,
            linewidths=2.4, color="magenta", zorder=6,
            label=f"Centro Jolina ({latitude:.2f}°N, {longitude:.2f}°E)",
        )
        axis.legend(loc="lower left", facecolor="black", labelcolor="white", framealpha=0.7)

    if zoom_radius_px is not None:
        if center_pixel is None:
            raise ValueError("zoom_radius_px requires cyclone_center_latlon")
        radius_x, radius_y = zoom_radius_px
        axis.set_xlim(max(0, center_pixel[0] - radius_x), min(width - 1, center_pixel[0] + radius_x))
        # Image coordinates increase downward.
        axis.set_ylim(min(height - 1, center_pixel[1] + radius_y), max(0, center_pixel[1] - radius_y))
    else:
        axis.set_xlim(0, width - 1)
        axis.set_ylim(height - 1, 0)

    axis.set_title(
        f"Jolina — IR 10.8 e moto apparente delle nubi\n"
        f"{field.timestamp_t:%Y-%m-%d %H:%M} → {field.timestamp_t1:%H:%M} UTC; "
        f"lunghezza frecce ×{arrow_exaggeration:g}"
    )
    axis.set_xlabel("colonna SEVIRI")
    axis.set_ylabel("riga SEVIRI")
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180, facecolor="white")
    return fig


def plot_motion_qa(
    image: np.ndarray,
    field: MotionField,
    output_path: str | Path | None = None,
    *,
    quiver_step: int = 20,
):
    diagnostics = image_space_diagnostics(field)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title(f"{field.channel} + apparent displacement")
    yy, xx = np.mgrid[0 : image.shape[0] : quiver_step, 0 : image.shape[1] : quiver_step]
    uu = field.dx[::quiver_step, ::quiver_step]
    vv = field.dy[::quiver_step, ::quiver_step]
    axes[0, 0].quiver(xx, yy, uu, vv, color="yellow", angles="xy", scale_units="xy", scale=1)

    panels = (
        (axes[0, 1], diagnostics["magnitude_px"], "Magnitude [pixel/frame]", "viridis"),
        (axes[1, 0], diagnostics["vorticity_proxy"], "Vorticity proxy [pixel/pixel]", "RdBu_r"),
        (axes[1, 1], diagnostics["divergence_proxy"], "Divergence proxy [pixel/pixel]", "RdBu_r"),
    )
    for axis, data, title, cmap in panels:
        shown = axis.imshow(data, cmap=cmap)
        axis.set_title(title)
        fig.colorbar(shown, ax=axis, shrink=0.8)
    for axis in axes.flat:
        axis.set_axis_off()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160)
    return fig


def save_overlay_animation(
    images: list[np.ndarray],
    fields: list[MotionField],
    output_path: str | Path,
    *,
    quiver_step: int = 25,
    fps: float = 2.0,
) -> Path:
    if len(images) != len(fields):
        raise ValueError("Provide one motion field for every displayed image")
    rendered = []
    for image, field in zip(images, fields):
        fig, axis = plt.subplots(figsize=(8, 6))
        axis.imshow(image, cmap="gray")
        yy, xx = np.mgrid[0 : image.shape[0] : quiver_step, 0 : image.shape[1] : quiver_step]
        axis.quiver(
            xx, yy, field.dx[::quiver_step, ::quiver_step],
            field.dy[::quiver_step, ::quiver_step], color="yellow",
            angles="xy", scale_units="xy", scale=1,
        )
        axis.set_axis_off()
        fig.canvas.draw()
        rendered.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(destination, rendered, duration=1000.0 / fps, loop=0)
    return destination
