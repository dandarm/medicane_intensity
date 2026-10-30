"""Extract physical IR 10.8 brightness temperatures from EUMETSAT native ZIPs."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from satpy import Scene


DEFAULT_DEMETRA_ROOT = Path("/media/isacDisk1/Demetra")


def _demetra_helpers(demetra_root: Path):
    utils_dir = demetra_root / "moduli" / "videomae" / "medicane_utils"
    if not utils_dir.is_dir():
        raise FileNotFoundError(f"Directory utility Demetra non trovata: {utils_dir}")
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
    from create_airmassRGB_from_hrseviri_local import (  # type: ignore
        extract_nat_from_zip,
        slot_dt_from_zip_name,
        spatial_cut_geos,
    )
    from geo_const import lat_max, lat_min, lon_max, lon_min  # type: ignore

    return extract_nat_from_zip, slot_dt_from_zip_name, spatial_cut_geos, (
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )


def extract_ir108_from_zip(
    zip_path: str | Path,
    output_dir: str | Path,
    *,
    demetra_root: str | Path = DEFAULT_DEMETRA_ROOT,
    overwrite: bool = False,
) -> Path:
    """Extract cropped IR_108 brightness temperature and save compressed NPZ.

    Cropping/orientation deliberately reuse Demetra's own geometry utility, so
    IR output remains pixel-aligned with its existing Mediterranean RGB frames.
    """
    source = Path(zip_path)
    destination_dir = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_dir.mkdir(parents=True, exist_ok=True)
    root = Path(demetra_root)
    extract_nat, slot_from_name, spatial_cut, bounds = _demetra_helpers(root)
    timestamp_hint = slot_from_name(source.name)
    if timestamp_hint is not None:
        destination = destination_dir / f"ir108_{timestamp_hint:%Y%m%d_%H%M}.npz"
        if destination.exists() and destination.stat().st_size > 0 and not overwrite:
            return destination

    with tempfile.TemporaryDirectory(prefix="medicane_ir108_") as temp_dir:
        nat_path = extract_nat(source, Path(temp_dir))
        scene = Scene(reader="seviri_l1b_native", filenames=[str(nat_path)])
        scene.load(["IR_108"])
        data = scene["IR_108"].rename({"x": "x_geostationary", "y": "y_geostationary"})
        x_increasing = bool(data.x_geostationary[0] < data.x_geostationary[-1])
        y_increasing = bool(data.y_geostationary[0] < data.y_geostationary[-1])
        lat_min, lat_max, lon_min, lon_max = bounds
        cropped = spatial_cut(
            data,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            x_increasing=x_increasing,
            y_increasing=y_increasing,
            flip_north_up=True,
        )
        brightness_temperature = np.asarray(cropped.compute().data, dtype=np.float32)
        timestamp = data.attrs.get("start_time")
        if not isinstance(timestamp, datetime):
            timestamp = timestamp_hint
        if timestamp is None:
            raise RuntimeError(f"Impossibile ricavare il timestamp da {source.name}")
        # Match the RSS five-minute slot naming used by Demetra.
        timestamp = timestamp.replace(
            minute=(timestamp.minute // 5) * 5, second=0, microsecond=0
        )
        destination = destination_dir / f"ir108_{timestamp:%Y%m%d_%H%M}.npz"
        np.savez_compressed(
            destination,
            brightness_temperature_k=brightness_temperature,
            x_geostationary=np.asarray(cropped.x_geostationary.values),
            y_geostationary=np.asarray(cropped.y_geostationary.values),
            timestamp=np.array(timestamp.isoformat()),
            channel=np.array("IR_108"),
            units=np.array(str(data.attrs.get("units", "K"))),
            source_zip=np.array(source.name),
        )
    return destination


def load_ir108_npz(path: str | Path) -> tuple[np.ndarray, datetime, dict[str, str]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        image = np.asarray(payload["brightness_temperature_k"], dtype=np.float32)
        timestamp = datetime.fromisoformat(str(payload["timestamp"].item()))
        metadata = {
            "channel": str(payload["channel"].item()),
            "units": str(payload["units"].item()),
            "source_zip": str(payload["source_zip"].item()),
        }
    return image, timestamp, metadata

