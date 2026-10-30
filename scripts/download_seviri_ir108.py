#!/usr/bin/env python3
"""Reuse Demetra's RSS downloader, then extract IR_108 without running inference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medicane_intensity.seviri import extract_ir108_from_zip  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scarica SEVIRI RSS ogni 5 minuti via Demetra ed estrae IR_108."
    )
    parser.add_argument("--start", required=True, help="Inizio, formato accettato da Demetra.")
    parser.add_argument("--end", required=True, help="Fine, formato accettato da Demetra.")
    parser.add_argument(
        "--demetra-root", default="/media/isacDisk1/Demetra", type=Path
    )
    parser.add_argument(
        "--output-root", default=PROJECT_ROOT / "data" / "downloads", type=Path
    )
    parser.add_argument(
        "--ir108-dir", default=PROJECT_ROOT / "data" / "ir108", type=Path
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--read-timeout", type=int, default=180)
    parser.add_argument("--overwrite-ir108", action="store_true")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Elabora soltanto gli ZIP già presenti sotto --output-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloader = args.demetra_root / "scripts" / "download_and_track_range.py"
    if not downloader.is_file():
        raise SystemExit(f"Downloader Demetra non trovato: {downloader}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.ir108_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        command = [
            sys.executable,
            str(downloader),
            "--start",
            args.start,
            "--end",
            args.end,
            "--download_source",
            "eumetsat",
            "--output_root",
            str(args.output_root),
            "--skip_inference",
            "--eumetsat_download_workers",
            str(args.workers),
            "--eumetsat_download_retries",
            str(args.retries),
            "--eumetsat_read_timeout",
            str(args.read_timeout),
        ]
        subprocess.run(command, check=True)

    zip_files = sorted(args.output_root.glob("range_*/raw_eumetsat/*.zip"))
    if not zip_files:
        raise SystemExit(f"Nessuno ZIP EUMETSAT trovato sotto {args.output_root}")
    created = 0
    failed = 0
    for index, zip_path in enumerate(zip_files, start=1):
        try:
            output = extract_ir108_from_zip(
                zip_path,
                args.ir108_dir,
                demetra_root=args.demetra_root,
                overwrite=args.overwrite_ir108,
            )
            created += 1
            print(f"[{index}/{len(zip_files)}] {output.name}")
        except Exception as error:  # Continue long batch and report every bad product.
            failed += 1
            print(f"[{index}/{len(zip_files)}] ERRORE {zip_path.name}: {error}", file=sys.stderr)
    print(f"IR_108 disponibili: {created}; errori: {failed}; directory: {args.ir108_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

