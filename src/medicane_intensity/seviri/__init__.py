"""SEVIRI download integration and channel extraction."""

from .ir108 import extract_ir108_from_zip, load_ir108_npz

__all__ = ["extract_ir108_from_zip", "load_ir108_npz"]

