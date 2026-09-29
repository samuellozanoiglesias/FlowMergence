"""
backend.py — Numerical backend selection.

All code uses `xp` as an alias for the array module, so the same code runs on CPU
(NumPy) or GPU (CuPy, whose API is compatible). cupy >= 12 is recommended.
CuPy is imported lazily so that CPU workers do not pay the import cost.
"""
from __future__ import annotations

import warnings

import numpy as np

_cp = None


def _try_cupy():
    global _cp
    if _cp is None:
        try:
            import cupy  # noqa: F401
            _cp = cupy
        except Exception:
            _cp = False
    return _cp or None


def get_xp(name: str = "numpy"):
    if name == "cupy":
        cp = _try_cupy()
        if cp is None:
            warnings.warn("CuPy not available: falling back to NumPy.")
            return np
        return cp
    return np


def get_array_module(a):
    """Return numpy or cupy depending on the type of `a`, without importing CuPy needlessly."""
    if type(a).__module__.startswith("cupy"):
        return _try_cupy()
    return np


def to_numpy(a):
    if type(a).__module__.startswith("cupy"):
        return _try_cupy().asnumpy(a)
    return np.asarray(a)


def to_float(a) -> float:
    return float(to_numpy(a))
