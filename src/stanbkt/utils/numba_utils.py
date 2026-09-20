"""Helpers for choosing numba parallel settings that are safe on the current machine."""

from __future__ import annotations

import importlib
import os


def _layer_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def threadsafe_parallel_available() -> bool:
    """Whether numba's parallel kernels can be called from several Python threads.

    The tbb and omp threading layers are safe for concurrent launches, workqueue is not.
    Returns False when only workqueue is usable, so callers can fall back to serial.
    """
    requested = os.environ.get("NUMBA_THREADING_LAYER", "").lower()
    if requested in ("tbb", "omp"):
        return True
    if requested == "workqueue":
        return False
    if _layer_available("numba.np.ufunc.tbbpool"):
        return True
    return _layer_available("numba.np.ufunc.omppool")


PARALLEL_SAFE = threadsafe_parallel_available()
