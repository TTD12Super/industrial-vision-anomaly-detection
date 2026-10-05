"""Small profiling helpers; they measure behavior without changing it."""

from __future__ import annotations

import platform
import resource
import time
from pathlib import Path
from typing import Any, Callable

import torch


def available_backends() -> dict[str, bool]:
    """Report whether each supported torch backend is available."""

    return {
        "cpu": True,
        "mps": bool(torch.backends.mps.is_available()),
        "cuda": bool(torch.cuda.is_available()),
    }


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous device work before reading a timer."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def memory_usage_mb() -> float:
    """Return the process high-water RSS in MiB."""

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return float(value / (1024 * 1024))
    return float(value / 1024)


def measure_callable(
    function: Callable[[], Any],
    device: torch.device | None = None,
) -> tuple[Any, float]:
    """Run one callable and return its value plus wall-clock seconds."""

    if device is not None:
        synchronize(device)
    start = time.perf_counter()
    value = function()
    if device is not None:
        synchronize(device)
    return value, time.perf_counter() - start


def file_size_mb(path: str | Path) -> float:
    """Return one artifact file size in MiB."""

    return float(Path(path).stat().st_size / (1024 * 1024))
