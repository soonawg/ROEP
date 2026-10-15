"""Benchmark backend registry."""

from __future__ import annotations

from typing import Dict

from .base import BackendNotReadyError, EpisodeBackend, EpisodeRunConfig

_BACKENDS: Dict[str, EpisodeBackend] = {}


def _load_backend(name: str) -> EpisodeBackend:
    key = str(name or "").strip().lower()
    if key in _BACKENDS:
        return _BACKENDS[key]

    if key == "libero":
        from .libero_backend import LiberoBackend

        backend = LiberoBackend()
    elif key == "rlbench":
        from .rlbench_backend import RLBenchBackend

        backend = RLBenchBackend()
    else:
        supported = ", ".join(sorted(["libero", "rlbench"]))
        raise ValueError(f"Unsupported benchmark backend '{name}'. Supported: {supported}")

    _BACKENDS[key] = backend
    return backend


def get_backend(name: str) -> EpisodeBackend:
    key = str(name or "").strip().lower() or "rlbench"
    return _load_backend(key)


def __getattr__(name: str):
    if name == "LiberoBackend":
        from .libero_backend import LiberoBackend

        return LiberoBackend
    if name == "RLBenchBackend":
        from .rlbench_backend import RLBenchBackend

        return RLBenchBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BackendNotReadyError",
    "EpisodeBackend",
    "EpisodeRunConfig",
    "LiberoBackend",
    "RLBenchBackend",
    "get_backend",
]
