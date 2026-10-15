"""LIBERO backend."""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, List, Tuple

from .base import BackendNotReadyError, EpisodeBackend, EpisodeRunConfig


def _resolve_callable(spec: str) -> Callable[[EpisodeRunConfig], Any]:
    if ":" not in spec:
        raise BackendNotReadyError(
            f"Invalid LIBERO runtime hook spec '{spec}'. Expected module.path:callable"
        )
    module_name, symbol_name = spec.split(":", 1)
    module_name = module_name.strip()
    symbol_name = symbol_name.strip()
    if not module_name or not symbol_name:
        raise BackendNotReadyError(
            f"Invalid LIBERO runtime hook spec '{spec}'. Both module and symbol are required."
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        raise BackendNotReadyError(
            f"LIBERO runtime hook import failed: {module_name} ({type(exc).__name__}: {exc})"
        ) from exc
    target = getattr(module, symbol_name, None)
    if not callable(target):
        raise BackendNotReadyError(f"LIBERO runtime hook is not callable: {spec}")
    return target


def _normalize_result(result: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(result, tuple) and len(result) == 2:
        episodes, steps = result
        if isinstance(episodes, list) and isinstance(steps, list):
            return episodes, steps
    if isinstance(result, dict):
        episodes = result.get("episode_records")
        steps = result.get("step_records")
        if isinstance(episodes, list) and isinstance(steps, list):
            return episodes, steps
    raise RuntimeError(
        "LIBERO runtime hook returned an invalid result. Expected "
        "(episode_records, step_records) or a dict with those keys."
    )


class LiberoBackend(EpisodeBackend):
    """Backend for LIBERO-based ROEPBench evaluation."""

    backend_name = "libero"

    def run_policy_episodes(self, config: EpisodeRunConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        task_name = str(config.libero_task_name or config.task).strip()
        suite = str(config.libero_suite).strip()
        if task_name == "":
            raise BackendNotReadyError(
                "LIBERO backend requires --libero_task_name or --task."
            )

        try:
            import libero  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise BackendNotReadyError(
                "LIBERO backend is selected but official LIBERO runtime is not installed or importable. "
                "Install LIBERO in the active environment or set "
                "ROEP_LIBERO_RUNTIME_HOOK=module.path:callable."
            ) from exc

        hook_spec = str(
            os.environ.get(
                "ROEP_LIBERO_RUNTIME_HOOK",
                "roepbench.libero.runtime_hook:run_libero_episodes",
            )
        ).strip()
        hook = _resolve_callable(hook_spec)
        return _normalize_result(hook(config))
