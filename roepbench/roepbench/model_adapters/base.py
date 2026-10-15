"""Base interfaces for model-specific policy adapters."""

from __future__ import annotations

import importlib
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class AdapterNotReadyError(RuntimeError):
    """Raised when adapter prerequisites are missing."""


@dataclass
class AdapterConfig:
    checkpoint_path: str
    device: str = "cuda"


class BaseModelAdapter(ABC):
    """Abstract base class for external model inference adapters."""

    model_name: str = "base"
    checkpoint_env_var: str = ""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self._loaded = False
        self._predict_fn: Optional[Callable[[Dict[str, Any], Optional[str]], Any]] = None

    def ensure_ready(self) -> None:
        """Validate runtime prerequisites before any rollout starts."""
        ckpt = str(self.config.checkpoint_path).strip()
        if ckpt == "":
            raise AdapterNotReadyError(
                f"{self.model_name}: missing checkpoint path. "
                f"Set {self.checkpoint_env_var} or pass --checkpoint."
            )
        path = Path(ckpt)
        if not path.exists():
            raise AdapterNotReadyError(
                f"{self.model_name}: checkpoint does not exist: {path}"
            )

    def load(self) -> None:
        """Load model weights into memory."""
        self.ensure_ready()
        self._load_impl()
        if self._predict_fn is None:
            raise AdapterNotReadyError(
                f"{self.model_name}: load completed but predict function is not configured."
            )
        self._loaded = True

    @abstractmethod
    def _load_impl(self) -> None:
        """Backend-specific loading implementation."""

    @abstractmethod
    def predict_action(
        self,
        obs_dict: Dict[str, Any],
        lang_goal: Optional[str] = None,
    ) -> Any:
        """Return policy action for current observation."""

    def _set_predict_fn(self, fn: Callable[[Dict[str, Any], Optional[str]], Any]) -> None:
        self._predict_fn = fn

    def _predict_via_loaded(self, obs_dict: Dict[str, Any], lang_goal: Optional[str] = None) -> Any:
        if self._predict_fn is None:
            raise AdapterNotReadyError(f"{self.model_name}: predict function is not initialized.")
        return self._predict_fn(obs_dict, lang_goal)

    @staticmethod
    def _resolve_symbol(spec: str) -> Any:
        if ":" not in spec:
            raise AdapterNotReadyError(
                f"Invalid adapter impl spec '{spec}'. Expected format 'module.submodule:Symbol'."
            )
        module_name, symbol_name = spec.split(":", 1)
        module_name = module_name.strip()
        symbol_name = symbol_name.strip()
        if not module_name or not symbol_name:
            raise AdapterNotReadyError(
                f"Invalid adapter impl spec '{spec}'. Both module and symbol are required."
            )
        try:
            module = importlib.import_module(module_name)
        except (ModuleNotFoundError, ImportError) as exc:
            raise AdapterNotReadyError(
                f"Adapter impl module import failed: {module_name} ({type(exc).__name__}: {exc})"
            ) from exc
        if not hasattr(module, symbol_name):
            raise AdapterNotReadyError(
                f"Adapter impl symbol not found: {module_name}:{symbol_name}"
            )
        return getattr(module, symbol_name)

    def _load_plugin_impl(self, impl_env_var: str) -> None:
        spec = str(os.environ.get(impl_env_var, "")).strip()
        if spec == "":
            raise AdapterNotReadyError(
                f"{self.model_name}: missing implementation binding. "
                f"Set {impl_env_var}='module.path:factory_or_class'."
            )

        target = self._resolve_symbol(spec)
        loaded_obj: Any
        if inspect.isclass(target):
            loaded_obj = target(
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device,
                model_name=self.model_name,
            )
        elif callable(target):
            loaded_obj = target(
                checkpoint_path=self.config.checkpoint_path,
                device=self.config.device,
                model_name=self.model_name,
            )
        else:
            raise AdapterNotReadyError(
                f"{self.model_name}: impl symbol '{spec}' is not callable/class."
            )

        if hasattr(loaded_obj, "predict_action") and callable(getattr(loaded_obj, "predict_action")):
            self._set_predict_fn(getattr(loaded_obj, "predict_action"))
            return
        if callable(loaded_obj):
            self._set_predict_fn(loaded_obj)
            return
        raise AdapterNotReadyError(
            f"{self.model_name}: loaded impl from '{spec}' has no callable predict interface."
        )
