"""X-VLA adapter with pluggable implementation binding."""

from __future__ import annotations

from pathlib import Path

from .base import AdapterNotReadyError, BaseModelAdapter


class XVLAAdapter(BaseModelAdapter):
    """Adapter for X-VLA policy inference."""

    model_name = "xvla"
    checkpoint_env_var = "ROEP_XVLA_CKPT"
    impl_env_var = "ROEP_XVLA_IMPL"

    def ensure_ready(self) -> None:
        """Allow either a local checkpoint path or a remote repo id."""
        ckpt = str(self.config.checkpoint_path).strip()
        if ckpt == "":
            raise AdapterNotReadyError(
                f"{self.model_name}: missing checkpoint path. "
                f"Set {self.checkpoint_env_var} or pass --checkpoint."
            )
        if Path(ckpt).exists():
            return
        if "/" in ckpt and " " not in ckpt:
            return
        raise AdapterNotReadyError(f"{self.model_name}: checkpoint does not exist: {ckpt}")

    def _load_impl(self) -> None:
        self._load_plugin_impl(self.impl_env_var)

    def predict_action(self, obs_dict, lang_goal=None):
        return self._predict_via_loaded(obs_dict=obs_dict, lang_goal=lang_goal)
