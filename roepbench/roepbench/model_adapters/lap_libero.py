"""LAP-LIBERO adapter with pluggable implementation binding."""

from __future__ import annotations

from pathlib import Path

from .base import AdapterNotReadyError, BaseModelAdapter


class LAPLiberoAdapter(BaseModelAdapter):
    """Adapter for LAP-3B-Libero policy inference."""

    model_name = "lap_libero"
    checkpoint_env_var = "ROEP_LAP_LIBERO_CKPT"
    impl_env_var = "ROEP_LAP_LIBERO_IMPL"

    def ensure_ready(self) -> None:
        """Require a local checkpoint directory for LAP."""
        ckpt = str(self.config.checkpoint_path).strip()
        if not ckpt:
            raise AdapterNotReadyError(
                f"{self.model_name}: missing checkpoint path. "
                f"Set {self.checkpoint_env_var} or pass --checkpoint."
            )
        if not Path(ckpt).exists():
            raise AdapterNotReadyError(
                f"{self.model_name}: checkpoint does not exist locally: {ckpt}"
            )

    def _load_impl(self) -> None:
        self._load_plugin_impl(self.impl_env_var)

    def predict_action(self, obs_dict, lang_goal=None):
        return self._predict_via_loaded(obs_dict=obs_dict, lang_goal=lang_goal)
