"""SmolVLA base adapter with pluggable implementation binding."""

from __future__ import annotations

from .base import BaseModelAdapter


class SmolVLAAdapter(BaseModelAdapter):
    """Adapter for SmolVLA base policy inference."""

    model_name = "smolvla_base"
    checkpoint_env_var = "ROEP_SMOLVLA_BASE_CKPT"
    impl_env_var = "ROEP_SMOLVLA_BASE_IMPL"

    def _load_impl(self) -> None:
        self._load_plugin_impl(self.impl_env_var)

    def predict_action(self, obs_dict, lang_goal=None):
        return self._predict_via_loaded(obs_dict=obs_dict, lang_goal=lang_goal)
