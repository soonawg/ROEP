"""Octo-Small adapter with pluggable implementation binding."""

from __future__ import annotations

from .base import BaseModelAdapter


class OctoSmallAdapter(BaseModelAdapter):
    """Adapter for Octo-Small policy inference."""

    model_name = "octo_small"
    checkpoint_env_var = "ROEP_OCTO_SMALL_CKPT"
    impl_env_var = "ROEP_OCTO_SMALL_IMPL"

    def _load_impl(self) -> None:
        self._load_plugin_impl(self.impl_env_var)

    def predict_action(self, obs_dict, lang_goal=None):
        return self._predict_via_loaded(obs_dict=obs_dict, lang_goal=lang_goal)
