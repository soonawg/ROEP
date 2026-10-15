"""RDT-170M adapter with pluggable implementation binding."""

from __future__ import annotations

from .base import BaseModelAdapter


class RDT170MAdapter(BaseModelAdapter):
    """Adapter for RDT-170M policy inference."""

    model_name = "rdt_170m"
    checkpoint_env_var = "ROEP_RDT_170M_CKPT"
    impl_env_var = "ROEP_RDT_170M_IMPL"

    def _load_impl(self) -> None:
        self._load_plugin_impl(self.impl_env_var)

    def predict_action(self, obs_dict, lang_goal=None):
        return self._predict_via_loaded(obs_dict=obs_dict, lang_goal=lang_goal)

