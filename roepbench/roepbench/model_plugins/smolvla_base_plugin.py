"""SmolVLA base plugin with native LeRobot loading when available."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..model_adapters.base import AdapterNotReadyError
from .common import to_env_action
from .runtime_bridge import maybe_build_runtime_bridge_hook


_ROBOT_STATE_SPECS = {
    "eef": {
        "pos": 3,
        "quat": 4,
    },
    "gripper": {
        "qpos": 2,
        "qvel": 2,
    },
    "joints": {
        "pos": 7,
        "vel": 7,
    },
}


def _resolve_callable(spec: str) -> Callable[[Dict[str, Any], Optional[str]], Any]:
    if ":" not in spec:
        raise AdapterNotReadyError(
            f"Invalid SmolVLA runtime hook spec '{spec}'. Expected module.path:callable"
        )
    module_name, symbol_name = spec.split(":", 1)
    module_name = module_name.strip()
    symbol_name = symbol_name.strip()
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        raise AdapterNotReadyError(
            f"SmolVLA runtime hook import failed: {module_name} ({type(exc).__name__}: {exc})"
        ) from exc
    target = getattr(module, symbol_name, None)
    if not callable(target):
        raise AdapterNotReadyError(f"SmolVLA runtime hook is not callable: {spec}")
    return target


def _batched_float_array(value: Any, expected_dim: int) -> np.ndarray:
    """Convert a leaf robot-state value into a float32 array with an explicit batch axis."""
    if value is None:
        return np.zeros((1, int(expected_dim)), dtype=np.float32)

    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.shape[-1] != int(expected_dim) and arr.size % int(expected_dim) == 0:
        arr = arr.reshape(-1, int(expected_dim))

    if arr.shape[-1] != int(expected_dim):
        raise ValueError(
            f"SmolVLA robot-state leaf expected trailing dim {expected_dim}, got shape {tuple(arr.shape)}"
        )
    return arr.astype(np.float32, copy=False)


def _normalize_robot_state(obs_dict: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Normalize ROEPBench robot_state payloads to LeRobot's batched LIBERO format."""
    raw_state = obs_dict.get("robot_state")
    if not isinstance(raw_state, dict):
        raw_state = {}

    normalized: Dict[str, Dict[str, np.ndarray]] = {}
    for group_name, leaves in _ROBOT_STATE_SPECS.items():
        raw_group = raw_state.get(group_name)
        if not isinstance(raw_group, dict):
            raw_group = {}
        normalized[group_name] = {}
        for leaf_name, expected_dim in leaves.items():
            fallback_key = f"{group_name.rstrip('s')}_{leaf_name}"
            if group_name == "eef":
                fallback_key = f"eef_{leaf_name}"
            elif group_name == "gripper":
                fallback_key = f"gripper_{leaf_name}"
            elif group_name == "joints":
                fallback_key = f"joint_{leaf_name}"
            normalized[group_name][leaf_name] = _batched_float_array(
                raw_group.get(leaf_name, obs_dict.get(fallback_key)),
                expected_dim=expected_dim,
            )
    return normalized


class _SmolVLAPluginAdapter:
    def __init__(self, checkpoint_path: str, device: str, model_name: str):
        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.model_name = str(model_name)

        self._hook: Optional[Callable[[Dict[str, Any], Optional[str]], Any]] = None
        self._torch: Any = None
        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None
        self._env_preprocessor: Any = None
        self._env_postprocessor: Any = None
        self._preprocess_observation: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self._action_key: str = "action"

        hook_spec = str(os.environ.get("ROEP_SMOLVLA_BASE_RUNTIME_HOOK", "")).strip()
        if hook_spec:
            self._hook = _resolve_callable(hook_spec)
            return

        bridge_hook = maybe_build_runtime_bridge_hook(
            model_key="smolvla_base",
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        if bridge_hook is not None:
            self._hook = bridge_hook
            return

        self._try_native_load()

    def _try_native_load(self) -> None:
        try:
            import torch  # type: ignore
            from lerobot.configs.policies import PreTrainedConfig  # type: ignore
            from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # type: ignore
            from lerobot.envs.factory import make_env_pre_post_processors  # type: ignore
            from lerobot.envs.utils import preprocess_observation  # type: ignore
            from lerobot.policies.factory import make_policy, make_pre_post_processors  # type: ignore
            from lerobot.utils.constants import ACTION  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "SmolVLA native load failed. Install LeRobot/SmolVLA deps in the model env or set "
                "ROEP_SMOLVLA_BASE_RUNTIME_HOOK=module.path:callable."
            ) from exc

        resolved_device = str(self.device).strip().lower() or "cpu"
        if resolved_device.startswith("cuda") and not bool(torch.cuda.is_available()):
            resolved_device = "cpu"
        self.device = resolved_device

        checkpoint = Path(self.checkpoint_path)
        local_only = checkpoint.exists()
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        cli_overrides = [f"--device={self.device}", "--use_amp=false"]
        try:
            policy_cfg = PreTrainedConfig.from_pretrained(
                self.checkpoint_path,
                local_files_only=local_only,
                cli_overrides=cli_overrides,
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "SmolVLA config load failed. Check checkpoint root/files and LeRobot compatibility. "
                f"checkpoint={self.checkpoint_path}"
            ) from exc

        policy_cfg.pretrained_path = checkpoint if local_only else Path(self.checkpoint_path)
        resolution = int(str(os.environ.get("ROEP_LIBERO_CAMERA_RESOLUTION", "256")).strip() or "256")
        env_cfg = LiberoEnvConfig(
            task=str(os.environ.get("ROEP_LIBERO_SUITE", "libero_spatial")),
            obs_type="pixels_agent_pos",
            observation_height=resolution,
            observation_width=resolution,
            control_mode=str(os.environ.get("ROEP_LIBERO_CONTROL_MODE", "relative")),
        )
        try:
            processor_device_override = {
                "device_processor": {
                    "device": self.device,
                }
            }
            self._policy = make_policy(policy_cfg, env_cfg=env_cfg)
            self._preprocessor, self._postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=self.checkpoint_path,
                preprocessor_overrides=processor_device_override,
                postprocessor_overrides=processor_device_override,
            )
            self._env_preprocessor, self._env_postprocessor = make_env_pre_post_processors(
                env_cfg=env_cfg,
                policy_cfg=policy_cfg,
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "SmolVLA native policy load failed. Check checkpoint, processor configs, and env feature compatibility. "
                f"checkpoint={self.checkpoint_path}"
            ) from exc

        self._torch = torch
        self._preprocess_observation = preprocess_observation
        self._action_key = ACTION

    @staticmethod
    def _to_hwc_uint8(image: Any) -> Optional[np.ndarray]:
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim != 3:
            return None
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr

    def _build_env_observation(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        front = self._to_hwc_uint8(
            obs_dict.get("front_rgb", obs_dict.get("agentview_rgb"))
        )
        wrist = self._to_hwc_uint8(obs_dict.get("wrist_rgb"))
        if front is None:
            raise RuntimeError("SmolVLA plugin: no front/agentview RGB observation available.")

        pixels = {"image": front}
        if wrist is not None:
            pixels["image2"] = wrist
        return {
            "pixels": pixels,
            "robot_state": _normalize_robot_state(obs_dict),
        }

    def predict_action(self, obs_dict: Dict[str, Any], lang_goal: Optional[str] = None):
        if self._hook is not None:
            return to_env_action(self._hook(obs_dict, lang_goal))

        assert self._policy is not None
        assert self._preprocess_observation is not None
        observation = self._build_env_observation(obs_dict)
        observation = self._preprocess_observation(observation)
        observation["task"] = str(lang_goal or "")
        observation = self._env_preprocessor(observation)
        observation = self._preprocessor(observation)
        if bool(obs_dict.get("_episode_reset", False)):
            self._policy.reset()
        with self._torch.inference_mode():
            action = self._policy.select_action(observation)
        action = self._postprocessor(action)
        action_transition = {self._action_key: action}
        action_transition = self._env_postprocessor(action_transition)
        action_value = action_transition[self._action_key]
        if hasattr(action_value, "detach"):
            action_value = action_value.detach().to("cpu").numpy()
        return to_env_action(np.asarray(action_value, dtype=np.float32).reshape(-1))


def build_adapter(checkpoint_path: str, device: str, model_name: str) -> Any:
    return _SmolVLAPluginAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )
