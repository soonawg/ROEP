"""LAP-3B-Libero plugin with runtime-bridge-first loading."""

from __future__ import annotations

import dataclasses
import importlib
import os
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional

import numpy as np

from ..model_adapters.base import AdapterNotReadyError
from .common import to_env_action
from .runtime_bridge import maybe_build_runtime_bridge_hook


def _resolve_callable(spec: str) -> Callable[[Dict[str, Any], Optional[str]], Any]:
    if ":" not in spec:
        raise AdapterNotReadyError(
            f"Invalid LAP runtime hook spec '{spec}'. Expected module.path:callable"
        )
    module_name, symbol_name = spec.split(":", 1)
    module_name = module_name.strip()
    symbol_name = symbol_name.strip()
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        raise AdapterNotReadyError(
            f"LAP runtime hook import failed: {module_name} ({type(exc).__name__}: {exc})"
        ) from exc
    target = getattr(module, symbol_name, None)
    if not callable(target):
        raise AdapterNotReadyError(f"LAP runtime hook is not callable: {spec}")
    return target


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


def _quat_to_rot6d(quat: Any) -> np.ndarray:
    arr = np.asarray(quat, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        out = np.zeros((6,), dtype=np.float32)
        out[: min(3, arr.size)] = arr[:3]
        return out

    x, y, z, w = arr[:4]
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return np.concatenate([rot[:, 0], rot[:, 1]], axis=0).astype(np.float32, copy=False)


def _build_request(obs_dict: Dict[str, Any], lang_goal: Optional[str]) -> Dict[str, Any]:
    front = _to_hwc_uint8(obs_dict.get("front_rgb", obs_dict.get("agentview_rgb")))
    wrist = _to_hwc_uint8(obs_dict.get("wrist_rgb", front))
    if front is None:
        raise RuntimeError("LAP plugin: no front/agentview RGB observation available.")
    if wrist is None:
        wrist = front

    eef_pos = np.asarray(obs_dict.get("eef_pos", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(
        obs_dict.get("eef_quat", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    gripper_qpos = np.asarray(obs_dict.get("gripper_qpos", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    gripper_state = float(np.clip(gripper_qpos[-2] / 0.04, 0.0, 1.0)) if gripper_qpos.size >= 2 else 0.0
    state = np.concatenate(
        [
            eef_pos[:3] if eef_pos.size >= 3 else np.pad(eef_pos[:3], (0, max(0, 3 - eef_pos.size))),
            _quat_to_rot6d(eef_quat),
            np.asarray([gripper_state], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return {
        "observation": {
            "base_0_rgb": front,
            "left_wrist_0_rgb": wrist,
            "state": state,
        },
        "prompt": str(lang_goal or ""),
    }


def _postprocess_action_chunk(raw_action: Any) -> np.ndarray:
    action_chunk = np.asarray(raw_action, dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.reshape(1, -1)
    elif action_chunk.ndim == 0:
        action_chunk = action_chunk.reshape(1, 1)
    if action_chunk.shape[-1] < 1:
        raise RuntimeError("LAP plugin: received empty action output.")
    action_chunk = action_chunk.copy()
    action_chunk[:, -1:] = 1.0 - 2.0 * action_chunk[:, -1:]
    action_chunk[:, -1:] = np.sign(action_chunk[:, -1:])
    return action_chunk.astype(np.float32, copy=False)


class _LAPLiberoPluginAdapter:
    def __init__(self, checkpoint_path: str, device: str, model_name: str):
        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.model_name = str(model_name)
        self._hook: Optional[Callable[[Dict[str, Any], Optional[str]], Any]] = None
        self._policy: Any = None
        self._replan_steps = max(1, int(str(os.environ.get("ROEP_LAP_LIBERO_REPLAN_STEPS", "2")).strip() or "2"))
        self._default_prompt = str(os.environ.get("ROEP_LAP_LIBERO_DEFAULT_PROMPT", "")).strip() or None
        self._action_queue: Deque[np.ndarray] = deque()

        hook_spec = str(os.environ.get("ROEP_LAP_LIBERO_RUNTIME_HOOK", "")).strip()
        if hook_spec:
            self._hook = _resolve_callable(hook_spec)
            return

        bridge_hook = maybe_build_runtime_bridge_hook(
            model_key="lap_libero",
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        if bridge_hook is not None:
            self._hook = bridge_hook
            return

        self._try_native_load()

    def _try_native_load(self) -> None:
        try:
            from lap.policies.policy_config_adapter import create_trained_policy  # type: ignore
            from lap.training import config as lap_config  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "LAP native load failed. Install LAP/openpi deps in the model env or set "
                "ROEP_LAP_LIBERO_RUNTIME_HOOK=module.path:callable."
            ) from exc

        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            raise AdapterNotReadyError(
                f"LAP checkpoint does not exist locally: {self.checkpoint_path}"
            )

        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        try:
            train_config = lap_config.get_config("lap_libero")
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(train_config.model, stop_action_to_vlm_grad=False),
            )
            self._policy = create_trained_policy(
                train_config,
                str(checkpoint),
                default_prompt=self._default_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "LAP native policy load failed. Check checkpoint layout, LAP config compatibility, and JAX runtime. "
                f"checkpoint={self.checkpoint_path}"
            ) from exc

    def _queue_actions(self, obs_dict: Dict[str, Any], lang_goal: Optional[str]) -> None:
        if self._policy is None:
            raise RuntimeError("LAP plugin is not initialized.")

        request = _build_request(obs_dict, lang_goal or self._default_prompt or "")
        response = self._policy.infer(request)
        if not isinstance(response, dict) or "actions" not in response:
            raise RuntimeError("LAP plugin: policy response missing 'actions'.")

        action_chunk = _postprocess_action_chunk(response["actions"])
        if action_chunk.shape[0] == 0:
            raise RuntimeError("LAP plugin: policy returned no actions.")

        for action in action_chunk[: self._replan_steps]:
            self._action_queue.append(action)

    def predict_action(self, obs_dict: Dict[str, Any], lang_goal: Optional[str] = None):
        if self._hook is not None:
            return to_env_action(self._hook(obs_dict, lang_goal), action_dim=7)

        if bool(obs_dict.get("_episode_reset", False)):
            self._action_queue.clear()
        if not self._action_queue:
            self._queue_actions(obs_dict, lang_goal)
        return to_env_action(self._action_queue.popleft(), action_dim=7)


def build_adapter(checkpoint_path: str, device: str, model_name: str) -> Any:
    """Factory used by ROEP adapter loader."""
    return _LAPLiberoPluginAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )
