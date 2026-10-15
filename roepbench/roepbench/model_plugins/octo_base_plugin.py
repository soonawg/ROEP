"""Octo-Base plugin template.

This plugin supports two modes:
1) Runtime hook mode (recommended for project-specific control):
   - set ROEP_OCTO_BASE_RUNTIME_HOOK=module.path:callable
   - callable signature: fn(obs_dict: dict, lang_goal: Optional[str]) -> action
2) Best-effort native Octo mode:
   - requires octo + jax stack in the same environment.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ..model_adapters.base import AdapterNotReadyError
from .common import get_preferred_rgb, to_env_action
from .runtime_bridge import maybe_build_runtime_bridge_hook


def _parse_bool(text: str, default: bool = False) -> bool:
    value = str(text or "").strip().lower()
    if value == "":
        return bool(default)
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _parse_float(text: str, default: float) -> float:
    try:
        return float(text)
    except Exception:  # noqa: BLE001
        return float(default)


def _resolve_callable(spec: str) -> Callable[[Dict[str, Any], Optional[str]], Any]:
    if ":" not in spec:
        raise AdapterNotReadyError(
            f"Invalid runtime hook spec '{spec}'. Expected module.path:callable"
        )
    mod_name, sym_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name.strip())
    except Exception as exc:  # noqa: BLE001
        raise AdapterNotReadyError(
            f"Runtime hook module import failed: {mod_name.strip()} ({type(exc).__name__}: {exc})"
        ) from exc
    fn = getattr(mod, sym_name.strip(), None)
    if not callable(fn):
        raise AdapterNotReadyError(f"Runtime hook is not callable: {spec}")
    return fn


def _resize_nearest_hw(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize HWC image with nearest-neighbor sampling using NumPy only."""
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={image.shape}")
    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    if src_h == int(target_h) and src_w == int(target_w):
        return image
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target shape: ({target_h}, {target_w})")
    y_idx = np.linspace(0, src_h - 1, int(target_h)).astype(np.int64)
    x_idx = np.linspace(0, src_w - 1, int(target_w)).astype(np.int64)
    return image[y_idx][:, x_idx]


def _camera_candidates(obs_key: str) -> Sequence[str]:
    if obs_key == "image_wrist":
        return ("wrist_rgb", "front_rgb")
    # image_primary and fallback
    return ("front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "overhead_rgb", "wrist_rgb")


def _get_rgb_for_key(obs_dict: Dict[str, Any], obs_key: str) -> Optional[np.ndarray]:
    for key in _camera_candidates(obs_key):
        arr = obs_dict.get(key, None)
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim != 3:
            continue
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr
    return None


def _prepare_image_for_spec(image_hwc: np.ndarray, spec_array: np.ndarray) -> np.ndarray:
    """Convert current RGB frame to match example_batch image spec (HWC)."""
    if image_hwc.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={image_hwc.shape}")
    if spec_array.ndim < 5:
        raise ValueError(f"Unexpected image spec shape: {spec_array.shape}")
    target_h, target_w, target_c = map(int, spec_array.shape[-3:])
    img = _resize_nearest_hw(image_hwc, target_h=target_h, target_w=target_w)

    if int(img.shape[-1]) != target_c:
        if target_c == 1:
            # Convert to luminance if model expects grayscale.
            gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
            img = gray[..., None]
        else:
            if int(img.shape[-1]) == 1:
                img = np.repeat(img, target_c, axis=-1)
            else:
                img = img[..., :target_c]

    if spec_array.dtype == np.uint8:
        return img.astype(np.uint8, copy=False)
    # Float-like specs: map [0,255] -> [0,1].
    out = img.astype(np.float32) / 255.0
    return out.astype(spec_array.dtype, copy=False)


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert XYZ intrinsic Euler angles to quaternion [x, y, z, w]."""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.asarray([qx, qy, qz, qw], dtype=np.float32)


def _normalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1)
    if q.size != 4:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1e-6:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return q / norm


class _OctoBasePluginAdapter:
    def __init__(self, checkpoint_path: str, device: str, model_name: str):
        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.model_name = str(model_name)

        self._hook: Optional[Callable[[Dict[str, Any], Optional[str]], Any]] = None
        self._model: Any = None
        self._jax: Any = None
        self._rng: Any = None
        self._dataset_stats: Any = None
        self._obs_example: Optional[Dict[str, Any]] = None
        self._step_index: int = 0
        self._layout = str(os.environ.get("ROEP_OCTO_BASE_ACTION_LAYOUT", "auto")).strip().lower()
        self._delta_scale = _parse_float(
            os.environ.get("ROEP_OCTO_BASE_DELTA_SCALE", "0.05"),
            default=0.05,
        )
        self._max_translation = _parse_float(
            os.environ.get("ROEP_OCTO_BASE_MAX_TRANSLATION", "0.10"),
            default=0.10,
        )
        self._gripper_threshold = _parse_float(
            os.environ.get("ROEP_OCTO_BASE_GRIPPER_THRESHOLD", "0.0"),
            default=0.0,
        )
        self._default_gripper = _parse_float(
            os.environ.get("ROEP_OCTO_BASE_DEFAULT_GRIPPER", "-1.0"),
            default=-1.0,
        )
        self._debug_actions = _parse_bool(
            os.environ.get("ROEP_OCTO_BASE_DEBUG_ACTION", "0"),
            default=False,
        )
        self._debug_action_limit = int(
            _parse_float(os.environ.get("ROEP_OCTO_BASE_DEBUG_ACTION_LIMIT", "20"), 20.0)
        )
        self._debug_action_count = 0

        hook_spec = str(os.environ.get("ROEP_OCTO_BASE_RUNTIME_HOOK", "")).strip()
        if hook_spec:
            self._hook = _resolve_callable(hook_spec)
            return

        bridge_hook = maybe_build_runtime_bridge_hook(
            model_key="octo_base",
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        if bridge_hook is not None:
            self._hook = bridge_hook
            return

        self._try_native_octo_load()

    def _try_native_octo_load(self) -> None:
        try:
            import jax  # type: ignore
            from octo.model.octo_model import OctoModel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "Octo native load failed. Either install octo+jax in this env "
                "or set ROEP_OCTO_BASE_RUNTIME_HOOK=module.path:callable."
            ) from exc

        self._jax = jax
        self._rng = jax.random.PRNGKey(0)
        try:
            backend = str(jax.default_backend())
            devices = jax.devices()
            logging.info("Octo JAX backend=%s devices=%s", backend, devices)
            require_gpu = str(os.environ.get("ROEP_OCTO_BASE_REQUIRE_GPU", "0")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if require_gpu and backend != "gpu":
                raise AdapterNotReadyError(
                    "ROEP_OCTO_BASE_REQUIRE_GPU=1 but JAX backend is not GPU. "
                    "Fix CUDA/JAX runtime or unset ROEP_OCTO_BASE_REQUIRE_GPU."
                )
        except AdapterNotReadyError:
            raise
        except Exception:
            # Backend introspection failure should not hide a potentially usable setup.
            pass
        try:
            self._model = OctoModel.load_pretrained(self.checkpoint_path)
            stats = getattr(self._model, "dataset_statistics", None)
            if isinstance(stats, dict):
                candidate = (
                    stats.get("bridge_dataset")
                    or stats.get("bridge_orig")
                    or stats.get("default")
                )
                if isinstance(candidate, dict) and "action" in candidate and isinstance(candidate.get("action"), dict):
                    candidate = candidate["action"]
                # Accept only direct unnormalization dicts that include mean/std or percentile bounds.
                if isinstance(candidate, dict) and (
                    ("mean" in candidate and "std" in candidate)
                    or ("p01" in candidate and "p99" in candidate)
                ):
                    self._dataset_stats = candidate
                else:
                    self._dataset_stats = None
            example_batch = getattr(self._model, "example_batch", None)
            if isinstance(example_batch, dict):
                obs = example_batch.get("observation", None)
                if isinstance(obs, dict):
                    self._obs_example = obs
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "Octo load_pretrained failed. Check checkpoint path/files and Octo/JAX version compatibility. "
                f"checkpoint={self.checkpoint_path} ({type(exc).__name__}: {exc})"
            ) from exc

    def _build_observations(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(self._obs_example, dict):
            rgb = get_preferred_rgb(obs_dict)
            if rgb is None:
                raise RuntimeError("Octo plugin: no RGB observation available.")
            return {
                "image_primary": np.asarray(rgb, dtype=np.uint8)[None, None, ...],
                "timestep_pad_mask": np.array([[True]], dtype=bool),
            }

        obs_example = self._obs_example
        out: Dict[str, Any] = {}

        # Infer history window from the checkpoint spec.
        history = 1
        tmask_spec = obs_example.get("timestep_pad_mask", None)
        if isinstance(tmask_spec, np.ndarray) and tmask_spec.ndim >= 2:
            history = int(tmask_spec.shape[1])
        history = max(1, history)

        for key, spec in obs_example.items():
            if key == "pad_mask_dict":
                if isinstance(spec, dict):
                    pad: Dict[str, Any] = {}
                    for pkey, parray in spec.items():
                        if isinstance(parray, np.ndarray) and parray.ndim >= 2:
                            pad[pkey] = np.ones((1, int(parray.shape[1])), dtype=parray.dtype)
                        else:
                            pad[pkey] = np.array([[True]], dtype=bool)
                    out["pad_mask_dict"] = pad
                continue

            if key == "timestep_pad_mask":
                if isinstance(spec, np.ndarray) and spec.ndim >= 2:
                    out[key] = np.ones((1, int(spec.shape[1])), dtype=spec.dtype)
                else:
                    out[key] = np.ones((1, history), dtype=bool)
                continue

            if key.startswith("image_") and isinstance(spec, np.ndarray):
                rgb = _get_rgb_for_key(obs_dict, key)
                if rgb is None and key != "image_primary":
                    rgb = _get_rgb_for_key(obs_dict, "image_primary")
                if rgb is None:
                    rgb = get_preferred_rgb(obs_dict)
                if rgb is None:
                    raise RuntimeError(f"Octo plugin: missing RGB for key '{key}'.")
                frame = _prepare_image_for_spec(rgb, spec)
                out[key] = np.repeat(frame[None, None, ...], history, axis=1).astype(spec.dtype, copy=False)
                continue

            if key == "timestep" and isinstance(spec, np.ndarray) and spec.ndim >= 2:
                width = int(spec.shape[1])
                start = max(0, self._step_index - width + 1)
                seq = np.arange(start, start + width, dtype=spec.dtype)
                out[key] = seq[None, :]
                continue

            if isinstance(spec, np.ndarray):
                # Keep non-image observation channels schema-compatible by zero-filling.
                out[key] = np.zeros_like(spec)

        self._step_index += 1
        return out

    def _convert_action(self, raw_action: Any) -> np.ndarray:
        arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return to_env_action(arr)

        layout = self._layout
        if layout == "auto":
            if arr.size >= 8:
                layout = "quat8"
            elif arr.size == 7:
                quat_norm = float(np.linalg.norm(arr[3:7]))
                layout = "quat7" if (0.2 <= quat_norm <= 2.5) else "euler7"
            elif arr.size == 6:
                layout = "euler6"
            else:
                layout = "passthrough"

        out = np.zeros((8,), dtype=np.float32)

        if layout == "quat8":
            pos_delta = arr[:3]
            quat = arr[3:7]
            grip_raw = float(arr[7])
        elif layout == "quat7":
            pos_delta = arr[:3]
            quat = arr[3:7]
            grip_raw = self._default_gripper
        elif layout == "euler7":
            pos_delta = arr[:3]
            quat = _quat_from_euler_xyz(float(arr[3]), float(arr[4]), float(arr[5]))
            grip_raw = float(arr[6])
        elif layout == "euler6":
            pos_delta = arr[:3]
            quat = _quat_from_euler_xyz(float(arr[3]), float(arr[4]), float(arr[5]))
            grip_raw = self._default_gripper
        elif layout == "passthrough":
            return to_env_action(arr)
        else:
            raise RuntimeError(
                "Unsupported ROEP_OCTO_BASE_ACTION_LAYOUT. "
                "Use one of: auto, quat8, quat7, euler7, euler6, passthrough."
            )

        pos_delta = np.asarray(pos_delta, dtype=np.float32) * float(self._delta_scale)
        trans_norm = float(np.linalg.norm(pos_delta))
        max_trans = float(max(1e-6, self._max_translation))
        if trans_norm > max_trans:
            pos_delta = pos_delta * (max_trans / trans_norm)

        quat = _normalize_quat(quat)
        gripper = 1.0 if float(grip_raw) >= float(self._gripper_threshold) else -1.0

        out[:3] = pos_delta
        out[3:7] = quat
        out[7] = float(gripper)

        if self._debug_actions and self._debug_action_count < self._debug_action_limit:
            logging.info(
                "Octo action convert step=%s layout=%s raw_shape=%s raw_head=%s out=%s",
                self._step_index,
                layout,
                tuple(arr.shape),
                np.asarray(arr[:8], dtype=np.float32).tolist(),
                out.tolist(),
            )
            self._debug_action_count += 1
        return out

    def predict_action(self, obs_dict: Dict[str, Any], lang_goal: Optional[str] = None) -> np.ndarray:
        if self._hook is not None:
            return self._convert_action(self._hook(obs_dict, lang_goal))

        prompt = str(lang_goal or "").strip() or "perform the task safely"
        observations = self._build_observations(obs_dict)

        tasks = self._model.create_tasks(texts=[prompt])
        kwargs: Dict[str, Any] = {
            "observations": observations,
            "tasks": tasks,
            "rng": self._rng,
        }
        if self._dataset_stats is not None:
            kwargs["unnormalization_statistics"] = self._dataset_stats

        action_seq = self._model.sample_actions(**kwargs)
        # Advance RNG deterministically.
        _, self._rng = self._jax.random.split(self._rng)
        action_np = np.asarray(action_seq)
        if action_np.ndim >= 3:
            raw = action_np[0, 0]
        elif action_np.ndim == 2:
            raw = action_np[0]
        else:
            raw = action_np
        return self._convert_action(raw)


def build_adapter(checkpoint_path: str, device: str, model_name: str) -> Any:
    """Factory used by ROEP adapter loader."""
    return _OctoBasePluginAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )
