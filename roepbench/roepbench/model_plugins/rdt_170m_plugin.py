"""RDT-170M plugin template (repo-native first).

This implementation supports:
1) Runtime hook mode (recommended for custom deployments):
   - ROEP_RDT_170M_RUNTIME_HOOK=module.path:callable
2) Repository-native mode:
   - ROEP_RDT_170M_REPO=/abs/path/to/RoboticsDiffusionTransformer
   - uses scripts.agilex_model.create_model + policy.step(...)
   - language embedding source policy:
     * campaign-safe default: ROEP_RDT_170M_TEXT_EMBED_HOOK=module.path:callable
     * static file path is blocked by default
       (ROEP_RDT_170M_TEXT_EMBED_PATH requires
        ROEP_RDT_170M_ALLOW_STATIC_TEXT_EMBED=true)
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager

import numpy as np
from PIL import Image

from ..model_adapters.base import AdapterNotReadyError
from .common import get_preferred_rgb, to_env_action
from .runtime_bridge import maybe_build_runtime_bridge_hook


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


class _RDT170MPluginAdapter:
    def __init__(self, checkpoint_path: str, device: str, model_name: str):
        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.model_name = str(model_name)

        self._hook: Optional[Callable[[Dict[str, Any], Optional[str]], Any]] = None
        self._policy: Any = None
        self._torch: Any = None
        self._text_embed_hook: Optional[Callable[..., Any]] = None
        self._fixed_text_embed: Any = None
        self._text_embed_cache: Dict[str, Any] = {}
        self._camera_keys: List[str] = [
            key.strip()
            for key in str(
                os.environ.get(
                    "ROEP_RDT_170M_CAMERA_KEYS",
                    "front_rgb,left_shoulder_rgb,right_shoulder_rgb",
                )
            ).split(",")
            if key.strip()
        ]
        self._state_dim = int(os.environ.get("ROEP_RDT_170M_STATE_DIM", "14"))

        hook_spec = str(os.environ.get("ROEP_RDT_170M_RUNTIME_HOOK", "")).strip()
        if hook_spec:
            self._hook = _resolve_callable(hook_spec)
            return

        bridge_hook = maybe_build_runtime_bridge_hook(
            model_key="rdt_170m",
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        if bridge_hook is not None:
            self._hook = bridge_hook
            return

        self._load_repo_native()

    @staticmethod
    @contextmanager
    def _pushd(path: Path):
        old = Path.cwd()
        os.chdir(str(path))
        try:
            yield
        finally:
            os.chdir(str(old))

    def _resolve_text_embedding(self, lang_goal: Optional[str]) -> Any:
        if self._fixed_text_embed is not None:
            return self._fixed_text_embed

        goal = str(lang_goal or "").strip() or "perform the task safely"
        if goal in self._text_embed_cache:
            return self._text_embed_cache[goal]

        if self._text_embed_hook is None:
            raise RuntimeError(
                "RDT text embedding source is not configured. "
                "Set ROEP_RDT_170M_TEXT_EMBED_HOOK or ROEP_RDT_170M_TEXT_EMBED_PATH."
            )

        try:
            value = self._text_embed_hook(goal, self.device)
        except TypeError:
            value = self._text_embed_hook(goal)

        if isinstance(value, dict) and "embeddings" in value:
            value = value["embeddings"]
        if self._torch is not None and hasattr(self._torch, "as_tensor"):
            try:
                value = self._torch.as_tensor(value)
            except Exception:
                pass
            if hasattr(value, "to"):
                value = value.to(self.device)
        self._text_embed_cache[goal] = value
        return value

    def _extract_proprio(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        for key in ("obs_low_dim", "low_dim_state", "low_dim", "robot_state"):
            raw = obs_dict.get(key, None)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            if arr.size >= self._state_dim:
                return arr[: self._state_dim]
            out = np.zeros((self._state_dim,), dtype=np.float32)
            out[: arr.size] = arr
            return out
        return np.zeros((self._state_dim,), dtype=np.float32)

    @staticmethod
    def _to_pil_rgb(arr: np.ndarray) -> Image.Image:
        img = np.asarray(arr)
        if img.ndim != 3:
            raise RuntimeError(f"Expected HWC/CHW image, got shape={img.shape}")
        if img.shape[0] in (1, 3, 4) and img.shape[1] > 8 and img.shape[2] > 8:
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(img)

    def _extract_images(self, obs_dict: Dict[str, Any]) -> List[Image.Image]:
        imgs: List[Image.Image] = []
        for key in self._camera_keys:
            raw = obs_dict.get(key, None)
            if not isinstance(raw, np.ndarray):
                continue
            try:
                imgs.append(self._to_pil_rgb(raw))
            except Exception:
                continue
        if imgs:
            return imgs

        fallback = get_preferred_rgb(obs_dict)
        if fallback is not None:
            return [self._to_pil_rgb(fallback)]
        return []

    def _load_repo_native(self) -> None:
        repo_path_raw = str(os.environ.get("ROEP_RDT_170M_REPO", "")).strip()
        if repo_path_raw == "":
            raise AdapterNotReadyError(
                "RDT repo-native mode requires ROEP_RDT_170M_REPO "
                "(path to RoboticsDiffusionTransformer repo), "
                "or set ROEP_RDT_170M_RUNTIME_HOOK=module.path:callable."
            )
        repo_path = Path(repo_path_raw).expanduser().resolve()
        if not repo_path.exists():
            raise AdapterNotReadyError(f"RDT repo path does not exist: {repo_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            with self._pushd(repo_path):
                import torch  # type: ignore
                from scripts.agilex_model import create_model  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "Failed to import RDT repo-native API "
                "(expected scripts.agilex_model.create_model). "
                f"repo={repo_path}"
            ) from exc

        self._torch = torch
        dtype_name = str(os.environ.get("ROEP_RDT_170M_DTYPE", "")).strip().lower()
        if dtype_name == "fp16":
            dtype = torch.float16
        elif dtype_name == "fp32":
            dtype = torch.float32
        elif dtype_name == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32

        chunk_size = int(os.environ.get("ROEP_RDT_170M_CHUNK_SIZE", "64"))
        episode_len = int(os.environ.get("ROEP_RDT_170M_EPISODE_LEN", "1000"))
        control_frequency = int(os.environ.get("ROEP_RDT_170M_CONTROL_FREQUENCY", "25"))
        vision_encoder = str(
            os.environ.get(
                "ROEP_RDT_170M_VISION_ENCODER",
                "google/siglip-so400m-patch14-384",
            )
        ).strip()
        camera_names = [
            x.strip()
            for x in str(
                os.environ.get(
                    "ROEP_RDT_170M_CAMERA_NAMES",
                    "cam_high,cam_right_wrist,cam_left_wrist",
                )
            ).split(",")
            if x.strip()
        ]
        args_cfg: Dict[str, Any] = {
            "episode_len": int(episode_len),
            "state_dim": int(self._state_dim),
            "chunk_size": int(chunk_size),
            "camera_names": list(camera_names),
        }

        try:
            with self._pushd(repo_path):
                policy_obj = create_model(
                    args=args_cfg,
                    dtype=dtype,
                    pretrained_vision_encoder_name_or_path=vision_encoder,
                    pretrained=self.checkpoint_path,
                    control_frequency=control_frequency,
                )
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "RDT create_model failed. Check checkpoint/repo compatibility and files. "
                f"checkpoint={self.checkpoint_path}, repo={repo_path}"
            ) from exc
        if isinstance(policy_obj, (tuple, list)):
            for item in policy_obj:
                if hasattr(item, "step") and callable(getattr(item, "step")):
                    self._policy = item
                    break
        elif isinstance(policy_obj, dict):
            for key in ("policy", "model"):
                item = policy_obj.get(key, None)
                if hasattr(item, "step") and callable(getattr(item, "step")):
                    self._policy = item
                    break
        elif hasattr(policy_obj, "step") and callable(getattr(policy_obj, "step")):
            self._policy = policy_obj

        if self._policy is None:
            raise AdapterNotReadyError(
                "RDT create_model did not return a policy object with step(...). "
                "Set ROEP_RDT_170M_RUNTIME_HOOK for custom integration."
            )

        embed_hook_spec = str(os.environ.get("ROEP_RDT_170M_TEXT_EMBED_HOOK", "")).strip()
        if embed_hook_spec:
            self._text_embed_hook = _resolve_callable(embed_hook_spec)
            return

        embed_path_raw = str(os.environ.get("ROEP_RDT_170M_TEXT_EMBED_PATH", "")).strip()
        if embed_path_raw:
            allow_static = str(
                os.environ.get("ROEP_RDT_170M_ALLOW_STATIC_TEXT_EMBED", "false")
            ).strip().lower() in {"1", "true", "yes", "on"}
            if not allow_static:
                raise AdapterNotReadyError(
                    "ROEP_RDT_170M_TEXT_EMBED_PATH uses one fixed embedding and can silently "
                    "invalidate multi-task/campaign evaluation. "
                    "Use ROEP_RDT_170M_TEXT_EMBED_HOOK for goal-conditioned embeddings, or set "
                    "ROEP_RDT_170M_ALLOW_STATIC_TEXT_EMBED=true only for explicit single-goal smoke tests."
                )
            embed_path = Path(embed_path_raw).expanduser().resolve()
            if not embed_path.exists():
                raise AdapterNotReadyError(
                    f"RDT text embedding file does not exist: {embed_path}"
                )
            try:
                payload = torch.load(str(embed_path), map_location=self.device)
            except Exception as exc:  # noqa: BLE001
                raise AdapterNotReadyError(
                    f"Failed to load RDT text embedding file: {embed_path}"
                ) from exc
            if isinstance(payload, dict) and "embeddings" in payload:
                payload = payload["embeddings"]
            if hasattr(payload, "to"):
                payload = payload.to(self.device)
            self._fixed_text_embed = payload
            return

        raise AdapterNotReadyError(
            "RDT repo-native mode requires goal-conditioned text embeddings. "
            "Set ROEP_RDT_170M_TEXT_EMBED_HOOK=module.path:callable. "
            "Static ROEP_RDT_170M_TEXT_EMBED_PATH is blocked by default and is only "
            "allowed with ROEP_RDT_170M_ALLOW_STATIC_TEXT_EMBED=true for explicit "
            "single-goal smoke tests."
        )

    def predict_action(self, obs_dict: Dict[str, Any], lang_goal: Optional[str] = None) -> np.ndarray:
        if self._hook is not None:
            return to_env_action(self._hook(obs_dict, lang_goal))

        images = self._extract_images(obs_dict)
        if not images:
            raise RuntimeError("RDT plugin: no usable RGB observations found.")
        proprio = self._extract_proprio(obs_dict)
        text_embeds = self._resolve_text_embedding(lang_goal)

        actions = self._policy.step(
            proprio=proprio,
            images=images,
            text_embeds=text_embeds,
        )
        arr = np.asarray(actions)
        if arr.ndim >= 3:
            raw = arr[0, 0]
        elif arr.ndim == 2:
            raw = arr[0]
        else:
            raw = arr
        return to_env_action(raw)


def build_adapter(checkpoint_path: str, device: str, model_name: str) -> Any:
    """Factory used by ROEP adapter loader."""
    return _RDT170MPluginAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )
