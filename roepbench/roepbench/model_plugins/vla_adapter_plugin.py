"""VLA-Adapter plugin using the official VLA-Adapter LIBERO inference path."""

from __future__ import annotations

import json
import math
import os
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

from ..model_adapters.base import AdapterNotReadyError
from .common import to_env_action


def _parse_bool(text: str, default: bool = False) -> bool:
    value = str(text or "").strip().lower()
    if value == "":
        return bool(default)
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _default_save_version(checkpoint_path: str) -> str:
    path = str(checkpoint_path or "").strip().rstrip("/")
    if not path:
        return "vla-adapter"
    tail = path.split("/")[-1].strip()
    return tail or "vla-adapter"


def _to_numpy_1d(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr


def _resolve_rgb(obs_dict: Dict[str, Any], *keys: str) -> Optional[np.ndarray]:
    for key in keys:
        arr = obs_dict.get(key)
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


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1)
    if quat.size < 4:
        out = np.zeros((3,), dtype=np.float32)
        out[: min(3, quat.size)] = quat[:3]
        return out
    quat = quat[:4].copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = float(np.sqrt(max(1.0 - quat[3] * quat[3], 0.0)))
    if math.isclose(den, 0.0):
        return np.zeros((3,), dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def _build_libero_observation(obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    full_image = _resolve_rgb(obs_dict, "front_rgb", "agentview_rgb", "wrist_rgb")
    if full_image is None:
        raise RuntimeError("VLA-Adapter plugin: no front/agentview RGB observation available.")

    wrist_image = _resolve_rgb(obs_dict, "wrist_rgb", "front_rgb", "agentview_rgb")
    if wrist_image is None:
        wrist_image = full_image

    eef_pos = _to_numpy_1d(obs_dict.get("eef_pos"))
    eef_quat = _to_numpy_1d(obs_dict.get("eef_quat"))
    gripper_qpos = _to_numpy_1d(obs_dict.get("gripper_qpos"))

    state = np.concatenate(
        [
            eef_pos[:3] if eef_pos.size else np.zeros((3,), dtype=np.float32),
            _quat2axisangle(eef_quat),
            gripper_qpos,
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    if state.size < 8:
        padded = np.zeros((8,), dtype=np.float32)
        padded[: state.size] = state
        state = padded
    elif state.size > 8:
        state = state[:8]

    return {
        "full_image": full_image,
        "wrist_image": wrist_image,
        "state": state,
    }


def _postprocess_action(raw_action: Any) -> np.ndarray:
    """Match the official VLA-Adapter LIBERO action post-processing."""
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1).copy()
    if action.size == 0:
        return action

    # Official eval normalizes the gripper channel from [0, 1] -> [-1, 1]
    # and then flips the sign again for the OpenVLA-family action convention.
    action[-1] = 2.0 * action[-1] - 1.0
    action[-1] = np.sign(action[-1])
    action[-1] *= -1.0
    return action


def _resolve_unnorm_key(norm_stats: Dict[str, Any], suite: str) -> str:
    suite_name = str(suite or "").strip().lower()
    candidates = [
        suite_name,
        f"{suite_name}_no_noops" if suite_name else "",
        "bridge_orig",
    ]
    for candidate in candidates:
        if candidate and candidate in norm_stats:
            return candidate
    if len(norm_stats) == 1:
        return str(next(iter(norm_stats.keys())))
    if norm_stats:
        return str(next(iter(norm_stats.keys())))
    raise AdapterNotReadyError("VLA-Adapter checkpoint does not expose norm_stats for action unnormalization.")


def _resolve_bf16_fallback(torch_mod: Any, device: str) -> str:
    """Resolve inference dtype fallback for GPUs without native bfloat16."""
    requested = str(os.environ.get("ROEP_VLA_ADAPTER_BF16_FALLBACK", "auto")).strip().lower()
    aliases = {
        "": "auto",
        "1": "auto",
        "true": "auto",
        "yes": "auto",
        "fp32": "float32",
        "float": "float32",
        "fp16": "float16",
        "half": "float16",
        "bf16": "none",
        "off": "none",
        "false": "none",
        "0": "none",
    }
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "float32", "float16", "none"}:
        raise AdapterNotReadyError(
            "Invalid ROEP_VLA_ADAPTER_BF16_FALLBACK. "
            "Use one of: auto, float32, float16, none."
        )
    if requested != "auto":
        return requested
    if not str(device).startswith("cuda") or not bool(torch_mod.cuda.is_available()):
        return "none"
    bf16_probe = getattr(torch_mod.cuda, "is_bf16_supported", None)
    if callable(bf16_probe):
        try:
            if bool(bf16_probe()):
                return "none"
        except Exception:
            return "none"
    return "float32"


class _VLAAdapterPluginAdapter:
    def __init__(self, checkpoint_path: str, device: str, model_name: str):
        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.model_name = str(model_name)

        self._cfg: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._action_head: Any = None
        self._proprio_projector: Any = None
        self._get_vla_action: Any = None
        self._action_queue: deque[np.ndarray] = deque()

        self._try_native_load()

    def _try_native_load(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoConfig,
                AutoImageProcessor,
                AutoModelForVision2Seq,
                AutoProcessor,
            )
            from experiments.robot.openvla_utils import (  # type: ignore
                find_checkpoint_file,
                get_action_head,
                get_processor,
                get_proprio_projector,
                get_vla,
                get_vla_action,
                load_component_state_dict,
            )
            from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig  # type: ignore
            from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # type: ignore
            from prismatic.models.action_heads import L1RegressionActionHead  # type: ignore
            from prismatic.models.projectors import ProprioProjector  # type: ignore
            from prismatic.extern.hf.processing_prismatic import (  # type: ignore
                PrismaticImageProcessor,
                PrismaticProcessor,
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "VLA-Adapter native load failed. Ensure the VLA-Adapter environment and repo imports are available."
            ) from exc

        cfg = SimpleNamespace(
            pretrained_checkpoint=self.checkpoint_path,
            model_family="openvla",
            load_in_8bit=False,
            load_in_4bit=False,
            use_film=False,
            num_images_in_input=int(os.environ.get("ROEP_VLA_ADAPTER_NUM_IMAGES_IN_INPUT", "2")),
            use_proprio=_parse_bool(os.environ.get("ROEP_VLA_ADAPTER_USE_PROPRIO", "1"), default=True),
            center_crop=True,
            num_open_loop_steps=int(os.environ.get("ROEP_VLA_ADAPTER_NUM_OPEN_LOOP_STEPS", "8")),
            save_version=str(
                os.environ.get(
                    "ROEP_VLA_ADAPTER_SAVE_VERSION",
                    _default_save_version(self.checkpoint_path),
                )
            ).strip()
            or _default_save_version(self.checkpoint_path),
            use_pro_version=_parse_bool(os.environ.get("ROEP_VLA_ADAPTER_USE_PRO_VERSION", "1"), default=True),
            use_l1_regression=_parse_bool(os.environ.get("ROEP_VLA_ADAPTER_USE_L1_REGRESSION", "1"), default=True),
            unnorm_key="",
        )

        try:
            bf16_fallback = _resolve_bf16_fallback(torch, self.device)
            if bf16_fallback == "float32":
                # Load directly as fp32. Loading as bf16 first and converting
                # afterward still produced NaN hidden states on RTX 2080 Ti.
                AutoConfig.register("openvla", OpenVLAConfig)
                AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
                AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
                AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
                model = AutoModelForVision2Seq.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=torch.float32,
                    load_in_8bit=cfg.load_in_8bit,
                    load_in_4bit=cfg.load_in_4bit,
                    low_cpu_mem_usage=False,
                    trust_remote_code=False,
                )
                model.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
                model.eval()
                model = model.to(torch.float32).to(self.device)
                stats_path = os.path.join(self.checkpoint_path, "dataset_statistics.json")
                if os.path.isfile(stats_path):
                    with open(stats_path, "r", encoding="utf-8") as f:
                        model.norm_stats = json.load(f)
            else:
                model = get_vla(cfg)
            if hasattr(model, "set_version"):
                model.set_version(cfg.save_version)

            # Official VLA-Adapter inference assumes native bf16. RTX 2080 Ti
            # cannot execute bf16 math safely; fp16 fallback produced NaN hidden
            # states in the Pro action head.
            if bf16_fallback == "float16":
                model = model.to(torch.float16)
                torch.bfloat16 = torch.float16  # type: ignore[attr-defined]
            cfg.bf16_fallback = bf16_fallback

            processor = get_processor(cfg)

            suite = str(os.environ.get("ROEP_LIBERO_SUITE", "libero_goal")).strip().lower()
            cfg.unnorm_key = _resolve_unnorm_key(getattr(model, "norm_stats", {}) or {}, suite)

            proprio_projector = None
            if bool(cfg.use_proprio) and bf16_fallback == "float32":
                proprio_projector = ProprioProjector(
                    llm_dim=model.llm_dim,
                    proprio_dim=8,
                ).to(self.device)
                proprio_projector.load_state_dict(
                    load_component_state_dict(find_checkpoint_file(self.checkpoint_path, "proprio_projector"))
                )
                proprio_projector = proprio_projector.to(torch.float32)
                proprio_projector.eval()
            elif bool(cfg.use_proprio):
                proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
                if self._should_force_fp32_aux_modules(torch):
                    proprio_projector = proprio_projector.to(torch.float32)
                proprio_projector.eval()

            action_head = None
            if bool(cfg.use_l1_regression) and bf16_fallback == "float32":
                action_head = L1RegressionActionHead(
                    input_dim=model.llm_dim,
                    hidden_dim=model.llm_dim,
                    action_dim=7,
                    use_pro_version=bool(cfg.use_pro_version),
                ).to(self.device)
                action_head.load_state_dict(
                    load_component_state_dict(find_checkpoint_file(self.checkpoint_path, "action_head"))
                )
                action_head = action_head.to(torch.float32)
                action_head.eval()
            elif bool(cfg.use_l1_regression):
                action_head = get_action_head(cfg, model.llm_dim)
                if self._should_force_fp32_aux_modules(torch):
                    action_head = action_head.to(torch.float32)
                action_head.eval()

            # Delay the fp32 alias until every checkpoint file has been loaded;
            # changing torch.bfloat16 before torch.load can crash CUDA restore
            # for these auxiliary component checkpoints.
            if bf16_fallback == "float32":
                torch.bfloat16 = torch.float32  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                f"VLA-Adapter native load failed: {type(exc).__name__}: {exc}"
            ) from exc

        self._cfg = cfg
        self._model = model
        self._processor = processor
        self._action_head = action_head
        self._proprio_projector = proprio_projector
        self._get_vla_action = get_vla_action

    def _should_force_fp32_aux_modules(self, torch_mod: Any) -> bool:
        if not str(self.device).startswith("cuda"):
            return False
        bf16_probe = getattr(torch_mod.cuda, "is_bf16_supported", None)
        if callable(bf16_probe):
            try:
                return not bool(bf16_probe())
            except Exception:
                return False
        return False

    def predict_action(self, obs_dict: Dict[str, Any], lang_goal: Optional[str] = None) -> np.ndarray:
        if self._model is None or self._processor is None or self._get_vla_action is None:
            raise RuntimeError("VLA-Adapter plugin not initialized.")

        if not self._action_queue:
            obs = _build_libero_observation(obs_dict)
            task_label = str(lang_goal or "").strip() or "perform the task safely"
            actions = self._get_vla_action(
                cfg=self._cfg,
                vla=self._model,
                processor=self._processor,
                obs=obs,
                task_label=task_label,
                action_head=self._action_head,
                proprio_projector=self._proprio_projector,
                noisy_action_projector=None,
                use_film=False,
                use_minivlm=True,
            )
            if isinstance(actions, np.ndarray):
                if actions.ndim == 1:
                    actions = actions.reshape(1, -1)
                action_batch = [actions[i] for i in range(actions.shape[0])]
            else:
                action_batch = list(actions or [])
            if not action_batch:
                raise RuntimeError("VLA-Adapter returned no actions.")
            processed_actions = []
            for action in action_batch:
                processed = _postprocess_action(action)
                if not np.isfinite(processed).all():
                    raise RuntimeError(
                        "VLA-Adapter returned non-finite action before env conversion. "
                        f"bf16_fallback={getattr(self._cfg, 'bf16_fallback', 'unknown')}"
                    )
                processed_actions.append(processed)
            self._action_queue.extend(processed_actions)

        return to_env_action(self._action_queue.popleft())


def build_adapter(checkpoint_path: str, device: str, model_name: str) -> Any:
    """Factory used by ROEP adapter loader."""
    return _VLAAdapterPluginAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )
