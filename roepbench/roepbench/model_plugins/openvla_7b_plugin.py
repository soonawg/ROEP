"""OpenVLA-7B plugin template with optional 4-bit loading."""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..model_adapters.base import AdapterNotReadyError
from .common import get_preferred_rgb, to_env_action
from .runtime_bridge import maybe_build_runtime_bridge_hook


def _parse_bool(text: str, default: bool = False) -> bool:
    v = str(text or "").strip().lower()
    if v == "":
        return bool(default)
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


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


def _cuda_supports_bfloat16(torch_mod: Any, device: str) -> bool:
    if not str(device).startswith("cuda"):
        return False
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return False
    bf16_probe = getattr(cuda, "is_bf16_supported", None)
    if callable(bf16_probe):
        try:
            return bool(bf16_probe())
        except Exception:
            pass
    capability_probe = getattr(cuda, "get_device_capability", None)
    if callable(capability_probe):
        try:
            major, _minor = capability_probe()
            return int(major) >= 8
        except Exception:
            return False
    return False


def _preferred_inference_dtype(torch_mod: Any, device: str) -> Any:
    if str(device).startswith("cuda"):
        if _cuda_supports_bfloat16(torch_mod, device):
            return torch_mod.bfloat16
        return torch_mod.float16
    return torch_mod.float32


def _supports_direct_image_instruction_call(model: Any) -> bool:
    predict_fn = getattr(model, "predict_action", None)
    if not callable(predict_fn):
        return False
    try:
        sig = inspect.signature(predict_fn)
    except Exception:
        return False
    params = set(sig.parameters)
    return "image" in params and "instruction" in params


def _move_inputs_to_device(
    inputs: Dict[str, Any],
    *,
    device: str,
    torch_mod: Any,
    float_dtype: Any,
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        try:
            is_fp = bool(
                hasattr(torch_mod, "is_floating_point") and torch_mod.is_floating_point(value)
            )
        except Exception:
            is_fp = False
        try:
            if is_fp:
                moved[key] = value.to(device=device, dtype=float_dtype)
            else:
                moved[key] = value.to(device=device)
        except Exception:
            moved[key] = value.to(device=device)
    return moved


class _OpenVLA7BPluginAdapter:
    def __init__(self, checkpoint_path: str, device: str, model_name: str):
        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.model_name = str(model_name)

        self._hook: Optional[Callable[[Dict[str, Any], Optional[str]], Any]] = None
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._inference_dtype: Any = None
        self._unnorm_key = str(os.environ.get("ROEP_OPENVLA_7B_UNNORM_KEY", "")).strip()

        hook_spec = str(os.environ.get("ROEP_OPENVLA_7B_RUNTIME_HOOK", "")).strip()
        if hook_spec:
            self._hook = _resolve_callable(hook_spec)
            return

        bridge_hook = maybe_build_runtime_bridge_hook(
            model_key="openvla_7b",
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
            from transformers import AutoProcessor  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "OpenVLA native load failed. Install required deps or set "
                "ROEP_OPENVLA_7B_RUNTIME_HOOK=module.path:callable."
            ) from exc

        model_cls = None
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore

            model_cls = AutoModelForVision2Seq
        except Exception:
            model_cls = None
        if model_cls is None:
            try:
                from transformers import AutoModelForImageTextToText  # type: ignore

                model_cls = AutoModelForImageTextToText
            except Exception:
                model_cls = None
        if model_cls is None:
            raise AdapterNotReadyError(
                "OpenVLA native load failed: neither AutoModelForVision2Seq nor "
                "AutoModelForImageTextToText is available in current transformers build. "
                "Use transformers 4.x or set ROEP_OPENVLA_7B_RUNTIME_HOOK=module.path:callable."
            )

        self._torch = torch
        resolved_device = str(self.device).strip().lower() or "cpu"
        if resolved_device.startswith("cuda") and not bool(torch.cuda.is_available()):
            resolved_device = "cpu"
        self.device = resolved_device
        self._inference_dtype = _preferred_inference_dtype(torch, self.device)
        local_files_only = bool(os.path.isdir(self.checkpoint_path))
        if os.path.isdir(self.checkpoint_path):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.checkpoint_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "OpenVLA AutoProcessor load failed. Check checkpoint root/files and transformers compatibility. "
                f"checkpoint={self.checkpoint_path}"
            ) from exc

        use_4bit = _parse_bool(os.environ.get("ROEP_OPENVLA_7B_4BIT", "true"), default=True)
        if not str(self.device).startswith("cuda"):
            use_4bit = False
        quant_cfg = None
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self._inference_dtype,
                )
            except Exception:
                quant_cfg = None

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = self._inference_dtype

        try:
            self._model = model_cls.from_pretrained(
                self.checkpoint_path,
                local_files_only=local_files_only,
                **model_kwargs,
            )
            self._model.eval()
            if quant_cfg is None and str(self.device).startswith("cuda"):
                self._model.to(self.device)
        except Exception as exc:  # noqa: BLE001
            raise AdapterNotReadyError(
                "OpenVLA model load failed. Check checkpoint root/files and quantization/dependency compatibility. "
                f"checkpoint={self.checkpoint_path}"
            ) from exc
        self._unnorm_key = self._resolve_unnorm_key()

    def _resolve_unnorm_key(self) -> str:
        explicit = str(self._unnorm_key or "").strip()
        if explicit:
            return explicit

        suite = str(os.environ.get("ROEP_LIBERO_SUITE", "")).strip().lower()
        candidate = suite if suite else "bridge_orig"
        norm_stats = getattr(self._model, "norm_stats", {}) or {}
        if isinstance(norm_stats, dict):
            if candidate in norm_stats:
                return candidate
            no_noops = f"{candidate}_no_noops"
            if no_noops in norm_stats:
                return no_noops
            if "bridge_orig" in norm_stats:
                return "bridge_orig"
            for key in norm_stats:
                return str(key)
        return candidate

    def predict_action(self, obs_dict: Dict[str, Any], lang_goal: Optional[str] = None) -> np.ndarray:
        if self._hook is not None:
            return to_env_action(self._hook(obs_dict, lang_goal))

        rgb = get_preferred_rgb(obs_dict)
        if rgb is None:
            raise RuntimeError("OpenVLA plugin: no RGB observation available.")

        from PIL import Image

        task_label = str(lang_goal or "").strip() or "perform the task safely"
        base_name = os.path.basename(str(self.checkpoint_path)).lower()
        if "openvla-v01" in base_name:
            prompt = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f"USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
            )
        else:
            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        image = Image.fromarray(np.asarray(rgb).astype(np.uint8, copy=False)).convert("RGB")

        if hasattr(self._model, "predict_action"):
            processor_errors: List[str] = []
            inputs: Optional[Dict[str, Any]] = None
            processor_variants = (
                lambda: self._processor(text=prompt, images=image, return_tensors="pt"),
                lambda: self._processor(prompt, image, return_tensors="pt"),
                lambda: self._processor(prompt, image),
            )
            for build_inputs in processor_variants:
                try:
                    built = build_inputs()
                    if hasattr(built, "items"):
                        inputs = dict(built.items())
                    else:
                        inputs = dict(built)
                    break
                except Exception as exc:  # noqa: BLE001
                    processor_errors.append(
                        f"{type(exc).__name__}: {exc}"
                    )

            if inputs is not None:
                try:
                    model_inputs = _move_inputs_to_device(
                        inputs,
                        device=self.device,
                        torch_mod=self._torch,
                        float_dtype=self._inference_dtype,
                    )
                    out = self._model.predict_action(
                        **model_inputs,
                        unnorm_key=self._unnorm_key,
                        do_sample=False,
                    )
                    return to_env_action(out)
                except Exception as exc:  # noqa: BLE001
                    processor_errors.append(
                        f"processor-path {type(exc).__name__}: {exc}"
                    )

            if _supports_direct_image_instruction_call(self._model):
                try:
                    out = self._model.predict_action(
                        image=rgb,
                        instruction=prompt,
                        unnorm_key=self._unnorm_key,
                    )
                    return to_env_action(out)
                except Exception as exc:  # noqa: BLE001
                    processor_errors.append(
                        f"direct-path {type(exc).__name__}: {exc}"
                    )

            details = "; ".join(processor_errors) if processor_errors else "no compatible prediction path"
            raise RuntimeError(f"OpenVLA predict_action failed: {details}")

        raise RuntimeError(
            "OpenVLA model does not expose predict_action in this build. "
            "Provide ROEP_OPENVLA_7B_RUNTIME_HOOK=module.path:callable."
        )


def build_adapter(checkpoint_path: str, device: str, model_name: str) -> Any:
    """Factory used by ROEP adapter loader."""
    return _OpenVLA7BPluginAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )
