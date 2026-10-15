from __future__ import annotations

import numpy as np
import pytest

from roepbench.model_plugins import openvla_7b_plugin as plugin


class _FakeCuda:
    def __init__(self, *, available: bool = True, bf16_supported: bool = False, capability=(7, 5)):
        self._available = available
        self._bf16_supported = bf16_supported
        self._capability = capability

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported

    def get_device_capability(self):
        return self._capability


class _FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self, *, cuda: _FakeCuda):
        self.cuda = cuda

    @staticmethod
    def is_floating_point(_value) -> bool:
        return False


def test_preferred_inference_dtype_uses_fp16_on_pre_ampere_cuda():
    fake_torch = _FakeTorch(cuda=_FakeCuda(bf16_supported=False, capability=(7, 5)))
    assert plugin._preferred_inference_dtype(fake_torch, "cuda") == fake_torch.float16


def test_preferred_inference_dtype_uses_bf16_when_supported():
    fake_torch = _FakeTorch(cuda=_FakeCuda(bf16_supported=True, capability=(8, 0)))
    assert plugin._preferred_inference_dtype(fake_torch, "cuda") == fake_torch.bfloat16


def test_direct_image_instruction_call_requires_explicit_signature():
    class _OpenVLALikeModel:
        def predict_action(self, input_ids=None, unnorm_key=None, **kwargs):
            return input_ids, unnorm_key, kwargs

    assert plugin._supports_direct_image_instruction_call(_OpenVLALikeModel()) is False


def test_predict_action_reports_processor_error_when_direct_call_is_unsupported():
    adapter = plugin._OpenVLA7BPluginAdapter.__new__(plugin._OpenVLA7BPluginAdapter)
    adapter._hook = None
    adapter._processor = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("processor boom"))
    adapter._model = type(
        "_Model",
        (),
        {
            "predict_action": lambda self, input_ids=None, unnorm_key=None, **kwargs: np.zeros(7),
        },
    )()
    adapter._torch = _FakeTorch(cuda=_FakeCuda())
    adapter._inference_dtype = adapter._torch.float16
    adapter._unnorm_key = "libero_goal"
    adapter.checkpoint_path = "model_ckpts/openvla-7b-finetuned-libero-goal"
    adapter.device = "cuda"

    obs_dict = {
        "front_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
    }

    with pytest.raises(RuntimeError, match="processor boom"):
        adapter.predict_action(obs_dict, lang_goal="Open the drawer safely")
