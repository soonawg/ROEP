from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from roepbench.model_adapters.base import AdapterNotReadyError
from roepbench.model_adapters.xvla import XVLAAdapter
from roepbench.model_plugins.xvla_plugin import (
    _XVLAPluginAdapter,
    _normalize_robot_state,
    _resolve_domain_id,
)


def test_xvla_adapter_accepts_remote_repo_id_checkpoint() -> None:
    adapter = XVLAAdapter.__new__(XVLAAdapter)
    adapter.model_name = "xvla"
    adapter.checkpoint_env_var = "ROEP_XVLA_CKPT"
    adapter.config = type("Cfg", (), {"checkpoint_path": "lerobot/xvla-libero"})()
    adapter.ensure_ready()


def test_xvla_normalize_robot_state_adds_rotation_matrix() -> None:
    obs_dict = {
        "eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper_qpos": np.array([0.4, 0.5], dtype=np.float32),
        "gripper_qvel": np.array([0.6, 0.7], dtype=np.float32),
        "joint_pos": np.arange(7, dtype=np.float32),
        "joint_vel": np.arange(7, dtype=np.float32) + 0.1,
    }

    robot_state = _normalize_robot_state(obs_dict)

    assert robot_state["eef"]["pos"].shape == (1, 3)
    assert robot_state["eef"]["quat"].shape == (1, 4)
    assert robot_state["eef"]["mat"].shape == (1, 3, 3)
    np.testing.assert_allclose(robot_state["eef"]["mat"][0], np.eye(3, dtype=np.float32))


def test_xvla_build_env_observation_uses_pixels_and_robot_state() -> None:
    adapter = object.__new__(_XVLAPluginAdapter)
    obs_dict = {
        "front_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "wrist_rgb": np.ones((32, 32, 3), dtype=np.uint8),
        "eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
        "gripper_qvel": np.array([0.3, 0.4], dtype=np.float32),
        "joint_pos": np.arange(7, dtype=np.float32),
        "joint_vel": np.arange(7, dtype=np.float32) + 0.5,
    }

    observation = adapter._build_env_observation(obs_dict)

    assert observation["pixels"]["image"].shape == (32, 32, 3)
    assert observation["pixels"]["image2"].shape == (32, 32, 3)
    assert observation["robot_state"]["eef"]["mat"].shape == (1, 3, 3)


def test_xvla_resolve_domain_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROEP_XVLA_DOMAIN_ID", "3")
    assert _resolve_domain_id() == 3


def test_xvla_rejects_invalid_domain_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROEP_XVLA_DOMAIN_ID", "not-an-int")
    with pytest.raises(AdapterNotReadyError, match="ROEP_XVLA_DOMAIN_ID"):
        _resolve_domain_id()


def test_xvla_domain_id_override_is_injected_before_select_action() -> None:
    captured: dict[str, object] = {}

    class _Policy:
        def reset(self) -> None:
            return None

        def select_action(self, observation):
            captured["domain_id"] = observation.get("domain_id")
            return np.zeros((1, 1, 10), dtype=np.float32)

    adapter = object.__new__(_XVLAPluginAdapter)
    adapter._hook = None
    adapter._policy = _Policy()
    adapter._preprocess_observation = lambda observation: observation
    adapter._env_preprocessor = lambda observation: observation
    adapter._preprocessor = lambda observation: observation
    adapter._postprocessor = lambda action: action
    adapter._env_postprocessor = lambda action: action
    adapter._torch = SimpleNamespace(
        tensor=lambda data, dtype=None, device=None: np.asarray(data, dtype=np.int64),
        long=np.int64,
        inference_mode=lambda: SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: False),
    )
    adapter._action_key = "action"
    adapter._domain_id_override = 3
    adapter.device = "cpu"

    obs_dict = {
        "_episode_reset": False,
        "front_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "wrist_rgb": np.ones((32, 32, 3), dtype=np.uint8),
        "eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
        "gripper_qvel": np.array([0.3, 0.4], dtype=np.float32),
        "joint_pos": np.arange(7, dtype=np.float32),
        "joint_vel": np.arange(7, dtype=np.float32) + 0.5,
    }

    # replace the context manager with a proper dummy object
    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    adapter._torch.inference_mode = lambda: _Ctx()
    adapter.predict_action(obs_dict, "task")
    np.testing.assert_array_equal(captured["domain_id"], np.array([3], dtype=np.int64))
