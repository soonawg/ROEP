from __future__ import annotations

from collections import deque

import numpy as np

from roepbench.model_adapters.lap_libero import LAPLiberoAdapter
from roepbench.model_plugins.lap_libero_plugin import (
    _LAPLiberoPluginAdapter,
    _build_request,
    _postprocess_action_chunk,
    _quat_to_rot6d,
)


def test_lap_libero_adapter_requires_local_checkpoint(tmp_path) -> None:
    ckpt = tmp_path / "lap"
    ckpt.mkdir()

    adapter = LAPLiberoAdapter.__new__(LAPLiberoAdapter)
    adapter.model_name = "lap_libero"
    adapter.checkpoint_env_var = "ROEP_LAP_LIBERO_CKPT"
    adapter.config = type("Cfg", (), {"checkpoint_path": str(ckpt)})()
    adapter.ensure_ready()


def test_lap_quat_to_rot6d_identity() -> None:
    rot6d = _quat_to_rot6d(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(rot6d, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32))


def test_lap_build_request_matches_upstream_shape_contract() -> None:
    obs_dict = {
        "front_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "wrist_rgb": np.ones((32, 32, 3), dtype=np.uint8),
        "eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper_qpos": np.array([0.02, 0.02], dtype=np.float32),
    }

    request = _build_request(obs_dict, "put the bowl on top of the cabinet")

    assert request["observation"]["base_0_rgb"].shape == (32, 32, 3)
    assert request["observation"]["left_wrist_0_rgb"].shape == (32, 32, 3)
    assert request["observation"]["state"].shape == (10,)
    assert request["prompt"] == "put the bowl on top of the cabinet"


def test_lap_postprocess_action_chunk_matches_official_gripper_convention() -> None:
    chunk = _postprocess_action_chunk(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
            ],
            dtype=np.float32,
        )
    )
    np.testing.assert_allclose(chunk[:, -1], np.array([-1.0, 1.0], dtype=np.float32))


def test_lap_plugin_uses_action_queue_and_clears_on_reset() -> None:
    adapter = object.__new__(_LAPLiberoPluginAdapter)
    adapter._hook = None
    adapter._policy = type(
        "Policy",
        (),
        {
            "infer": lambda self, request: {
                "actions": np.array(
                    [
                        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.25],
                        [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.75],
                    ],
                    dtype=np.float32,
                )
            }
        },
    )()
    adapter._replan_steps = 2
    adapter._default_prompt = None
    adapter._action_queue = deque()

    obs = {
        "_episode_reset": True,
        "front_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "wrist_rgb": np.ones((32, 32, 3), dtype=np.uint8),
        "eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper_qpos": np.array([0.02, 0.02], dtype=np.float32),
    }

    first = adapter.predict_action(obs, "task")
    second = adapter.predict_action({**obs, "_episode_reset": False}, "task")

    np.testing.assert_allclose(first[:3], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert float(first[6]) == 1.0
    np.testing.assert_allclose(second[:3], np.array([0.4, 0.5, 0.6], dtype=np.float32))
    assert float(second[6]) == -1.0
