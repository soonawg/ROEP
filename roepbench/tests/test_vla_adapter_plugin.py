from __future__ import annotations

from collections import deque

import numpy as np

from roepbench.model_plugins import vla_adapter_plugin as plugin


def test_resolve_unnorm_key_prefers_suite_no_noops():
    norm_stats = {
        "bridge_orig": {},
        "libero_goal_no_noops": {},
    }
    assert plugin._resolve_unnorm_key(norm_stats, "libero_goal") == "libero_goal_no_noops"


def test_default_save_version_uses_checkpoint_tail():
    assert (
        plugin._default_save_version("/tmp/models/vla_adapter_libero_goal_pro")
        == "vla_adapter_libero_goal_pro"
    )
    assert plugin._default_save_version("VLA-Adapter/LIBERO-Goal-Pro") == "LIBERO-Goal-Pro"


def test_build_libero_observation_constructs_8d_state():
    obs = {
        "front_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "wrist_rgb": np.ones((32, 32, 3), dtype=np.uint8),
        "eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper_qpos": np.array([0.4, 0.5], dtype=np.float32),
    }

    built = plugin._build_libero_observation(obs)

    assert built["full_image"].shape == (32, 32, 3)
    assert built["wrist_image"].shape == (32, 32, 3)
    assert built["state"].shape == (8,)
    np.testing.assert_allclose(built["state"][:3], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(built["state"][3:6], np.zeros((3,), dtype=np.float32))
    np.testing.assert_allclose(built["state"][6:], np.array([0.4, 0.5], dtype=np.float32))


def test_postprocess_action_matches_official_libero_gripper_convention():
    raw = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.0], dtype=np.float32)
    processed = plugin._postprocess_action(raw)
    np.testing.assert_allclose(processed[:-1], raw[:-1])
    assert processed[-1] == 1.0

    raw_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    processed_open = plugin._postprocess_action(raw_open)
    assert processed_open[-1] == -1.0


def test_predict_action_consumes_open_loop_queue_before_requery():
    calls = {"count": 0}

    def fake_get_vla_action(**_kwargs):
        calls["count"] += 1
        return [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0], dtype=np.float32),
            np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 1.0], dtype=np.float32),
        ]

    adapter = plugin._VLAAdapterPluginAdapter.__new__(plugin._VLAAdapterPluginAdapter)
    adapter._cfg = object()
    adapter._model = object()
    adapter._processor = object()
    adapter._action_head = object()
    adapter._proprio_projector = object()
    adapter._get_vla_action = fake_get_vla_action
    adapter._action_queue = deque()

    obs = {
        "front_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "wrist_rgb": np.ones((32, 32, 3), dtype=np.uint8),
        "eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "gripper_qpos": np.array([0.4, 0.5], dtype=np.float32),
    }

    first = adapter.predict_action(obs, "put the bowl on top of the cabinet")
    second = adapter.predict_action(obs, "put the bowl on top of the cabinet")

    assert calls["count"] == 1
    np.testing.assert_allclose(first[:6], np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32))
    assert first[6] == 1.0
    np.testing.assert_allclose(second[:6], np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], dtype=np.float32))
    assert second[6] == -1.0
