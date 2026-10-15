from __future__ import annotations

import numpy as np

from roepbench.model_adapters.pi0_fast import PI0FastAdapter
from roepbench.model_plugins.pi0_fast_plugin import _PI0FastPluginAdapter, _normalize_robot_state


def test_pi0_fast_adapter_accepts_remote_repo_id_checkpoint() -> None:
    adapter = PI0FastAdapter.__new__(PI0FastAdapter)
    adapter.model_name = "pi0_fast"
    adapter.checkpoint_env_var = "ROEP_PI0_FAST_CKPT"
    adapter.config = type("Cfg", (), {"checkpoint_path": "lerobot/pi0fast-libero-v044"})()
    adapter.ensure_ready()


def test_pi0_fast_normalize_robot_state_batches_inputs() -> None:
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
    assert robot_state["gripper"]["qpos"].shape == (1, 2)
    assert robot_state["joints"]["pos"].shape == (1, 7)


def test_pi0_fast_build_env_observation_uses_pixels_and_robot_state() -> None:
    adapter = object.__new__(_PI0FastPluginAdapter)
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
    assert observation["robot_state"]["eef"]["pos"].shape == (1, 3)
