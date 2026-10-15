import unittest

import numpy as np

from roepbench.model_plugins.smolvla_base_plugin import (
    _SmolVLAPluginAdapter,
    _normalize_robot_state,
)


class SmolVLABatchingTests(unittest.TestCase):
    def test_normalize_robot_state_adds_singleton_batch(self) -> None:
        obs_dict = {
            "robot_state": {
                "eef": {
                    "pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                },
                "gripper": {
                    "qpos": np.array([0.4, 0.5], dtype=np.float32),
                    "qvel": np.array([0.6, 0.7], dtype=np.float32),
                },
                "joints": {
                    "pos": np.arange(7, dtype=np.float32),
                    "vel": np.arange(7, dtype=np.float32) + 0.1,
                },
            }
        }

        robot_state = _normalize_robot_state(obs_dict)

        self.assertEqual(robot_state["eef"]["pos"].shape, (1, 3))
        self.assertEqual(robot_state["eef"]["quat"].shape, (1, 4))
        self.assertEqual(robot_state["gripper"]["qpos"].shape, (1, 2))
        self.assertEqual(robot_state["gripper"]["qvel"].shape, (1, 2))
        self.assertEqual(robot_state["joints"]["pos"].shape, (1, 7))
        self.assertEqual(robot_state["joints"]["vel"].shape, (1, 7))

    def test_build_env_observation_uses_batched_robot_state(self) -> None:
        adapter = object.__new__(_SmolVLAPluginAdapter)
        obs_dict = {
            "front_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((32, 32, 3), dtype=np.uint8),
            "eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
            "gripper_qvel": np.array([0.3, 0.4], dtype=np.float32),
            "joint_pos": np.arange(7, dtype=np.float32),
            "joint_vel": np.arange(7, dtype=np.float32) + 0.5,
        }

        observation = adapter._build_env_observation(obs_dict)

        self.assertIn("pixels", observation)
        self.assertIn("robot_state", observation)
        self.assertEqual(observation["pixels"]["image"].shape, (32, 32, 3))
        self.assertEqual(observation["robot_state"]["eef"]["pos"].shape, (1, 3))
        self.assertEqual(observation["robot_state"]["eef"]["quat"].shape, (1, 4))
        self.assertEqual(observation["robot_state"]["gripper"]["qpos"].shape, (1, 2))
        self.assertEqual(observation["robot_state"]["joints"]["pos"].shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
