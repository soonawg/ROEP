"""Native LIBERO rollout hook for ROEPBench."""

from __future__ import annotations

import csv
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..backends.base import BackendNotReadyError, EpisodeRunConfig
from .contracts import build_episode_record, build_step_record


SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

DEFAULT_SETTLE_STEPS = 10
DEFAULT_CAMERA_RESOLUTION = 256
LIBERO_ACTION_DIM = 7


def _debug_enabled() -> bool:
    value = str(os.environ.get("ROEP_LIBERO_DEBUG", "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _debug(message: str) -> None:
    if not _debug_enabled():
        return
    print(f"[libero-debug] {message}", file=sys.stderr, flush=True)


def _numpy():
    import numpy as np

    return np


def _apply_perturbations(*args: Any, **kwargs: Any):
    from ..perturbations import apply_perturbations

    return apply_perturbations(*args, **kwargs)


def _to_numpy_1d_float32(value: Any):
    np = _numpy()
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _to_float_list(value: Any) -> List[float]:
    arr = _to_numpy_1d_float32(value)
    if arr is None:
        return []
    return arr.astype(_numpy().float32, copy=False).reshape(-1).tolist()


def _to_jsonable(value: Any) -> Any:
    np = _numpy()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _figure_capture_dir() -> Optional[Path]:
    raw = str(os.environ.get("ROEP_FIGURE_CAPTURE_DIR", "")).strip()
    if raw == "":
        return None
    return Path(raw).expanduser().resolve()


def _figure_capture_label(config: EpisodeRunConfig) -> str:
    raw = str(os.environ.get("ROEP_FIGURE_CAPTURE_LABEL", "")).strip()
    if raw:
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in config.run_id)


def _figure_capture_every() -> int:
    raw = str(os.environ.get("ROEP_FIGURE_CAPTURE_EVERY", "20")).strip()
    try:
        value = int(raw)
    except ValueError:
        value = 20
    return max(value, 1)


def _save_rgb_png(path: Path, image: Any) -> bool:
    np = _numpy()
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False
    arr = image
    if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    try:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)
        return True
    except Exception:
        return False


def _capture_rollout_frame(
    *,
    config: EpisodeRunConfig,
    suite_name: str,
    task_name: str,
    task_description: str,
    condition: str,
    episode_idx: int,
    step_idx: int,
    obs_dict: Dict[str, Any],
    perturbed_obs: Dict[str, Any],
    final: bool = False,
) -> None:
    capture_dir = _figure_capture_dir()
    if capture_dir is None:
        return
    if not final and step_idx != 0 and step_idx % _figure_capture_every() != 0:
        return

    label = _figure_capture_label(config)
    frame_kind = "final" if final else f"step{step_idx:04d}"
    stem = f"{label}__ep{episode_idx:03d}__{frame_kind}"
    perturbed_path = capture_dir / f"{stem}__perturbed_front.png"
    raw_path = capture_dir / f"{stem}__raw_front.png"
    saved_perturbed = _save_rgb_png(perturbed_path, perturbed_obs.get("front_rgb"))
    saved_raw = _save_rgb_png(raw_path, obs_dict.get("front_rgb"))
    if not (saved_perturbed or saved_raw):
        return

    manifest_path = capture_dir / "figure2_capture_manifest.csv"
    exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "run_id",
                "suite",
                "task",
                "task_description",
                "condition",
                "episode_idx",
                "step_idx",
                "frame_kind",
                "raw_front_path",
                "perturbed_front_path",
                "occlusion_severity",
                "depth_dropout_prob",
                "shield_enabled",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "label": label,
                "run_id": config.run_id,
                "suite": suite_name,
                "task": task_name,
                "task_description": task_description,
                "condition": condition,
                "episode_idx": int(episode_idx),
                "step_idx": int(step_idx),
                "frame_kind": frame_kind,
                "raw_front_path": str(raw_path if saved_raw else ""),
                "perturbed_front_path": str(perturbed_path if saved_perturbed else ""),
                "occlusion_severity": float(config.occlusion_severity),
                "depth_dropout_prob": float(config.depth_dropout_prob),
                "shield_enabled": bool(config.safety_shield_checkpoint),
            }
        )


def _stable_int(*parts: Any) -> int:
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % (2**32)


def _suite_name(config: EpisodeRunConfig) -> str:
    suite = str(config.libero_suite or config.libero_benchmark_name or "").strip().lower()
    if suite == "":
        raise BackendNotReadyError(
            "LIBERO backend requires libero_suite/libero_benchmark_name to be set."
        )
    return suite


def _camera_resolution() -> int:
    raw = str(os.environ.get("ROEP_LIBERO_CAMERA_RESOLUTION", "")).strip()
    if raw == "":
        return DEFAULT_CAMERA_RESOLUTION
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CAMERA_RESOLUTION
    return max(64, value)


def _settle_steps() -> int:
    raw = str(os.environ.get("ROEP_LIBERO_SETTLE_STEPS", "")).strip()
    if raw == "":
        return DEFAULT_SETTLE_STEPS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SETTLE_STEPS
    return max(0, value)


def _control_mode() -> str:
    value = str(os.environ.get("ROEP_LIBERO_CONTROL_MODE", "relative")).strip().lower()
    if value not in {"relative", "absolute"}:
        return "relative"
    return value


def _image_preprocess_mode() -> str:
    value = str(os.environ.get("ROEP_LIBERO_IMAGE_PREPROCESS_MODE", "current")).strip().lower()
    if value not in {"current", "lerobot_native"}:
        return "current"
    return value


def _load_libero_modules() -> Dict[str, Any]:
    _debug("importing LIBERO modules")
    try:
        import libero  # type: ignore  # noqa: F401
        from libero.libero import benchmark, get_libero_path  # type: ignore
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise BackendNotReadyError(
            "Official LIBERO package is not importable in the active env. "
            "Install LIBERO into roep_libero_core first."
        ) from exc
    return {
        "benchmark": benchmark,
        "get_libero_path": get_libero_path,
        "OffScreenRenderEnv": OffScreenRenderEnv,
    }


def _resolve_task(benchmark_module: Any, config: EpisodeRunConfig) -> Tuple[Any, int, Any]:
    suite_name = _suite_name(config)
    _debug(f"resolving suite={suite_name}")
    suite_dict = benchmark_module.get_benchmark_dict()
    if suite_name not in suite_dict:
        available = ", ".join(sorted(suite_dict))
        raise BackendNotReadyError(
            f"Unknown LIBERO suite '{suite_name}'. Available: {available}"
        )
    suite = suite_dict[suite_name]()

    task_id_raw = str(config.libero_task_id or "").strip()
    if task_id_raw != "":
        task_id = int(task_id_raw)
        if task_id < 0 or task_id >= int(suite.get_num_tasks()):
            raise BackendNotReadyError(
                f"LIBERO task_id out of range: {task_id} for suite '{suite_name}'"
            )
        return suite, task_id, suite.get_task(task_id)

    requested = str(config.libero_task_name or config.task or "").strip()
    requested_lower = requested.lower()
    task_names = list(suite.get_task_names())
    for idx, task_name in enumerate(task_names):
        if str(task_name).strip().lower() == requested_lower:
            return suite, idx, suite.get_task(idx)
    for idx in range(int(suite.get_num_tasks())):
        task = suite.get_task(idx)
        if str(getattr(task, "language", "")).strip().lower() == requested_lower:
            return suite, idx, task

    preview = ", ".join(task_names[: min(5, len(task_names))])
    raise BackendNotReadyError(
        f"Could not resolve LIBERO task '{requested}' in suite '{suite_name}'. "
        f"Example task names: {preview}"
    )


def _load_init_states(config: EpisodeRunConfig, suite: Any, task: Any, task_id: int) -> Any:
    path = str(config.libero_init_states_path or "").strip()
    if path:
        _debug(f"loading init states from path={path}")
        try:
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise BackendNotReadyError(
                "ROEP_LIBERO_INIT_STATES_PATH is set but torch is not importable in the core env."
            ) from exc
        init_states_path = Path(path)
        if not init_states_path.exists():
            raise BackendNotReadyError(f"LIBERO init states file does not exist: {init_states_path}")
        return torch.load(str(init_states_path), weights_only=False)  # nosec B614

    try:
        _debug(f"loading default init states for task_id={task_id}")
        return suite.get_task_init_states(task_id)
    except Exception as exc:  # noqa: BLE001
        raise BackendNotReadyError(
            f"Failed to load default LIBERO init states for task '{task.name}' (task_id={task_id})."
        ) from exc


def _build_env(mods: Dict[str, Any], task: Any, resolution: int) -> Any:
    task_bddl_file = os.path.join(
        mods["get_libero_path"]("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    try:
        _debug(f"constructing OffScreenRenderEnv task={task.name} resolution={resolution}")
        env = mods["OffScreenRenderEnv"](**env_args)
        env.reset()
    except Exception as exc:  # noqa: BLE001
        raise BackendNotReadyError(
            f"Failed to construct LIBERO OffScreenRenderEnv for task '{task.name}'."
        ) from exc
    return env


def _extract_obs_dict(raw_obs: Dict[str, Any], env: Any = None) -> Dict[str, Any]:
    np = _numpy()
    front = raw_obs.get("agentview_image")
    wrist = raw_obs.get("robot0_eye_in_hand_image")
    if _image_preprocess_mode() == "current":
        if isinstance(front, np.ndarray):
            front = front[::-1, ::-1].copy()
        if isinstance(wrist, np.ndarray):
            wrist = wrist[::-1, ::-1].copy()

    eef_mat = None
    if env is not None:
        try:
            eef_mat = env.robots[0].controller.ee_ori_mat
        except Exception:
            eef_mat = None

    robot_state = {
        "eef": {
            "pos": raw_obs.get("robot0_eef_pos"),
            "quat": raw_obs.get("robot0_eef_quat"),
            "mat": eef_mat,
        },
        "gripper": {
            "qpos": raw_obs.get("robot0_gripper_qpos"),
            "qvel": raw_obs.get("robot0_gripper_qvel"),
        },
        "joints": {
            "pos": raw_obs.get("robot0_joint_pos"),
            "vel": raw_obs.get("robot0_joint_vel"),
        },
    }
    low_dim_parts = []
    for key in (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot0_joint_pos",
        "robot0_joint_vel",
    ):
        arr = _to_numpy_1d_float32(raw_obs.get(key))
        if arr is not None and arr.size > 0:
            low_dim_parts.append(arr)
    low_dim = (
        np.concatenate(low_dim_parts, axis=0).astype(np.float32, copy=False)
        if low_dim_parts
        else None
    )
    return {
        "front_rgb": front,
        "wrist_rgb": wrist,
        "agentview_rgb": front,
        "robot_state": robot_state,
        "eef_pos": raw_obs.get("robot0_eef_pos"),
        "eef_quat": raw_obs.get("robot0_eef_quat"),
        "eef_mat": eef_mat,
        "gripper_qpos": raw_obs.get("robot0_gripper_qpos"),
        "gripper_qvel": raw_obs.get("robot0_gripper_qvel"),
        "joint_pos": raw_obs.get("robot0_joint_pos"),
        "joint_vel": raw_obs.get("robot0_joint_vel"),
        "low_dim": low_dim,
        "obs_low_dim": low_dim,
        "low_dim_state": low_dim,
        "benchmark_backend": "libero",
    }


def _normalized_policy_action(raw_action: Any, policy_name: str) -> np.ndarray:
    np = _numpy()
    arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if arr.size < LIBERO_ACTION_DIM:
        padded = np.zeros((LIBERO_ACTION_DIM,), dtype=np.float32)
        if arr.size > 0:
            padded[: arr.size] = arr
        arr = padded
    elif arr.size > LIBERO_ACTION_DIM:
        arr = arr[:LIBERO_ACTION_DIM]

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32, copy=False)
    model_name = str(policy_name or "").strip().lower()
    gripper = float(arr[-1])

    if "openvla" in model_name:
        if -0.05 <= gripper <= 1.05:
            gripper = 2.0 * gripper - 1.0
        gripper = 1.0 if gripper >= 0.0 else -1.0
        gripper *= -1.0
    else:
        if -0.05 <= gripper <= 1.05:
            gripper = 1.0 if gripper >= 0.5 else -1.0
        else:
            gripper = 1.0 if gripper >= 0.0 else -1.0

    arr[-1] = gripper
    if "xvla" in model_name and _control_mode() == "absolute":
        return arr
    return np.clip(arr, -1.0, 1.0)


def _normalized_env_action(raw_action: Any) -> Any:
    np = _numpy()
    arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if arr.size < LIBERO_ACTION_DIM:
        padded = np.zeros((LIBERO_ACTION_DIM,), dtype=np.float32)
        if arr.size > 0:
            padded[: arr.size] = arr
        arr = padded
    elif arr.size > LIBERO_ACTION_DIM:
        arr = arr[:LIBERO_ACTION_DIM]
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32, copy=False)
    if _control_mode() == "absolute":
        return arr
    return np.clip(arr, -1.0, 1.0)


def _suite_max_steps(config: EpisodeRunConfig, suite_name: str) -> int:
    configured = int(config.max_steps)
    if configured > 0 and configured != 120:
        return configured
    return int(SUITE_MAX_STEPS.get(suite_name, max(1, configured)))


def _task_description(task: Any, config: EpisodeRunConfig) -> str:
    text = str(getattr(task, "language", "")).strip()
    if text:
        return text
    return str(config.libero_task_name or config.task)


def _build_safety_shield(config: EpisodeRunConfig) -> Any:
    ckpt = str(config.safety_shield_checkpoint or "").strip()
    if ckpt == "":
        return None
    try:
        from ..recovery.safety_shield import SafetyShield
    except Exception as exc:  # noqa: BLE001
        raise BackendNotReadyError(
            "SafetyShield is not importable in the active LIBERO env."
        ) from exc
    try:
        return SafetyShield(
            checkpoint_path=ckpt,
            device=str(config.safety_shield_device or "cpu"),
            tau_override=config.safety_shield_tau,
            mdsp_mode=str(config.safety_shield_mdsp_mode or "lift_z_scale_xy"),
            action_low=float(config.safety_shield_action_low),
            action_high=float(config.safety_shield_action_high),
            refusal_risk_threshold=float(config.safety_shield_refusal_risk_threshold),
            verify_projected_action=bool(config.safety_shield_verify_projected_action),
            z_lift=float(config.safety_shield_z_lift),
            xy_scale=float(config.safety_shield_xy_scale),
            max_translation_norm=float(config.safety_shield_max_translation_norm),
        )
    except Exception as exc:  # noqa: BLE001
        raise BackendNotReadyError(
            f"Failed to construct SafetyShield for LIBERO runtime ({type(exc).__name__}: {exc})"
        ) from exc


def run_libero_episodes(config: EpisodeRunConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    _debug("run_libero_episodes start")
    mods = _load_libero_modules()
    np = _numpy()
    shield = _build_safety_shield(config)
    shield_enabled = shield is not None
    suite, task_id, task = _resolve_task(mods["benchmark"], config)
    suite_name = _suite_name(config)
    task_name = str(getattr(task, "name", config.task))
    task_description = _task_description(task, config)
    init_states = _load_init_states(config, suite, task, task_id)
    env = _build_env(mods, task, resolution=_camera_resolution())
    max_steps = _suite_max_steps(config, suite_name)
    settle_steps = _settle_steps()
    control_mode = _control_mode()

    if control_mode not in {"relative", "absolute"}:
        control_mode = "relative"

    episode_records: List[Dict[str, Any]] = []
    step_records: List[Dict[str, Any]] = []

    try:
        try:
            env.seed(int(config.seed))
        except Exception:
            pass

        for robot in getattr(env, "robots", []):
            try:
                robot.controller.use_delta = control_mode == "relative"
            except Exception:
                pass

        for episode_idx in range(int(config.episodes)):
            _debug(f"episode {episode_idx}: reset")
            start_time = time.time()
            total_reward = 0.0
            success = False
            timeout = False
            safe_refusal = False
            error = ""
            num_steps = 0
            last_obs_summary: Dict[str, Any] = {}
            occluded_pixels = 0
            dropped_depth_pixels = 0
            attack_visual_applied_steps = 0
            shield_intervention_count = 0
            shield_safe_refusal_count = 0
            shield_risk_scores: List[float] = []
            shield_last_info: Dict[str, Any] = {}
            shield_info_history: List[Dict[str, Any]] = []
            raw_obs = env.reset()
            demo_idx = int(episode_idx % len(init_states))
            _debug(f"episode {episode_idx}: set_init_state demo_idx={demo_idx}")
            raw_obs = env.set_init_state(init_states[demo_idx])

            for _ in range(settle_steps):
                raw_obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            for robot in getattr(env, "robots", []):
                try:
                    robot.controller.use_delta = control_mode == "relative"
                except Exception:
                    pass
            _debug(f"episode {episode_idx}: settle complete")

            episode_step_start = len(step_records)
            for step_idx in range(max_steps):
                _debug(f"episode {episode_idx}: step {step_idx} extract_obs")
                obs_dict = _extract_obs_dict(raw_obs, env=env)
                rng = np.random.RandomState(
                    _stable_int(config.run_id, config.seed, episode_idx, step_idx)
                )
                _debug(f"episode {episode_idx}: step {step_idx} apply_perturbations")
                perturbed_obs, pert_stats = _apply_perturbations(
                    obs_dict,
                    occlusion_severity=float(config.occlusion_severity),
                    depth_dropout_prob=float(config.depth_dropout_prob),
                    gaussian_blur_sigma=0.0,
                    obs_blackout_active=False,
                    rng=rng,
                )
                perturbed_obs["_episode_reset"] = bool(step_idx == 0)
                perturbed_obs["_episode_idx"] = int(episode_idx)
                perturbed_obs["_step_idx"] = int(step_idx)
                perturbed_obs["libero_suite"] = suite_name
                perturbed_obs["libero_task_name"] = task_description
                occluded_pixels += int(pert_stats.get("occluded_pixels", 0))
                dropped_depth_pixels += int(pert_stats.get("dropped_depth_pixels", 0))
                if int(pert_stats.get("occluded_pixels", 0)) > 0:
                    attack_visual_applied_steps += 1
                capture_condition = (
                    "full_medium"
                    if shield_enabled
                    and (
                        float(config.occlusion_severity) > 0.0
                        or float(config.depth_dropout_prob) > 0.0
                    )
                    else "recovery_no_ood"
                    if shield_enabled
                    else "perturb_medium"
                    if (
                        float(config.occlusion_severity) > 0.0
                        or float(config.depth_dropout_prob) > 0.0
                    )
                    else "clean"
                )
                _capture_rollout_frame(
                    config=config,
                    suite_name=suite_name,
                    task_name=task_name,
                    task_description=task_description,
                    condition=capture_condition,
                    episode_idx=episode_idx,
                    step_idx=step_idx,
                    obs_dict=obs_dict,
                    perturbed_obs=perturbed_obs,
                )

                try:
                    if config.policy_fn is None:
                        action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
                    else:
                        _debug(f"episode {episode_idx}: step {step_idx} policy_fn")
                        action = _normalized_policy_action(
                            config.policy_fn(perturbed_obs, task_description),
                            config.policy_name,
                        )
                except Exception as exc:  # noqa: BLE001
                    if config.strict_runtime_errors:
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                    break

                recovery_active = False
                shield_info: Dict[str, Any] = {}
                step_low_dim = _to_float_list(perturbed_obs.get("low_dim"))
                policy_action = action
                env_action = action
                if shield_enabled:
                    try:
                        shield_action, recovery_active, shield_info = shield.step(
                            obs=perturbed_obs,
                            proposed_action=policy_action,
                        )
                    except Exception as shield_exc:  # noqa: BLE001
                        if config.strict_runtime_errors:
                            raise RuntimeError(
                                f"safety shield failed: {type(shield_exc).__name__}: {shield_exc}"
                            ) from shield_exc
                        shield_info = {
                            "intervened": False,
                            "intervention_reason": "shield_error",
                            "shield_error": f"{type(shield_exc).__name__}: {shield_exc}",
                        }
                        shield_action = policy_action
                        recovery_active = False
                    shield_info = _to_jsonable(shield_info)
                    shield_last_info = dict(shield_info)
                    shield_info_history.append(
                        {
                            "step_idx": int(step_idx + 1),
                            **dict(shield_info),
                        }
                    )
                    risk_score = shield_info.get("risk_score")
                    if risk_score is not None:
                        shield_risk_scores.append(float(risk_score))
                    if bool(shield_info.get("intervened", recovery_active)):
                        shield_intervention_count += 1
                    if isinstance(shield_action, str) and shield_action == "__SAFE_REFUSAL__":
                        num_steps = step_idx + 1
                        safe_refusal = True
                        shield_safe_refusal_count += 1
                        step_records.append(
                            build_step_record(
                                run_id=config.run_id,
                                task=task_name,
                                seed=config.seed,
                                episode_idx=episode_idx,
                                step_idx=step_idx,
                                action=policy_action.tolist(),
                                lang_goal=task_description,
                                action_mode_profile=config.action_mode_profile,
                                ood_active=bool(
                                    float(config.occlusion_severity) > 0.0
                                    or float(config.depth_dropout_prob) > 0.0
                                ),
                                threat_model="none",
                                attack_channel="none",
                                attack_applied=False,
                                done=True,
                                reward=0.0,
                                extra={
                                    "env_step_executed": False,
                                    "safe_refusal": True,
                                    "shield_enabled": True,
                                    "recovery_active": bool(recovery_active),
                                    "shield_info": dict(shield_info),
                                    "risk_score": shield_info.get("risk_score"),
                                    "tau": shield_info.get("tau"),
                                    "error": None,
                                    "obs_low_dim": step_low_dim,
                                    "action_outcome_label": "",
                                },
                            )
                        )
                        break
                    env_action = _normalized_env_action(shield_action)

                _debug(f"episode {episode_idx}: step {step_idx} env.step")
                raw_obs, reward, done, info = env.step(env_action.tolist())
                total_reward += float(reward)
                num_steps = step_idx + 1
                is_success = False
                try:
                    is_success = bool(env.check_success())
                except Exception:
                    is_success = bool(done)
                terminated = bool(done) or bool(is_success)
                success = success or bool(is_success)
                last_obs_summary = {
                    "front_rgb_shape": list(perturbed_obs["front_rgb"].shape)
                    if isinstance(perturbed_obs.get("front_rgb"), np.ndarray)
                    else None,
                    "eef_pos": (
                        np.asarray(perturbed_obs.get("eef_pos"), dtype=np.float32).reshape(-1).tolist()
                        if perturbed_obs.get("eef_pos") is not None
                        else None
                    ),
                }

                step_records.append(
                    build_step_record(
                        run_id=config.run_id,
                        task=task_name,
                        seed=config.seed,
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        action=env_action.tolist(),
                        lang_goal=task_description,
                        action_mode_profile=config.action_mode_profile,
                        ood_active=bool(
                            float(config.occlusion_severity) > 0.0
                            or float(config.depth_dropout_prob) > 0.0
                        ),
                        threat_model="none",
                        attack_channel="none",
                        attack_applied=False,
                        done=terminated,
                        reward=float(reward),
                        extra={
                            "action_outcome_label": "",
                            "info": {
                                "done": bool(done),
                                "is_success": bool(
                                    info.get("is_success", is_success)
                                ) if isinstance(info, dict) else bool(is_success),
                            },
                            "perturbation": {
                                "occluded_pixels": int(pert_stats.get("occluded_pixels", 0)),
                                "dropped_depth_pixels": int(pert_stats.get("dropped_depth_pixels", 0)),
                            },
                            "env_step_executed": True,
                            "safe_refusal": False,
                            "shield_enabled": bool(shield_enabled),
                            "recovery_active": bool(recovery_active),
                            "shield_info": dict(shield_info),
                            "risk_score": shield_info.get("risk_score"),
                            "tau": shield_info.get("tau"),
                            "error": None,
                            "obs_low_dim": step_low_dim,
                        },
                    )
                )

                if terminated or step_idx + 1 >= max_steps:
                    _capture_rollout_frame(
                        config=config,
                        suite_name=suite_name,
                        task_name=task_name,
                        task_description=task_description,
                        condition=capture_condition,
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        obs_dict=obs_dict,
                        perturbed_obs=perturbed_obs,
                        final=True,
                    )

                if terminated:
                    break

            if not success and not safe_refusal and not error and num_steps >= max_steps:
                timeout = True

            if safe_refusal:
                action_outcome_label = "safe_refusal"
            elif success:
                action_outcome_label = "success"
            elif timeout:
                action_outcome_label = "timeout"
            elif error:
                action_outcome_label = "invalid_command"
            else:
                action_outcome_label = "unsafe_attempt"

            for i in range(episode_step_start, len(step_records)):
                step_records[i]["action_outcome_label"] = action_outcome_label

            unsafe_attempt = action_outcome_label == "unsafe_attempt"
            invalid_command = action_outcome_label == "invalid_command"
            safe_refusal = action_outcome_label == "safe_refusal"
            first_intervention_step = None
            first_intervention_reason = ""
            for entry in shield_info_history:
                if bool(entry.get("intervened", False)):
                    first_intervention_step = int(entry.get("step_idx"))
                    first_intervention_reason = str(entry.get("intervention_reason", "")).strip()
                    break

            episode_records.append(
                build_episode_record(
                    run_id=config.run_id,
                    task=task_name,
                    seed=config.seed,
                    episode_idx=episode_idx,
                    eval_demo_seed=demo_idx,
                    lang_goal=task_description,
                    success=bool(success),
                    num_steps=int(num_steps),
                    total_reward=float(total_reward),
                    start_time=start_time,
                    end_time=time.time(),
                    policy_name=config.policy_name or "none",
                    action_outcome_label=action_outcome_label,
                    threat_model="none",
                    attack_channel="none",
                    attack_applied=False,
                    ood_active=bool(
                        float(config.occlusion_severity) > 0.0
                        or float(config.depth_dropout_prob) > 0.0
                    ),
                    action_mode_profile=config.action_mode_profile,
                    extra={
                        "success_source": "task_signal",
                        "terminated": bool(success or safe_refusal),
                        "timeout": bool(timeout),
                        "safe_refusal": bool(safe_refusal),
                        "unsafe_attempt": bool(unsafe_attempt),
                        "invalid_command": bool(invalid_command),
                        "unsafe_action_rate": float(1.0 if unsafe_attempt else 0.0),
                        "unsafe_action_rate_proxy": float(1.0 if unsafe_attempt else 0.0),
                        "unsafe_action_rate_exception_all": float(
                            1.0 if (unsafe_attempt or invalid_command) else 0.0
                        ),
                        "unsafe_action_rate_total": float(
                            1.0 if (unsafe_attempt or invalid_command) else 0.0
                        ),
                        "recovery_triggered": bool(shield_intervention_count > 0),
                        "recovery_count": int(shield_intervention_count),
                        "recovery_steps": int(shield_intervention_count),
                        "recovery_action_applied_steps": int(shield_intervention_count),
                        "recovery_first_step": first_intervention_step,
                        "intervention_count": int(shield_intervention_count),
                        "recovery_enabled_effective": bool(shield_enabled),
                        "recovery_disabled_reason": "" if shield_enabled else "libero_baseline_only",
                        "detector_type": "safety_shield" if shield_enabled else "none",
                        "detector_candidate_count": int(shield_intervention_count),
                        "detector_trigger_count": int(shield_intervention_count),
                        "recovery_mode": "safety_shield" if shield_enabled else "none",
                        "shield_enabled": bool(shield_enabled),
                        "shield_intervention_count": int(shield_intervention_count),
                        "shield_safe_refusal_count": int(shield_safe_refusal_count),
                        "shield_mean_risk_score": (
                            float(np.mean(shield_risk_scores)) if shield_risk_scores else None
                        ),
                        "shield_max_risk_score": (
                            float(np.max(shield_risk_scores)) if shield_risk_scores else None
                        ),
                        "shield_last_info": dict(shield_last_info),
                        "shield_info_history": list(shield_info_history),
                        "debug_integrity": False,
                        "ood_gate_open_steps": 0,
                        "blackout_on_recovery_trigger": False,
                        "trigger_induced_blackout_steps": 0,
                        "recovery_state": "RECOVER" if shield_intervention_count > 0 else "NORMAL",
                        "recovery_watch_trigger_count": 0,
                        "recovery_risk_trigger_count": int(shield_intervention_count),
                        "recovery_state_transition_count": int(1 if shield_intervention_count > 0 else 0),
                        "recovery_trigger_reason": first_intervention_reason,
                        "unsafe_action_count_invalid": int(1 if invalid_command else 0),
                        "risk_suite_enabled": bool(config.risk_suite_enabled),
                        "risk_attack_counts_as_ood": bool(config.risk_attack_counts_as_ood),
                        "attack_sampling_mode": str(config.risk_attack_sampling_mode),
                        "attack_probability": float(config.risk_attack_probability),
                        "attack_severity": float(config.risk_attack_severity),
                        "attack_original_lang_goal": task_description,
                        "attack_attacked_lang_goal": task_description,
                        "attack_visual_overlay_pixels": int(occluded_pixels),
                        "attack_visual_applied_steps": int(attack_visual_applied_steps),
                        "error": str(error),
                        "perturbation": {
                            "occlusion_severity": float(config.occlusion_severity),
                            "depth_dropout_prob": float(config.depth_dropout_prob),
                            "occluded_pixels": int(occluded_pixels),
                            "dropped_depth_pixels": int(dropped_depth_pixels),
                        },
                        "observation": last_obs_summary,
                    },
                )
            )
    finally:
        try:
            env.close()
        except Exception:
            pass

    return episode_records, step_records


def build_minimal_example_records(config: EpisodeRunConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Example record constructor for local testing of the artifact contract only."""
    start = time.time()
    steps = [
        build_step_record(
            run_id=config.run_id,
            task=config.task,
            seed=config.seed,
            episode_idx=0,
            step_idx=0,
            action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            lang_goal=config.libero_task_name or config.task,
            action_mode_profile=config.action_mode_profile,
            done=True,
            reward=0.0,
            extra={"note": "artifact-contract example only; not a real LIBERO rollout"},
        )
    ]
    episodes = [
        build_episode_record(
            run_id=config.run_id,
            task=config.task,
            seed=config.seed,
            episode_idx=0,
            eval_demo_seed=0,
            lang_goal=config.libero_task_name or config.task,
            success=False,
            num_steps=1,
            total_reward=0.0,
            start_time=start,
            end_time=time.time(),
            policy_name=config.policy_name,
            action_mode_profile=config.action_mode_profile,
            action_outcome_label="invalid_command",
            extra={"note": "artifact-contract example only; not a real LIBERO rollout"},
        )
    ]
    return episodes, steps
