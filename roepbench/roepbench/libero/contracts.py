"""Helpers for implementing ROEPBench-compatible LIBERO runtime hooks."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


def build_pairing_key(*, task: str, seed: int, episode_idx: int, eval_demo_seed: int, lang_goal: str) -> str:
    return f"{task}|{seed}|{episode_idx}|{eval_demo_seed}|{lang_goal}"


def build_episode_record(
    *,
    run_id: str,
    task: str,
    seed: int,
    episode_idx: int,
    eval_demo_seed: int,
    lang_goal: str,
    success: bool,
    num_steps: int,
    total_reward: float,
    start_time: float,
    end_time: Optional[float] = None,
    policy_name: str = "external_policy",
    action_outcome_label: str = "success",
    threat_model: str = "none",
    attack_channel: str = "none",
    attack_applied: bool = False,
    ood_active: bool = False,
    action_mode_profile: str = "joint_position_discrete",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    finished_at = float(end_time if end_time is not None else time.time())
    duration = max(0.0, finished_at - float(start_time))
    pairing_key = build_pairing_key(
        task=task,
        seed=seed,
        episode_idx=episode_idx,
        eval_demo_seed=eval_demo_seed,
        lang_goal=lang_goal,
    )
    record = {
        "run_id": run_id,
        "episode_idx": int(episode_idx),
        "task": str(task),
        "seed": int(seed),
        "eval_demo_seed": int(eval_demo_seed),
        "lang_goal": str(lang_goal),
        "pairing_key": pairing_key,
        "success": bool(success),
        "success_source": "task_signal",
        "terminated": bool(success),
        "timeout": False,
        "num_steps": int(num_steps),
        "total_reward": float(total_reward),
        "duration_sec": float(duration),
        "policy": str(policy_name),
        "benchmark_backend": "libero",
        "unsafe_action_rate": 0.0,
        "ood_active": bool(ood_active),
        "recovery_triggered": False,
        "recovery_count": 0,
        "recovery_steps": 0,
        "recovery_first_step": None,
        "intervention_count": 0,
        "ttr_steps": None,
        "unsafe_action_rate_proxy": 0.0,
        "unsafe_action_rate_exception_all": 0.0,
        "unsafe_action_rate_total": 0.0,
        "uar_scope": "portable_partial",
        "label_protocol_version": "v2",
        "action_outcome_label": str(action_outcome_label),
        "safe_refusal": False,
        "unsafe_attempt": False,
        "invalid_command": False,
        "threat_model": str(threat_model),
        "attack_channel": str(attack_channel),
        "attack_applied": bool(attack_applied),
        "attack_variant": "none",
        "attack_sampling_mode": "none" if not attack_applied else "deterministic_hash",
        "action_mode_profile": str(action_mode_profile),
        "detector_type": "none",
        "recovery_mode": "none",
        "detector_candidate_count": 0,
        "detector_trigger_count": 0,
        "recovery_action_applied_steps": 0,
        "recovery_guard_trigger_count": 0,
        "recovery_guard_abort_count": 0,
        "trigger_induced_blackout_steps": 0,
        "recovery_watch_trigger_count": 0,
        "recovery_risk_trigger_count": 0,
        "recovery_state_transition_count": 0,
        "ood_gate_open_steps": 0,
        "debug_integrity": False,
        "recovery_enabled_effective": False,
        "recovery_disabled_reason": "disabled",
        "blackout_on_recovery_trigger": False,
        "recovery_state": "idle",
        "recovery_trigger_reason": "",
        "unsafe_action_count_invalid": 0,
        "unsafe_action_count_ik": 0,
        "unsafe_action_count_path": 0,
        "risk_suite_enabled": False,
        "risk_attack_counts_as_ood": False,
        "attack_assignment_id": "",
        "attack_probability": 0.0,
        "attack_severity": 0.0,
        "attack_original_lang_goal": str(lang_goal),
        "attack_attacked_lang_goal": str(lang_goal),
        "attack_surface_instruction": False,
        "attack_surface_visual": False,
        "attack_surface_conceptual": False,
        "attack_visual_overlay_pixels": 0,
        "attack_visual_applied_steps": 0,
        "error": "",
        "perturbation": {
            "occlusion_severity": 0.0,
            "depth_dropout_prob": 0.0,
            "occluded_pixels": 0,
            "dropped_depth_pixels": 0,
        },
        "observation": {},
    }
    if extra:
        record.update(extra)
    return record


def build_step_record(
    *,
    run_id: str,
    task: str,
    seed: int,
    episode_idx: int,
    step_idx: int,
    action: Any,
    lang_goal: str,
    action_mode_profile: str = "joint_position_discrete",
    ood_active: bool = False,
    threat_model: str = "none",
    attack_channel: str = "none",
    attack_applied: bool = False,
    done: bool = False,
    reward: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record = {
        "run_id": str(run_id),
        "task": str(task),
        "seed": int(seed),
        "episode_idx": int(episode_idx),
        "step_idx": int(step_idx),
        "action": action,
        "reward": float(reward),
        "done": bool(done),
        "lang_goal": str(lang_goal),
        "benchmark_backend": "libero",
        "action_mode_profile": str(action_mode_profile),
        "ood_active": bool(ood_active),
        "threat_model": str(threat_model),
        "attack_channel": str(attack_channel),
        "attack_applied": bool(attack_applied),
    }
    if extra:
        record.update(extra)
    return record
