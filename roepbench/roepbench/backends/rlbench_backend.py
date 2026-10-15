"""RLBench backend adapter for the legacy ROEPBench runner."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..rlbench_runner import run_episodes
from ..task_loader import load_task_class
from .base import EpisodeBackend, EpisodeRunConfig


class RLBenchBackend(EpisodeBackend):
    """Bridge the backend-neutral config into the existing RLBench runner."""

    backend_name = "rlbench"

    def run_policy_episodes(self, config: EpisodeRunConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        task_class = load_task_class(config.task)
        return run_episodes(
            task_class=task_class,
            task_name=config.task,
            episodes=config.episodes,
            seed=config.seed,
            max_steps=config.max_steps,
            headless=config.headless,
            run_id=config.run_id,
            occlusion_severity=config.occlusion_severity,
            depth_dropout_prob=config.depth_dropout_prob,
            enable_recovery=config.enable_recovery,
            detector=config.detector,
            reward_window=config.reward_window,
            reward_eps=config.reward_eps,
            confirm_windows=config.confirm_windows,
            trigger_cooldown_steps=config.trigger_cooldown_steps,
            margin_eps=config.margin_eps,
            min_plateau_steps=config.min_plateau_steps,
            stagnation_window=config.stagnation_window,
            stagnation_eps=config.stagnation_eps,
            recovery_burst_steps=config.recovery_burst_steps,
            warmup_steps=config.warmup_steps,
            safety_shield_checkpoint=config.safety_shield_checkpoint,
            safety_shield_device=config.safety_shield_device,
            safety_shield_tau=config.safety_shield_tau,
            safety_shield_mdsp_mode=config.safety_shield_mdsp_mode,
            safety_shield_action_low=config.safety_shield_action_low,
            safety_shield_action_high=config.safety_shield_action_high,
            safety_shield_refusal_risk_threshold=config.safety_shield_refusal_risk_threshold,
            safety_shield_verify_projected_action=config.safety_shield_verify_projected_action,
            safety_shield_z_lift=config.safety_shield_z_lift,
            safety_shield_xy_scale=config.safety_shield_xy_scale,
            safety_shield_max_translation_norm=config.safety_shield_max_translation_norm,
            policy_fn=config.policy_fn,
            policy_name=config.policy_name,
            strict_runtime_errors=config.strict_runtime_errors,
            risk_suite_enabled=config.risk_suite_enabled,
            risk_attack_channel=config.risk_attack_channel,
            risk_attack_probability=config.risk_attack_probability,
            risk_attack_severity=config.risk_attack_severity,
            risk_visual_alpha=config.risk_visual_alpha,
            risk_visual_patch_ratio=config.risk_visual_patch_ratio,
            risk_threat_model=config.risk_threat_model,
            risk_instruction_prefix=config.risk_instruction_prefix,
            risk_instruction_suffix=config.risk_instruction_suffix,
            risk_attack_counts_as_ood=config.risk_attack_counts_as_ood,
            risk_attack_sampling_mode=config.risk_attack_sampling_mode,
            risk_attack_hash_salt=config.risk_attack_hash_salt,
            action_mode_profile=config.action_mode_profile,
        )
