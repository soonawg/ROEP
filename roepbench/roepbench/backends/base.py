"""Backend-neutral episode runner interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class BackendNotReadyError(RuntimeError):
    """Raised when a benchmark backend is selected but not runnable."""


@dataclass
class EpisodeRunConfig:
    """Normalized runtime config shared by benchmark backends."""

    benchmark_backend: str
    task: str
    episodes: int
    seed: int
    max_steps: int
    headless: bool
    run_id: str
    policy_fn: Optional[Callable[[Dict[str, Any], Optional[str]], Any]] = None
    policy_name: str = "none"
    occlusion_severity: float = 0.0
    depth_dropout_prob: float = 0.0
    enable_recovery: bool = True
    detector: str = "reward_plateau"
    reward_window: int = 25
    reward_eps: float = 0.0
    confirm_windows: int = 3
    trigger_cooldown_steps: int = 30
    margin_eps: float = 0.0
    min_plateau_steps: int = 24
    stagnation_window: int = 15
    stagnation_eps: float = 1e-4
    recovery_burst_steps: int = 8
    warmup_steps: int = 10
    safety_shield_checkpoint: Optional[str] = None
    safety_shield_device: str = "cpu"
    safety_shield_tau: Optional[float] = None
    safety_shield_mdsp_mode: str = "lift_z_scale_xy"
    safety_shield_action_low: float = -1.0
    safety_shield_action_high: float = 1.0
    safety_shield_refusal_risk_threshold: float = 0.995
    safety_shield_verify_projected_action: bool = False
    safety_shield_z_lift: float = 0.05
    safety_shield_xy_scale: float = 0.5
    safety_shield_max_translation_norm: float = 0.25
    strict_runtime_errors: bool = False
    risk_suite_enabled: bool = False
    risk_attack_channel: str = "none"
    risk_attack_probability: float = 0.0
    risk_attack_severity: float = 0.0
    risk_visual_alpha: float = 0.0
    risk_visual_patch_ratio: float = 0.0
    risk_threat_model: str = "none"
    risk_instruction_prefix: str = ""
    risk_instruction_suffix: str = ""
    risk_attack_counts_as_ood: bool = False
    risk_attack_sampling_mode: str = "deterministic_hash"
    risk_attack_hash_salt: str = "roepbench"
    action_mode_profile: str = "joint_position_discrete"
    libero_suite: str = ""
    libero_task_name: str = ""
    libero_benchmark_name: str = ""
    libero_task_id: str = ""
    libero_init_states_path: str = ""
    backend_options: Dict[str, Any] = field(default_factory=dict)


class EpisodeBackend(ABC):
    """Abstract episode backend."""

    backend_name: str = "base"

    @abstractmethod
    def run_policy_episodes(self, config: EpisodeRunConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run evaluation episodes and return episode/step records."""
