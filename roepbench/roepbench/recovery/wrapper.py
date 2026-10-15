from typing import Dict, List, Optional

import numpy as np

from .detectors import (
    LowDimDeltaDetector,
    RewardPlateauConfirmDetector,
    RewardPlateauDetector,
)
from .primitives import RandomBurstRecovery, RegraspRecovery, SafeRetractRecovery


class RecoverVLAWrapper:
    """Rule-based RecoverVLA v0 wrapper that orchestrates detector + recovery primitive."""

    def __init__(
        self,
        enable_recovery: bool,
        detector: str,
        warmup_steps: int,
        reward_window: int,
        reward_eps: float,
        stagnation_window: int,
        stagnation_eps: float,
        recovery_burst_steps: int,
        primitive_sequence: Optional[List[str]] = None,
        confirm_windows: int = 3,
        trigger_cooldown_steps: int = 30,
        margin_eps: float = 0.0,
        min_plateau_steps: int = 24,
    ):
        self.enable_recovery = bool(enable_recovery)
        self.detector_type = str(detector)
        self.warmup_steps = max(0, int(warmup_steps))
        self.reward_window = max(1, int(reward_window))
        self.reward_eps = float(reward_eps)
        self.confirm_windows = max(1, int(confirm_windows))
        self.trigger_cooldown_steps = max(0, int(trigger_cooldown_steps))
        self.margin_eps = float(max(0.0, margin_eps))
        self.min_plateau_steps = max(1, int(min_plateau_steps))
        self.stagnation_window = max(1, int(stagnation_window))
        self.stagnation_eps = float(stagnation_eps)
        self.primitive = RandomBurstRecovery(recovery_burst_steps)
        self.primitive_map = {
            "random_burst": RandomBurstRecovery(recovery_burst_steps),
            "safe_retract_like": SafeRetractRecovery(recovery_burst_steps),
            "regrasp_like": RegraspRecovery(recovery_burst_steps),
        }
        self.primitive_sequence = (
            primitive_sequence
            if primitive_sequence
            else ["random_burst", "safe_retract_like", "regrasp_like"]
        )
        self._primitive_idx = 0
        self.current_primitive_mode = "random_burst"
        self._last_primitive_mode = "random_burst"
        self.recovery_primitive_sequence: List[str] = []

        if self.detector_type == "reward_plateau":
            self.detector = RewardPlateauDetector(
                window=self.reward_window,
                eps=self.reward_eps,
            )
        elif self.detector_type == "plateau_with_confirm":
            self.detector = RewardPlateauConfirmDetector(
                window=self.reward_window,
                eps=self.reward_eps,
                confirm_windows=self.confirm_windows,
                margin_eps=self.margin_eps,
                min_plateau_steps=self.min_plateau_steps,
            )
        elif self.detector_type == "plateau_strict":
            self.detector = RewardPlateauConfirmDetector(
                window=max(self.reward_window, 30),
                eps=self.reward_eps,
                confirm_windows=max(self.confirm_windows, 4),
                margin_eps=self.margin_eps,
                min_plateau_steps=max(self.min_plateau_steps, 30),
            )
        elif self.detector_type == "plateau_with_margin":
            self.detector = RewardPlateauConfirmDetector(
                window=max(self.reward_window, 20),
                eps=self.reward_eps,
                confirm_windows=max(2, self.confirm_windows),
                margin_eps=max(self.margin_eps, 1e-4),
                min_plateau_steps=max(self.min_plateau_steps, 20),
            )
        elif self.detector_type == "plateau_with_confirm_v2":
            self.detector = RewardPlateauConfirmDetector(
                window=self.reward_window,
                eps=self.reward_eps,
                confirm_windows=max(self.confirm_windows, 4),
                margin_eps=self.margin_eps,
                min_plateau_steps=self.min_plateau_steps,
            )
        elif self.detector_type == "plateau_strict_v2":
            self.detector = RewardPlateauConfirmDetector(
                window=max(self.reward_window, 32),
                eps=self.reward_eps,
                confirm_windows=max(self.confirm_windows, 5),
                margin_eps=self.margin_eps,
                min_plateau_steps=max(self.min_plateau_steps, 32),
            )
        elif self.detector_type == "plateau_margin_v2":
            self.detector = RewardPlateauConfirmDetector(
                window=max(self.reward_window, 24),
                eps=self.reward_eps,
                confirm_windows=max(self.confirm_windows, 4),
                margin_eps=max(self.margin_eps, 2e-4),
                min_plateau_steps=max(self.min_plateau_steps, 24),
            )
        else:
            self.detector = LowDimDeltaDetector(
                window=self.stagnation_window,
                eps=self.stagnation_eps,
            )

        self.recovery_triggered = False
        self.recovery_count = 0
        self.recovery_steps = 0
        self.recovery_action_applied_steps = 0
        self.recovery_first_step: Optional[int] = None
        self.recovery_attempts_exhausted = False
        self._last_detector_trigger_step: Optional[int] = None
        self.detector_candidate_count = 0
        self.detector_trigger_count = 0

    def on_step(self, num_steps: int, reward: float, low_dim: Optional[np.ndarray]) -> None:
        if self.detector_type in (
            "reward_plateau",
            "plateau_with_confirm",
            "plateau_strict",
            "plateau_with_margin",
            "plateau_with_confirm_v2",
            "plateau_strict_v2",
            "plateau_margin_v2",
        ):
            self.detector.update(float(reward))
        else:
            self.detector.update(low_dim)

    def should_trigger(self, num_steps: int, commit: bool = True) -> bool:
        if not self.enable_recovery:
            return False
        raw_flag = bool(
            self.detector.should_trigger(num_steps=num_steps, warmup_steps=self.warmup_steps)
        )
        if raw_flag:
            self.detector_candidate_count += 1

        if self._last_detector_trigger_step is not None:
            if int(num_steps) - int(self._last_detector_trigger_step) < self.trigger_cooldown_steps:
                return False

        if raw_flag:
            if commit:
                self._last_detector_trigger_step = int(num_steps)
                self.detector_trigger_count += 1
        return raw_flag

    def commit_trigger(self, num_steps: int) -> bool:
        """Commit a previously evaluated trigger event.

        This is used by integration layers that maintain an outer state machine
        (e.g., WATCH -> RECOVER) and need to separate "candidate observed"
        from "trigger committed".
        """
        if not self.enable_recovery:
            return False
        if self._last_detector_trigger_step is not None:
            if int(num_steps) - int(self._last_detector_trigger_step) < self.trigger_cooldown_steps:
                return False
        self._last_detector_trigger_step = int(num_steps)
        self.detector_trigger_count += 1
        return True

    def start_recovery(self, num_steps: int, max_steps: int) -> int:
        self.recovery_triggered = True
        self.recovery_count += 1
        if self.recovery_first_step is None:
            self.recovery_first_step = int(num_steps)
        mode = self.primitive_sequence[self._primitive_idx % len(self.primitive_sequence)]
        self._primitive_idx += 1
        self.current_primitive_mode = str(mode)
        self._last_primitive_mode = str(mode)
        self.recovery_primitive_sequence.append(str(mode))
        primitive = self.primitive_map.get(str(mode), self.primitive)
        burst_steps = primitive.get_burst_steps(current_step=num_steps, max_steps=max_steps)
        self.recovery_attempts_exhausted = burst_steps <= 0
        return burst_steps

    def on_recovery_step(self) -> None:
        self.recovery_steps += 1
        self.recovery_action_applied_steps += 1

    def reset_after_burst(self) -> None:
        self.detector.reset()

    def to_log_fields(self) -> Dict:
        eff_window = getattr(self.detector, "window", None)
        eff_confirm = getattr(self.detector, "confirm_windows", None)
        eff_margin = getattr(self.detector, "margin_eps", None)
        eff_min_plateau = getattr(self.detector, "min_plateau_steps", None)
        return {
            "recovery_triggered": bool(self.recovery_triggered),
            "recovery_count": int(self.recovery_count),
            "recovery_steps": int(self.recovery_steps),
            "recovery_action_applied_steps": int(self.recovery_action_applied_steps),
            "recovery_first_step": self.recovery_first_step,
            "detector_candidate_count": int(self.detector_candidate_count),
            "detector_trigger_count": int(self.detector_trigger_count),
            "detector_type": self.detector_type,
            "reward_window": self.reward_window,
            "reward_eps": self.reward_eps,
            "confirm_windows": self.confirm_windows,
            "trigger_cooldown_steps": self.trigger_cooldown_steps,
            "margin_eps": self.margin_eps,
            "min_plateau_steps": self.min_plateau_steps,
            "detector_effective_window": int(eff_window) if eff_window is not None else None,
            "detector_effective_confirm_windows": int(eff_confirm) if eff_confirm is not None else None,
            "detector_effective_margin_eps": float(eff_margin) if eff_margin is not None else None,
            "detector_effective_min_plateau_steps": (
                int(eff_min_plateau) if eff_min_plateau is not None else None
            ),
            "stagnation_eps": self.stagnation_eps if self.detector_type == "low_dim" else None,
            "stagnation_window": self.stagnation_window if self.detector_type == "low_dim" else None,
            "recovery_mode": self._last_primitive_mode,
            "recovery_primitive_sequence": list(self.recovery_primitive_sequence),
            "recovery_last_primitive": self._last_primitive_mode,
            "recovery_attempts_exhausted": bool(self.recovery_attempts_exhausted),
        }
