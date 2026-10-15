from typing import List, Optional

import numpy as np


class RewardPlateauDetector:
    """Detects stagnation when reward range in a recent window is near-constant."""

    def __init__(self, window: int = 25, eps: float = 0.0):
        self.window = max(1, int(window))
        self.eps = float(eps)
        self.reward_history: List[float] = []

    def reset(self) -> None:
        self.reward_history = []

    def update(self, reward: float) -> None:
        self.reward_history.append(float(reward))

    def should_trigger(self, num_steps: int, warmup_steps: int) -> bool:
        if int(num_steps) < int(warmup_steps):
            return False
        if len(self.reward_history) < self.window:
            return False
        recent_rewards = self.reward_history[-self.window :]
        reward_span = float(max(recent_rewards) - min(recent_rewards))
        return reward_span <= self.eps


class RewardPlateauConfirmDetector:
    """Plateau detector with temporal confirmation over consecutive windows."""

    def __init__(
        self,
        window: int = 25,
        eps: float = 0.0,
        confirm_windows: int = 3,
        margin_eps: float = 0.0,
        min_plateau_steps: int = 24,
    ):
        self.window = max(1, int(window))
        self.eps = float(eps)
        self.confirm_windows = max(1, int(confirm_windows))
        self.margin_eps = float(max(0.0, margin_eps))
        self.min_plateau_steps = max(1, int(min_plateau_steps))
        self.reward_history: List[float] = []

    def reset(self) -> None:
        self.reward_history = []

    def update(self, reward: float) -> None:
        self.reward_history.append(float(reward))

    def _window_span_ok(self, end_offset: int) -> bool:
        if end_offset < 0:
            return False
        end = len(self.reward_history) - end_offset
        start = end - self.window
        if start < 0:
            return False
        recent_rewards = self.reward_history[start:end]
        reward_span = float(max(recent_rewards) - min(recent_rewards))
        # margin_eps allows controlled tolerance while still guarding noisy plateaus.
        if reward_span > (self.eps + self.margin_eps):
            return False
        if self.margin_eps > 0.0:
            reward_var = float(np.var(recent_rewards))
            if reward_var > self.margin_eps:
                return False
        return True

    def should_trigger(self, num_steps: int, warmup_steps: int) -> bool:
        if int(num_steps) < int(warmup_steps):
            return False
        needed = max(self.window + self.confirm_windows - 1, self.min_plateau_steps)
        if len(self.reward_history) < needed:
            return False
        for off in range(self.confirm_windows):
            if not self._window_span_ok(end_offset=off):
                return False
        return True


class LowDimDeltaDetector:
    """Backward-compatible low_dim delta stagnation detector."""

    def __init__(self, window: int = 15, eps: float = 1e-4):
        self.window = max(1, int(window))
        self.eps = float(eps)
        self.prev_low_dim: Optional[np.ndarray] = None
        self.delta_history: List[float] = []

    def reset(self) -> None:
        self.prev_low_dim = None
        self.delta_history = []

    def update(self, low_dim: Optional[np.ndarray]) -> None:
        if not isinstance(low_dim, np.ndarray):
            self.prev_low_dim = None
            return

        cur_low_dim = low_dim.astype(np.float32).ravel()
        if self.prev_low_dim is not None and self.prev_low_dim.shape == cur_low_dim.shape:
            delta_t = float(np.linalg.norm(cur_low_dim - self.prev_low_dim, ord=2))
            self.delta_history.append(delta_t)
        self.prev_low_dim = cur_low_dim.copy()

    def should_trigger(self, num_steps: int, warmup_steps: int) -> bool:
        if int(num_steps) < int(warmup_steps):
            return False
        if len(self.delta_history) < self.window:
            return False
        mean_delta = float(np.mean(self.delta_history[-self.window :]))
        return mean_delta < self.eps
