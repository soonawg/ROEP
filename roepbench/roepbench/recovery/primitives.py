from typing import Any, Dict, Tuple, Union

import numpy as np


SAFE_REFUSAL_ACTION = "__SAFE_REFUSAL__"


def _to_numpy_action_1d(action: Any) -> np.ndarray:
    if isinstance(action, np.ndarray):
        return action.astype(np.float32, copy=False).ravel()
    return np.asarray(action, dtype=np.float32).ravel()


def project_action_to_safe_space(
    proposed_action: Any,
    low_dim_state: Any = None,
    risk_score: float = None,
    mode: str = "lift_z_scale_xy",
    action_low: float = -1.0,
    action_high: float = 1.0,
    z_lift: float = 0.05,
    xy_scale: float = 0.5,
    max_translation_norm: float = 0.25,
) -> Tuple[Union[np.ndarray, str], Dict[str, Any]]:
    """Project action to a conservative safe space (MDSP V1).

    Returns:
        (projected_action_or_safe_refusal, metadata)
    """
    del low_dim_state  # reserved for richer future constraints

    meta: Dict[str, Any] = {
        "mode": str(mode),
        "risk_score": None if risk_score is None else float(risk_score),
        "applied": False,
        "reason": "",
        "translation_norm": None,
    }

    try:
        action = _to_numpy_action_1d(proposed_action)
    except Exception:
        meta["reason"] = "action_parse_error"
        return SAFE_REFUSAL_ACTION, meta

    if action.size == 0:
        meta["reason"] = "empty_action"
        return SAFE_REFUSAL_ACTION, meta
    if not np.all(np.isfinite(action)):
        meta["reason"] = "non_finite_action"
        return SAFE_REFUSAL_ACTION, meta

    out = action.copy()
    risk_scale = 1.0
    if risk_score is not None:
        risk_scale = float(np.clip(float(risk_score), 0.0, 1.0))

    if str(mode) == "lift_z_scale_xy":
        if out.shape[0] >= 3:
            out[0:2] *= float(xy_scale)
            out[2] += float(z_lift) * float(risk_scale)
        else:
            out *= float(xy_scale)
    elif str(mode) == "scale_all":
        out *= float(xy_scale)
    else:
        # Unknown mode: apply safe fallback scaling.
        out *= float(xy_scale)
        meta["reason"] = "unknown_mode_fallback_scale_all"

    out = np.clip(out, float(action_low), float(action_high)).astype(np.float32, copy=False)

    trans_dim = int(min(3, out.shape[0]))
    trans_norm = float(np.linalg.norm(out[:trans_dim], ord=2)) if trans_dim > 0 else 0.0
    meta["translation_norm"] = trans_norm
    if trans_norm > float(max_translation_norm):
        meta["reason"] = "translation_norm_exceeded"
        return SAFE_REFUSAL_ACTION, meta

    if not np.all(np.isfinite(out)):
        meta["reason"] = "projected_non_finite"
        return SAFE_REFUSAL_ACTION, meta

    if meta["reason"] == "":
        meta["reason"] = "projected"
    meta["applied"] = True
    return out, meta


class RandomBurstRecovery:
    """Recovery primitive: execute a fixed-size random burst."""

    def __init__(self, burst_steps: int = 8):
        self.burst_steps = max(1, int(burst_steps))

    def get_burst_steps(self, current_step: int, max_steps: int) -> int:
        remaining = int(max_steps) - int(current_step)
        if remaining <= 0:
            return 0
        return min(self.burst_steps, remaining)


class SafeRetractRecovery(RandomBurstRecovery):
    """Recovery primitive label for safe retract-like action shaping."""

    mode = "safe_retract_like"


class RegraspRecovery(RandomBurstRecovery):
    """Recovery primitive label for regrasp-like action shaping."""

    mode = "regrasp_like"
