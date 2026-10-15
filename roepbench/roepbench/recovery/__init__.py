from .detectors import LowDimDeltaDetector, RewardPlateauDetector
from .primitives import (
    SAFE_REFUSAL_ACTION,
    RandomBurstRecovery,
    project_action_to_safe_space,
)
from .wrapper import RecoverVLAWrapper

try:
    from .safety_shield import LightweightPRM, SafetyShield
except Exception:  # pragma: no cover - optional torch dependency at runtime
    LightweightPRM = None
    SafetyShield = None

__all__ = [
    "RewardPlateauDetector",
    "LowDimDeltaDetector",
    "RandomBurstRecovery",
    "project_action_to_safe_space",
    "SAFE_REFUSAL_ACTION",
    "RecoverVLAWrapper",
    "LightweightPRM",
    "SafetyShield",
]
