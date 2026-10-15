"""Shared helpers for external-model plugin templates."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


DEFAULT_ACTION_DIM = 8


def get_preferred_rgb(obs_dict: Dict[str, Any]) -> Optional[np.ndarray]:
    """Pick an RGB observation in HWC uint8 format if available."""
    candidates = [
        "front_rgb",
        "wrist_rgb",
        "left_shoulder_rgb",
        "right_shoulder_rgb",
        "overhead_rgb",
    ]
    for key in candidates:
        arr = obs_dict.get(key, None)
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim != 3:
            continue
        # CHW -> HWC
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 8 and arr.shape[2] > 8:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr
    return None


def to_env_action(raw_action: Any, action_dim: int = DEFAULT_ACTION_DIM) -> np.ndarray:
    """Convert model output to fixed-dimensional float32 env action."""
    arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if arr.size == action_dim:
        return arr
    if arr.size > action_dim:
        return arr[:action_dim]
    out = np.zeros((int(action_dim),), dtype=np.float32)
    if arr.size > 0:
        out[: arr.size] = arr
    return out
