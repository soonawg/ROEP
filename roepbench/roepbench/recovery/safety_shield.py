from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from .primitives import SAFE_REFUSAL_ACTION, project_action_to_safe_space


def _pick_device(mode: str) -> torch.device:
    m = str(mode).strip().lower()
    if m == "cpu":
        return torch.device("cpu")
    if m == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_numpy_1d_float32(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).ravel()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float32, copy=False).ravel()
    return np.asarray(value, dtype=np.float32).ravel()


class LightweightPRM(nn.Module):
    """Lightweight MLP PRM with built-in z-score normalization."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        dims = [int(input_dim)] + [int(d) for d in hidden_dims] + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.register_buffer("norm_mean", torch.zeros(int(input_dim), dtype=torch.float32))
        self.register_buffer("norm_std", torch.ones(int(input_dim), dtype=torch.float32))
        self._eps = 1e-6

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean_arr = np.asarray(mean, dtype=np.float32).ravel()
        std_arr = np.asarray(std, dtype=np.float32).ravel()
        if mean_arr.shape != std_arr.shape:
            raise ValueError("mean/std shape mismatch")
        if mean_arr.shape[0] != int(self.norm_mean.shape[0]):
            raise ValueError("normalization shape mismatch with model input_dim")
        std_arr = np.maximum(std_arr, self._eps)
        self.norm_mean.copy_(torch.from_numpy(mean_arr))
        self.norm_std.copy_(torch.from_numpy(std_arr))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        if x_raw.ndim != 2:
            raise ValueError(f"Expected shape [N, D], got {tuple(x_raw.shape)}")
        x = (x_raw - self.norm_mean) / torch.clamp(self.norm_std, min=self._eps)
        return self.net(x).squeeze(-1)


class SafetyShield:
    """Runtime safety wrapper: PRM risk scoring + calibrated threshold + MDSP."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
        tau_override: Optional[float] = None,
        mdsp_mode: str = "lift_z_scale_xy",
        action_low: float = -1.0,
        action_high: float = 1.0,
        refusal_risk_threshold: float = 0.995,
        verify_projected_action: bool = False,
        z_lift: float = 0.05,
        xy_scale: float = 0.5,
        max_translation_norm: float = 0.25,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = _pick_device(device)
        payload = torch.load(str(self.checkpoint_path), map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid checkpoint: expected dict payload")
        missing = [k for k in ("state_dict", "input_dim", "hidden_dims") if k not in payload]
        if missing:
            raise RuntimeError(f"Invalid checkpoint: missing keys {missing}")
        if tau_override is None and "tau" not in payload:
            raise RuntimeError("Invalid checkpoint: missing tau and no tau_override provided")

        self.input_dim = int(payload["input_dim"])
        self.hidden_dims = [int(v) for v in payload["hidden_dims"]]
        self.label_mapping = payload.get("label_mapping", {"risk": 1, "safe": 0})
        self.tau = float(tau_override if tau_override is not None else payload["tau"])

        self.model = LightweightPRM(input_dim=self.input_dim, hidden_dims=self.hidden_dims)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.mdsp_mode = str(mdsp_mode)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.refusal_risk_threshold = float(refusal_risk_threshold)
        self.verify_projected_action = bool(verify_projected_action)
        self.z_lift = float(z_lift)
        self.xy_scale = float(xy_scale)
        self.max_translation_norm = float(max_translation_norm)

    def extract_low_dim(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            if "obs_low_dim" in obs:
                return _to_numpy_1d_float32(obs["obs_low_dim"])
            if "low_dim_state" in obs:
                return _to_numpy_1d_float32(obs["low_dim_state"])
            inner = obs.get("observation")
            if isinstance(inner, dict) and "low_dim_state" in inner:
                return _to_numpy_1d_float32(inner["low_dim_state"])
            if "low_dim" in obs:
                return _to_numpy_1d_float32(obs["low_dim"])
            raise ValueError("Could not extract low-dim state from obs dict")
        # Fallback: allow direct low-dim tensor/array/list input.
        return _to_numpy_1d_float32(obs)

    def _risk_from_feature(self, feature_1d: np.ndarray) -> float:
        if feature_1d.ndim != 1:
            raise ValueError("feature_1d must be 1D")
        if int(feature_1d.shape[0]) != int(self.input_dim):
            raise ValueError(
                f"Feature dim mismatch: got {feature_1d.shape[0]}, expected {self.input_dim}"
            )
        if self.device.type == "cpu":
            x_t = torch.from_numpy(feature_1d.reshape(1, -1))
        else:
            x_t = torch.as_tensor(feature_1d.reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x_t)
            risk = torch.sigmoid(logits).detach().cpu().item()
        return float(risk)

    def compute_risk(self, obs: Any, proposed_action: Any) -> Tuple[float, np.ndarray]:
        low_dim = self.extract_low_dim(obs)
        action = _to_numpy_1d_float32(proposed_action)
        feature = np.concatenate([low_dim, action], axis=0).astype(np.float32, copy=False)
        risk_score = self._risk_from_feature(feature)
        return float(risk_score), feature

    def _risk_from_lowdim_action(self, low_dim: np.ndarray, action: np.ndarray) -> float:
        feature = np.concatenate([low_dim, action], axis=0).astype(np.float32, copy=False)
        return self._risk_from_feature(feature)

    def step(
        self,
        obs: Any,
        proposed_action: Any,
    ) -> Tuple[Union[np.ndarray, str], bool, Dict[str, Any]]:
        low_dim = self.extract_low_dim(obs)
        action = _to_numpy_1d_float32(proposed_action)
        risk_score = self._risk_from_lowdim_action(low_dim=low_dim, action=action)

        if risk_score < self.tau:
            return (
                action,
                False,
                {
                    "risk_score": float(risk_score),
                    "tau": float(self.tau),
                    "intervened": False,
                    "intervention_reason": "below_tau",
                },
            )

        projected, proj_meta = project_action_to_safe_space(
            proposed_action=action,
            low_dim_state=low_dim,
            risk_score=risk_score,
            mode=self.mdsp_mode,
            action_low=self.action_low,
            action_high=self.action_high,
            z_lift=self.z_lift,
            xy_scale=self.xy_scale,
            max_translation_norm=self.max_translation_norm,
        )

        if isinstance(projected, str) and projected == SAFE_REFUSAL_ACTION:
            return (
                SAFE_REFUSAL_ACTION,
                True,
                {
                    "risk_score": float(risk_score),
                    "tau": float(self.tau),
                    "intervened": True,
                    "intervention_reason": "safe_refusal_projection",
                    "projection_meta": proj_meta,
                },
            )

        info: Dict[str, Any] = {
            "risk_score": float(risk_score),
            "tau": float(self.tau),
            "intervened": True,
            "intervention_reason": "projected",
            "projection_meta": proj_meta,
        }

        if self.verify_projected_action:
            projected_risk = self._risk_from_lowdim_action(
                low_dim=low_dim,
                action=_to_numpy_1d_float32(projected),
            )
            info["projected_risk_score"] = float(projected_risk)
            if projected_risk >= self.refusal_risk_threshold:
                info["intervention_reason"] = "safe_refusal_projected_risk_high"
                return SAFE_REFUSAL_ACTION, True, info

        return _to_numpy_1d_float32(projected), True, info
