#!/usr/bin/env python3
"""Train a lightweight Process Reward Model (PRM) from ROEPBench JSONL runs.

Phase-1 implementation highlights:
- Episode-level split (leakage-safe): train/val/calib
- Risk labeling: trailing H steps in positive-outcome episodes => 1
- Safe labeling: steps in safe-outcome episodes => 0
- Optional fallback-positive outcomes when primary positives are absent
- Model-internal z-score normalization via register_buffer
- Quantile-based calibration threshold tau from safe scores
"""

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _parse_bool(text: str) -> bool:
    s = str(text).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _parse_outcome_set(text: str) -> set:
    items = [t.strip().lower() for t in str(text).split(",") if t.strip()]
    return set(items)


def _to_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _normalize_outcome_label(row: Dict[str, Any]) -> str:
    label = str(row.get("action_outcome_label", "")).strip().lower()
    if label:
        return label
    if bool(row.get("success", False)):
        return "success"
    return "other_failure"


def _extract_vector(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).ravel()
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value, dtype=np.float32).ravel()
        except Exception:
            return None
    if isinstance(value, dict):
        for key in ("values", "data", "vector", "array"):
            if key in value:
                out = _extract_vector(value[key])
                if out is not None:
                    return out
    return None


def _extract_obs_low_dim(row: Dict[str, Any], strict_keys: bool) -> Optional[np.ndarray]:
    # Strict mode enforces the agreed key contract.
    if strict_keys:
        return _extract_vector(row.get("obs_low_dim"))
    # Fallback extraction for non-contract logs.
    out = _extract_vector(row.get("obs_low_dim"))
    if out is not None:
        return out
    out = _extract_vector(row.get("low_dim_state"))
    if out is not None:
        return out
    obs = row.get("observation")
    if isinstance(obs, dict):
        out = _extract_vector(obs.get("low_dim_state"))
        if out is not None:
            return out
    return None


def _extract_action(row: Dict[str, Any], strict_keys: bool) -> Optional[np.ndarray]:
    if strict_keys:
        return _extract_vector(row.get("action"))
    for key in ("action", "policy_action", "proposed_action"):
        out = _extract_vector(row.get(key))
        if out is not None:
            return out
    return None


def _jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hash_to_unit(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(h, 16) / float(16**16 - 1)


@dataclass
class Sample:
    episode_id: str
    x: np.ndarray
    y: int


class LightweightPRM(nn.Module):
    """Tiny MLP risk predictor with built-in z-score normalization buffers."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        dims = [int(input_dim)] + [int(d) for d in hidden_dims] + [1]
        layers: List[nn.Module] = []
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
            raise ValueError(f"Expected 2D tensor [N, D], got shape={tuple(x_raw.shape)}")
        x = (x_raw - self.norm_mean) / torch.clamp(self.norm_std, min=self._eps)
        logits = self.net(x).squeeze(-1)
        return logits


def _parse_hidden_dims(text: str) -> List[int]:
    dims = [int(t.strip()) for t in str(text).split(",") if t.strip()]
    if not dims:
        raise ValueError("--hidden_dims must contain at least one positive integer")
    if any(d <= 0 for d in dims):
        raise ValueError("--hidden_dims must be positive integers")
    return dims


def _build_dataset(
    run_dirs: Iterable[Path],
    horizon_steps: int,
    strict_keys: bool,
    safe_outcomes: set,
    risk_outcomes: set,
    risk_episode_prefix_as_safe: bool = False,
) -> Tuple[List[Sample], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "run_dirs_seen": 0,
        "run_dirs_used": 0,
        "rows_total": 0,
        "rows_missing_required_keys": 0,
        "rows_missing_feature_vector": 0,
        "rows_dim_mismatch": 0,
        "episodes_total": 0,
        "episodes_safe": 0,
        "episodes_risk": 0,
        "episodes_skipped_other_outcome": 0,
        "samples_safe": 0,
        "samples_risk": 0,
        "episodes_by_final_outcome": {},
        "safe_outcomes": sorted(list(safe_outcomes)),
        "risk_outcomes": sorted(list(risk_outcomes)),
        "risk_episode_prefix_as_safe": bool(risk_episode_prefix_as_safe),
    }

    samples: List[Sample] = []
    expected_obs_dim: Optional[int] = None
    expected_action_dim: Optional[int] = None

    for run_dir in sorted(run_dirs):
        stats["run_dirs_seen"] += 1
        # Step-level training uses step_logs.jsonl. Keep backward compatibility
        # with older runs that inlined step rows into episode_logs.jsonl.
        path = run_dir / "step_logs.jsonl"
        if not path.exists():
            path = run_dir / "episode_logs.jsonl"
        if not path.exists():
            continue
        rows = _jsonl_rows(path)
        if not rows:
            continue
        stats["run_dirs_used"] += 1
        stats["rows_total"] += len(rows)

        episodes: Dict[str, List[Tuple[int, int, Optional[np.ndarray], str]]] = {}
        # tuple: (sort_step, row_order, feature_or_none, outcome_label)
        for i, row in enumerate(rows):
            if strict_keys:
                required = ("obs_low_dim", "action", "action_outcome_label", "episode_idx", "run_id")
                missing = [k for k in required if k not in row]
                if missing:
                    stats["rows_missing_required_keys"] += 1
                    continue

            run_id = str(row.get("run_id", run_dir.name))
            ep_idx = _to_int(row.get("episode_idx"), -1)
            ep_key = f"{run_dir.name}|{run_id}|{ep_idx}"

            obs_vec = _extract_obs_low_dim(row, strict_keys=strict_keys)
            act_vec = _extract_action(row, strict_keys=strict_keys)

            feature: Optional[np.ndarray] = None
            if obs_vec is not None and act_vec is not None:
                if expected_obs_dim is None:
                    expected_obs_dim = int(obs_vec.shape[0])
                if expected_action_dim is None:
                    expected_action_dim = int(act_vec.shape[0])
                if (
                    int(obs_vec.shape[0]) == int(expected_obs_dim)
                    and int(act_vec.shape[0]) == int(expected_action_dim)
                ):
                    feature = np.concatenate([obs_vec, act_vec], axis=0).astype(np.float32)
                else:
                    stats["rows_dim_mismatch"] += 1
            else:
                stats["rows_missing_feature_vector"] += 1

            step_idx = _to_int(row.get("step_idx"), i)
            outcome = _normalize_outcome_label(row)
            episodes.setdefault(ep_key, []).append((step_idx, i, feature, outcome))

        for ep_key, items in episodes.items():
            stats["episodes_total"] += 1
            items_sorted = sorted(items, key=lambda t: (t[0], t[1]))
            final_outcome = str(items_sorted[-1][3]).lower()
            by_outcome = dict(stats["episodes_by_final_outcome"])
            by_outcome[final_outcome] = int(by_outcome.get(final_outcome, 0)) + 1
            stats["episodes_by_final_outcome"] = by_outcome
            valid = [it for it in items_sorted if it[2] is not None]
            if not valid:
                continue

            if final_outcome in safe_outcomes:
                stats["episodes_safe"] += 1
                for _, _, feat, _ in valid:
                    samples.append(Sample(episode_id=ep_key, x=feat, y=0))
                stats["samples_safe"] += len(valid)
            elif final_outcome in risk_outcomes:
                stats["episodes_risk"] += 1
                n = min(int(horizon_steps), len(valid))
                if bool(risk_episode_prefix_as_safe) and n < len(valid):
                    prefix = valid[:-n]
                    for _, _, feat, _ in prefix:
                        samples.append(Sample(episode_id=ep_key, x=feat, y=0))
                    stats["samples_safe"] += len(prefix)
                tail = valid[-n:]
                for _, _, feat, _ in tail:
                    samples.append(Sample(episode_id=ep_key, x=feat, y=1))
                stats["samples_risk"] += len(tail)
            else:
                stats["episodes_skipped_other_outcome"] += 1

    if expected_obs_dim is None or expected_action_dim is None:
        expected_obs_dim = 0
        expected_action_dim = 0
    stats["obs_dim"] = int(expected_obs_dim)
    stats["action_dim"] = int(expected_action_dim)
    stats["input_dim"] = int(expected_obs_dim + expected_action_dim)
    return samples, stats


def _split_episodes(
    episode_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    train_eps: set = set()
    val_eps: set = set()
    calib_eps: set = set()
    cutoff_val = float(train_ratio + val_ratio)
    for ep in sorted(set(episode_ids)):
        u = _hash_to_unit(f"{seed}|{ep}")
        if u < train_ratio:
            train_eps.add(ep)
        elif u < cutoff_val:
            val_eps.add(ep)
        else:
            calib_eps.add(ep)

    # Keep splits non-empty where possible.
    train_list = sorted(train_eps)
    if not val_eps and len(train_list) > 1:
        moved = train_list.pop()
        train_eps.remove(moved)
        val_eps.add(moved)
    if not calib_eps and len(train_list) > 1:
        moved = train_list.pop()
        train_eps.remove(moved)
        calib_eps.add(moved)
    return train_eps, val_eps, calib_eps


def _samples_to_arrays(samples: Sequence[Sample], allowed_eps: set) -> Tuple[np.ndarray, np.ndarray]:
    xs = [s.x for s in samples if s.episode_id in allowed_eps]
    ys = [s.y for s in samples if s.episode_id in allowed_eps]
    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return x, y


def _pick_device(mode: str) -> torch.device:
    m = str(mode).strip().lower()
    if m == "cpu":
        return torch.device("cpu")
    if m == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_metrics(logits: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).to(torch.float32)
    y = y_true.to(torch.float32)
    acc = float((preds == y).to(torch.float32).mean().item())
    tp = float(((preds == 1.0) & (y == 1.0)).sum().item())
    fp = float(((preds == 1.0) & (y == 0.0)).sum().item())
    fn = float(((preds == 0.0) & (y == 1.0)).sum().item())
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    return {"accuracy": acc, "precision": float(prec), "recall": float(rec)}


def _calibrated_tau(
    model: LightweightPRM,
    device: torch.device,
    alpha: float,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, str, int]:
    def safe_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.size == 0 or y.size == 0:
            return np.zeros((0,), dtype=np.float32)
        safe_mask = (y == 0.0)
        if int(np.sum(safe_mask)) == 0:
            return np.zeros((0,), dtype=np.float32)
        x_safe = x[safe_mask]
        with torch.no_grad():
            inp = torch.from_numpy(x_safe).to(device=device, dtype=torch.float32)
            probs = torch.sigmoid(model(inp)).detach().cpu().numpy().astype(np.float32)
        return probs

    for source in ("calib", "val", "all"):
        x_s, y_s = splits[source]
        scores = safe_scores(x_s, y_s)
        if scores.size > 0:
            q = float(np.quantile(scores, 1.0 - float(alpha)))
            return q, source, int(scores.size)
    return 0.95, "fallback_constant_0.95", 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Train lightweight PRM from step-level ROEPBench JSONL logs.")
    ap.add_argument("--seed0_dir", type=Path, default=Path("external/peract/ckpts/multi/PERACT_BC/seed0"))
    ap.add_argument("--glob", type=str, required=True, help="Run directory glob, e.g. roepbench_*__phase3_x")
    ap.add_argument("--out_dir", type=Path, default=Path("checkpoints/lightweight_prm"))
    ap.add_argument("--horizon_steps", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dims", type=str, default="128,64,32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--strict_keys", type=str, default="true")
    ap.add_argument("--min_rows", type=int, default=100)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument(
        "--safe_outcomes",
        type=str,
        default="success",
        help="Comma-separated final outcomes used as class 0 (safe).",
    )
    ap.add_argument(
        "--positive_outcomes",
        type=str,
        default="unsafe_attempt",
        help="Comma-separated final outcomes used as class 1 (risk).",
    )
    ap.add_argument(
        "--fallback_positive_outcomes",
        type=str,
        default="timeout",
        help=(
            "If no positive samples are found with --positive_outcomes, "
            "retry using positive_outcomes U fallback_positive_outcomes."
        ),
    )
    ap.add_argument(
        "--fallback_safe_outcomes",
        type=str,
        default="timeout",
        help=(
            "If data is still insufficient or one-sided after positive fallback, "
            "retry using safe_outcomes U (fallback_safe_outcomes - effective_positive_outcomes)."
        ),
    )
    ap.add_argument(
        "--fallback_risk_episode_prefix_as_safe",
        type=str,
        default="true",
        help=(
            "If samples become positive-only after outcome fallbacks, "
            "treat non-tail steps of risk episodes as safe negatives (weak supervision fallback)."
        ),
    )
    args = ap.parse_args()

    t0 = time.time()
    strict_keys = _parse_bool(args.strict_keys)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    safe_outcomes = _parse_outcome_set(args.safe_outcomes)
    primary_risk_outcomes = _parse_outcome_set(args.positive_outcomes)
    fallback_risk_outcomes = _parse_outcome_set(args.fallback_positive_outcomes)
    fallback_safe_outcomes = _parse_outcome_set(args.fallback_safe_outcomes)
    fallback_risk_episode_prefix_as_safe = _parse_bool(args.fallback_risk_episode_prefix_as_safe)

    if args.horizon_steps <= 0:
        raise ValueError("--horizon_steps must be > 0")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must be in (0, 1)")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch_size must be > 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.min_rows <= 0:
        raise ValueError("--min_rows must be > 0")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0,1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be in [0,1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train_ratio + --val_ratio must be < 1")
    if not safe_outcomes:
        raise ValueError("--safe_outcomes must include at least one outcome label")
    if not primary_risk_outcomes:
        raise ValueError("--positive_outcomes must include at least one outcome label")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _pick_device(args.device)

    run_dirs = sorted([p for p in args.seed0_dir.glob(args.glob) if p.is_dir()])
    if not run_dirs:
        raise RuntimeError(f"No run dirs matched glob '{args.glob}' in {args.seed0_dir}")

    effective_risk_outcomes = set(primary_risk_outcomes)
    effective_safe_outcomes = set(safe_outcomes)
    fallback_positive_applied = False
    fallback_safe_applied = False

    risk_prefix_as_safe_applied = False

    def _rebuild(eff_safe: set, eff_risk: set, risk_prefix_as_safe: bool = False):
        s, ds = _build_dataset(
            run_dirs=run_dirs,
            horizon_steps=int(args.horizon_steps),
            strict_keys=bool(strict_keys),
            safe_outcomes=eff_safe,
            risk_outcomes=eff_risk,
            risk_episode_prefix_as_safe=bool(risk_prefix_as_safe),
        )
        y = np.asarray([x.y for x in s], dtype=np.int64) if s else np.zeros((0,), dtype=np.int64)
        return s, ds, int(np.sum(y == 1)), int(np.sum(y == 0))

    samples, data_stats, num_pos, num_neg = _rebuild(
        effective_safe_outcomes, effective_risk_outcomes
    )
    if num_pos == 0 and fallback_risk_outcomes:
        effective_risk_outcomes = set(primary_risk_outcomes) | set(fallback_risk_outcomes)
        samples, data_stats, num_pos, num_neg = _rebuild(
            effective_safe_outcomes, effective_risk_outcomes
        )
        fallback_positive_applied = True
        print(
            "[fallback] no positive samples with primary outcomes "
            f"{sorted(primary_risk_outcomes)}; retrying with {sorted(effective_risk_outcomes)}"
        )

    need_safe_fallback = (len(samples) < int(args.min_rows)) or (num_neg == 0)
    if need_safe_fallback and fallback_safe_outcomes:
        addable_safe = set(fallback_safe_outcomes) - set(effective_risk_outcomes)
        if addable_safe:
            effective_safe_outcomes = set(effective_safe_outcomes) | set(addable_safe)
            samples, data_stats, num_pos, num_neg = _rebuild(
                effective_safe_outcomes, effective_risk_outcomes
            )
            fallback_safe_applied = True
            print(
                "[fallback] insufficient/one-sided data with safe outcomes "
                f"{sorted(safe_outcomes)}; retrying with {sorted(effective_safe_outcomes)}"
            )

    if (
        fallback_risk_episode_prefix_as_safe
        and num_pos > 0
        and num_neg == 0
    ):
        samples, data_stats, num_pos, num_neg = _rebuild(
            effective_safe_outcomes,
            effective_risk_outcomes,
            risk_prefix_as_safe=True,
        )
        risk_prefix_as_safe_applied = True
        print(
            "[fallback] positive-only dataset after outcome labeling; "
            "retrying with risk-episode prefix-as-safe weak supervision."
        )

    if len(samples) < int(args.min_rows):
        raise RuntimeError(
            f"Insufficient labeled rows: got {len(samples)}, require >= {int(args.min_rows)}"
        )
    if num_pos == 0 or num_neg == 0:
        raise RuntimeError(
            f"Invalid class distribution: positives={num_pos}, negatives={num_neg}"
        )

    labels = np.asarray([s.y for s in samples], dtype=np.int64)
    num_pos = int(np.sum(labels == 1))
    num_neg = int(np.sum(labels == 0))

    episode_ids = [s.episode_id for s in samples]
    train_eps, val_eps, calib_eps = _split_episodes(
        episode_ids=episode_ids,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    if not train_eps:
        raise RuntimeError("Train split has zero episodes; cannot train.")

    x_train, y_train = _samples_to_arrays(samples, train_eps)
    x_val, y_val = _samples_to_arrays(samples, val_eps)
    x_calib, y_calib = _samples_to_arrays(samples, calib_eps)
    x_all = np.stack([s.x for s in samples], axis=0).astype(np.float32)
    y_all = np.asarray([s.y for s in samples], dtype=np.float32)

    if x_train.size == 0:
        raise RuntimeError("Train split has zero samples; cannot train.")
    if x_train.shape[1] <= 0:
        raise RuntimeError("input_dim is zero; check obs_low_dim/action extraction.")

    train_pos = int(np.sum(y_train == 1.0))
    train_neg = int(np.sum(y_train == 0.0))
    if train_pos == 0 or train_neg == 0:
        raise RuntimeError(
            f"Train split class issue: positives={train_pos}, negatives={train_neg}"
        )

    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)

    model = LightweightPRM(input_dim=int(x_train.shape[1]), hidden_dims=hidden_dims).to(device)
    model.set_normalization(mean=mean, std=std)

    train_ds = TensorDataset(
        torch.from_numpy(x_train).to(torch.float32),
        torch.from_numpy(y_train).to(torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)

    pos_weight = torch.tensor([float(train_neg / max(1, train_pos))], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    x_val_t = torch.from_numpy(x_val).to(device=device, dtype=torch.float32) if x_val.size else None
    y_val_t = torch.from_numpy(y_val).to(device=device, dtype=torch.float32) if y_val.size else None

    history: List[Dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    patience = 7
    no_improve = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            bsz = int(xb.shape[0])
            running += float(loss.item()) * bsz
            seen += bsz
        train_loss = running / max(1, seen)

        model.eval()
        with torch.no_grad():
            train_logits_eval = model(torch.from_numpy(x_train).to(device=device, dtype=torch.float32))
            train_metrics = _eval_metrics(train_logits_eval, torch.from_numpy(y_train).to(device=device, dtype=torch.float32))

            if x_val_t is not None and y_val_t is not None and int(x_val_t.shape[0]) > 0:
                val_logits = model(x_val_t)
                val_loss = float(criterion(val_logits, y_val_t).item())
                val_metrics = _eval_metrics(val_logits, y_val_t)
            else:
                val_loss = train_loss
                val_metrics = {
                    "accuracy": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                }

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
        }
        history.append(row)

        improved = val_loss < (best_val_loss - 1e-9)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    tau, tau_source, tau_safe_n = _calibrated_tau(
        model=model,
        device=device,
        alpha=float(args.alpha),
        splits={
            "calib": (x_calib, y_calib),
            "val": (x_val, y_val),
            "all": (x_all, y_all),
        },
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "lightweight_prm.pth"
    meta_path = args.out_dir / "lightweight_prm_metadata.json"
    report_path = args.out_dir / "train_report.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "hidden_dims": [int(d) for d in hidden_dims],
            "horizon_steps": int(args.horizon_steps),
            "alpha": float(args.alpha),
            "tau": float(tau),
            "label_mapping": {"risk": 1, "safe": 0},
            "safe_outcomes": sorted(list(effective_safe_outcomes)),
            "positive_outcomes": sorted(list(effective_risk_outcomes)),
        },
        ckpt_path,
    )

    split_stats = {
        "episodes_train": int(len(train_eps)),
        "episodes_val": int(len(val_eps)),
        "episodes_calib": int(len(calib_eps)),
        "samples_train": int(x_train.shape[0]),
        "samples_val": int(x_val.shape[0]),
        "samples_calib": int(x_calib.shape[0]),
    }
    class_stats = {
        "all_positive": int(num_pos),
        "all_negative": int(num_neg),
        "train_positive": int(train_pos),
        "train_negative": int(train_neg),
    }

    metadata: Dict[str, Any] = {
        "created_unix_sec": float(time.time()),
        "elapsed_sec": float(time.time() - t0),
        "seed0_dir": str(args.seed0_dir),
        "glob": str(args.glob),
        "strict_keys": bool(strict_keys),
        "device_used": str(device),
        "horizon_steps": int(args.horizon_steps),
        "alpha": float(args.alpha),
        "tau": float(tau),
        "tau_source": str(tau_source),
        "tau_safe_count": int(tau_safe_n),
        "label_mapping": {"risk": 1, "safe": 0},
        "safe_outcomes_requested": sorted(list(safe_outcomes)),
        "safe_outcomes_effective": sorted(list(effective_safe_outcomes)),
        "positive_outcomes_requested": sorted(list(primary_risk_outcomes)),
        "fallback_positive_outcomes": sorted(list(fallback_risk_outcomes)),
        "fallback_safe_outcomes": sorted(list(fallback_safe_outcomes)),
        "positive_outcomes_effective": sorted(list(effective_risk_outcomes)),
        "fallback_positive_applied": bool(fallback_positive_applied),
        "fallback_safe_applied": bool(fallback_safe_applied),
        "fallback_risk_episode_prefix_as_safe_enabled": bool(
            fallback_risk_episode_prefix_as_safe
        ),
        "fallback_risk_episode_prefix_as_safe_applied": bool(
            risk_prefix_as_safe_applied
        ),
        "feature_schema": {
            "obs_low_dim_dim": int(data_stats.get("obs_dim", 0)),
            "action_dim": int(data_stats.get("action_dim", 0)),
            "input_dim": int(data_stats.get("input_dim", 0)),
        },
        "data_stats": data_stats,
        "split_stats": split_stats,
        "class_stats": class_stats,
        "training": {
            "epochs_requested": int(args.epochs),
            "epochs_completed": int(len(history)),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "hidden_dims": [int(d) for d in hidden_dims],
        },
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    report_path.write_text(
        json.dumps({"history": history}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote: {ckpt_path}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote: {report_path}")
    print(
        "Summary: "
        f"samples={len(samples)} "
        f"train/val/calib={split_stats['samples_train']}/{split_stats['samples_val']}/{split_stats['samples_calib']} "
        f"tau={tau:.6f} source={tau_source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
