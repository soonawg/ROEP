#!/usr/bin/env python3
"""Assess whether an external model is admissible for paper-track evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_scaleup_matrix as scaleup_matrix  # noqa: E402


EXIT_ADMITTED = 0
EXIT_REJECTED = 10
EXIT_INCONCLUSIVE = 11


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _flatten_numeric(value: Any) -> List[float]:
    out: List[float] = []
    if isinstance(value, (int, float)):
        out.append(float(value))
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_flatten_numeric(item))
    return out


def _iter_step_records(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    yield rec
    except Exception:  # noqa: BLE001
        return


def _collect_control_metrics(
    *,
    seed0_dir: Path,
    namespace: str,
    seeds_csv: str,
    action_abs_min: float,
) -> Dict[str, Any]:
    seeds = [int(x) for x in _parse_csv_list(seeds_csv)]
    step_records_count = 0
    numeric_action_records = 0
    nonzero_action_records = 0
    action_abs_max = 0.0
    action_mode_profiles = set()
    gripper_values = set()
    source_count = 0
    missing_sources = 0

    for seed in seeds:
        run_dir = seed0_dir / f"roepbench_baseline_s{seed}__{namespace}"
        step_path = run_dir / "step_logs.jsonl"
        if not step_path.exists():
            missing_sources += 1
            continue
        source_count += 1
        for rec in _iter_step_records(step_path):
            step_records_count += 1
            amp = str(rec.get("action_mode_profile", "")).strip()
            if amp:
                action_mode_profiles.add(amp)
            action = _flatten_numeric(rec.get("action"))
            if not action:
                continue
            numeric_action_records += 1
            local_abs_max = max(abs(x) for x in action)
            action_abs_max = max(action_abs_max, local_abs_max)
            if local_abs_max > float(action_abs_min):
                nonzero_action_records += 1
            gripper_values.add(round(float(action[-1]), 6))

    control_gate_pass = True
    control_gate_reason = ""
    if step_records_count <= 0:
        control_gate_pass = False
        control_gate_reason = "no baseline step logs found in control namespace"
    elif not action_mode_profiles:
        control_gate_pass = False
        control_gate_reason = "action_mode_profile missing from baseline step logs"
    elif numeric_action_records <= 0:
        control_gate_pass = False
        control_gate_reason = "no numeric actions found in baseline step logs"
    elif action_abs_max <= float(action_abs_min):
        control_gate_pass = False
        control_gate_reason = (
            f"action_abs_max={action_abs_max:.6g} <= action_abs_min={float(action_abs_min):.6g}"
        )

    return {
        "control_namespace": namespace,
        "control_step_records": int(step_records_count),
        "control_step_log_sources": int(source_count),
        "control_missing_sources": int(missing_sources),
        "control_action_mode_profiles": sorted(action_mode_profiles),
        "control_numeric_action_records": int(numeric_action_records),
        "control_nonzero_action_records": int(nonzero_action_records),
        "control_nonzero_action_fraction": (
            float(nonzero_action_records) / float(numeric_action_records)
            if numeric_action_records > 0
            else 0.0
        ),
        "control_action_abs_max": float(action_abs_max),
        "control_gripper_unique_count": int(len(gripper_values)),
        "control_gripper_unique_values_preview": sorted(gripper_values)[:16],
        "control_gripper_constant": bool(len(gripper_values) == 1 and numeric_action_records > 0),
        "control_gate_pass": bool(control_gate_pass),
        "control_gate_reason": control_gate_reason,
    }


def _load_policy_from_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    cfg = scaleup_matrix._load_json_compatible_yaml(path)  # type: ignore[attr-defined]
    obj = cfg.get("external_model_policy", {})
    return obj if isinstance(obj, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Assess paper-track admission for an external model.")
    ap.add_argument("--seed0_dir", required=True, type=Path)
    ap.add_argument("--cache_namespace", required=True, type=str)
    ap.add_argument("--control_namespace", type=str, default="")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--task", type=str, default="")
    ap.add_argument("--scenario", type=str, default="")
    ap.add_argument("--min_baseline_sr", type=float, default=None)
    ap.add_argument("--min_success_total", type=int, default=None)
    ap.add_argument("--control_action_abs_min", type=float, default=None)
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    policy: Dict[str, Any] = {}
    if args.config is not None:
        policy = _load_policy_from_config(args.config.resolve())

    min_baseline_sr = (
        float(args.min_baseline_sr)
        if args.min_baseline_sr is not None
        else float(policy.get("paper_min_baseline_sr", 0.05))
    )
    min_success_total = (
        int(args.min_success_total)
        if args.min_success_total is not None
        else int(policy.get("paper_min_success_total", 1))
    )
    control_action_abs_min = (
        float(args.control_action_abs_min)
        if args.control_action_abs_min is not None
        else float(policy.get("paper_control_action_abs_min", 1e-6))
    )

    cache_metrics = scaleup_matrix._compute_capability_metrics(  # type: ignore[attr-defined]
        seed0_dir=args.seed0_dir.resolve(),
        cache_namespace=str(args.cache_namespace),
        seeds_csv=str(args.seeds),
    )
    control_namespace = str(args.control_namespace or args.cache_namespace)
    control_metrics = _collect_control_metrics(
        seed0_dir=args.seed0_dir.resolve(),
        namespace=control_namespace,
        seeds_csv=str(args.seeds),
        action_abs_min=control_action_abs_min,
    )

    baseline_sr = float(cache_metrics.get("baseline_sr", 0.0))
    success_total = int(cache_metrics.get("success_total", 0))
    episode_total = int(cache_metrics.get("episode_total", 0))
    baseline_run_count = int(cache_metrics.get("baseline_run_count", 0))
    missing_sources = int(cache_metrics.get("missing_sources", 0))

    status = "admitted"
    reason = ""
    if baseline_run_count <= 0 or missing_sources > 0 or episode_total <= 0:
        status = "inconclusive_missing_sources"
        reason = (
            f"cache sources incomplete: baseline_run_count={baseline_run_count}, "
            f"missing_sources={missing_sources}, episode_total={episode_total}"
        )
    elif not bool(control_metrics.get("control_gate_pass", False)):
        status = "rejected_control_gate"
        reason = str(control_metrics.get("control_gate_reason", "")).strip() or "control gate failed"
    elif success_total < int(min_success_total):
        status = "rejected_capability_floor"
        reason = f"success_total={success_total} < min_success_total={int(min_success_total)}"
    elif baseline_sr < float(min_baseline_sr):
        status = "rejected_capability_floor"
        reason = f"baseline_sr={baseline_sr:.4f} < min_baseline_sr={float(min_baseline_sr):.4f}"

    report = {
        "model": str(args.model),
        "task": str(args.task),
        "scenario": str(args.scenario),
        "cache_namespace": str(args.cache_namespace),
        "control_namespace": str(control_namespace),
        "paper_min_baseline_sr": float(min_baseline_sr),
        "paper_min_success_total": int(min_success_total),
        "paper_control_action_abs_min": float(control_action_abs_min),
        "admission_status": status,
        "admission_reason": reason,
        **cache_metrics,
        **control_metrics,
    }

    if args.out_json is not None:
        out_path = args.out_json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "[admission] "
        f"model={args.model or 'unknown'} status={status} "
        f"baseline_sr={baseline_sr:.4f} success_total={success_total} "
        f"control_gate_pass={bool(control_metrics.get('control_gate_pass', False))}"
    )
    if reason:
        print(f"[admission] reason={reason}")

    if status == "admitted":
        return EXIT_ADMITTED
    if status.startswith("inconclusive"):
        return EXIT_INCONCLUSIVE
    return EXIT_REJECTED


if __name__ == "__main__":
    raise SystemExit(main())
