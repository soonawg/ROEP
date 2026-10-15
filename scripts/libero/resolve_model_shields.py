#!/usr/bin/env python3
"""Resolve canonical model-specific shield bindings from the bootstrap summary."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_scaleup_module():
    script_path = PROJECT_ROOT / "scripts" / "run_scaleup_matrix.py"
    spec = importlib.util.spec_from_file_location("run_scaleup_matrix", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config helper from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cfg(path: Path) -> Dict[str, Any]:
    scaleup = _load_scaleup_module()
    return scaleup._load_json_compatible_yaml(path.resolve())  # type: ignore[attr-defined]


def _summary_path(args: argparse.Namespace) -> Path:
    if args.summary_json:
        return args.summary_json.resolve()
    cfg = _load_cfg(args.config)
    protocol = cfg.get("protocol_first", {}) if isinstance(cfg.get("protocol_first"), dict) else {}
    raw = str(protocol.get("shield_bootstrap_summary_json", "")).strip()
    if raw:
        return Path(raw).resolve()
    prm_cfg = cfg.get("prm_training", {}) if isinstance(cfg.get("prm_training"), dict) else {}
    out_root = Path(str(prm_cfg.get("out_root", PROJECT_ROOT / "checkpoints" / "libero_prm_v2")))
    return (out_root / "shield_bootstrap_summary.json").resolve()


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _env_token(model: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(model).upper())


def _load_summary(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.is_dir():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _pick_shield(rows: List[Dict[str, Any]], *, model: str, task: str, scenario: str) -> Dict[str, str]:
    for row in rows:
        if str(row.get("model", "")).strip() != model:
            continue
        if str(row.get("task", "")).strip() != task:
            continue
        if str(row.get("scenario", "")).strip() != scenario:
            continue
        if str(row.get("status", "")).strip() not in {"trained", "reused"}:
            continue
        ckpt = str(row.get("checkpoint_path", "")).strip()
        tau = str(row.get("tau", "")).strip()
        if ckpt and tau:
            return {
                "checkpoint": ckpt,
                "tau": tau,
                "source": f"summary-{str(row.get('status', '')).strip()}",
            }
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve shield exports for a model/task/scenario triple.")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "libero_research_v2.yaml")
    ap.add_argument("--summary_json", type=Path, default=None)
    ap.add_argument("--models", type=str, required=True)
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument(
        "--format",
        type=str,
        default="exports",
        choices=["exports", "json"],
        help="exports: shell export lines, json: mapping payload",
    )
    args = ap.parse_args()

    rows = _load_summary(_summary_path(args))
    resolved: Dict[str, Dict[str, str]] = {}
    for model in _parse_csv(args.models):
        shield = _pick_shield(rows, model=model, task=args.task, scenario=args.scenario)
        if shield:
            resolved[model] = shield

    if args.format == "json":
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0

    for model, shield in resolved.items():
        token = _env_token(model)
        print(f"export ROEP_EXTERNAL_{token}_SHIELD_CKPT={json.dumps(shield['checkpoint'])}")
        print(f"export ROEP_EXTERNAL_{token}_SHIELD_TAU={json.dumps(shield['tau'])}")
        print(f"export ROEP_EXTERNAL_{token}_SHIELD_SOURCE={json.dumps(shield['source'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
