#!/usr/bin/env python3
"""Assess whether a LIBERO perturbation run set is ready for OOD analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


EXIT_READY = 0
EXIT_BLOCKED = 10
EXIT_INCONCLUSIVE = 11


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_row(rows: List[Dict[str, str]], condition: str, profile: str) -> Optional[Dict[str, str]]:
    for row in rows:
        if row.get("condition") == condition and row.get("profile") == profile:
            return row
    return None


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def main() -> int:
    ap = argparse.ArgumentParser(description="Assess LIBERO perturbation readiness.")
    ap.add_argument("--results_dir", type=Path, required=True)
    ap.add_argument("--expected_num_runs", type=int, required=True)
    ap.add_argument("--profiles", type=str, default="mild,medium,strong")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--task", type=str, default="")
    ap.add_argument("--scenario", type=str, default="")
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    main_path = results_dir / "main_table.csv"
    profiles = [x.strip() for x in str(args.profiles).split(",") if x.strip()]

    status = "perturbation_ready"
    reason = ""
    available_rows: List[str] = []
    missing_rows: List[str] = []
    low_support_rows: List[str] = []

    if not main_path.exists():
        status = "perturbation_blocked"
        reason = f"missing aggregate output: {main_path}"
        rows: List[Dict[str, str]] = []
    else:
        rows = _read_csv(main_path)
        baseline = _find_row(rows, "baseline", "clean")
        if baseline is None:
            status = "perturbation_blocked"
            reason = "missing baseline[clean] row in main_table.csv"
        else:
            available_rows.append("baseline[clean]")

        for profile in profiles:
            row = _find_row(rows, "perturb", profile)
            label = f"perturb[{profile}]"
            if row is None:
                missing_rows.append(label)
                continue
            available_rows.append(label)
            if _to_int(row.get("num_runs", 0), default=0) < int(args.expected_num_runs):
                low_support_rows.append(label)

        medium = _find_row(rows, "perturb", "medium")
        if medium is not None:
            medium_sr = _to_float(medium.get("success_rate_mean"))
            medium_uar = _to_float(medium.get("unsafe_attempt_rate_mean"))
            if math.isnan(medium_sr) or math.isnan(medium_uar):
                status = "perturbation_blocked"
                reason = "medium perturbation metrics are non-numeric"
        if status == "perturbation_ready" and missing_rows:
            status = "perturbation_blocked"
            reason = "missing required perturbation rows: " + ", ".join(missing_rows)
        elif status == "perturbation_ready" and low_support_rows:
            status = "inconclusive_low_support"
            reason = (
                f"insufficient run support (<{int(args.expected_num_runs)}) for: "
                + ", ".join(low_support_rows)
            )

    report = {
        "model": str(args.model),
        "task": str(args.task),
        "scenario": str(args.scenario),
        "results_dir": str(results_dir),
        "expected_num_runs": int(args.expected_num_runs),
        "profiles": profiles,
        "perturbation_status": status,
        "perturbation_reason": reason,
        "available_rows": available_rows,
        "missing_rows": missing_rows,
        "low_support_rows": low_support_rows,
    }

    if args.out_json is not None:
        out_path = args.out_json.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "[perturbation] "
        f"model={args.model or 'unknown'} status={status} "
        f"results_dir={results_dir}"
    )
    if reason:
        print(f"[perturbation] reason={reason}")

    if status == "perturbation_ready":
        return EXIT_READY
    if status == "inconclusive_low_support":
        return EXIT_INCONCLUSIVE
    return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
