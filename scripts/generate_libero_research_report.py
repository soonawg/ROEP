#!/usr/bin/env python3
"""Generate LIBERO research tables and benchmark-view reports from ROEPBench outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


EXTERNAL_BENCHMARK_MATRIX = [
    {
        "benchmark": "ResponsibleRobotBench",
        "focus": "safety + intervention + cost",
        "reproducible_view": "success, unsafe_attempt_rate, safe_refusal_rate, intervention cost proxy",
        "paper_level_only": "benchmark-specific hazard prompts and human-in-the-loop cost policy",
        "roepbench_positioning": "adds paired harm, noise-floor correction, recovery/full diagnostics",
    },
    {
        "benchmark": "RoboEval",
        "focus": "diagnostic outcome decomposition",
        "reproducible_view": "success, safe_refusal, unsafe_attempt, timeout, invalid_command taxonomy",
        "paper_level_only": "benchmark-specific stage annotations and behavior-quality labels",
        "roepbench_positioning": "adds paired-condition gates and explicit deployment risk accounting",
    },
    {
        "benchmark": "Trustworthy Evaluation",
        "focus": "trust/authenticity/action quality",
        "reproducible_view": "success, safety proxy, efficiency proxy, intervention proxy",
        "paper_level_only": "teleoperation authenticity labels and benchmark-specific quality judges",
        "roepbench_positioning": "keeps closed-loop paired evaluation instead of judge-only scoring",
    },
    {
        "benchmark": "LIBERO-X / LIBERO-PRO / VLA-Arena",
        "focus": "robustness and OOD stress",
        "reproducible_view": "clean SR, OOD SR drop, episode length, robustness-only slices",
        "paper_level_only": "their native task families and perturbation generators",
        "roepbench_positioning": "turns robustness into a safety-aware deployment gate, not SR-only",
    },
    {
        "benchmark": "CALVIN / VLABench",
        "focus": "long-horizon progress and reasoning",
        "reproducible_view": "limited; mostly clean/OOD capability framing only",
        "paper_level_only": "their long-horizon task chains and progress-score semantics",
        "roepbench_positioning": "not a direct substitute; focuses on deployment risk and recovery",
    },
    {
        "benchmark": "VLA-Risk",
        "focus": "attack benchmark / cautionary contrast",
        "reproducible_view": "unsafe-attempt-oriented risk slices and threat-model rows",
        "paper_level_only": "their attack families, physical tests, and multimodal adversarial tasks",
        "roepbench_positioning": "explicitly avoids ASR-only claims; uses refusal/unsafe labels and paired harm",
    },
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_models(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _to_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _find_row(rows: Iterable[Dict[str, str]], condition: str, profile: str) -> Optional[Dict[str, str]]:
    for row in rows:
        if row.get("condition") == condition and row.get("profile") == profile:
            return row
    return None


def _hr_row(rows: Iterable[Dict[str, str]], pair_type: str, profile: str) -> Optional[Dict[str, str]]:
    for row in rows:
        if row.get("pair_type") == pair_type and row.get("profile") == profile:
            return row
    return None


def _default_report_path(root: Path, model: str, task: str, scenario: str, suffix: str = "") -> Path:
    name = f"{model}__{task}__{scenario}{suffix}"
    return root / name


def _metric(row: Optional[Dict[str, str]], key: str) -> float:
    if row is None:
        return float("nan")
    return _to_float(row.get(key))


def _pair_metric(row: Optional[Dict[str, str]], key: str) -> float:
    if row is None:
        return float("nan")
    return _to_float(row.get(key))


def _should_emit_profile_row(row: Optional[Dict[str, str]], *, omit_not_run_profiles: bool) -> bool:
    if row is not None:
        return True
    return not omit_not_run_profiles


def _status_text(value: object) -> str:
    return str(value).strip()


def _is_skipped_status(value: object) -> bool:
    return _status_text(value) in {"skipped", "not_run", "blocked"}


def _baseline_success_value(
    baseline: Optional[Dict[str, str]],
    admission_report: Dict[str, object],
    *,
    clean_decision: str,
) -> object:
    if clean_decision == "clean_not_admitted":
        fallback = admission_report.get("baseline_sr", "")
        try:
            return float(fallback) if fallback not in ("", None) else ""
        except Exception:
            return ""
    value = _metric(baseline, "success_rate_mean")
    if value == value:
        return value
    fallback = admission_report.get("baseline_sr", "")
    try:
        return float(fallback) if fallback not in ("", None) else ""
    except Exception:
        return ""


def _blank_metrics(keys: Iterable[str]) -> Dict[str, object]:
    return {key: "" for key in keys}


def _noise_floor_metrics(
    *,
    noise_main_rows: List[Dict[str, str]],
    noise_hr_rows: List[Dict[str, str]],
) -> Dict[str, float]:
    baseline = _find_row(noise_main_rows, "baseline", "clean")
    baseline_clone = _find_row(noise_main_rows, "baseline_clone", "clean")
    perturb_medium = _find_row(noise_main_rows, "perturb", "medium")
    perturb_clone_medium = _find_row(noise_main_rows, "perturb_clone", "medium")
    baseline_hr = _hr_row(noise_hr_rows, "baseline_clone_vs_baseline", "clean")
    medium_hr = _hr_row(noise_hr_rows, "perturb_clone_vs_perturb", "medium")
    return {
        "clean_noise_sr_delta": _metric(baseline_clone, "success_rate_mean")
        - _metric(baseline, "success_rate_mean"),
        "clean_noise_unsafe_delta": _metric(baseline_clone, "unsafe_attempt_rate_mean")
        - _metric(baseline, "unsafe_attempt_rate_mean"),
        "clean_noise_hr": _pair_metric(baseline_hr, "hr"),
        "clean_noise_pair_mismatch": _pair_metric(baseline_hr, "pair_mismatch_rate_union"),
        "medium_noise_sr_delta": _metric(perturb_clone_medium, "success_rate_mean")
        - _metric(perturb_medium, "success_rate_mean"),
        "medium_noise_unsafe_delta": _metric(perturb_clone_medium, "unsafe_attempt_rate_mean")
        - _metric(perturb_medium, "unsafe_attempt_rate_mean"),
        "medium_noise_hr": _pair_metric(medium_hr, "hr"),
        "medium_noise_pair_mismatch": _pair_metric(medium_hr, "pair_mismatch_rate_union"),
    }


def _format_float(value: float) -> str:
    if value != value:
        return "NA"
    return f"{value:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate LIBERO research summary tables.")
    ap.add_argument("--status_csv", type=Path, required=True)
    ap.add_argument("--phase_status_csv", type=Path, required=True)
    ap.add_argument("--admission_root", type=Path, required=True)
    ap.add_argument("--research_root", type=Path, required=True)
    ap.add_argument("--noise_root", type=Path, required=True)
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--scenario", type=str, required=True)
    ap.add_argument("--models", type=str, default="openvla_7b,smolvla_base,spatialvla,vla_adapter")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument(
        "--omit_not_run_profiles",
        action="store_true",
        help="Omit manuscript-facing rows for profiles that were not executed.",
    )
    args = ap.parse_args()

    models = _parse_models(args.models)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_status_rows = _read_csv(args.status_csv.resolve())
    phase_status_rows = _read_csv(args.phase_status_csv.resolve())
    clean_status_map = {row.get("model", ""): row for row in clean_status_rows}
    phase_status_map = {row.get("model", ""): row for row in phase_status_rows}

    clean_table_rows: List[Dict[str, object]] = []
    table_a_rows: List[Dict[str, object]] = []
    table_b_rows: List[Dict[str, object]] = []
    table_c_rows: List[Dict[str, object]] = []
    benchmark_view_rows: List[Dict[str, object]] = []
    model_summary_rows: List[Dict[str, object]] = []

    for model in models:
        clean_status = clean_status_map.get(model, {})
        phase_status = phase_status_map.get(model, {})
        clean_decision = _status_text(clean_status.get("decision", ""))
        perturbation_status = _status_text(phase_status.get("perturbation", ""))
        recovery_status = _status_text(phase_status.get("recovery", ""))
        admission_path = Path(
            clean_status.get("admission_report")
            or _default_report_path(args.admission_root.resolve(), model, args.task, args.scenario).with_suffix(".json")
        )
        admission_report = _load_json(admission_path)

        research_results_dir = Path(
            phase_status.get("research_results_dir")
            or _default_report_path(args.research_root.resolve(), model, args.task, args.scenario)
        )
        noise_results_dir = Path(
            phase_status.get("noise_floor_dir")
            or _default_report_path(args.noise_root.resolve(), model, args.task, args.scenario)
        )

        research_main_rows = _read_csv(research_results_dir / "main_table.csv")
        research_hr_rows = _read_csv(research_results_dir / "hr_pairwise.csv")
        noise_main_rows = _read_csv(noise_results_dir / "main_table.csv")
        noise_hr_rows = _read_csv(noise_results_dir / "hr_pairwise.csv")
        noise_metrics = _noise_floor_metrics(
            noise_main_rows=noise_main_rows,
            noise_hr_rows=noise_hr_rows,
        )

        baseline = _find_row(research_main_rows, "baseline", "clean")
        recovery_clean = _find_row(research_main_rows, "recovery", "no_ood")
        clean_hr = _hr_row(research_hr_rows, "recovery_vs_baseline", "clean")
        baseline_success_value = _baseline_success_value(
            baseline,
            admission_report,
            clean_decision=clean_decision,
        )

        clean_table_rows.append(
            {
                "model": model,
                "decision": clean_status.get("decision", ""),
                "official_canary": clean_status.get("official_canary", ""),
                "roep_canary": clean_status.get("roep_canary", ""),
                "admission": clean_status.get("admission", ""),
                "baseline_sr": admission_report.get("baseline_sr", ""),
                "success_total": admission_report.get("success_total", ""),
                "control_gate_pass": admission_report.get("control_gate_pass", ""),
                "blocker_reason": clean_status.get("blocker_reason", ""),
                "admission_report": str(admission_path),
            }
        )

        table_a_rows.append(
            {
                "model": model,
                "clean_decision": clean_status.get("decision", ""),
                "admission_status": admission_report.get("admission_status", clean_status.get("admission", "")),
                "baseline_sr": admission_report.get("baseline_sr", ""),
                "success_total": admission_report.get("success_total", ""),
                "control_gate_pass": admission_report.get("control_gate_pass", ""),
                "baseline_success_rate_mean": _metric(baseline, "success_rate_mean"),
                "baseline_unsafe_attempt_rate_mean": _metric(baseline, "unsafe_attempt_rate_mean"),
                "baseline_safe_refusal_rate_mean": _metric(baseline, "safe_refusal_rate_mean"),
                "baseline_mean_steps_mean": _metric(baseline, "mean_steps_mean"),
                "baseline_mean_ic_mean": _metric(baseline, "mean_ic_mean"),
                "baseline_mean_ttr_steps_mean": _metric(baseline, "mean_ttr_steps_mean"),
            }
        )

        emitted_skipped_perturb = False
        emitted_skipped_recovery = False
        for profile in ("mild", "medium", "strong"):
            perturb = _find_row(research_main_rows, "perturb", profile)
            full = _find_row(research_main_rows, "full", profile)
            full_hr = _hr_row(research_hr_rows, "full_vs_perturb", profile)

            emit_perturb_row = (
                perturbation_status == "passed"
                and _should_emit_profile_row(perturb, omit_not_run_profiles=args.omit_not_run_profiles)
            )
            if emit_perturb_row:
                table_b_rows.append(
                    {
                        "model": model,
                        "profile": profile,
                        "perturbation_status": perturbation_status,
                        "evidence_state": "measured",
                        "evidence_note": "",
                        "baseline_success_rate_mean": baseline_success_value,
                        "perturb_success_rate_mean": _metric(perturb, "success_rate_mean"),
                        "sr_drop_vs_clean": _to_float(str(baseline_success_value), default=float("nan"))
                        - _metric(perturb, "success_rate_mean"),
                        "perturb_unsafe_attempt_rate_mean": _metric(perturb, "unsafe_attempt_rate_mean"),
                        "perturb_safe_refusal_rate_mean": _metric(perturb, "safe_refusal_rate_mean"),
                        "perturb_mean_steps_mean": _metric(perturb, "mean_steps_mean"),
                        "perturb_mean_ic_mean": _metric(perturb, "mean_ic_mean"),
                        "perturb_mean_ttr_steps_mean": _metric(perturb, "mean_ttr_steps_mean"),
                        "medium_noise_hr": noise_metrics["medium_noise_hr"] if profile == "medium" else "",
                        "medium_noise_sr_delta": noise_metrics["medium_noise_sr_delta"] if profile == "medium" else "",
                        "medium_noise_pair_mismatch": noise_metrics["medium_noise_pair_mismatch"]
                        if profile == "medium"
                        else "",
                    }
                )
            elif (
                not emitted_skipped_perturb
                and args.omit_not_run_profiles
                and profile == "medium"
                and clean_decision == "clean_not_admitted"
                and _is_skipped_status(perturbation_status)
            ):
                note = "skipped after clean_not_admitted" if clean_decision == "clean_not_admitted" else "not run"
                table_b_rows.append(
                    {
                        "model": model,
                        "profile": profile,
                        "perturbation_status": perturbation_status,
                        "evidence_state": "skipped",
                        "evidence_note": note,
                        "baseline_success_rate_mean": baseline_success_value,
                        **_blank_metrics(
                            [
                                "perturb_success_rate_mean",
                                "sr_drop_vs_clean",
                                "perturb_unsafe_attempt_rate_mean",
                                "perturb_safe_refusal_rate_mean",
                                "perturb_mean_steps_mean",
                                "perturb_mean_ic_mean",
                                "perturb_mean_ttr_steps_mean",
                                "medium_noise_hr",
                                "medium_noise_sr_delta",
                                "medium_noise_pair_mismatch",
                            ]
                        ),
                    }
                )
                emitted_skipped_perturb = True

            emit_full_row = (
                recovery_status == "passed"
                and _should_emit_profile_row(full, omit_not_run_profiles=args.omit_not_run_profiles)
            )
            if emit_full_row:
                table_c_rows.append(
                    {
                        "model": model,
                        "profile": profile,
                        "recovery_status": recovery_status,
                        "evidence_state": "measured",
                        "evidence_note": "",
                        "perturb_success_rate_mean": _metric(perturb, "success_rate_mean"),
                        "full_success_rate_mean": _metric(full, "success_rate_mean"),
                        "full_sr_delta_vs_perturb": _metric(full, "success_rate_mean")
                        - _metric(perturb, "success_rate_mean"),
                        "full_unsafe_attempt_rate_mean": _metric(full, "unsafe_attempt_rate_mean"),
                        "full_unsafe_delta_vs_perturb": _metric(full, "unsafe_attempt_rate_mean")
                        - _metric(perturb, "unsafe_attempt_rate_mean"),
                        "full_safe_refusal_rate_mean": _metric(full, "safe_refusal_rate_mean"),
                        "full_safe_refusal_delta_vs_perturb": _metric(full, "safe_refusal_rate_mean")
                        - _metric(perturb, "safe_refusal_rate_mean"),
                        "full_mean_recovery_action_applied_steps_mean": _metric(
                            full, "mean_recovery_action_applied_steps_mean"
                        ),
                        "hr_full_vs_perturb_mean": _pair_metric(full_hr, "hr"),
                        "hr_full_vs_perturb_pair_mismatch_rate_union": _pair_metric(
                            full_hr, "pair_mismatch_rate_union"
                        ),
                    }
                )
            elif (
                not emitted_skipped_recovery
                and args.omit_not_run_profiles
                and profile == "medium"
                and clean_decision == "clean_not_admitted"
                and _is_skipped_status(recovery_status)
            ):
                note = "skipped after clean_not_admitted" if clean_decision == "clean_not_admitted" else "not run"
                table_c_rows.append(
                    {
                        "model": model,
                        "profile": profile,
                        "recovery_status": recovery_status,
                        "evidence_state": "skipped",
                        "evidence_note": note,
                        **_blank_metrics(
                            [
                                "perturb_success_rate_mean",
                                "full_success_rate_mean",
                                "full_sr_delta_vs_perturb",
                                "full_unsafe_attempt_rate_mean",
                                "full_unsafe_delta_vs_perturb",
                                "full_safe_refusal_rate_mean",
                                "full_safe_refusal_delta_vs_perturb",
                                "full_mean_recovery_action_applied_steps_mean",
                                "hr_full_vs_perturb_mean",
                                "hr_full_vs_perturb_pair_mismatch_rate_union",
                            ]
                        ),
                    }
                )
                emitted_skipped_recovery = True

        if recovery_status == "passed" and _should_emit_profile_row(
            recovery_clean, omit_not_run_profiles=args.omit_not_run_profiles
        ):
            table_c_rows.append(
                {
                    "model": model,
                    "profile": "clean",
                    "recovery_status": recovery_status,
                    "evidence_state": "measured",
                    "evidence_note": "",
                    "perturb_success_rate_mean": baseline_success_value,
                    "full_success_rate_mean": _metric(recovery_clean, "success_rate_mean"),
                    "full_sr_delta_vs_perturb": _metric(recovery_clean, "success_rate_mean")
                    - _to_float(str(baseline_success_value), default=float("nan")),
                    "full_unsafe_attempt_rate_mean": _metric(recovery_clean, "unsafe_attempt_rate_mean"),
                    "full_unsafe_delta_vs_perturb": _metric(recovery_clean, "unsafe_attempt_rate_mean")
                    - _metric(baseline, "unsafe_attempt_rate_mean"),
                    "full_safe_refusal_rate_mean": _metric(recovery_clean, "safe_refusal_rate_mean"),
                    "full_safe_refusal_delta_vs_perturb": _metric(recovery_clean, "safe_refusal_rate_mean")
                    - _metric(baseline, "safe_refusal_rate_mean"),
                    "full_mean_recovery_action_applied_steps_mean": _metric(
                        recovery_clean, "mean_recovery_action_applied_steps_mean"
                    ),
                    "hr_full_vs_perturb_mean": _pair_metric(clean_hr, "hr"),
                    "hr_full_vs_perturb_pair_mismatch_rate_union": _pair_metric(
                        clean_hr, "pair_mismatch_rate_union"
                    ),
                }
            )

        view_specs = {
            "LIBERO-style": [
                ("baseline", "clean", baseline),
                (
                    "perturb",
                    "medium",
                    _find_row(research_main_rows, "perturb", "medium") if perturbation_status == "passed" else None,
                ),
            ],
            "robustness-only": [
                (
                    "perturb",
                    profile,
                    _find_row(research_main_rows, "perturb", profile) if perturbation_status == "passed" else None,
                )
                for profile in ("mild", "medium", "strong")
            ],
            "ResponsibleRobotBench-like": [
                ("baseline", "clean", baseline),
                (
                    "perturb",
                    "medium",
                    _find_row(research_main_rows, "perturb", "medium") if perturbation_status == "passed" else None,
                ),
                ("full", "medium", _find_row(research_main_rows, "full", "medium") if recovery_status == "passed" else None),
            ],
            "RoboEval-like": [
                ("baseline", "clean", baseline),
                (
                    "perturb",
                    "medium",
                    _find_row(research_main_rows, "perturb", "medium") if perturbation_status == "passed" else None,
                ),
                ("full", "medium", _find_row(research_main_rows, "full", "medium") if recovery_status == "passed" else None),
            ],
            "ROEPBench": [
                ("baseline", "clean", baseline),
                (
                    "perturb",
                    "medium",
                    _find_row(research_main_rows, "perturb", "medium") if perturbation_status == "passed" else None,
                ),
                ("recovery", "no_ood", recovery_clean if recovery_status == "passed" else None),
                ("full", "medium", _find_row(research_main_rows, "full", "medium") if recovery_status == "passed" else None),
            ],
        }

        for view_name, rows in view_specs.items():
            for condition, profile, row in rows:
                if row is None:
                    continue
                metrics = {
                    "success_rate_mean": _metric(row, "success_rate_mean"),
                    "unsafe_attempt_rate_mean": _metric(row, "unsafe_attempt_rate_mean"),
                    "safe_refusal_rate_mean": _metric(row, "safe_refusal_rate_mean"),
                    "mean_steps_mean": _metric(row, "mean_steps_mean"),
                }
                if view_name in {"ResponsibleRobotBench-like", "ROEPBench"}:
                    metrics["mean_ic_mean"] = _metric(row, "mean_ic_mean")
                    metrics["mean_ttr_steps_mean"] = _metric(row, "mean_ttr_steps_mean")
                if view_name == "RoboEval-like":
                    metrics["timeout_rate_mean"] = _metric(row, "timeout_rate_mean")
                    metrics["invalid_command_rate_mean"] = _metric(row, "invalid_command_rate_mean")
                if view_name == "ROEPBench":
                    pair_type = "recovery_vs_baseline" if condition == "recovery" else "full_vs_perturb"
                    pair_profile = "clean" if condition == "recovery" else profile
                    pair_row = _hr_row(research_hr_rows, pair_type, pair_profile)
                    metrics["paired_hr_mean"] = _pair_metric(pair_row, "hr")
                    metrics["pair_mismatch_rate_union"] = _pair_metric(pair_row, "pair_mismatch_rate_union")
                    metrics["mean_recovery_action_applied_steps_mean"] = _metric(
                        row, "mean_recovery_action_applied_steps_mean"
                    )
                    metrics["clean_noise_hr"] = noise_metrics["clean_noise_hr"]
                    metrics["medium_noise_hr"] = noise_metrics["medium_noise_hr"]
                    metrics["medium_noise_sr_delta"] = noise_metrics["medium_noise_sr_delta"]
                if view_name in {"LIBERO-style", "robustness-only"}:
                    metrics["sr_drop_vs_clean"] = _metric(baseline, "success_rate_mean") - _metric(
                        row, "success_rate_mean"
                    )

                for metric_name, value in metrics.items():
                    benchmark_view_rows.append(
                        {
                            "model": model,
                            "view": view_name,
                            "condition": condition,
                            "profile": profile,
                            "metric": metric_name,
                            "value": value,
                        }
                    )

        model_summary_rows.append(
            {
                "model": model,
                "clean_decision": clean_status.get("decision", ""),
                "admission_status": admission_report.get("admission_status", ""),
                "perturbation_status": phase_status.get("perturbation", ""),
                "noise_floor_status": phase_status.get("noise_floor", ""),
                "recovery_status": phase_status.get("recovery", ""),
                "formal_gate_status": phase_status.get("formal_gate_status", ""),
                "research_decision": phase_status.get("decision", ""),
                "blocker_reason": phase_status.get("blocker_reason", clean_status.get("blocker_reason", "")),
                "research_results_dir": str(research_results_dir),
                "noise_results_dir": str(noise_results_dir),
            }
        )

    _write_csv(out_dir / "clean_decision_table.csv", clean_table_rows)
    _write_csv(out_dir / "table_a_clean_capability.csv", table_a_rows)
    _write_csv(out_dir / "table_b_perturbation_robustness.csv", table_b_rows)
    _write_csv(out_dir / "table_c_recovery_full_effect.csv", table_c_rows)
    _write_csv(out_dir / "benchmark_views.csv", benchmark_view_rows)
    _write_csv(out_dir / "model_research_summary.csv", model_summary_rows)
    _write_csv(out_dir / "benchmark_comparison_matrix.csv", EXTERNAL_BENCHMARK_MATRIX)

    summary_lines = [
        "# LIBERO Research Report",
        "",
        f"- task: `{args.task}`",
        f"- scenario: `{args.scenario}`",
        f"- models: `{', '.join(models)}`",
        "",
        "## Current Model Status",
        "",
        "| Model | Clean Decision | Admission | Perturbation | Recovery | Final Research Decision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in model_summary_rows:
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    _to_text(row["model"]),
                    _to_text(row["clean_decision"]),
                    _to_text(row["admission_status"]),
                    _to_text(row["perturbation_status"]),
                    _to_text(row["recovery_status"]),
                    _to_text(row["research_decision"]),
                ]
            )
            + " |"
        )

    summary_lines.extend(
        [
            "",
            "## Key Report Artifacts",
            "",
            "- `clean_decision_table.csv`: clean admission and gating status.",
            "- `table_a_clean_capability.csv`: clean capability anchor table.",
            "- `table_b_perturbation_robustness.csv`: perturbation robustness table.",
            "- `table_c_recovery_full_effect.csv`: recovery/full comparison table.",
            "- `benchmark_views.csv`: same runs projected into comparator benchmark views.",
            "- `benchmark_comparison_matrix.csv`: paper-level benchmark framing matrix.",
            "",
            "## Benchmark Comparison Framing",
            "",
            "| Benchmark | Focus | Reproducible From Current Runs | ROEPBench Positioning |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in EXTERNAL_BENCHMARK_MATRIX:
        summary_lines.append(
            "| "
            + " | ".join(
                [
                    row["benchmark"],
                    row["focus"],
                    row["reproducible_view"],
                    row["roepbench_positioning"],
                ]
            )
            + " |"
        )

    (out_dir / "research_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[report] wrote LIBERO research report to {out_dir}")
    for artifact in (
        "clean_decision_table.csv",
        "table_a_clean_capability.csv",
        "table_b_perturbation_robustness.csv",
        "table_c_recovery_full_effect.csv",
        "benchmark_views.csv",
        "benchmark_comparison_matrix.csv",
        "research_summary.md",
    ):
        print(f"[report] {out_dir / artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
