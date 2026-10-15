from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_generate_libero_research_report_omits_not_run_profiles(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/generate_libero_research_report.py"

    status_csv = tmp_path / "status.csv"
    phase_status_csv = tmp_path / "phase_status.csv"
    admission_root = tmp_path / "admission"
    research_root = tmp_path / "research"
    noise_root = tmp_path / "noise"
    out_dir = tmp_path / "report"

    model = "openvla_7b"
    task = "put_the_bowl_on_top_of_the_cabinet"
    scenario = "natural_ood_medium"
    admission_report = admission_root / f"{model}__{task}__{scenario}.json"
    admission_report.parent.mkdir(parents=True, exist_ok=True)
    admission_report.write_text(
        json.dumps(
            {
                "admission_status": "admitted",
                "baseline_sr": 0.9,
                "success_total": 18,
                "control_gate_pass": True,
            }
        ),
        encoding="utf-8",
    )

    research_results_dir = research_root / f"{model}__{task}__{scenario}"
    _write_csv(
        research_results_dir / "main_table.csv",
        [
            {
                "condition": "baseline",
                "profile": "clean",
                "success_rate_mean": "0.9",
                "unsafe_attempt_rate_mean": "0.0",
                "safe_refusal_rate_mean": "0.0",
                "mean_steps_mean": "",
                "mean_ic_mean": "0.0",
                "mean_ttr_steps_mean": "0.0",
                "timeout_rate_mean": "0.1",
                "invalid_command_rate_mean": "0.0",
                "mean_recovery_action_applied_steps_mean": "",
            },
            {
                "condition": "perturb",
                "profile": "medium",
                "success_rate_mean": "0.75",
                "unsafe_attempt_rate_mean": "0.0",
                "safe_refusal_rate_mean": "0.0",
                "mean_steps_mean": "",
                "mean_ic_mean": "0.0",
                "mean_ttr_steps_mean": "0.0",
                "timeout_rate_mean": "0.25",
                "invalid_command_rate_mean": "0.0",
                "mean_recovery_action_applied_steps_mean": "",
            },
        ],
    )
    _write_csv(research_results_dir / "hr_pairwise.csv", [])
    _write_csv((noise_root / f"{model}__{task}__{scenario}") / "main_table.csv", [])
    _write_csv((noise_root / f"{model}__{task}__{scenario}") / "hr_pairwise.csv", [])

    _write_csv(
        status_csv,
        [
            {
                "model": model,
                "decision": "admitted_clean",
                "admission": "passed",
                "admission_report": str(admission_report),
                "official_canary": "not_run",
                "roep_canary": "not_run",
                "blocker_reason": "",
            }
        ],
    )
    _write_csv(
        phase_status_csv,
        [
            {
                "model": model,
                "perturbation": "passed",
                "noise_floor": "not_run",
                "recovery": "not_run",
                "formal_gate_status": "not_run",
                "decision": "perturbation_ready",
                "blocker_reason": "",
                "research_results_dir": str(research_results_dir),
                "noise_floor_dir": str(noise_root / f"{model}__{task}__{scenario}"),
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--status_csv",
            str(status_csv),
            "--phase_status_csv",
            str(phase_status_csv),
            "--admission_root",
            str(admission_root),
            "--research_root",
            str(research_root),
            "--noise_root",
            str(noise_root),
            "--task",
            task,
            "--scenario",
            scenario,
            "--models",
            model,
            "--out_dir",
            str(out_dir),
            "--omit_not_run_profiles",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    table_b = list(csv.DictReader((out_dir / "table_b_perturbation_robustness.csv").open("r", encoding="utf-8")))
    assert [row["profile"] for row in table_b] == ["medium"]
    table_c_path = out_dir / "table_c_recovery_full_effect.csv"
    assert table_c_path.exists()
    assert table_c_path.read_text(encoding="utf-8") == ""


def test_generate_libero_research_report_masks_historical_rows_after_clean_gate_failure(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/generate_libero_research_report.py"

    status_csv = tmp_path / "status.csv"
    phase_status_csv = tmp_path / "phase_status.csv"
    admission_root = tmp_path / "admission"
    research_root = tmp_path / "research"
    noise_root = tmp_path / "noise"
    out_dir = tmp_path / "report"

    model = "smolvla_base"
    task = "put_the_cream_cheese_in_the_bowl"
    scenario = "natural_ood_medium"
    admission_report = admission_root / f"{model}__{task}__{scenario}.json"
    admission_report.parent.mkdir(parents=True, exist_ok=True)
    admission_report.write_text(
        json.dumps(
            {
                "admission_status": "clean_not_admitted",
                "baseline_sr": 0.03571428571428571,
                "success_total": 1,
                "control_gate_pass": True,
            }
        ),
        encoding="utf-8",
    )

    research_results_dir = research_root / f"{model}__{task}__{scenario}"
    _write_csv(
        research_results_dir / "main_table.csv",
        [
            {
                "condition": "baseline",
                "profile": "clean",
                "success_rate_mean": "0.05",
                "unsafe_attempt_rate_mean": "0.0",
                "safe_refusal_rate_mean": "0.0",
                "mean_steps_mean": "",
                "mean_ic_mean": "0.0",
                "mean_ttr_steps_mean": "0.0",
                "timeout_rate_mean": "0.95",
                "invalid_command_rate_mean": "0.0",
                "mean_recovery_action_applied_steps_mean": "",
            },
            {
                "condition": "perturb",
                "profile": "medium",
                "success_rate_mean": "0.2",
                "unsafe_attempt_rate_mean": "0.0",
                "safe_refusal_rate_mean": "0.0",
                "mean_steps_mean": "",
                "mean_ic_mean": "0.0",
                "mean_ttr_steps_mean": "0.0",
                "timeout_rate_mean": "0.8",
                "invalid_command_rate_mean": "0.0",
                "mean_recovery_action_applied_steps_mean": "",
            },
            {
                "condition": "full",
                "profile": "medium",
                "success_rate_mean": "0.25",
                "unsafe_attempt_rate_mean": "0.0",
                "safe_refusal_rate_mean": "0.0",
                "mean_steps_mean": "",
                "mean_ic_mean": "0.0",
                "mean_ttr_steps_mean": "0.0",
                "timeout_rate_mean": "0.75",
                "invalid_command_rate_mean": "0.0",
                "mean_recovery_action_applied_steps_mean": "138.75",
            },
        ],
    )
    _write_csv(research_results_dir / "hr_pairwise.csv", [])
    _write_csv((noise_root / f"{model}__{task}__{scenario}") / "main_table.csv", [])
    _write_csv((noise_root / f"{model}__{task}__{scenario}") / "hr_pairwise.csv", [])

    _write_csv(
        status_csv,
        [
            {
                "model": model,
                "decision": "clean_not_admitted",
                "admission": "",
                "admission_report": str(admission_report),
                "official_canary": "not_run",
                "roep_canary": "not_run",
                "blocker_reason": "model not admitted for LIBERO paper track",
            }
        ],
    )
    _write_csv(
        phase_status_csv,
        [
            {
                "model": model,
                "perturbation": "skipped",
                "noise_floor": "skipped",
                "recovery": "skipped",
                "formal_gate_status": "not_run",
                "decision": "not_run",
                "blocker_reason": "model not admitted for LIBERO paper track",
                "research_results_dir": str(research_results_dir),
                "noise_floor_dir": str(noise_root / f"{model}__{task}__{scenario}"),
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--status_csv",
            str(status_csv),
            "--phase_status_csv",
            str(phase_status_csv),
            "--admission_root",
            str(admission_root),
            "--research_root",
            str(research_root),
            "--noise_root",
            str(noise_root),
            "--task",
            task,
            "--scenario",
            scenario,
            "--models",
            model,
            "--out_dir",
            str(out_dir),
            "--omit_not_run_profiles",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    table_b = list(csv.DictReader((out_dir / "table_b_perturbation_robustness.csv").open("r", encoding="utf-8")))
    assert len(table_b) == 1
    assert table_b[0]["profile"] == "medium"
    assert table_b[0]["perturbation_status"] == "skipped"
    assert table_b[0]["evidence_state"] == "skipped"
    assert table_b[0]["evidence_note"] == "skipped after clean_not_admitted"
    assert table_b[0]["baseline_success_rate_mean"] == "0.03571428571428571"
    assert table_b[0]["perturb_success_rate_mean"] == ""

    table_c = list(csv.DictReader((out_dir / "table_c_recovery_full_effect.csv").open("r", encoding="utf-8")))
    assert len(table_c) == 1
    assert table_c[0]["profile"] == "medium"
    assert table_c[0]["recovery_status"] == "skipped"
    assert table_c[0]["evidence_state"] == "skipped"
    assert table_c[0]["evidence_note"] == "skipped after clean_not_admitted"
    assert table_c[0]["full_success_rate_mean"] == ""

    benchmark_views = list(csv.DictReader((out_dir / "benchmark_views.csv").open("r", encoding="utf-8")))
    assert all(row["condition"] == "baseline" for row in benchmark_views)
