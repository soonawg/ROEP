#!/usr/bin/env python3
"""Run protocol-first LIBERO tracks with task-aware budgets and shield selection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_scaleup_matrix as scaleup_matrix  # noqa: E402


def _load_cfg(path: Path) -> Dict[str, Any]:
    return scaleup_matrix._load_json_compatible_yaml(path)  # type: ignore[attr-defined]


def _parse_summary(path: Path) -> List[Dict[str, Any]]:
    if str(path).strip() in {"", "."}:
        return []
    if not path.exists() or path.is_dir():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise RuntimeError(f"Unexpected shield bootstrap summary format: {path}")


def _env_token(model: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(model).upper())


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _pick_shield(summary_rows: List[Dict[str, Any]], model: str, task: str, scenario: str) -> Dict[str, str]:
    for row in summary_rows:
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
            return {"checkpoint": ckpt, "tau": tau}
    return {}


def _target_map(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    protocol = cfg.get("protocol_first", {})
    targets = protocol.get("targets", []) if isinstance(protocol, dict) else []
    out: Dict[str, Dict[str, Any]] = {}
    for target in targets:
        if not isinstance(target, dict):
            continue
        name = str(target.get("name", "")).strip()
        if name:
            out[name] = target
    return out


def _read_report_row(report_dir: Path) -> Dict[str, str]:
    summary_path = report_dir / "model_research_summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[0]
    return {str(k): str(v) for k, v in row.items()}


def _run_status(proc_rc: int, report_row: Dict[str, str]) -> str:
    if proc_rc == 0:
        return "done"
    if report_row.get("formal_gate_status") == "INCONCLUSIVE":
        return "inconclusive"
    if report_row.get("research_decision") in {"recovery_blocked", "perturbation_blocked"}:
        return "blocked"
    return "failed"


def _clear_shield_env(env: Dict[str, str]) -> None:
    for candidate in ("openvla_7b", "smolvla_base", "spatialvla", "vla_adapter"):
        token = _env_token(candidate)
        env.pop(f"ROEP_EXTERNAL_{token}_SHIELD_CKPT", None)
        env.pop(f"ROEP_EXTERNAL_{token}_SHIELD_TAU", None)


def _common_env(
    *,
    config_path: Path,
    suite: str,
    task: str,
    scenario: str,
    model: str,
    admission_root: Path,
    research_root: Path,
    noise_root: Path,
    report_dir: Path,
    admission_ns_prefix: str,
    research_ns_prefix: str,
    noise_ns_prefix: str,
) -> Dict[str, str]:
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "CONFIG_PATH": str(config_path),
        "MODELS_CSV": model,
        "ROEP_BENCHMARK_BACKEND": "libero",
        "ROEP_LIBERO_SUITE": suite,
        "ROEP_LIBERO_TASK_NAME": task,
        "TASK": task,
        "SCENARIO": scenario,
        "ADMISSION_ROOT": str(admission_root),
        "RESEARCH_ROOT": str(research_root),
        "NOISE_ROOT": str(noise_root),
        "RESEARCH_REPORT_DIR": str(report_dir),
        "ADMISSION_NS_PREFIX": admission_ns_prefix,
        "RESEARCH_NS_PREFIX": research_ns_prefix,
        "NOISE_NS_PREFIX": noise_ns_prefix,
        "RUN_CORE_CANARY": "0",
        "RUN_OFFICIAL_CANARIES": "0",
        "RUN_VALIDATE": "1",
        "REUSE_EXISTING_PHASE_OUTPUTS": "1",
        "CONTINUE_ON_STAGE_ERROR": "1",
    }


def _apply_generic_budget(
    env: Dict[str, str],
    *,
    budget: Dict[str, Any],
    profiles_default: str,
    run_smoke: bool,
    run_admission: bool,
    run_perturb: bool,
    run_noise_floor: bool,
    run_recovery: bool,
    run_full: bool,
) -> None:
    episodes = str(budget.get("episodes", 4))
    seeds = str(budget.get("seeds", "0,1,2,3,4"))
    max_steps = str(budget.get("max_steps", 120))
    profiles = str(budget.get("profiles", profiles_default))
    research_profiles = str(budget.get("research_profiles", profiles))
    env.update(
        {
            "RUN_SMOKE": "1" if run_smoke else "0",
            "RUN_ADMISSION": "1" if run_admission else "0",
            "RUN_PERTURB": "1" if run_perturb else "0",
            "RUN_NOISE_FLOOR": "1" if run_noise_floor else "0",
            "RUN_RECOVERY": "1" if run_recovery else "0",
            "RUN_FULL": "1" if run_full else "0",
            "PROFILES": profiles,
            "RESEARCH_PROFILES": research_profiles,
            "SMOKE_EPISODES": "1",
            "SMOKE_SEEDS": "0",
            "SMOKE_MAX_STEPS": max_steps,
            "ADMISSION_EPISODES": episodes,
            "ADMISSION_SEEDS": seeds,
            "ADMISSION_MAX_STEPS": max_steps,
            "PERTURB_EPISODES": episodes,
            "PERTURB_SEEDS": seeds,
            "PERTURB_MAX_STEPS": max_steps,
            "NOISE_EPISODES": episodes,
            "NOISE_SEEDS": seeds,
            "NOISE_MAX_STEPS": max_steps,
            "RECOVERY_EPISODES": episodes,
            "RECOVERY_SEEDS": seeds,
            "RECOVERY_MAX_STEPS": max_steps,
        }
    )


def _apply_minimal_gate_budget(
    env: Dict[str, str],
    *,
    minimal_gate: Dict[str, Any],
    profiles_default: str,
    run_noise_floor: bool,
    run_recovery: bool,
    run_full: bool,
) -> None:
    profiles = str(minimal_gate.get("profiles", profiles_default))
    research_profiles = str(minimal_gate.get("research_profiles", profiles))
    env.update(
        {
            "RUN_SMOKE": str(minimal_gate.get("run_smoke", 1)),
            "RUN_ADMISSION": str(minimal_gate.get("run_admission", 1)),
            "RUN_PERTURB": "1",
            "RUN_NOISE_FLOOR": "1" if run_noise_floor else "0",
            "RUN_RECOVERY": "1" if run_recovery else "0",
            "RUN_FULL": "1" if run_full else "0",
            "PROFILES": profiles,
            "RESEARCH_PROFILES": research_profiles,
            "SMOKE_EPISODES": str(minimal_gate.get("smoke_episodes", 1)),
            "SMOKE_SEEDS": str(minimal_gate.get("smoke_seeds", "0")),
            "SMOKE_MAX_STEPS": str(minimal_gate.get("smoke_max_steps", 120)),
            "ADMISSION_EPISODES": str(minimal_gate.get("admission_episodes", 2)),
            "ADMISSION_SEEDS": str(minimal_gate.get("admission_seeds", "0,1")),
            "ADMISSION_MAX_STEPS": str(minimal_gate.get("admission_max_steps", 120)),
            "PERTURB_EPISODES": str(minimal_gate.get("perturb_episodes", 1)),
            "PERTURB_SEEDS": str(minimal_gate.get("perturb_seeds", "0")),
            "PERTURB_MAX_STEPS": str(minimal_gate.get("perturb_max_steps", 120)),
            "NOISE_EPISODES": str(minimal_gate.get("noise_episodes", 1)),
            "NOISE_SEEDS": str(minimal_gate.get("noise_seeds", "0")),
            "NOISE_MAX_STEPS": str(minimal_gate.get("noise_max_steps", 120)),
            "RECOVERY_EPISODES": str(minimal_gate.get("recovery_episodes", 1)),
            "RECOVERY_SEEDS": str(minimal_gate.get("recovery_seeds", "0")),
            "RECOVERY_MAX_STEPS": str(minimal_gate.get("recovery_max_steps", 120)),
        }
    )


def _run_pipeline(env: Dict[str, str], *, execute: bool, result: Dict[str, Any], report_dir: Path) -> Dict[str, Any]:
    cmd = ["bash", str(PROJECT_ROOT / "scripts" / "libero" / "run_minimal_gate.sh")]
    result["command"] = " ".join(shlex.quote(part) for part in cmd)
    if not execute:
        result["status"] = "planned"
        return result

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    report_row = _read_report_row(report_dir)
    result["returncode"] = int(proc.returncode)
    if report_row:
        result["formal_gate_status"] = report_row.get("formal_gate_status", "")
        result["research_decision"] = report_row.get("research_decision", "")
        result["blocker_reason"] = report_row.get("blocker_reason", "")
        result["clean_decision"] = report_row.get("clean_decision", "")
        result["perturbation_status"] = report_row.get("perturbation_status", "")
    result["status"] = _run_status(proc.returncode, report_row)
    return result


def _run_target(
    *,
    cfg: Dict[str, Any],
    config_path: Path,
    target: Dict[str, Any],
    summary_rows: List[Dict[str, Any]],
    execute: bool,
    mode: str,
    support_close_namespace_tag: str,
    reuse_existing_phase_outputs: str,
    budget_episodes_override: str,
    budget_seeds_override: str,
) -> Dict[str, Any]:
    protocol = cfg.get("protocol_first", {}) if isinstance(cfg.get("protocol_first"), dict) else {}
    global_cfg = cfg.get("global", {}) if isinstance(cfg.get("global"), dict) else {}
    minimal_gate = (
        cfg.get("calibration_reset", {}).get("minimal_gate", {})
        if isinstance(cfg.get("calibration_reset"), dict)
        else {}
    )
    support_close = (
        dict(protocol.get("support_close_budget", {}))
        if isinstance(protocol.get("support_close_budget"), dict)
        else {}
    )
    target_name = str(target["name"])
    model = str(target["model"])
    task = str(target["task"])
    scenario = str(target.get("scenario", global_cfg.get("scenario", "natural_ood_medium")))
    suite = str(global_cfg.get("suite", "libero_goal"))
    results_root = Path(str(protocol.get("results_root", PROJECT_ROOT / "results/libero_protocol_first")))
    target_root = results_root / target_name

    if mode == "support_close":
        namespace_tag = str(support_close_namespace_tag).strip() or "v1_paperstage"
        if str(budget_episodes_override).strip():
            support_close["episodes"] = str(budget_episodes_override).strip()
        if str(budget_seeds_override).strip():
            support_close["seeds"] = str(budget_seeds_override).strip()
        admission_root = target_root / "admission"
        research_root = target_root / "research"
        noise_root = target_root / "noise"
        report_dir = target_root / "report"
        env = os.environ.copy()
        env.update(
            _common_env(
                config_path=config_path,
                suite=suite,
                task=task,
                scenario=scenario,
                model=model,
                admission_root=admission_root,
                research_root=research_root,
                noise_root=noise_root,
                report_dir=report_dir,
                admission_ns_prefix=f"protocol_first_{target_name}_support_close_admission_{namespace_tag}",
                research_ns_prefix=f"protocol_first_{target_name}_support_close_{namespace_tag}",
                noise_ns_prefix=f"protocol_first_{target_name}_support_close_noise_{namespace_tag}",
            )
        )
        env["REUSE_EXISTING_PHASE_OUTPUTS"] = str(reuse_existing_phase_outputs)
        _apply_generic_budget(
            env,
            budget=support_close,
            profiles_default=str(global_cfg.get("profiles", "medium")),
            run_smoke=False,
            run_admission=True,
            run_perturb=True,
            run_noise_floor=bool(support_close.get("run_noise_floor", True)),
            run_recovery=bool(support_close.get("run_recovery", True)),
            run_full=bool(support_close.get("run_full", True)),
        )
    else:
        admission_root = PROJECT_ROOT / "results/libero_research_reset/external_model_validation"
        research_root = target_root / "research"
        noise_root = target_root / "noise"
        report_dir = target_root / "report"
        env = os.environ.copy()
        env.update(
            _common_env(
                config_path=config_path,
                suite=suite,
                task=task,
                scenario=scenario,
                model=model,
                admission_root=admission_root,
                research_root=research_root,
                noise_root=noise_root,
                report_dir=report_dir,
                admission_ns_prefix=str(
                    cfg.get("results_layout", {}).get(
                        "admission_namespace_prefix", "libero_reset_min_gate_admission_v1"
                    )
                ),
                research_ns_prefix=f"protocol_first_{target_name}_research_v1",
                noise_ns_prefix=f"protocol_first_{target_name}_noise_v1",
            )
        )
        _apply_minimal_gate_budget(
            env,
            minimal_gate=minimal_gate,
            profiles_default=str(global_cfg.get("profiles", "medium")),
            run_noise_floor=bool(target.get("run_noise_floor", True)),
            run_recovery=bool(target.get("run_recovery", True)),
            run_full=bool(target.get("run_full", True)),
        )

    _clear_shield_env(env)
    shield = _pick_shield(summary_rows, model=model, task=task, scenario=scenario)
    if shield:
        token = _env_token(model)
        env[f"ROEP_EXTERNAL_{token}_SHIELD_CKPT"] = shield["checkpoint"]
        env[f"ROEP_EXTERNAL_{token}_SHIELD_TAU"] = shield["tau"]

    result: Dict[str, Any] = {
        "mode": mode,
        "target": target_name,
        "model": model,
        "task": task,
        "scenario": scenario,
        "role": str(target.get("role", "")),
        "paper_table_role": str(target.get("paper_table_role", "")),
        "admission_root": str(admission_root),
        "research_root": str(research_root),
        "noise_root": str(noise_root),
        "report_dir": str(report_dir),
        "shield_checkpoint": shield.get("checkpoint", ""),
        "shield_tau": shield.get("tau", ""),
        "support_close_namespace_tag": str(support_close_namespace_tag) if mode == "support_close" else "",
        "reuse_existing_phase_outputs": str(reuse_existing_phase_outputs) if mode == "support_close" else "",
        "budget_episodes_override": str(budget_episodes_override) if mode == "support_close" else "",
        "budget_seeds_override": str(budget_seeds_override) if mode == "support_close" else "",
    }
    return _run_pipeline(env, execute=execute, result=result, report_dir=report_dir)


def _run_recovery_recalibration(
    *,
    cfg: Dict[str, Any],
    config_path: Path,
    summary_rows: List[Dict[str, Any]],
    execute: bool,
) -> List[Dict[str, Any]]:
    del summary_rows  # Recalibration intentionally runs clean/perturb only without shield bindings.
    return _run_recalibration_campaign(
        cfg=cfg,
        config_path=config_path,
        execute=execute,
        block_key="recovery_recalibration",
        mode_name="recovery_recalibration",
        default_root_name="recovery_recalibration",
        default_namespace_prefix="protocol_first_recovery_recalibration_v1",
    )


def _run_recalibration_campaign(
    *,
    cfg: Dict[str, Any],
    config_path: Path,
    execute: bool,
    block_key: str,
    mode_name: str,
    default_root_name: str,
    default_namespace_prefix: str,
) -> List[Dict[str, Any]]:
    protocol = cfg.get("protocol_first", {}) if isinstance(cfg.get("protocol_first"), dict) else {}
    global_cfg = cfg.get("global", {}) if isinstance(cfg.get("global"), dict) else {}
    recal = protocol.get(block_key, {}) if isinstance(protocol.get(block_key), dict) else {}
    results_root = Path(str(protocol.get("results_root", PROJECT_ROOT / "results/libero_protocol_first")))
    recal_root = Path(str(recal.get("results_root", results_root / default_root_name)))
    suite = str(global_cfg.get("suite", "libero_goal"))
    scenario = str(global_cfg.get("scenario", "natural_ood_medium"))
    tasks = [task for task in recal.get("task_pool", []) if str(task).strip()]
    models = _split_csv(str(recal.get("models_csv", "")))
    namespace_prefix = str(recal.get("namespace_prefix", default_namespace_prefix))

    runs: List[Dict[str, Any]] = []
    for task in tasks:
        for model in models:
            pair_root = recal_root / str(task) / str(model)
            admission_root = recal_root / "admission"
            research_root = pair_root / "research"
            noise_root = pair_root / "noise"
            report_dir = pair_root / "report"

            env = os.environ.copy()
            env.update(
                _common_env(
                    config_path=config_path,
                    suite=suite,
                    task=str(task),
                    scenario=scenario,
                    model=str(model),
                    admission_root=admission_root,
                    research_root=research_root,
                    noise_root=noise_root,
                    report_dir=report_dir,
                    admission_ns_prefix=f"{namespace_prefix}_admission",
                    research_ns_prefix=namespace_prefix,
                    noise_ns_prefix=f"{namespace_prefix}_noise",
                )
            )
            _apply_generic_budget(
                env,
                budget=recal,
                profiles_default=str(global_cfg.get("profiles", "medium")),
                run_smoke=False,
                run_admission=bool(recal.get("run_admission", True)),
                run_perturb=bool(recal.get("run_perturb", True)),
                run_noise_floor=bool(recal.get("run_noise_floor", False)),
                run_recovery=bool(recal.get("run_recovery", False)),
                run_full=bool(recal.get("run_full", False)),
            )
            _clear_shield_env(env)

            result: Dict[str, Any] = {
                "mode": mode_name,
                "target": "",
                "model": str(model),
                "task": str(task),
                "scenario": scenario,
                "role": "common_recovery_recalibration",
                "paper_table_role": "not_selected",
                "admission_root": str(admission_root),
                "research_root": str(research_root),
                "noise_root": str(noise_root),
                "report_dir": str(report_dir),
                "shield_checkpoint": "",
                "shield_tau": "",
            }
            runs.append(_run_pipeline(env, execute=execute, result=result, report_dir=report_dir))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description="Run protocol-first LIBERO track targets.")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/libero_research_v2.yaml")
    ap.add_argument(
        "--mode",
        type=str,
        default="targets",
        choices=["targets", "support_close", "recovery_recalibration", "recovery_recalibration_expansion"],
        help="targets: existing target map, support_close: step-up rerun current targets, recovery_recalibration: 4-task clean+perturb sweep, recovery_recalibration_expansion: full libero_goal clean+perturb sweep.",
    )
    ap.add_argument("--target", type=str, default="all", help="Target name from protocol_first.targets or 'all'.")
    ap.add_argument(
        "--support_close_namespace_tag",
        type=str,
        default="v1_paperstage",
        help="Namespace suffix for support_close runs. Use a new tag for step-up reruns.",
    )
    ap.add_argument(
        "--reuse_existing_phase_outputs",
        type=str,
        default="1",
        choices=["0", "1"],
        help="Whether support_close may reuse existing phase outputs.",
    )
    ap.add_argument(
        "--budget_episodes",
        type=str,
        default="",
        help="Optional support_close budget override for episodes.",
    )
    ap.add_argument(
        "--budget_seeds",
        type=str,
        default="",
        help="Optional support_close budget override for comma-separated seeds.",
    )
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    cfg = _load_cfg(args.config.resolve())
    config_path = args.config.resolve()
    protocol = cfg.get("protocol_first", {}) if isinstance(cfg.get("protocol_first"), dict) else {}
    summary_path = Path(str(protocol.get("shield_bootstrap_summary_json", "")))
    summary_rows = _parse_summary(summary_path)
    targets = _target_map(cfg)

    results_root = Path(str(protocol.get("results_root", PROJECT_ROOT / "results/libero_protocol_first")))
    results_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]]
    if args.mode == "recovery_recalibration":
        runs = _run_recovery_recalibration(
            cfg=cfg,
            config_path=config_path,
            summary_rows=summary_rows,
            execute=args.execute,
        )
        ledger_path = results_root / "recovery_recalibration" / "recovery_recalibration_ledger.json"
    elif args.mode == "recovery_recalibration_expansion":
        runs = _run_recalibration_campaign(
            cfg=cfg,
            config_path=config_path,
            execute=args.execute,
            block_key="recovery_recalibration_expansion",
            mode_name="recovery_recalibration_expansion",
            default_root_name="recovery_recalibration_expansion",
            default_namespace_prefix="protocol_first_recovery_recalibration_expansion_v1",
        )
        ledger_path = results_root / "recovery_recalibration_expansion" / "recovery_recalibration_expansion_ledger.json"
    else:
        selected: List[Dict[str, Any]]
        if args.target == "all":
            selected = list(targets.values())
        else:
            if args.target not in targets:
                raise SystemExit(f"Unknown protocol-first target: {args.target}")
            selected = [targets[args.target]]
        runs = [
            _run_target(
                cfg=cfg,
                config_path=config_path,
                target=target,
                summary_rows=summary_rows,
                execute=args.execute,
                mode=args.mode,
                support_close_namespace_tag=args.support_close_namespace_tag,
                reuse_existing_phase_outputs=args.reuse_existing_phase_outputs,
                budget_episodes_override=args.budget_episodes,
                budget_seeds_override=args.budget_seeds,
            )
            for target in selected
        ]
        ledger_path = results_root / f"protocol_first_{args.mode}_ledger.json"

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = {
        "config_path": str(args.config.resolve()),
        "mode": args.mode,
        "shield_bootstrap_summary_json": str(summary_path),
        "execute": bool(args.execute),
        "support_close_namespace_tag": str(args.support_close_namespace_tag),
        "reuse_existing_phase_outputs": str(args.reuse_existing_phase_outputs),
        "budget_episodes": str(args.budget_episodes),
        "budget_seeds": str(args.budget_seeds),
        "runs": runs,
    }
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[protocol-first] ledger={ledger_path}")

    overall_rc = 0
    for run in runs:
        print(
            "[protocol-first] "
            f"mode={run.get('mode', '')} "
            f"target={run.get('target', '')} "
            f"model={run['model']} task={run['task']} "
            f"status={run['status']} shield={run.get('shield_checkpoint', '')}"
        )
        run_rc = int(run.get("returncode", 0))
        if run_rc != 0 and overall_rc == 0:
            overall_rc = run_rc
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
