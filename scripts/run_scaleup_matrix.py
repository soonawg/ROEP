#!/usr/bin/env python3
"""Scale-up experiment orchestrator for ROEPBench.

Goals:
1) Schedule multi-task, multi-model, multi-threat, multi-tau evaluations.
2) Reuse baseline/perturb cache across tau sweeps for PerAct backend.
3) Support resume via ledger so large runs can be restarted safely.
4) Chain run -> validate -> aggregate -> gate automatically.

The config file is stored as JSON-compatible YAML (`.yaml` extension) so this
script can parse it with Python stdlib only.
"""

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SNAPSHOT_FILES = [
    "main_table.csv",
    "stats_table.csv",
    "hr_pairwise.csv",
    "hr_pairwise_details.csv",
]

PYTHON_EXE = sys.executable or "python"
BLOCKED_EXIT_CODE = 42


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "na"


def _tau_token(tau: float) -> str:
    return f"{tau:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _parse_csv_list(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_csv_floats(text: Optional[str]) -> List[float]:
    if not text:
        return []
    out: List[float] = []
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _load_json_compatible_yaml(path: Path) -> Dict[str, Any]:
    data = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Config parse failed: {path}. "
            "Use JSON-compatible YAML in this repo (PyYAML is intentionally not required)."
        ) from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return obj


def _build_shell(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _append_cli_arg(parts: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    flag = f"--{key}"
    if isinstance(value, bool):
        parts.extend([flag, "true" if value else "false"])
        return
    parts.extend([flag, str(value)])


def _run_shell(
    cmd: str,
    cwd: Path,
    execute: bool,
    capture_output: bool = False,
) -> Tuple[int, str]:
    print(cmd)
    if not execute:
        return 0, ""

    if capture_output:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            check=False,
            text=True,
            capture_output=True,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if out.strip():
            print(out.rstrip())
        return int(proc.returncode), out

    proc = subprocess.run(cmd, shell=True, cwd=str(cwd), check=False)
    return int(proc.returncode), ""


def _read_ledger(path: Path, campaign_name: str, resume: bool) -> Dict[str, Any]:
    if resume and path.exists():
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid ledger format: {path}")
        return obj

    return {
        "campaign_name": campaign_name,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "steps": {},
    }


def _write_ledger(path: Path, ledger: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger["updated_at"] = _now_iso()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _is_step_done(ledger: Dict[str, Any], step_key: str) -> bool:
    step = ledger.get("steps", {}).get(step_key)
    if not isinstance(step, dict):
        return False
    return step.get("status") == "done"


def _step_status(ledger: Dict[str, Any], step_key: str) -> str:
    step = ledger.get("steps", {}).get(step_key)
    if not isinstance(step, dict):
        return ""
    return str(step.get("status", "")).strip().lower()


def _step_flag(ledger: Dict[str, Any], step_key: str, flag: str) -> bool:
    step = ledger.get("steps", {}).get(step_key)
    if not isinstance(step, dict):
        return False
    return bool(step.get(flag, False))


def _set_step(
    ledger: Dict[str, Any],
    step_key: str,
    status: str,
    command: Optional[str] = None,
    returncode: Optional[int] = None,
    note: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    steps = ledger.setdefault("steps", {})
    current = steps.get(step_key, {}) if isinstance(steps.get(step_key), dict) else {}
    attempts = int(current.get("attempts", 0)) + 1
    payload: Dict[str, Any] = {
        "status": status,
        "updated_at": _now_iso(),
        "attempts": attempts,
    }
    if command is not None:
        payload["command"] = command
    if returncode is not None:
        payload["returncode"] = int(returncode)
    if note:
        payload["note"] = note
    if isinstance(extra, dict):
        for k, v in extra.items():
            payload[str(k)] = v
    steps[step_key] = payload


def _last_nonempty_line(text: str) -> str:
    for raw in reversed((text or "").splitlines()):
        line = raw.strip()
        if line:
            return line
    return ""


def _extract_blocked_reason(output: str) -> str:
    for raw in reversed((output or "").splitlines()):
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("[blocked]"):
            detail = line[len("[blocked]") :].strip()
            return detail or line
    return ""


def _extract_error_reason(output: str) -> str:
    wrapper_prefix = "[error] external command failed (model="
    fallback_error = ""
    for raw in reversed((output or "").splitlines()):
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("[error]"):
            if not fallback_error:
                fallback_error = line
            if line.lower().startswith(wrapper_prefix):
                continue
            detail = line[len("[error]") :].strip()
            return detail or line
    if fallback_error:
        detail = fallback_error[len("[error]") :].strip()
        return detail or fallback_error
    return _last_nonempty_line(output)


def _mark_combo_blocked(
    ledger: Dict[str, Any],
    combo_slug: str,
    reason: str,
    command: Optional[str] = None,
    returncode: Optional[int] = None,
) -> None:
    _set_step(
        ledger,
        f"{combo_slug}::combo",
        "blocked",
        command=command,
        returncode=returncode,
        note=reason,
        extra={"blocked_reason": reason},
    )


def _episode_count(demo_path: Path, task: str) -> int:
    epi_dir = demo_path / task / "all_variations" / "episodes"
    if not epi_dir.exists():
        return 0
    return len([p for p in epi_dir.iterdir() if p.is_dir() and p.name.startswith("episode")])


def _validate_preflight(
    cfg: Dict[str, Any],
    project_root: Path,
    selected_tasks: Optional[Sequence[str]] = None,
    selected_models: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    paths = cfg["paths"]
    demo_path = Path(paths["demo_path"]).resolve()
    seed0_dir = Path(paths["seed0_dir"]).resolve()
    prm_cfg = dict(cfg.get("prm_training", {}))
    prm_enabled = bool(prm_cfg.get("enabled", False))

    required_scripts = [
        project_root / "scripts" / "run_roep_experiments.py",
        project_root / "scripts" / "validate_roep_runs.py",
        project_root / "scripts" / "aggregate_roep_metrics.py",
        project_root / "scripts" / "evaluate_phase_gates.py",
    ]
    if prm_enabled:
        required_scripts.append(project_root / "scripts" / "train_lightweight_prm.py")
    missing = [str(p) for p in required_scripts if not p.exists()]
    if missing:
        raise FileNotFoundError("Required scripts missing:\n- " + "\n- ".join(missing))

    if not demo_path.exists():
        raise FileNotFoundError(f"demo_path not found: {demo_path}")
    if not seed0_dir.exists():
        raise FileNotFoundError(f"seed0_dir not found: {seed0_dir}")

    tasks = list(selected_tasks) if selected_tasks else list(cfg["global"]["tasks"])
    episodes = int(cfg["global"]["episodes"])
    for task in tasks:
        count = _episode_count(demo_path, task)
        print(f"[preflight] task={task} demos={count}")
        if count < episodes:
            demo_cmd = (
                "python external/RLBench/tools/dataset_generator.py "
                f"--tasks=\"{task}\" "
                f"--save_path=\"{demo_path}\" "
                "--image_size=128,128 "
                "--renderer=opengl "
                f"--episodes_per_task={episodes} "
                "--processes=1 "
                "--all_variations=True"
            )
            raise RuntimeError(
                f"Task '{task}' has only {count} demos, but episodes={episodes}. "
                "Lower episodes or add demos first.\n"
                "Recommended commands:\n"
                f"  bash {project_root / 'scripts' / 'start_xvfb.sh'}\n"
                f"  {demo_cmd}"
            )

    models = list(selected_models) if selected_models is not None else [
        m for m in cfg["models"] if bool(m.get("enabled", False))
    ]
    if not models:
        raise RuntimeError("No enabled model in config.")
    needs_shield_ckpt = False
    for model in models:
        backend = str(model.get("backend", "")).strip().lower()
        if backend not in ("peract", "shell"):
            raise ValueError(f"Unsupported backend for model={model.get('name')}: {backend}")
        if backend == "peract":
            needs_shield_ckpt = True
        if backend == "shell":
            ext_runner = project_root / "scripts" / "run_external_model_eval.py"
            if not ext_runner.exists():
                raise FileNotFoundError(
                    f"shell backend requires external runner script: {ext_runner}"
                )
            if not model.get("cache_command_template") or not model.get("tau_command_template"):
                raise ValueError(
                    f"shell backend requires cache_command_template/tau_command_template: {model.get('name')}"
                )

    if needs_shield_ckpt and not prm_enabled:
        shield_ckpt = Path(cfg["shield"]["checkpoint"]).resolve()
        if not shield_ckpt.exists():
            raise FileNotFoundError(f"safety shield checkpoint missing: {shield_ckpt}")


def _merge_dicts(*items: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        for k, v in item.items():
            out[k] = v
    return out


def _enabled_combos(
    cfg: Dict[str, Any],
    only_models: Iterable[str],
    only_tasks: Iterable[str],
    only_scenarios: Iterable[str],
    only_paper_candidates: bool = False,
    only_platform_candidates: bool = False,
) -> List[Dict[str, Any]]:
    model_filter = {_slug(x) for x in only_models}
    task_filter = {_slug(x) for x in only_tasks}
    scenario_filter = {_slug(x) for x in only_scenarios}

    combos: List[Dict[str, Any]] = []
    tasks = list(cfg["global"]["tasks"])
    for model in cfg["models"]:
        if not bool(model.get("enabled", False)):
            continue
        model_name = str(model["name"])
        if model_filter and _slug(model_name) not in model_filter:
            continue
        track_flags = _model_track_flags(model)
        if only_paper_candidates and not bool(track_flags["paper_candidate"]):
            continue
        if only_platform_candidates and not bool(track_flags["platform_candidate"]):
            continue

        for task in tasks:
            if task_filter and _slug(task) not in task_filter:
                continue
            for scenario in cfg["scenarios"]:
                if not bool(scenario.get("enabled", False)):
                    continue
                scenario_name = str(scenario["name"])
                if scenario_filter and _slug(scenario_name) not in scenario_filter:
                    continue

                combos.append(
                    {
                        "model": model,
                        "task": task,
                        "scenario": scenario,
                    }
                )
    return combos


def _run_ids_for_conditions(
    conditions_csv: str,
    profiles_csv: str,
    seeds_csv: str,
) -> List[str]:
    conditions = _parse_csv_list(conditions_csv)
    profiles = _parse_csv_list(profiles_csv)
    seeds = [int(x) for x in _parse_csv_list(seeds_csv)]

    out: List[str] = []
    for seed in seeds:
        if "baseline" in conditions:
            out.append(f"baseline_s{seed}")
        if "baseline_clone" in conditions:
            out.append(f"baseline_clone_s{seed}")
        if "recovery" in conditions:
            out.append(f"recovery_no_ood_s{seed}")

        for profile in profiles:
            if "perturb" in conditions:
                out.append(f"perturb_{profile}_s{seed}")
            if "perturb_clone" in conditions:
                out.append(f"perturb_clone_{profile}_s{seed}")
            if "full" in conditions:
                out.append(f"full_{profile}_s{seed}")
    return out


def _run_dirs_exist(seed0_dir: Path, namespace: str, run_ids: Sequence[str]) -> bool:
    for run_id in run_ids:
        run_dir = seed0_dir / f"roepbench_{run_id}__{namespace}"
        if not run_dir.exists():
            return False
    return True


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _run_dirs_have_expected_episodes(
    seed0_dir: Path,
    namespace: str,
    run_ids: Sequence[str],
    expected_episodes: int,
) -> bool:
    if expected_episodes <= 0:
        return _run_dirs_exist(seed0_dir, namespace, run_ids)
    for run_id in run_ids:
        run_dir = seed0_dir / f"roepbench_{run_id}__{namespace}"
        if not run_dir.exists():
            return False
        ep_logs = run_dir / "episode_logs.jsonl"
        if not ep_logs.exists():
            return False
        try:
            n = _count_lines(ep_logs)
        except OSError:
            return False
        if n != expected_episodes:
            return False
    return True


def _clear_run_dirs(seed0_dir: Path, namespace: str, run_ids: Sequence[str]) -> None:
    for run_id in run_ids:
        run_dir = seed0_dir / f"roepbench_{run_id}__{namespace}"
        if run_dir.is_symlink():
            run_dir.unlink()
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)


def _aggregate_outputs_exist(results_dir: Path) -> bool:
    for filename in SNAPSHOT_FILES:
        if not (results_dir / filename).exists():
            return False
    return True


def _prm_outputs_exist(prm_out_dir: Path) -> bool:
    return (prm_out_dir / "lightweight_prm.pth").exists() and (
        prm_out_dir / "lightweight_prm_metadata.json"
    ).exists()


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return int(default)


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def _read_summary_row(path: Path) -> Optional[Dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if isinstance(row, dict):
                    return {str(k): str(v) for k, v in row.items()}
    except Exception:  # noqa: BLE001
        return None
    return None


def _count_success_from_episode_logs(path: Path) -> Tuple[int, int]:
    episodes = 0
    successes = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                episodes += 1
                if bool(rec.get("success", False)):
                    successes += 1
    except Exception:  # noqa: BLE001
        return 0, 0
    return episodes, successes


def _compute_capability_metrics(
    seed0_dir: Path,
    cache_namespace: str,
    seeds_csv: str,
) -> Dict[str, Any]:
    seeds = [int(x) for x in _parse_csv_list(seeds_csv)]
    total_episodes = 0
    total_successes = 0
    baseline_run_count = 0
    summary_sources = 0
    jsonl_sources = 0
    missing_sources = 0

    for seed in seeds:
        run_id = f"baseline_s{seed}"
        run_dir = seed0_dir / f"roepbench_{run_id}__{cache_namespace}"
        if not run_dir.exists():
            missing_sources += 1
            continue
        baseline_run_count += 1

        summary_path = run_dir / "summary.csv"
        if summary_path.exists():
            row = _read_summary_row(summary_path)
            if row:
                episodes = _parse_int(row.get("episodes"), default=0)
                successes = _parse_int(row.get("success_count"), default=-1)
                if episodes > 0 and successes >= 0:
                    total_episodes += episodes
                    total_successes += successes
                    summary_sources += 1
                    continue

        ep_logs = run_dir / "episode_logs.jsonl"
        if ep_logs.exists():
            episodes, successes = _count_success_from_episode_logs(ep_logs)
            if episodes > 0:
                total_episodes += episodes
                total_successes += successes
                jsonl_sources += 1
                continue

        missing_sources += 1

    baseline_sr = (
        float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    )
    return {
        "baseline_sr": baseline_sr,
        "success_total": int(total_successes),
        "episode_total": int(total_episodes),
        "baseline_run_count": int(baseline_run_count),
        "summary_sources": int(summary_sources),
        "jsonl_sources": int(jsonl_sources),
        "missing_sources": int(missing_sources),
    }


def _link_or_copy_dir(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def _hydrate_cache_runs(
    seed0_dir: Path,
    cache_namespace: str,
    target_namespace: str,
    cache_conditions: str,
    profiles_csv: str,
    seeds_csv: str,
    overwrite_existing: bool = False,
) -> None:
    run_ids = _run_ids_for_conditions(cache_conditions, profiles_csv, seeds_csv)
    for run_id in run_ids:
        src = seed0_dir / f"roepbench_{run_id}__{cache_namespace}"
        dst = seed0_dir / f"roepbench_{run_id}__{target_namespace}"
        if not src.exists():
            raise FileNotFoundError(
                f"Cache source missing for hydrate: {src}. "
                "Run cache step successfully first."
            )
        if overwrite_existing:
            if dst.is_symlink():
                dst.unlink()
            elif dst.exists():
                shutil.rmtree(dst)
        _link_or_copy_dir(src, dst)


def _build_peract_cmd(
    project_root: Path,
    demo_path: Path,
    task: str,
    episodes: int,
    seeds_csv: str,
    namespace: str,
    conditions_csv: str,
    profiles_csv: str,
    merged_args: Dict[str, Any],
    execute: bool,
) -> str:
    parts: List[str] = [
        PYTHON_EXE,
        str(project_root / "scripts" / "run_roep_experiments.py"),
        "--demo_path",
        str(demo_path),
        "--task",
        task,
        "--conditions",
        conditions_csv,
        "--profiles",
        profiles_csv,
        "--episodes",
        str(episodes),
        "--seeds",
        seeds_csv,
        "--namespace",
        namespace,
    ]

    for key, value in merged_args.items():
        _append_cli_arg(parts, key, value)

    if execute:
        parts.append("--execute")
    return _build_shell(parts)


def _build_shell_backend_cmd(
    template: str,
    context: Dict[str, Any],
) -> str:
    # Templates are expected to include their own quoting where needed.
    return template.format(**context)


def _build_passthrough_cli_args(extra_args: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in extra_args.items():
        _append_cli_arg(parts, key, value)
    return " ".join(shlex.quote(str(p)) for p in parts)


def _default_action_mode_profile_for_model(model_name: str) -> str:
    token = _slug(model_name)
    if token in {"octo_base", "octo_small"}:
        return "ee_pose_ik_delta_discrete"
    return "joint_position_discrete"


def _model_track_flags(model: Dict[str, Any]) -> Dict[str, Any]:
    backend = str(model.get("backend", "")).strip().lower()
    default_external = backend == "shell"
    paper_candidate = bool(model.get("paper_candidate", default_external))
    platform_candidate = bool(model.get("platform_candidate", default_external))
    paper_role = str(
        model.get(
            "paper_role",
            "candidate" if paper_candidate else ("baseline" if backend == "peract" else "auxiliary"),
        )
    ).strip() or "candidate"
    return {
        "paper_candidate": paper_candidate,
        "platform_candidate": platform_candidate,
        "paper_role": paper_role,
    }


def _build_validate_cmd(
    project_root: Path,
    seed0_dir: Path,
    namespace: str,
    flags: Dict[str, Any],
) -> str:
    parts = [
        PYTHON_EXE,
        str(project_root / "scripts" / "validate_roep_runs.py"),
        "--seed0_dir",
        str(seed0_dir),
        "--glob",
        f"roepbench_*__{namespace}",
        "--expected_episodes",
        str(int(flags.get("expected_episodes", 10))),
    ]

    for key, value in flags.items():
        if key == "expected_episodes":
            continue
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
            continue
        if value is None:
            continue
        parts.extend([f"--{key}", str(value)])
    return _build_shell(parts)


def _build_aggregate_cmd(
    project_root: Path,
    seed0_dir: Path,
    namespace: str,
    out_dir: Path,
) -> str:
    parts = [
        PYTHON_EXE,
        str(project_root / "scripts" / "aggregate_roep_metrics.py"),
        "--seed0_dir",
        str(seed0_dir),
        "--glob",
        f"roepbench_*__{namespace}",
        "--out_dir",
        str(out_dir),
    ]
    return _build_shell(parts)


def _build_gate_cmd(
    project_root: Path,
    results_dir: Path,
    gate_flags: Dict[str, Any],
) -> str:
    parts = [
        PYTHON_EXE,
        str(project_root / "scripts" / "evaluate_phase_gates.py"),
        "--results_dir",
        str(results_dir),
    ]
    for key, value in gate_flags.items():
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
            continue
        if value is None:
            continue
        parts.extend([f"--{key}", str(value)])
    return _build_shell(parts)


def _build_prm_train_cmd(
    project_root: Path,
    seed0_dir: Path,
    cache_namespace: str,
    prm_out_dir: Path,
    prm_cfg: Dict[str, Any],
) -> str:
    parts = [
        PYTHON_EXE,
        str(project_root / "scripts" / "train_lightweight_prm.py"),
        "--seed0_dir",
        str(seed0_dir),
        "--glob",
        f"roepbench_*__{cache_namespace}",
        "--out_dir",
        str(prm_out_dir),
    ]

    scalar_keys = [
        "horizon_steps",
        "alpha",
        "epochs",
        "batch_size",
        "lr",
        "hidden_dims",
        "seed",
        "device",
        "strict_keys",
        "min_rows",
        "train_ratio",
        "val_ratio",
        "safe_outcomes",
        "positive_outcomes",
        "fallback_positive_outcomes",
        "fallback_safe_outcomes",
    ]
    for key in scalar_keys:
        if key not in prm_cfg:
            continue
        value = prm_cfg.get(key)
        if value is None:
            continue
        parts.extend([f"--{key}", str(value)])
    return _build_shell(parts)


def _snapshot_results(
    project_root: Path,
    campaign_name: str,
    model_name: str,
    task: str,
    scenario_name: str,
    tau: float,
    namespace: str,
) -> Path:
    token = _tau_token(tau)
    base = (
        project_root
        / "results"
        / "scaleup"
        / _slug(campaign_name)
        / _slug(model_name)
        / _slug(task)
        / _slug(scenario_name)
        / f"tau_{token}"
    )
    base.mkdir(parents=True, exist_ok=True)

    src_dir = project_root / "results" / namespace
    for filename in SNAPSHOT_FILES:
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, base / filename)

    meta = {
        "snapshot_at": _now_iso(),
        "campaign_name": campaign_name,
        "model": model_name,
        "task": task,
        "scenario": scenario_name,
        "tau": tau,
        "namespace": namespace,
        "results_dir": str(src_dir),
    }
    with (base / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ROEPBench scale-up matrix with resume ledger.")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scaleup_master_v1.yaml"),
        help="JSON-compatible YAML config path.",
    )
    ap.add_argument("--resume", action="store_true", help="Resume from existing ledger.")
    ap.add_argument("--execute", action="store_true", help="Actually run commands.")
    ap.add_argument("--preflight_only", action="store_true", help="Run preflight and exit.")
    ap.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to next step even if a command fails.",
    )
    ap.add_argument("--only_model", type=str, default="", help="Comma-separated model filter.")
    ap.add_argument("--only_task", type=str, default="", help="Comma-separated task filter.")
    ap.add_argument("--only_scenario", type=str, default="", help="Comma-separated scenario filter.")
    ap.add_argument("--tau_values", type=str, default="", help="Override tau grid with comma-separated values.")
    ap.add_argument("--max_combos", type=int, default=0, help="Limit number of combos for quick tests.")
    ap.add_argument(
        "--only_paper_candidates",
        action="store_true",
        help="Run only models marked as paper_candidate in config (default: external shell models).",
    )
    ap.add_argument(
        "--only_platform_candidates",
        action="store_true",
        help="Run only models marked as platform_candidate in config (default: external shell models).",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = _load_json_compatible_yaml(args.config.resolve())

    campaign_name = str(cfg.get("campaign_name", "scaleup_v1"))
    namespace_prefix = str(cfg.get("namespace_prefix", campaign_name))

    only_models = _parse_csv_list(args.only_model)
    only_tasks = _parse_csv_list(args.only_task)
    only_scenarios = _parse_csv_list(args.only_scenario)

    combos = _enabled_combos(
        cfg,
        only_models,
        only_tasks,
        only_scenarios,
        only_paper_candidates=bool(args.only_paper_candidates),
        only_platform_candidates=bool(args.only_platform_candidates),
    )
    if args.max_combos and args.max_combos > 0:
        combos = combos[: args.max_combos]

    if not combos:
        raise RuntimeError("No enabled combinations after filters.")

    selected_tasks = sorted({str(c["task"]) for c in combos})
    selected_models = [c["model"] for c in combos]
    _validate_preflight(
        cfg,
        project_root,
        selected_tasks=selected_tasks,
        selected_models=selected_models,
    )

    tau_grid = _parse_csv_floats(args.tau_values)
    if not tau_grid:
        tau_grid = [float(x) for x in cfg["shield"]["tau_grid"]]
    if not tau_grid:
        raise RuntimeError("tau grid is empty.")

    paths_cfg = cfg["paths"]
    demo_path = Path(paths_cfg["demo_path"]).resolve()
    seed0_dir = Path(paths_cfg["seed0_dir"]).resolve()
    noise_floor_results = Path(paths_cfg["noise_floor_results_dir"]).resolve()
    baseline_results_dir = str(paths_cfg.get("baseline_results_dir", "")).strip()

    global_cfg = cfg["global"]
    episodes = int(global_cfg["episodes"])
    seeds_csv = str(global_cfg["seeds"])
    default_profiles = str(global_cfg["profiles"])
    cache_conditions = str(global_cfg.get("conditions_cache", "baseline,perturb"))
    tau_conditions = str(global_cfg.get("conditions_tau", "recovery,full"))
    capability_floor_min_sr = _parse_float(
        global_cfg.get("capability_floor_min_sr", 0.05),
        default=0.05,
    )
    prm_cfg = dict(cfg.get("prm_training", {}))
    prm_enabled = bool(prm_cfg.get("enabled", False))
    prm_cfg.setdefault("seed", 42)
    prm_cfg.setdefault("device", "auto")
    prm_cfg.setdefault("horizon_steps", 5)
    prm_cfg.setdefault("alpha", 0.05)
    prm_cfg.setdefault("epochs", 40)
    prm_cfg.setdefault("batch_size", 256)
    prm_cfg.setdefault("lr", 1e-3)
    prm_cfg.setdefault("hidden_dims", "128,64,32")
    prm_cfg.setdefault("strict_keys", "true")
    prm_cfg.setdefault("min_rows", 100)
    prm_cfg.setdefault("train_ratio", 0.8)
    prm_cfg.setdefault("val_ratio", 0.1)
    prm_cfg.setdefault("safe_outcomes", "success")
    prm_cfg.setdefault("positive_outcomes", "unsafe_attempt")
    prm_cfg.setdefault("fallback_positive_outcomes", "timeout")
    prm_cfg.setdefault("fallback_safe_outcomes", "timeout")
    prm_out_root = Path(
        str(
            prm_cfg.get(
                "out_root",
                project_root / "checkpoints" / "scaleup_prm",
            )
        )
    ).resolve()

    validate_flags = dict(cfg.get("validate_flags", {}))
    gate_flags = dict(cfg.get("gate_flags", {}))
    gate_flags["noise_floor_dir"] = str(noise_floor_results)
    if baseline_results_dir:
        gate_flags["baseline_results_dir"] = baseline_results_dir

    ledger_path = project_root / "results" / "scaleup" / _slug(campaign_name) / "ledger.json"
    ledger = _read_ledger(ledger_path, campaign_name=campaign_name, resume=args.resume)

    print(f"[info] campaign={campaign_name}")
    print(f"[info] combos={len(combos)}, tau_grid={tau_grid}")
    print(f"[info] prm_training_enabled={prm_enabled}")
    print(f"[info] ledger={ledger_path}")

    if args.preflight_only:
        print("[preflight] OK")
        return 0

    fail_fast = not args.continue_on_error

    for idx, combo in enumerate(combos, start=1):
        model = combo["model"]
        scenario = combo["scenario"]
        task = combo["task"]

        model_name = str(model["name"])
        backend = str(model["backend"]).strip().lower()
        scenario_name = str(scenario["name"])
        profiles_csv = str(scenario.get("profiles", default_profiles))
        track_flags = _model_track_flags(model)
        combo_track_extra = {
            "paper_candidate": bool(track_flags["paper_candidate"]),
            "platform_candidate": bool(track_flags["platform_candidate"]),
            "paper_role": str(track_flags["paper_role"]),
        }
        combo_paper_admission_status = (
            "pending" if bool(track_flags["paper_candidate"]) else "not_applicable"
        )
        combo_paper_admission_reason = ""

        combo_slug = f"{_slug(model_name)}__{_slug(task)}__{_slug(scenario_name)}"
        namespace_base = f"{namespace_prefix}__{combo_slug}"
        cache_namespace = f"{namespace_base}__cache"
        combo_status_key = f"{combo_slug}::combo"

        print(
            f"\n[combo {idx}/{len(combos)}] model={model_name} task={task} "
            f"scenario={scenario_name} backend={backend}"
        )
        if args.resume and _is_step_done(ledger, combo_status_key):
            if _step_flag(ledger, combo_status_key, "capability_floor"):
                print(
                    f"[skip] {combo_status_key} already marked capability-floor "
                    "in ledger (resume mode)"
                )
                continue
        if args.resume and _step_status(ledger, combo_status_key) == "blocked":
            combo_step = ledger.get("steps", {}).get(combo_status_key, {})
            blocked_rc = combo_step.get("returncode")
            try:
                blocked_rc_int = int(blocked_rc)
            except Exception:  # noqa: BLE001
                blocked_rc_int = -1
            if blocked_rc_int == BLOCKED_EXIT_CODE:
                print(f"[skip] {combo_status_key} already blocked in ledger (resume mode)")
                continue
            print(
                f"[resume-fix] {combo_status_key} was blocked with rc={blocked_rc} (legacy/no-rc entry). "
                "Retrying this combo."
            )
        combo_blocked = False
        combo_failed = False

        model_base_args = dict(model.get("base_args", {}))
        scenario_base_args = dict(scenario.get("base_args", {}))
        scenario_risk_args = dict(scenario.get("risk_args", {}))
        model_action_mode_profile = str(model.get("action_mode_profile", "")).strip()
        combo_action_mode_profile = (
            model_action_mode_profile
            if model_action_mode_profile
            else _default_action_mode_profile_for_model(model_name)
        )
        cache_run_ids = _run_ids_for_conditions(cache_conditions, profiles_csv, seeds_csv)
        tau_run_ids = _run_ids_for_conditions(tau_conditions, profiles_csv, seeds_csv)
        combo_shield_ckpt = str(cfg["shield"]["checkpoint"])

        # 1) Cache step (baseline + perturb)
        cache_step_key = f"{combo_slug}::cache::run"
        can_skip_cache = False
        cache_was_rerun = False
        if args.resume and _is_step_done(ledger, cache_step_key):
            if backend in {"peract", "shell"} and args.execute:
                can_skip_cache = _run_dirs_have_expected_episodes(
                    seed0_dir,
                    cache_namespace,
                    cache_run_ids,
                    episodes,
                )
                if not can_skip_cache:
                    print(
                        f"[resume-fix] {cache_step_key} marked done, but cache dirs are missing/incomplete. "
                        "Re-running cache step."
                )
            else:
                can_skip_cache = True
        if can_skip_cache:
            print(f"[skip] {cache_step_key} already done")
        else:
            cache_was_rerun = bool(args.execute and args.resume and _is_step_done(ledger, cache_step_key))
            if backend == "peract":
                if args.execute:
                    _clear_run_dirs(seed0_dir, cache_namespace, cache_run_ids)
                cache_args = _merge_dicts(model_base_args, scenario_base_args, scenario_risk_args)
                cache_cmd = _build_peract_cmd(
                    project_root=project_root,
                    demo_path=demo_path,
                    task=task,
                    episodes=episodes,
                    seeds_csv=seeds_csv,
                    namespace=cache_namespace,
                    conditions_csv=cache_conditions,
                    profiles_csv=profiles_csv,
                    merged_args=cache_args,
                    execute=args.execute,
                )
            else:
                template = str(model["cache_command_template"])
                ctx = {
                    "project_root": str(project_root),
                    "demo_path": str(demo_path),
                    "task": task,
                    "episodes": episodes,
                    "seeds": seeds_csv,
                    "namespace": cache_namespace,
                    "scenario": scenario_name,
                    "profiles": profiles_csv,
                    "seed0_dir": str(seed0_dir),
                    "python_exec": str(PYTHON_EXE),
                    "matrix_python": str(PYTHON_EXE),
                    "execute_flag": "--execute" if args.execute else "",
                    "action_mode_profile": combo_action_mode_profile,
                }
                cache_cmd = _build_shell_backend_cmd(template, ctx)
                risk_passthrough = _build_passthrough_cli_args(scenario_risk_args)
                if risk_passthrough:
                    cache_cmd = f"{cache_cmd} {risk_passthrough}"
                if "--action_mode_profile" not in cache_cmd:
                    cache_cmd = (
                        f"{cache_cmd} --action_mode_profile "
                        f"{shlex.quote(str(combo_action_mode_profile))}"
                    )

            rc, cache_out = _run_shell(
                cache_cmd,
                cwd=project_root,
                execute=args.execute,
                capture_output=(backend == "shell"),
            )
            if rc != 0 and args.execute and backend == "shell":
                blocked_detail = _extract_blocked_reason(cache_out)
                if rc == BLOCKED_EXIT_CODE:
                    reason = blocked_detail or (
                        f"external model cache blocked (model={model_name}, task={task}, "
                        f"scenario={scenario_name}, rc={rc})"
                    )
                    combo_paper_admission_status = (
                        "blocked" if bool(track_flags["paper_candidate"]) else "not_applicable"
                    )
                    combo_paper_admission_reason = reason
                    _set_step(
                        ledger,
                        cache_step_key,
                        "blocked",
                        command=cache_cmd,
                        returncode=rc,
                        note=reason,
                        extra={
                            "blocked_reason": reason,
                            "action_mode_profile": combo_action_mode_profile,
                        },
                    )
                    _mark_combo_blocked(
                        ledger,
                        combo_slug=combo_slug,
                        reason=reason,
                        command=cache_cmd,
                        returncode=rc,
                    )
                    _set_step(
                        ledger,
                        combo_status_key,
                        "blocked",
                        command=cache_cmd,
                        returncode=rc,
                        note=reason,
                        extra={
                            **combo_track_extra,
                            "blocked_reason": reason,
                            "paper_admission_status": combo_paper_admission_status,
                            "paper_admission_reason": combo_paper_admission_reason,
                        },
                    )
                    _write_ledger(ledger_path, ledger)
                    print(f"[blocked] {reason}")
                    combo_blocked = True
                    continue

                error_detail = _extract_error_reason(cache_out)
                reason = error_detail or (
                    f"external model cache failed (model={model_name}, task={task}, "
                    f"scenario={scenario_name}, rc={rc})"
                )
                combo_paper_admission_status = (
                    "failed" if bool(track_flags["paper_candidate"]) else "not_applicable"
                )
                combo_paper_admission_reason = reason
                _set_step(
                    ledger,
                    cache_step_key,
                    "failed",
                    command=cache_cmd,
                    returncode=rc,
                    note=reason,
                    extra={
                        "failure_reason": reason,
                        "action_mode_profile": combo_action_mode_profile,
                    },
                )
                _set_step(
                    ledger,
                    combo_status_key,
                    "failed",
                    command=cache_cmd,
                    returncode=rc,
                    note=reason,
                    extra={
                        **combo_track_extra,
                        "failure_reason": reason,
                        "paper_admission_status": combo_paper_admission_status,
                        "paper_admission_reason": combo_paper_admission_reason,
                    },
                )
                _write_ledger(ledger_path, ledger)
                print(f"[error] {reason}")
                combo_failed = True
                if fail_fast:
                    raise RuntimeError(f"Cache step failed: {cache_step_key}")
                continue

            status = "done" if (args.execute and rc == 0) else ("planned" if not args.execute else "failed")
            _set_step(
                ledger,
                cache_step_key,
                status,
                command=cache_cmd,
                returncode=rc,
                extra={"action_mode_profile": combo_action_mode_profile},
            )
            _write_ledger(ledger_path, ledger)
            if args.execute and rc != 0:
                combo_failed = True
            if rc != 0 and fail_fast and args.execute:
                raise RuntimeError(f"Cache step failed: {cache_step_key}")

        # 1.25) Capability-floor precheck from cache baseline runs.
        capability_key = f"{combo_slug}::capability::check"
        can_skip_capability = False
        if args.resume and _is_step_done(ledger, capability_key):
            can_skip_capability = not cache_was_rerun
        if can_skip_capability:
            print(f"[skip] {capability_key} already done")
            capability_metrics = ledger.get("steps", {}).get(capability_key, {})
            combo_capability_floor = bool(
                capability_metrics.get("capability_floor", False)
            )
            capability_reason = str(capability_metrics.get("capability_floor_reason", ""))
            if bool(track_flags["paper_candidate"]):
                combo_paper_admission_status = str(
                    capability_metrics.get(
                        "paper_admission_status",
                        "rejected_capability_floor" if combo_capability_floor else "admitted_capability",
                    )
                ).strip() or combo_paper_admission_status
                combo_paper_admission_reason = str(capability_metrics.get("capability_floor_reason", "")).strip()
                if not combo_paper_admission_reason and not combo_capability_floor:
                    combo_paper_admission_reason = (
                        f"baseline_sr={_parse_float(capability_metrics.get('baseline_sr'), default=0.0):.4f}, "
                        f"success_total={_parse_int(capability_metrics.get('success_total'), default=0)}"
                    )
            if combo_capability_floor:
                print(
                    f"[capability-floor] skip combo {combo_slug}: "
                    f"{capability_reason or 'baseline capability floor hit'}"
                )
                continue
        else:
            capability_metrics = _compute_capability_metrics(
                seed0_dir=seed0_dir,
                cache_namespace=cache_namespace,
                seeds_csv=seeds_csv,
            )
            baseline_sr = _parse_float(capability_metrics.get("baseline_sr"), default=0.0)
            success_total = _parse_int(capability_metrics.get("success_total"), default=0)
            episode_total = _parse_int(capability_metrics.get("episode_total"), default=0)
            missing_sources = _parse_int(capability_metrics.get("missing_sources"), default=0)
            baseline_run_count = _parse_int(capability_metrics.get("baseline_run_count"), default=0)

            # Missing baseline cache sources should be handled as cache integrity failure,
            # not as capability floor.
            if args.execute and (
                baseline_run_count <= 0 or missing_sources > 0 or episode_total <= 0
            ):
                reason = (
                    "capability precheck source missing/incomplete: "
                    f"baseline_run_count={baseline_run_count}, "
                    f"missing_sources={missing_sources}, episode_total={episode_total}"
                )
                combo_paper_admission_status = (
                    "failed" if bool(track_flags["paper_candidate"]) else "not_applicable"
                )
                combo_paper_admission_reason = reason
                _set_step(
                    ledger,
                    capability_key,
                    "failed",
                    note=reason,
                    extra={
                        **capability_metrics,
                        "failure_reason": reason,
                    },
                )
                _set_step(
                    ledger,
                    combo_status_key,
                    "failed",
                    note=reason,
                    extra={
                        **combo_track_extra,
                        "failure_reason": reason,
                        "paper_admission_status": combo_paper_admission_status,
                        "paper_admission_reason": combo_paper_admission_reason,
                    },
                )
                _write_ledger(ledger_path, ledger)
                combo_failed = True
                print(f"[error] {reason}")
                if fail_fast:
                    raise RuntimeError(f"Capability precheck failed: {capability_key}")
                continue

            capability_floor = (success_total == 0) or (baseline_sr < capability_floor_min_sr)
            if success_total == 0:
                capability_reason = (
                    f"baseline success_total == 0 in cache namespace "
                    f"(baseline_sr={baseline_sr:.4f}, episodes={capability_metrics.get('episode_total', 0)})"
                )
            elif baseline_sr < capability_floor_min_sr:
                capability_reason = (
                    f"baseline_sr={baseline_sr:.4f} < capability_floor_min_sr={capability_floor_min_sr:.4f}"
                )
            else:
                capability_reason = ""

            _set_step(
                ledger,
                capability_key,
                "done" if args.execute else "planned",
                note=(
                    capability_reason
                    if capability_floor
                    else (
                        f"capability ok: baseline_sr={baseline_sr:.4f}, "
                        f"success_total={success_total}"
                    )
                ),
                extra={
                    "capability_floor": bool(capability_floor),
                    "capability_floor_reason": capability_reason,
                    "capability_floor_min_sr": float(capability_floor_min_sr),
                    "paper_admission_status": (
                        "rejected_capability_floor"
                        if (capability_floor and bool(track_flags["paper_candidate"]))
                        else (
                            "admitted_capability"
                            if bool(track_flags["paper_candidate"])
                            else "not_applicable"
                        )
                    ),
                    **capability_metrics,
                },
            )
            _write_ledger(ledger_path, ledger)

            if capability_floor:
                combo_paper_admission_status = (
                    "rejected_capability_floor"
                    if bool(track_flags["paper_candidate"])
                    else "not_applicable"
                )
                combo_paper_admission_reason = capability_reason
                print(f"[capability-floor] {combo_slug}: {capability_reason}")
                _set_step(
                    ledger,
                    combo_status_key,
                    "done" if args.execute else "planned",
                    note=f"capability-floor: {capability_reason}",
                    extra={
                        **combo_track_extra,
                        "capability_floor": True,
                        "capability_floor_reason": capability_reason,
                        "capability_floor_min_sr": float(capability_floor_min_sr),
                        "skip_tau": True,
                        "paper_admission_status": combo_paper_admission_status,
                        "paper_admission_reason": combo_paper_admission_reason,
                        **capability_metrics,
                    },
                )

                # Mark downstream steps as intentionally skipped for resume safety.
                skip_extra = {
                    "skipped_by_capability_floor": True,
                    "capability_floor_reason": capability_reason,
                }
                if prm_enabled and backend in {"peract", "shell"}:
                    _set_step(
                        ledger,
                        f"{combo_slug}::prm::train",
                        "done" if args.execute else "planned",
                        note="skipped by capability floor",
                        extra=skip_extra,
                    )
                for tau in tau_grid:
                    tau_tok = _tau_token(tau)
                    for key in (
                        f"{combo_slug}::tau::{tau_tok}::hydrate",
                        f"{combo_slug}::tau::{tau_tok}::run",
                        f"{combo_slug}::tau::{tau_tok}::validate",
                        f"{combo_slug}::tau::{tau_tok}::aggregate",
                        f"{combo_slug}::tau::{tau_tok}::gate",
                        f"{combo_slug}::tau::{tau_tok}::snapshot",
                    ):
                        _set_step(
                            ledger,
                            key,
                            "done" if args.execute else "planned",
                            note="skipped by capability floor",
                            extra=skip_extra,
                        )
                _write_ledger(ledger_path, ledger)
                continue
            if bool(track_flags["paper_candidate"]) and combo_paper_admission_status == "pending":
                combo_paper_admission_status = "admitted_capability"
                combo_paper_admission_reason = (
                    f"baseline_sr={baseline_sr:.4f}, success_total={success_total}"
                )

        # 1.5) Task+scenario-specific PRM train from cache runs.
        prm_step_key = f"{combo_slug}::prm::train"
        prm_out_dir = (
            prm_out_root
            / _slug(campaign_name)
            / _slug(model_name)
            / _slug(task)
            / _slug(scenario_name)
            / "prm"
        )
        if prm_enabled and backend in {"peract", "shell"}:
            can_skip_prm = False
            if args.resume and _is_step_done(ledger, prm_step_key):
                if _step_flag(ledger, prm_step_key, "skipped_by_capability_floor"):
                    can_skip_prm = True
                elif args.execute:
                    can_skip_prm = _prm_outputs_exist(prm_out_dir)
                    if not can_skip_prm:
                        print(
                            f"[resume-fix] {prm_step_key} marked done, but PRM outputs are missing. "
                            "Re-running PRM training."
                        )
                else:
                    can_skip_prm = True
            if can_skip_prm:
                print(f"[skip] {prm_step_key} already done")
                if args.execute:
                    candidate_ckpt = prm_out_dir / "lightweight_prm.pth"
                    if candidate_ckpt.exists():
                        combo_shield_ckpt = str(candidate_ckpt)
            else:
                prm_cmd = _build_prm_train_cmd(
                    project_root=project_root,
                    seed0_dir=seed0_dir,
                    cache_namespace=cache_namespace,
                    prm_out_dir=prm_out_dir,
                    prm_cfg=prm_cfg,
                )
                rc, prm_out = _run_shell(
                    prm_cmd,
                    cwd=project_root,
                    execute=args.execute,
                    capture_output=True,
                )
                if args.execute and rc != 0:
                    tail = prm_out.strip().splitlines()[-1] if prm_out.strip() else ""
                    if backend == "peract":
                        fallback_ckpt = str(cfg["shield"]["checkpoint"])
                        note = (
                            "prm training failed; fallback checkpoint used "
                            f"(prm_out_dir={prm_out_dir}, fallback={fallback_ckpt})"
                        )
                        if tail:
                            note = f"{note}; error={tail}"
                        _set_step(
                            ledger,
                            prm_step_key,
                            "done",
                            command=prm_cmd,
                            returncode=rc,
                            note=note,
                            extra={
                                "prm_fallback_used": True,
                                "fallback_checkpoint": fallback_ckpt,
                                "prm_out_dir": str(prm_out_dir),
                            },
                        )
                        combo_shield_ckpt = fallback_ckpt
                        print(
                            f"[warn] PRM train failed for {combo_slug}. Using fallback checkpoint: {fallback_ckpt}"
                        )
                    else:
                        note = (
                            "prm training failed for external backend; "
                            "no fallback checkpoint is allowed for shell backend "
                            f"(prm_out_dir={prm_out_dir})"
                        )
                        if tail:
                            note = f"{note}; error={tail}"
                        combo_paper_admission_reason = combo_paper_admission_reason or note
                        _set_step(
                            ledger,
                            prm_step_key,
                            "failed",
                            command=prm_cmd,
                            returncode=rc,
                            note=note,
                            extra={
                                "prm_fallback_used": False,
                                "prm_out_dir": str(prm_out_dir),
                                "failure_reason": note,
                            },
                        )
                        _set_step(
                            ledger,
                            combo_status_key,
                            "failed",
                            command=prm_cmd,
                            returncode=rc,
                            note=note,
                            extra={
                                **combo_track_extra,
                                "failure_reason": note,
                                "paper_admission_status": combo_paper_admission_status,
                                "paper_admission_reason": combo_paper_admission_reason,
                            },
                        )
                        _write_ledger(ledger_path, ledger)
                        print(f"[error] {note}")
                        combo_failed = True
                        if fail_fast:
                            raise RuntimeError(f"PRM training failed: {prm_step_key}")
                        continue
                else:
                    status = "done" if (args.execute and rc == 0) else ("planned" if not args.execute else "failed")
                    note = f"prm_out_dir={prm_out_dir}"
                    _set_step(
                        ledger,
                        prm_step_key,
                        status,
                        command=prm_cmd,
                        returncode=rc,
                        note=note,
                        extra={
                            "prm_fallback_used": False,
                            "prm_out_dir": str(prm_out_dir),
                        },
                    )
                    if args.execute and rc == 0:
                        combo_shield_ckpt = str(prm_out_dir / "lightweight_prm.pth")
                _write_ledger(ledger_path, ledger)

        # 2) Tau sweep (recovery + full)
        for tau in tau_grid:
            tau_tok = _tau_token(tau)
            tau_namespace = f"{namespace_base}__tau_{tau_tok}"

            hydrate_key = f"{combo_slug}::tau::{tau_tok}::hydrate"
            run_key = f"{combo_slug}::tau::{tau_tok}::run"
            val_key = f"{combo_slug}::tau::{tau_tok}::validate"
            agg_key = f"{combo_slug}::tau::{tau_tok}::aggregate"
            gate_key = f"{combo_slug}::tau::{tau_tok}::gate"
            snap_key = f"{combo_slug}::tau::{tau_tok}::snapshot"

            can_skip_hydrate = False
            if args.resume and _is_step_done(ledger, hydrate_key):
                if _step_flag(ledger, hydrate_key, "skipped_by_capability_floor"):
                    can_skip_hydrate = True
                elif args.execute:
                    can_skip_hydrate = _run_dirs_have_expected_episodes(
                        seed0_dir,
                        tau_namespace,
                        cache_run_ids,
                        episodes,
                    )
                    if not can_skip_hydrate:
                        print(
                            f"[resume-fix] {hydrate_key} marked done, but hydrated dirs are missing/incomplete. "
                            "Re-hydrating cache."
                        )
                else:
                    can_skip_hydrate = True
            if can_skip_hydrate:
                print(f"[skip] {hydrate_key} already done")
            else:
                try:
                    if args.execute:
                        _hydrate_cache_runs(
                            seed0_dir=seed0_dir,
                            cache_namespace=cache_namespace,
                            target_namespace=tau_namespace,
                            cache_conditions=cache_conditions,
                            profiles_csv=profiles_csv,
                            seeds_csv=seeds_csv,
                            overwrite_existing=True,
                        )
                        note = "cache hydrated"
                        status = "done"
                    else:
                        print(
                            f"[dry-run] hydrate cache namespace {cache_namespace} -> {tau_namespace}"
                        )
                        note = "dry-run"
                        status = "planned"
                    _set_step(ledger, hydrate_key, status, note=note)
                except Exception as exc:  # noqa: BLE001
                    _set_step(
                        ledger,
                        hydrate_key,
                        "failed",
                        note=f"hydrate failed: {exc}",
                    )
                    _write_ledger(ledger_path, ledger)
                    if args.execute:
                        combo_failed = True
                    if fail_fast and args.execute:
                        raise
                _write_ledger(ledger_path, ledger)

            can_skip_run = False
            if args.resume and _is_step_done(ledger, run_key):
                if _step_flag(ledger, run_key, "skipped_by_capability_floor"):
                    can_skip_run = True
                elif backend == "peract" and args.execute:
                    can_skip_run = _run_dirs_have_expected_episodes(
                        seed0_dir,
                        tau_namespace,
                        tau_run_ids,
                        episodes,
                    )
                    if not can_skip_run:
                        print(
                            f"[resume-fix] {run_key} marked done, but tau run dirs are missing/incomplete. "
                            "Re-running tau step."
                        )
                else:
                    can_skip_run = True
            if can_skip_run:
                print(f"[skip] {run_key} already done")
            else:
                if backend == "peract":
                    if args.execute:
                        _clear_run_dirs(seed0_dir, tau_namespace, tau_run_ids)
                    shield_args = {
                        "safety_shield_checkpoint": combo_shield_ckpt,
                        "safety_shield_device": cfg["shield"].get("device", "cuda"),
                        "safety_shield_tau": tau,
                        "safety_shield_mdsp_mode": cfg["shield"].get(
                            "mdsp_mode", "lift_z_scale_xy"
                        ),
                        "safety_shield_refusal_risk_threshold": cfg["shield"].get(
                            "refusal_risk_threshold", 0.995
                        ),
                        "safety_shield_verify_projected_action": cfg["shield"].get(
                            "verify_projected_action", "false"
                        ),
                        "safety_shield_z_lift": cfg["shield"].get("z_lift", 0.05),
                        "safety_shield_xy_scale": cfg["shield"].get("xy_scale", 0.60),
                        "safety_shield_max_translation_norm": cfg["shield"].get(
                            "max_translation_norm", 1.05
                        ),
                        "safety_shield_apply_conditions": cfg["shield"].get(
                            "apply_conditions", "recovery,full"
                        ),
                    }
                    tau_args = _merge_dicts(
                        model_base_args,
                        scenario_base_args,
                        scenario_risk_args,
                        shield_args,
                    )
                    run_cmd = _build_peract_cmd(
                        project_root=project_root,
                        demo_path=demo_path,
                        task=task,
                        episodes=episodes,
                        seeds_csv=seeds_csv,
                        namespace=tau_namespace,
                        conditions_csv=tau_conditions,
                        profiles_csv=profiles_csv,
                        merged_args=tau_args,
                        execute=args.execute,
                    )
                else:
                    template = str(model["tau_command_template"])
                    ctx = {
                        "project_root": str(project_root),
                        "demo_path": str(demo_path),
                        "task": task,
                        "episodes": episodes,
                        "seeds": seeds_csv,
                        "namespace": tau_namespace,
                        "scenario": scenario_name,
                        "profiles": profiles_csv,
                        "tau": tau,
                        "seed0_dir": str(seed0_dir),
                        "python_exec": str(PYTHON_EXE),
                        "matrix_python": str(PYTHON_EXE),
                        "execute_flag": "--execute" if args.execute else "",
                        "action_mode_profile": combo_action_mode_profile,
                    }
                    run_cmd = _build_shell_backend_cmd(template, ctx)
                    risk_passthrough = _build_passthrough_cli_args(scenario_risk_args)
                    if risk_passthrough:
                        run_cmd = f"{run_cmd} {risk_passthrough}"
                    if "--action_mode_profile" not in run_cmd:
                        run_cmd = (
                            f"{run_cmd} --action_mode_profile "
                            f"{shlex.quote(str(combo_action_mode_profile))}"
                        )

                rc, run_out = _run_shell(
                    run_cmd,
                    cwd=project_root,
                    execute=args.execute,
                    capture_output=(backend == "shell"),
                )
                if rc != 0 and args.execute and backend == "shell":
                    blocked_detail = _extract_blocked_reason(run_out)
                    if rc == BLOCKED_EXIT_CODE:
                        reason = blocked_detail or (
                            f"external model tau run blocked (model={model_name}, task={task}, "
                            f"scenario={scenario_name}, tau={tau}, rc={rc})"
                        )
                        combo_paper_admission_reason = combo_paper_admission_reason or reason
                        _set_step(
                            ledger,
                            run_key,
                            "blocked",
                            command=run_cmd,
                            returncode=rc,
                            note=reason,
                            extra={
                                "blocked_reason": reason,
                                "action_mode_profile": combo_action_mode_profile,
                            },
                        )
                        _mark_combo_blocked(
                            ledger,
                            combo_slug=combo_slug,
                            reason=reason,
                            command=run_cmd,
                            returncode=rc,
                        )
                        _set_step(
                            ledger,
                            combo_status_key,
                            "blocked",
                            command=run_cmd,
                            returncode=rc,
                            note=reason,
                            extra={
                                **combo_track_extra,
                                "blocked_reason": reason,
                                "paper_admission_status": combo_paper_admission_status,
                                "paper_admission_reason": combo_paper_admission_reason,
                            },
                        )
                        _write_ledger(ledger_path, ledger)
                        print(f"[blocked] {reason}")
                        combo_blocked = True
                        break

                    error_detail = _extract_error_reason(run_out)
                    reason = error_detail or (
                        f"external model tau run failed (model={model_name}, task={task}, "
                        f"scenario={scenario_name}, tau={tau}, rc={rc})"
                    )
                    combo_paper_admission_reason = combo_paper_admission_reason or reason
                    _set_step(
                        ledger,
                        run_key,
                        "failed",
                        command=run_cmd,
                        returncode=rc,
                        note=reason,
                        extra={
                            "failure_reason": reason,
                            "action_mode_profile": combo_action_mode_profile,
                        },
                    )
                    _set_step(
                        ledger,
                        combo_status_key,
                        "failed",
                        command=run_cmd,
                        returncode=rc,
                        note=reason,
                        extra={
                            **combo_track_extra,
                            "failure_reason": reason,
                            "paper_admission_status": combo_paper_admission_status,
                            "paper_admission_reason": combo_paper_admission_reason,
                        },
                    )
                    _write_ledger(ledger_path, ledger)
                    print(f"[error] {reason}")
                    combo_failed = True
                    if fail_fast:
                        raise RuntimeError(f"Tau run failed: {run_key}")
                    continue

                status = "done" if (args.execute and rc == 0) else ("planned" if not args.execute else "failed")
                _set_step(
                    ledger,
                    run_key,
                    status,
                    command=run_cmd,
                    returncode=rc,
                    extra={"action_mode_profile": combo_action_mode_profile},
                )
                _write_ledger(ledger_path, ledger)
                if args.execute and rc != 0:
                    combo_failed = True
                if rc != 0 and fail_fast and args.execute:
                    raise RuntimeError(f"Tau run failed: {run_key}")

            # Validate
            can_skip_validate = False
            if args.resume and _is_step_done(ledger, val_key):
                if _step_flag(ledger, val_key, "skipped_by_capability_floor"):
                    can_skip_validate = True
                elif backend == "peract" and args.execute:
                    can_skip_validate = _run_dirs_have_expected_episodes(
                        seed0_dir,
                        tau_namespace,
                        tau_run_ids,
                        episodes,
                    )
                    if not can_skip_validate:
                        print(
                            f"[resume-fix] {val_key} marked done, but required run dirs are missing/incomplete. "
                            "Re-running validate."
                        )
                else:
                    can_skip_validate = True
            if can_skip_validate:
                print(f"[skip] {val_key} already done")
            else:
                validate_cmd = _build_validate_cmd(
                    project_root=project_root,
                    seed0_dir=seed0_dir,
                    namespace=tau_namespace,
                    flags=validate_flags,
                )
                rc, _ = _run_shell(validate_cmd, cwd=project_root, execute=args.execute)
                status = "done" if (args.execute and rc == 0) else ("planned" if not args.execute else "failed")
                _set_step(ledger, val_key, status, command=validate_cmd, returncode=rc)
                _write_ledger(ledger_path, ledger)
                if args.execute and rc != 0:
                    combo_failed = True
                if rc != 0 and fail_fast and args.execute:
                    raise RuntimeError(f"Validate failed: {val_key}")

            # Aggregate
            out_dir = project_root / "results" / tau_namespace
            can_skip_aggregate = False
            if args.resume and _is_step_done(ledger, agg_key):
                if _step_flag(ledger, agg_key, "skipped_by_capability_floor"):
                    can_skip_aggregate = True
                elif args.execute:
                    can_skip_aggregate = _aggregate_outputs_exist(out_dir)
                    if not can_skip_aggregate:
                        print(
                            f"[resume-fix] {agg_key} marked done, but aggregate outputs are missing. "
                            "Re-running aggregate."
                        )
                else:
                    can_skip_aggregate = True
            if can_skip_aggregate:
                print(f"[skip] {agg_key} already done")
            else:
                agg_cmd = _build_aggregate_cmd(
                    project_root=project_root,
                    seed0_dir=seed0_dir,
                    namespace=tau_namespace,
                    out_dir=out_dir,
                )
                rc, _ = _run_shell(agg_cmd, cwd=project_root, execute=args.execute)
                status = "done" if (args.execute and rc == 0) else ("planned" if not args.execute else "failed")
                _set_step(ledger, agg_key, status, command=agg_cmd, returncode=rc)
                _write_ledger(ledger_path, ledger)
                if args.execute and rc != 0:
                    combo_failed = True
                if rc != 0 and fail_fast and args.execute:
                    raise RuntimeError(f"Aggregate failed: {agg_key}")

            # Gate
            gate_snap_dir = (
                project_root
                / "results"
                / "scaleup"
                / _slug(campaign_name)
                / _slug(model_name)
                / _slug(task)
                / _slug(scenario_name)
                / f"tau_{tau_tok}"
            )
            gate_stdout_file = gate_snap_dir / "gate_stdout.txt"
            can_skip_gate = False
            if args.resume and _is_step_done(ledger, gate_key):
                if _step_flag(ledger, gate_key, "skipped_by_capability_floor"):
                    can_skip_gate = True
                elif args.execute:
                    can_skip_gate = gate_stdout_file.exists()
                    if not can_skip_gate:
                        print(
                            f"[resume-fix] {gate_key} marked done, but gate output snapshot is missing. "
                            "Re-running gate."
                        )
                else:
                    can_skip_gate = True
            if can_skip_gate:
                print(f"[skip] {gate_key} already done")
            else:
                gate_cmd = _build_gate_cmd(
                    project_root=project_root,
                    results_dir=project_root / "results" / tau_namespace,
                    gate_flags=gate_flags,
                )
                rc, gate_output = _run_shell(
                    gate_cmd,
                    cwd=project_root,
                    execute=args.execute,
                    capture_output=True,
                )
                status = "done" if (args.execute and rc == 0) else ("planned" if not args.execute else "failed")
                _set_step(ledger, gate_key, status, command=gate_cmd, returncode=rc)
                _write_ledger(ledger_path, ledger)
                if args.execute:
                    gate_snap_dir.mkdir(parents=True, exist_ok=True)
                    gate_stdout_file.write_text(gate_output, encoding="utf-8")
                if args.execute and rc != 0:
                    combo_failed = True
                if rc != 0 and fail_fast and args.execute:
                    raise RuntimeError(f"Gate failed: {gate_key}")

            # Snapshot
            if args.resume and _is_step_done(ledger, snap_key):
                print(f"[skip] {snap_key} already done")
            else:
                try:
                    if args.execute:
                        dst = _snapshot_results(
                            project_root=project_root,
                            campaign_name=campaign_name,
                            model_name=model_name,
                            task=task,
                            scenario_name=scenario_name,
                            tau=tau,
                            namespace=tau_namespace,
                        )
                        _set_step(ledger, snap_key, "done", note=f"snapshot={dst}")
                    else:
                        print(f"[dry-run] snapshot results for namespace={tau_namespace}")
                        _set_step(ledger, snap_key, "planned", note="dry-run")
                except Exception as exc:  # noqa: BLE001
                    _set_step(ledger, snap_key, "failed", note=f"snapshot failed: {exc}")
                    _write_ledger(ledger_path, ledger)
                    if args.execute:
                        combo_failed = True
                    if fail_fast and args.execute:
                        raise
                _write_ledger(ledger_path, ledger)

        if args.execute:
            if not combo_blocked and combo_failed:
                if _step_status(ledger, combo_status_key) != "failed":
                    _set_step(
                        ledger,
                        combo_status_key,
                        "failed",
                        note="combo completed with step failures",
                        extra={
                            **combo_track_extra,
                            "paper_admission_status": combo_paper_admission_status,
                            "paper_admission_reason": combo_paper_admission_reason,
                        },
                    )
                    _write_ledger(ledger_path, ledger)
            elif not combo_blocked and (not combo_failed):
                _set_step(
                    ledger,
                    combo_status_key,
                    "done",
                    note="combo completed",
                    extra={
                        **combo_track_extra,
                        "paper_admission_status": combo_paper_admission_status,
                        "paper_admission_reason": combo_paper_admission_reason,
                    },
                )
                _write_ledger(ledger_path, ledger)

    print("\n[done] scaleup matrix pipeline completed")
    print(f"[done] ledger: {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
