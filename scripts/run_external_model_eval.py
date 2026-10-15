#!/usr/bin/env python3
"""External model launcher for scale-up matrix shell backends.

This script is intentionally thin: it resolves an execution command template
from environment variables and runs it. This keeps orchestration generic while
allowing each external model to provide its own runtime entrypoint.

Exit codes:
- 0: success
- 42: blocked (required command template is not configured)
- non-zero: subprocess failure from external command
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict


BLOCKED_EXIT_CODE = 42


def _slug_upper(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_").upper() or "MODEL"


def _resolve_command_template(model: str, stage: str, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    model_token = _slug_upper(model)
    stage_token = stage.strip().upper()
    keys = [
        f"ROEP_EXTERNAL_{model_token}_{stage_token}_CMD",
        f"ROEP_EXTERNAL_{model_token}_CMD",
        f"ROEP_EXTERNAL_DEFAULT_{stage_token}_CMD",
    ]
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _resolve_python_launcher(model: str, stage: str, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    model_token = _slug_upper(model)
    stage_token = stage.strip().upper()
    keys = [
        f"ROEP_EXTERNAL_{model_token}_{stage_token}_PYTHON",
        f"ROEP_EXTERNAL_{model_token}_PYTHON",
        f"ROEP_EXTERNAL_DEFAULT_{stage_token}_PYTHON",
        "ROEP_EXTERNAL_MODEL_PYTHON",
        "ROEP_EXTERNAL_PYTHON",
    ]
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return str(sys.executable or "python")


def _resolve_eval_python(explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    env_value = str(os.environ.get("ROEP_EXTERNAL_EVAL_PYTHON", "")).strip()
    if env_value:
        return env_value
    return str(sys.executable or "python")


def _run(cmd: str, cwd: Path, execute: bool) -> int:
    print(cmd)
    if not execute:
        return 0
    proc = subprocess.run(cmd, shell=True, cwd=str(cwd), check=False)
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run external-model eval command for ROEP scale-up.")
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--stage", required=True, choices=["cache", "tau"])
    ap.add_argument("--task", required=True, type=str)
    ap.add_argument("--namespace", required=True, type=str)
    ap.add_argument("--episodes", required=True, type=int)
    ap.add_argument("--seeds", required=True, type=str)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--scenario", type=str, default="")
    ap.add_argument("--profiles", type=str, default="medium")
    ap.add_argument("--action_mode_profile", type=str, default="")
    ap.add_argument("--benchmark_backend", type=str, default=os.environ.get("ROEP_BENCHMARK_BACKEND", ""))
    ap.add_argument("--libero_suite", type=str, default=os.environ.get("ROEP_LIBERO_SUITE", ""))
    ap.add_argument("--libero_task_name", type=str, default=os.environ.get("ROEP_LIBERO_TASK_NAME", ""))
    ap.add_argument(
        "--libero_benchmark_name",
        type=str,
        default=os.environ.get("ROEP_LIBERO_BENCHMARK_NAME", ""),
    )
    ap.add_argument("--libero_task_id", type=str, default=os.environ.get("ROEP_LIBERO_TASK_ID", ""))
    ap.add_argument(
        "--libero_init_states_path",
        type=str,
        default=os.environ.get("ROEP_LIBERO_INIT_STATES_PATH", ""),
    )
    ap.add_argument(
        "--command_template",
        type=str,
        default="",
        help="Optional template override. If unset, environment variables are used.",
    )
    ap.add_argument(
        "--python_exec",
        type=str,
        default="",
        help=(
            "Optional launcher used by templates via {model_python}. "
            "If unset, resolves from ROEP_EXTERNAL_<MODEL>[_<STAGE>]_PYTHON."
        ),
    )
    ap.add_argument(
        "--eval_python",
        type=str,
        default="",
        help=(
            "Optional evaluator python launcher used by templates via {eval_python}. "
            "Defaults to ROEP_EXTERNAL_EVAL_PYTHON or current interpreter."
        ),
    )
    ap.add_argument(
        "--seed0_dir",
        type=str,
        default=os.environ.get("ROEP_SEED0_DIR", ""),
        help="Optional seed0_dir passthrough for external templates.",
    )
    ap.add_argument("--execute", action="store_true")
    args, passthrough = ap.parse_known_args()

    project_root = Path(__file__).resolve().parents[1]
    template = _resolve_command_template(args.model, args.stage, args.command_template)
    model_python = _resolve_python_launcher(args.model, args.stage, args.python_exec)
    eval_python = _resolve_eval_python(args.eval_python)
    if not template:
        model_token = _slug_upper(args.model)
        stage_token = args.stage.upper()
        print(
            "[blocked] external model command template is not configured.\n"
            f"  required env (preferred): ROEP_EXTERNAL_{model_token}_{stage_token}_CMD\n"
            f"  fallback env: ROEP_EXTERNAL_{model_token}_CMD or ROEP_EXTERNAL_DEFAULT_{stage_token}_CMD\n"
            "  template variables: {project_root} {task} {namespace} {episodes} {seeds} "
            "{tau} {stage} {model} {scenario} {profiles} {action_mode_profile} "
            "{benchmark_backend} {libero_suite} {libero_task_name} {libero_benchmark_name} "
            "{libero_task_id} {libero_init_states_path} {execute_flag} "
            "{model_python} {python_exec} {eval_python} {seed0_dir}",
            file=sys.stderr,
        )
        return BLOCKED_EXIT_CODE

    context: Dict[str, str] = {
        "project_root": str(project_root),
        "task": str(args.task),
        "namespace": str(args.namespace),
        "episodes": str(int(args.episodes)),
        "seeds": str(args.seeds),
        "tau": "" if args.tau is None else str(float(args.tau)),
        "stage": str(args.stage),
        "model": str(args.model),
        "scenario": str(args.scenario),
        "profiles": str(args.profiles),
        "action_mode_profile": str(args.action_mode_profile),
        "benchmark_backend": str(args.benchmark_backend),
        "libero_suite": str(args.libero_suite),
        "libero_task_name": str(args.libero_task_name),
        "libero_benchmark_name": str(args.libero_benchmark_name),
        "libero_task_id": str(args.libero_task_id),
        "libero_init_states_path": str(args.libero_init_states_path),
        "execute_flag": "--execute" if args.execute else "",
        "model_python": str(model_python),
        "python_exec": str(model_python),
        "eval_python": str(eval_python),
        "seed0_dir": str(args.seed0_dir or ""),
    }
    try:
        cmd = template.format(**context)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[blocked] invalid command template for model={args.model} stage={args.stage}: {exc}",
            file=sys.stderr,
        )
        return BLOCKED_EXIT_CODE

    if passthrough:
        cmd = f"{cmd} " + " ".join(shlex.quote(str(x)) for x in passthrough)

    rc = _run(cmd=cmd, cwd=project_root, execute=args.execute)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
