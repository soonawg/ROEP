#!/usr/bin/env python3
"""External model adapter runner (identity-safe, rollout-enabled).

This script executes benchmark rollouts using a model adapter and writes the
same artifact set expected by ROEP validators:
- episode_logs.jsonl
- step_logs.jsonl
- summary.csv
- episode_manifest.csv

Exit codes:
- 0: success
- 42: blocked (missing adapter/model prerequisites)
- 1: runtime failure
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = PROJECT_ROOT / "roepbench"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from roepbench.backends import BackendNotReadyError, EpisodeRunConfig, get_backend  # noqa: E402
from roepbench.logging_utils import write_episode_jsonl, write_step_jsonl, write_summary_csv  # noqa: E402
from roepbench.model_adapters import (  # noqa: E402
    AdapterConfig,
    AdapterNotReadyError,
    LAPLiberoAdapter,
    OctoBaseAdapter,
    OpenVLA7BAdapter,
    PI05LiberoAdapter,
    PI0FastAdapter,
    RDT170MAdapter,
    SmolVLAAdapter,
    SpatialVLAAdapter,
    VLAAdapter,
    XVLAAdapter,
)


BLOCKED_EXIT_CODE = 42

ADAPTERS: Dict[str, Type] = {
    "lap_libero": LAPLiberoAdapter,
    "octo_base": OctoBaseAdapter,
    "pi05_libero": PI05LiberoAdapter,
    "pi0_fast": PI0FastAdapter,
    "rdt_170m": RDT170MAdapter,
    "openvla_7b": OpenVLA7BAdapter,
    "smolvla_base": SmolVLAAdapter,
    "spatialvla": SpatialVLAAdapter,
    "vla_adapter": VLAAdapter,
    "xvla": XVLAAdapter,
    # Backward-compatible aliases.
    "lap": LAPLiberoAdapter,
    "pi05": PI05LiberoAdapter,
    "vit_bc_lang": RDT170MAdapter,
    "octo_small": OctoBaseAdapter,
    "openvla": OpenVLA7BAdapter,
}

MODEL_DEFAULT_ACTION_MODE_PROFILE: Dict[str, str] = {
    "octo_base": "ee_pose_ik_delta_discrete",
    "octo_small": "ee_pose_ik_delta_discrete",
}

LIBERO_DEFAULT_ACTION_MODE_PROFILE = "libero_ee_rel_pose_gripper"

PROFILE_TO_OOD: Dict[str, Tuple[float, float]] = {
    "clean": (0.0, 0.0),
    "mild": (0.10, 0.10),
    "medium": (0.30, 0.20),
    "strong": (0.50, 0.30),
}

MODEL_DEFAULT_BENCHMARK_BACKEND: Dict[str, str] = {
    "lap_libero": "libero",
    "pi05_libero": "libero",
    "smolvla_base": "libero",
    "spatialvla": "libero",
    "vla_adapter": "libero",
    "xvla": "libero",
    "pi0_fast": "libero",
}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text == "":
        return bool(default)
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _enforce_requested_cuda_if_needed(device: str) -> Optional[str]:
    requested = str(device or "").strip().lower()
    if not requested.startswith("cuda"):
        return None
    if not _parse_bool(os.environ.get("ROEP_REQUIRE_CUDA", "1"), default=True):
        return None
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return f"device={device} requested but torch import failed before CUDA probe: {type(exc).__name__}: {exc}"

    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    if not available:
        return (
            f"device={device} requested but CUDA is unavailable in the current execution context "
            f"(torch.cuda.device_count={device_count})."
        )
    try:
        _ = torch.zeros(1, device="cuda")
    except Exception as exc:  # noqa: BLE001
        return (
            f"device={device} requested but CUDA tensor allocation failed "
            f"({type(exc).__name__}: {exc})."
        )
    return None


def _resolve_action_mode_profile(model: str, explicit: str, benchmark_backend: str) -> str:
    value = str(explicit or "").strip()
    if value:
        return value
    if str(benchmark_backend or "").strip().lower() == "libero":
        return LIBERO_DEFAULT_ACTION_MODE_PROFILE
    return MODEL_DEFAULT_ACTION_MODE_PROFILE.get(str(model), "joint_position_discrete")


def _prewarm_libero_runtime_if_needed(benchmark_backend: str) -> None:
    if str(benchmark_backend or "").strip().lower() != "libero":
        return
    import libero  # type: ignore  # noqa: F401
    from libero.libero import benchmark as _benchmark  # type: ignore  # noqa: F401
    from libero.libero.envs import OffScreenRenderEnv as _OffScreenRenderEnv  # type: ignore  # noqa: F401


def _risk_defaults_from_scenario(scenario_name: str) -> Dict[str, Any]:
    scen = str(scenario_name or "").strip().lower()
    # Natural OOD by default (no synthetic attack injection).
    out: Dict[str, Any] = {
        "risk_suite_enabled": False,
        "risk_attack_channel": "none",
        "risk_attack_probability": 0.0,
        "risk_attack_severity": 0.0,
        "risk_visual_alpha": 0.0,
        "risk_visual_patch_ratio": 0.0,
        "risk_threat_model": "natural_ood_v1" if "natural" in scen else "none",
        "risk_instruction_prefix": "",
        "risk_instruction_suffix": "",
        "risk_attack_counts_as_ood": False,
        "risk_attack_sampling_mode": "deterministic_hash",
        # Keep default salt namespace-invariant for cache/tau parity.
        "risk_attack_hash_salt": "external_model_default",
    }

    if "threat" not in scen:
        return out

    channel = "hybrid"
    if "instruction" in scen:
        channel = "instruction"
    elif "visual" in scen:
        channel = "visual"
    elif "conceptual" in scen:
        channel = "conceptual"

    severity = 0.35
    if "mild" in scen:
        severity = 0.20
    elif "strong" in scen:
        severity = 0.60

    out.update(
        {
            "risk_suite_enabled": True,
            "risk_attack_channel": channel,
            "risk_attack_probability": 1.0,
            "risk_attack_severity": float(severity),
            "risk_visual_alpha": 0.4,
            "risk_visual_patch_ratio": 0.12,
            "risk_threat_model": "risk_suite_v1",
            "risk_attack_counts_as_ood": False,
            "risk_attack_sampling_mode": "deterministic_hash",
            # Keep default salt namespace-invariant for cache/tau parity.
            "risk_attack_hash_salt": "external_threat_default",
        }
    )
    return out


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_csv(text: str) -> List[int]:
    return [int(token) for token in _parse_csv_list(text)]


def _default_ckpt(model: str) -> str:
    candidate_keys = {
        "lap_libero": ["ROEP_LAP_LIBERO_CKPT", "ROEP_LAP_CKPT"],
        "octo_base": ["ROEP_OCTO_BASE_CKPT", "ROEP_OCTO_SMALL_CKPT"],
        "pi05_libero": ["ROEP_PI05_LIBERO_CKPT", "ROEP_PI05_CKPT"],
        "pi0_fast": ["ROEP_PI0_FAST_CKPT", "ROEP_PI0_CKPT"],
        "rdt_170m": ["ROEP_RDT_170M_CKPT", "ROEP_VIT_BC_LANG_CKPT"],
        "openvla_7b": ["ROEP_OPENVLA_7B_CKPT", "ROEP_OPENVLA_CKPT"],
        "smolvla_base": ["ROEP_SMOLVLA_BASE_CKPT"],
        "spatialvla": ["ROEP_SPATIALVLA_CKPT"],
        "vla_adapter": ["ROEP_VLA_ADAPTER_CKPT"],
        "xvla": ["ROEP_XVLA_CKPT"],
        # Backward-compatible aliases.
        "lap": ["ROEP_LAP_CKPT", "ROEP_LAP_LIBERO_CKPT"],
        "pi05": ["ROEP_PI05_CKPT", "ROEP_PI05_LIBERO_CKPT"],
        "vit_bc_lang": ["ROEP_VIT_BC_LANG_CKPT", "ROEP_RDT_170M_CKPT"],
        "octo_small": ["ROEP_OCTO_SMALL_CKPT", "ROEP_OCTO_BASE_CKPT"],
        "openvla": ["ROEP_OPENVLA_CKPT", "ROEP_OPENVLA_7B_CKPT"],
    }[model]
    for key in candidate_keys:
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    return ""


def _find_xvfb_script() -> Path:
    candidates = [
        PROJECT_ROOT / "scripts" / "start_xvfb.sh",
        PROJECT_ROOT / "roepbench" / "scripts" / "start_xvfb.sh",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find scripts/start_xvfb.sh")


def _start_xvfb_if_needed() -> None:
    if str(os.environ.get("DISPLAY", "")).strip() != "":
        return
    script = _find_xvfb_script()
    subprocess.run(["bash", str(script)], check=True)


def _resolve_benchmark_backend(model: str, explicit: str) -> str:
    value = str(explicit or "").strip().lower()
    if value:
        return value
    return MODEL_DEFAULT_BENCHMARK_BACKEND.get(str(model), "rlbench")


def _build_run_specs(stage: str, profiles: List[str]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if stage == "cache":
        specs.append(
            {
                "condition": "baseline",
                "profile": "clean",
                "run_prefix": "baseline",
                "occlusion": 0.0,
                "depth": 0.0,
                "shield": False,
                "ood_profile_id": "clean",
            }
        )
        for profile in profiles:
            if profile not in PROFILE_TO_OOD:
                raise RuntimeError(f"Unknown profile '{profile}' for cache stage.")
            occ, dep = PROFILE_TO_OOD[profile]
            specs.append(
                {
                    "condition": "perturb",
                    "profile": profile,
                    "run_prefix": f"perturb_{profile}",
                    "occlusion": occ,
                    "depth": dep,
                    "shield": False,
                    "ood_profile_id": profile,
                }
            )
        return specs

    if stage == "noise":
        specs.append(
            {
                "condition": "baseline_clone",
                "profile": "clean",
                "run_prefix": "baseline_clone",
                "occlusion": 0.0,
                "depth": 0.0,
                "shield": False,
                "ood_profile_id": "clean",
            }
        )
        for profile in profiles:
            if profile not in PROFILE_TO_OOD:
                raise RuntimeError(f"Unknown profile '{profile}' for noise stage.")
            occ, dep = PROFILE_TO_OOD[profile]
            specs.append(
                {
                    "condition": "perturb_clone",
                    "profile": profile,
                    "run_prefix": f"perturb_clone_{profile}",
                    "occlusion": occ,
                    "depth": dep,
                    "shield": False,
                    "ood_profile_id": profile,
                }
            )
        return specs

    if stage in {"tau", "recovery", "full"}:
        if stage in {"tau", "recovery"}:
            specs.append(
                {
                    "condition": "recovery",
                    "profile": "no_ood",
                    "run_prefix": "recovery_no_ood",
                    "occlusion": 0.0,
                    "depth": 0.0,
                    "shield": True,
                    "ood_profile_id": "clean",
                }
            )
        if stage in {"tau", "full"}:
            for profile in profiles:
                if profile not in PROFILE_TO_OOD:
                    raise RuntimeError(f"Unknown profile '{profile}' for {stage} stage.")
                occ, dep = PROFILE_TO_OOD[profile]
                specs.append(
                    {
                        "condition": "full",
                        "profile": profile,
                        "run_prefix": f"full_{profile}",
                        "occlusion": occ,
                        "depth": dep,
                        "shield": True,
                        "ood_profile_id": profile,
                    }
                )
        return specs

    raise RuntimeError(f"Unsupported stage: {stage}")


def _write_episode_manifest(
    out_dir: Path,
    run_id: str,
    task: str,
    seed: int,
    ood_profile_id: str,
    records: List[Dict[str, Any]],
) -> None:
    manifest_path = out_dir / "episode_manifest.csv"
    fieldnames = [
        "run_id",
        "episode_idx",
        "task",
        "seed",
        "eval_demo_seed",
        "lang_goal",
        "ood_active",
        "ood_profile_id",
        "pairing_key",
        "threat_model",
        "attack_channel",
        "attack_applied",
        "attack_variant",
    ]
    rows: List[Dict[str, Any]] = []
    for rec in records:
        epi = int(rec.get("episode_idx", 0))
        lang_goal = str(rec.get("lang_goal", ""))
        row_seed = int(rec.get("seed", seed))
        eval_demo_seed = int(rec.get("eval_demo_seed", epi))
        pairing_key = str(rec.get("pairing_key", "")).strip()
        if pairing_key == "":
            pairing_key = f"{task}|{row_seed}|{epi}|{eval_demo_seed}|{lang_goal}"
        rows.append(
            {
                "run_id": run_id,
                "episode_idx": epi,
                "task": task,
                "seed": row_seed,
                "eval_demo_seed": eval_demo_seed,
                "lang_goal": lang_goal,
                "ood_active": bool(rec.get("ood_active", False)),
                "ood_profile_id": str(ood_profile_id),
                "pairing_key": pairing_key,
                "threat_model": str(rec.get("threat_model", "none")),
                "attack_channel": str(rec.get("attack_channel", "none")),
                "attack_applied": bool(rec.get("attack_applied", False)),
                "attack_variant": str(rec.get("attack_variant", "none")),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_one(
    *,
    adapter: Any,
    benchmark_backend: str,
    task: str,
    episodes: int,
    seed: int,
    namespace: str,
    seed0_dir: Path,
    run_prefix: str,
    occlusion: float,
    depth: float,
    use_shield: bool,
    shield_ckpt: str,
    shield_device: str,
    tau: Optional[float],
    shield_mdsp_mode: str,
    shield_refusal_risk_threshold: float,
    shield_verify_projected_action: bool,
    shield_z_lift: float,
    shield_xy_scale: float,
    shield_max_translation_norm: float,
    ood_profile_id: str,
    max_steps: int,
    headless: bool,
    risk_args: Dict[str, Any],
    action_mode_profile: str,
    libero_suite: str,
    libero_task_name: str,
    libero_benchmark_name: str,
    libero_task_id: str,
    libero_init_states_path: str,
) -> None:
    run_id = f"{run_prefix}_s{seed}"
    out_dir = seed0_dir / f"roepbench_{run_id}__{namespace}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _policy(obs_dict: Dict[str, Any], lang_goal: Optional[str]) -> Any:
        return adapter.predict_action(obs_dict=obs_dict, lang_goal=lang_goal)

    backend = get_backend(benchmark_backend)
    config = EpisodeRunConfig(
        benchmark_backend=str(benchmark_backend),
        task=task,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        headless=headless,
        run_id=out_dir.name,
        occlusion_severity=float(occlusion),
        depth_dropout_prob=float(depth),
        enable_recovery=bool(use_shield),
        detector="reward_plateau",
        reward_window=10,
        reward_eps=0.0,
        confirm_windows=2,
        trigger_cooldown_steps=20,
        margin_eps=0.0,
        min_plateau_steps=10,
        stagnation_window=15,
        stagnation_eps=1e-4,
        recovery_burst_steps=1,
        warmup_steps=6,
        safety_shield_checkpoint=(shield_ckpt if use_shield else None),
        safety_shield_device=shield_device,
        safety_shield_tau=tau,
        safety_shield_mdsp_mode=shield_mdsp_mode,
        safety_shield_action_low=-1.0,
        safety_shield_action_high=1.0,
        safety_shield_refusal_risk_threshold=shield_refusal_risk_threshold,
        safety_shield_verify_projected_action=shield_verify_projected_action,
        safety_shield_z_lift=shield_z_lift,
        safety_shield_xy_scale=shield_xy_scale,
        safety_shield_max_translation_norm=shield_max_translation_norm,
        policy_fn=_policy,
        policy_name=str(adapter.model_name),
        strict_runtime_errors=True,
        risk_suite_enabled=bool(risk_args.get("risk_suite_enabled", False)),
        risk_attack_channel=str(risk_args.get("risk_attack_channel", "none")),
        risk_attack_probability=float(risk_args.get("risk_attack_probability", 0.0)),
        risk_attack_severity=float(risk_args.get("risk_attack_severity", 0.0)),
        risk_visual_alpha=float(risk_args.get("risk_visual_alpha", 0.0)),
        risk_visual_patch_ratio=float(risk_args.get("risk_visual_patch_ratio", 0.0)),
        risk_threat_model=str(risk_args.get("risk_threat_model", "none")),
        risk_instruction_prefix=str(risk_args.get("risk_instruction_prefix", "")),
        risk_instruction_suffix=str(risk_args.get("risk_instruction_suffix", "")),
        risk_attack_counts_as_ood=bool(risk_args.get("risk_attack_counts_as_ood", False)),
        risk_attack_sampling_mode=str(risk_args.get("risk_attack_sampling_mode", "deterministic_hash")),
        risk_attack_hash_salt=str(risk_args.get("risk_attack_hash_salt", "external_model")),
        action_mode_profile=str(action_mode_profile),
        libero_suite=str(libero_suite),
        libero_task_name=str(libero_task_name),
        libero_benchmark_name=str(libero_benchmark_name),
        libero_task_id=str(libero_task_id),
        libero_init_states_path=str(libero_init_states_path),
    )
    records, step_records = backend.run_policy_episodes(config)

    write_episode_jsonl(out_dir / "episode_logs.jsonl", records)
    write_step_jsonl(out_dir / "step_logs.jsonl", step_records)
    write_summary_csv(out_dir / "summary.csv", records)
    _write_episode_manifest(
        out_dir=out_dir,
        run_id=out_dir.name,
        task=task,
        seed=seed,
        ood_profile_id=ood_profile_id,
        records=records,
    )

    print(f"[ok] {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description="External model adapter eval runner.")
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS.keys()))
    ap.add_argument("--stage", required=True, choices=["cache", "noise", "recovery", "full", "tau"])
    ap.add_argument("--task", required=True, type=str)
    ap.add_argument("--namespace", required=True, type=str)
    ap.add_argument("--episodes", required=True, type=int)
    ap.add_argument("--seeds", required=True, type=str)
    ap.add_argument("--profiles", default="medium", type=str)
    ap.add_argument("--scenario", default="", type=str)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--checkpoint", default="", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument(
        "--benchmark_backend",
        default=os.environ.get("ROEP_BENCHMARK_BACKEND", ""),
        type=str,
    )
    ap.add_argument("--libero_suite", default=os.environ.get("ROEP_LIBERO_SUITE", ""), type=str)
    ap.add_argument(
        "--libero_task_name",
        default=os.environ.get("ROEP_LIBERO_TASK_NAME", ""),
        type=str,
    )
    ap.add_argument(
        "--libero_benchmark_name",
        default=os.environ.get("ROEP_LIBERO_BENCHMARK_NAME", ""),
        type=str,
    )
    ap.add_argument(
        "--libero_task_id",
        default=os.environ.get("ROEP_LIBERO_TASK_ID", ""),
        type=str,
    )
    ap.add_argument(
        "--libero_init_states_path",
        default=os.environ.get("ROEP_LIBERO_INIT_STATES_PATH", ""),
        type=str,
    )
    ap.add_argument("--seed0_dir", default=os.environ.get("ROEP_SEED0_DIR", ""), type=str)
    ap.add_argument(
        "--safety_shield_checkpoint",
        default=os.environ.get("ROEP_EXTERNAL_SHIELD_CKPT", ""),
        type=str,
    )
    ap.add_argument(
        "--safety_shield_device",
        default=os.environ.get("ROEP_SAFETY_SHIELD_DEVICE", "cuda"),
        type=str,
    )
    ap.add_argument(
        "--safety_shield_mdsp_mode",
        default=os.environ.get("ROEP_EXTERNAL_SHIELD_MDSP_MODE", "lift_z_scale_xy"),
        type=str,
    )
    ap.add_argument(
        "--safety_shield_refusal_risk_threshold",
        default=float(os.environ.get("ROEP_EXTERNAL_SHIELD_REFUSAL_RISK_THRESHOLD", "0.995")),
        type=float,
    )
    ap.add_argument(
        "--safety_shield_verify_projected_action",
        action=argparse.BooleanOptionalAction,
        default=_parse_bool(os.environ.get("ROEP_EXTERNAL_SHIELD_VERIFY_PROJECTED_ACTION", "false")),
    )
    ap.add_argument(
        "--safety_shield_z_lift",
        default=float(os.environ.get("ROEP_EXTERNAL_SHIELD_Z_LIFT", "0.05")),
        type=float,
    )
    ap.add_argument(
        "--safety_shield_xy_scale",
        default=float(os.environ.get("ROEP_EXTERNAL_SHIELD_XY_SCALE", "0.8")),
        type=float,
    )
    ap.add_argument(
        "--safety_shield_max_translation_norm",
        default=float(os.environ.get("ROEP_EXTERNAL_SHIELD_MAX_TRANSLATION_NORM", "1.05")),
        type=float,
    )
    ap.add_argument("--max_steps", default=120, type=int)
    ap.add_argument(
        "--skip_clean_baseline",
        action="store_true",
        help=(
            "For cache/noise diagnostic runs, skip baseline/baseline_clone and run only "
            "profile perturbation specs. Default is false to preserve main protocol behavior."
        ),
    )
    ap.add_argument("--risk_suite_enabled", default="", type=str)
    ap.add_argument("--risk_attack_channel", default="", type=str)
    ap.add_argument("--risk_attack_probability", default=None, type=float)
    ap.add_argument("--risk_attack_severity", default=None, type=float)
    ap.add_argument("--risk_visual_alpha", default=None, type=float)
    ap.add_argument("--risk_visual_patch_ratio", default=None, type=float)
    ap.add_argument("--risk_threat_model", default="", type=str)
    ap.add_argument("--risk_instruction_prefix", default="", type=str)
    ap.add_argument("--risk_instruction_suffix", default="", type=str)
    ap.add_argument("--risk_attack_counts_as_ood", default="", type=str)
    ap.add_argument("--risk_attack_sampling_mode", default="", type=str)
    ap.add_argument("--risk_attack_hash_salt", default="", type=str)
    ap.add_argument("--action_mode_profile", default="", type=str)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--use_xvfb", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if not args.execute:
        print("[dry-run] run_model_adapter_eval requires --execute for actual rollout.")
        return 0

    if args.use_xvfb:
        _start_xvfb_if_needed()

    scenario_name = str(args.scenario or "").strip().lower()
    risk_args = _risk_defaults_from_scenario(scenario_name=scenario_name)
    if str(args.risk_suite_enabled).strip() != "":
        risk_args["risk_suite_enabled"] = _parse_bool(args.risk_suite_enabled, default=False)
    if str(args.risk_attack_channel).strip() != "":
        risk_args["risk_attack_channel"] = str(args.risk_attack_channel).strip().lower()
    if args.risk_attack_probability is not None:
        risk_args["risk_attack_probability"] = float(args.risk_attack_probability)
    if args.risk_attack_severity is not None:
        risk_args["risk_attack_severity"] = float(args.risk_attack_severity)
    if args.risk_visual_alpha is not None:
        risk_args["risk_visual_alpha"] = float(args.risk_visual_alpha)
    if args.risk_visual_patch_ratio is not None:
        risk_args["risk_visual_patch_ratio"] = float(args.risk_visual_patch_ratio)
    if str(args.risk_threat_model).strip() != "":
        risk_args["risk_threat_model"] = str(args.risk_threat_model).strip()
    if str(args.risk_instruction_prefix).strip() != "":
        risk_args["risk_instruction_prefix"] = str(args.risk_instruction_prefix)
    if str(args.risk_instruction_suffix).strip() != "":
        risk_args["risk_instruction_suffix"] = str(args.risk_instruction_suffix)
    if str(args.risk_attack_counts_as_ood).strip() != "":
        risk_args["risk_attack_counts_as_ood"] = _parse_bool(
            args.risk_attack_counts_as_ood,
            default=False,
        )
    if str(args.risk_attack_sampling_mode).strip() != "":
        risk_args["risk_attack_sampling_mode"] = str(args.risk_attack_sampling_mode).strip().lower()
    if str(args.risk_attack_hash_salt).strip() != "":
        risk_args["risk_attack_hash_salt"] = str(args.risk_attack_hash_salt).strip()

    seed0_dir_raw = str(args.seed0_dir).strip()
    if seed0_dir_raw == "":
        print(
            "[blocked] seed0_dir is not set. Pass --seed0_dir or set ROEP_SEED0_DIR.",
            file=sys.stderr,
        )
        return BLOCKED_EXIT_CODE
    seed0_dir = Path(seed0_dir_raw).resolve()
    seed0_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(args.model)
    benchmark_backend = _resolve_benchmark_backend(model_name, args.benchmark_backend)
    action_mode_profile = _resolve_action_mode_profile(
        model_name,
        args.action_mode_profile,
        benchmark_backend,
    )
    cuda_error = _enforce_requested_cuda_if_needed(str(args.device))
    if cuda_error is not None:
        print(f"[blocked] {cuda_error}", file=sys.stderr)
        return BLOCKED_EXIT_CODE
    _prewarm_libero_runtime_if_needed(benchmark_backend)
    adapter_cls = ADAPTERS[model_name]
    ckpt = str(args.checkpoint).strip() or _default_ckpt(model_name)
    adapter = adapter_cls(AdapterConfig(checkpoint_path=ckpt, device=str(args.device)))

    print(
        f"[adapter] model={model_name} backend={benchmark_backend} stage={args.stage} task={args.task} "
        f"namespace={args.namespace} episodes={args.episodes} seeds={args.seeds} "
        f"action_mode_profile={action_mode_profile}"
    )
    if bool(risk_args.get("risk_suite_enabled", False)):
        print(
            "[risk] enabled "
            f"channel={risk_args.get('risk_attack_channel')} "
            f"p={risk_args.get('risk_attack_probability')} "
            f"severity={risk_args.get('risk_attack_severity')} "
            f"threat_model={risk_args.get('risk_threat_model')}"
        )
    try:
        adapter.load()
    except AdapterNotReadyError as exc:
        print(f"[blocked] adapter not ready: {exc}", file=sys.stderr)
        return BLOCKED_EXIT_CODE
    except NotImplementedError as exc:
        print(f"[blocked] adapter skeleton only: {exc}", file=sys.stderr)
        return BLOCKED_EXIT_CODE
    except Exception as exc:  # noqa: BLE001
        print(f"[error] adapter load failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    profiles = _parse_csv_list(args.profiles)
    if not profiles:
        profiles = ["medium"]
    seeds = _parse_int_csv(args.seeds)
    specs = _build_run_specs(stage=args.stage, profiles=profiles)
    if bool(args.skip_clean_baseline) and args.stage in {"cache", "noise"}:
        specs = [
            spec
            for spec in specs
            if str(spec.get("condition", "")) not in {"baseline", "baseline_clone"}
        ]

    shield_ckpt = str(args.safety_shield_checkpoint).strip()
    shield_required = any(bool(spec["shield"]) for spec in specs)
    if shield_required:
        if args.tau is None:
            print(
                f"[blocked] {args.stage} stage requires --tau for fixed-tau safety shield runs.",
                file=sys.stderr,
            )
            return BLOCKED_EXIT_CODE
        if shield_ckpt == "":
            print(
                f"[blocked] {args.stage} stage requires --safety_shield_checkpoint "
                "or ROEP_EXTERNAL_SHIELD_CKPT.",
                file=sys.stderr,
            )
            return BLOCKED_EXIT_CODE
        if not Path(shield_ckpt).exists():
            print(
                f"[blocked] safety shield checkpoint does not exist: {shield_ckpt}",
                file=sys.stderr,
            )
            return BLOCKED_EXIT_CODE

    for seed in seeds:
        for spec in specs:
            try:
                _run_one(
                    adapter=adapter,
                    benchmark_backend=benchmark_backend,
                    task=str(args.task),
                    episodes=int(args.episodes),
                    seed=int(seed),
                    namespace=str(args.namespace),
                    seed0_dir=seed0_dir,
                    run_prefix=str(spec["run_prefix"]),
                    occlusion=float(spec["occlusion"]),
                    depth=float(spec["depth"]),
                    use_shield=bool(spec["shield"]),
                    shield_ckpt=shield_ckpt,
                    shield_device=str(args.safety_shield_device),
                    tau=(float(args.tau) if args.tau is not None else None),
                    shield_mdsp_mode=str(args.safety_shield_mdsp_mode),
                    shield_refusal_risk_threshold=float(args.safety_shield_refusal_risk_threshold),
                    shield_verify_projected_action=bool(args.safety_shield_verify_projected_action),
                    shield_z_lift=float(args.safety_shield_z_lift),
                    shield_xy_scale=float(args.safety_shield_xy_scale),
                    shield_max_translation_norm=float(args.safety_shield_max_translation_norm),
                    ood_profile_id=str(spec["ood_profile_id"]),
                    max_steps=int(args.max_steps),
                    headless=bool(args.headless),
                    risk_args=risk_args,
                    action_mode_profile=action_mode_profile,
                    libero_suite=str(args.libero_suite),
                    libero_task_name=str(args.libero_task_name),
                    libero_benchmark_name=str(args.libero_benchmark_name),
                    libero_task_id=str(args.libero_task_id),
                    libero_init_states_path=str(args.libero_init_states_path),
                )
            except BackendNotReadyError as exc:
                print(f"[blocked] benchmark backend not ready: {exc}", file=sys.stderr)
                return BLOCKED_EXIT_CODE
            except RuntimeError as exc:
                msg = str(exc)
                if "Feature dim mismatch" in msg and "safety shield failed" in msg:
                    print(
                        "[blocked] safety shield checkpoint is incompatible with current model "
                        f"features: {msg}. Train/use a model-specific PRM checkpoint.",
                        file=sys.stderr,
                    )
                    return BLOCKED_EXIT_CODE
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
