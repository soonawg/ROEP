#!/usr/bin/env python3
"""Run or print ROEPBench+PerAct experiment commands.

This script standardizes command generation so runs are reproducible and
out_dir/run_id are always separated by condition and seed.
"""

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _profile_to_kwargs(name: str) -> Dict[str, str]:
    table = {
        "clean": {
            "occlusion": "0.0",
            "depth": "0.0",
            "blur": "0.0",
            "blackout": "0",
        },
        "mild": {
            "occlusion": "0.1",
            "depth": "0.1",
            "blur": "0.4",
            "blackout": "1",
        },
        "medium": {
            "occlusion": "0.3",
            "depth": "0.2",
            "blur": "0.8",
            "blackout": "2",
        },
        "strong": {
            "occlusion": "0.5",
            "depth": "0.3",
            "blur": "1.2",
            "blackout": "3",
        },
        "mixed_ood": {
            "occlusion": "0.3",
            "depth": "0.2",
            "blur": "0.8",
            "blackout": "2",
        },
    }
    if name not in table:
        raise ValueError(f"Unknown profile: {name}")
    return table[name]


def _recovery_profile_kwargs(name: str) -> Dict[str, str]:
    table = {
        "conservative_v1": {
            "reward_window": "8",
            "warmup_steps": "5",
            "recovery_burst_steps": "8",
            "recovery_action_scale": "0.25",
            "recovery_clip": "0.5",
            "max_recovery_events_per_episode": "2",
            "min_steps_between_recoveries": "5",
            "disable_recovery_after_first_success": "true",
        },
        "aggressive_v1": {
            "reward_window": "5",
            "warmup_steps": "3",
            "recovery_burst_steps": "8",
            "recovery_action_scale": "0.4",
            "recovery_clip": "0.8",
            "max_recovery_events_per_episode": "4",
            "min_steps_between_recoveries": "2",
            "disable_recovery_after_first_success": "false",
        },
        "short_horizon_v1": {
            "reward_window": "2",
            "warmup_steps": "1",
            "recovery_burst_steps": "1",
            "recovery_action_scale": "0.03",
            "recovery_clip": "0.06",
            "max_recovery_events_per_episode": "1",
            "min_steps_between_recoveries": "2",
            "disable_recovery_after_first_success": "true",
        },
    }
    if name not in table:
        raise ValueError(f"Unknown recovery profile set: {name}")
    return table[name]


def _safety_shield_profile_kwargs(name: str) -> Dict[str, str]:
    table = {
        "paper_tunedfix_v1": {
            "safety_shield_tau": "0.88",
            "safety_shield_mdsp_mode": "lift_z_scale_xy",
            "safety_shield_refusal_risk_threshold": "0.995",
            "safety_shield_verify_projected_action": "false",
            "safety_shield_z_lift": "0.05",
            "safety_shield_xy_scale": "0.8",
            "safety_shield_max_translation_norm": "1.05",
        }
    }
    if name not in table:
        raise ValueError(f"Unknown safety shield profile set: {name}")
    return table[name]


def _quote(arg: str) -> str:
    return shlex.quote(arg)


def _build_base_cmd(
    peract_dir: Path,
    gpu: int,
    task: str,
    logdir: Path,
    weight: int,
    episodes: int,
    from_eps: int,
    demo_path: Path,
    train_seed: int,
) -> List[str]:
    return [
        f"cd {_quote(str(peract_dir))}",
        "&&",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "python",
        "eval.py",
        f"rlbench.tasks=[{task}]",
        f"framework.logdir={_quote(str(logdir))}",
        f"framework.eval_type={weight}",
        f"framework.eval_episodes={episodes}",
        f"framework.eval_from_eps_number={from_eps}",
        f"rlbench.demo_path={_quote(str(demo_path))}",
        f"framework.start_seed={train_seed}",
    ]


def _build_condition_args(
    condition: str,
    profile: str,
    seed: int,
    out_dir: Path,
    run_id: str,
    recovery_profile_set: str,
    detector: str,
    reward_window: Optional[int],
    reward_eps: Optional[float],
    confirm_windows: Optional[int],
    trigger_cooldown_steps: Optional[int],
    margin_eps: Optional[float],
    min_plateau_steps: Optional[int],
    warmup_steps: Optional[int],
    recovery_burst_steps: Optional[int],
    recovery_burst_steps_clean: Optional[int],
    recovery_burst_steps_ood: Optional[int],
    recovery_action_scale: Optional[float],
    recovery_action_scale_clean: Optional[float],
    recovery_action_scale_ood: Optional[float],
    recovery_clip: Optional[float],
    recovery_clip_clean: Optional[float],
    recovery_clip_ood: Optional[float],
    max_recovery_events_per_episode: Optional[int],
    max_recovery_events_clean: Optional[int],
    max_recovery_events_ood: Optional[int],
    min_steps_between_recoveries: Optional[int],
    disable_recovery_after_first_success: Optional[str],
    recovery_only_when_ood: Optional[str],
    debug_integrity: Optional[str],
    recovery_blend_alpha: Optional[float],
    shadow_recovery: Optional[str],
    recovery_drop_window: Optional[int],
    recovery_drop_tol: Optional[float],
    blackout_on_recovery_trigger: Optional[str],
    recovery_guard_enabled: Optional[str],
    recovery_guard_delta_window: Optional[int],
    recovery_guard_abort_on_risk_spike: Optional[str],
    recovery_watch_steps: Optional[int],
    recovery_verify_steps: Optional[int],
    primitive_sequence: str,
    risk_suite_enabled: Optional[str],
    risk_attack_channel: Optional[str],
    risk_attack_probability: Optional[float],
    risk_attack_severity: Optional[float],
    risk_visual_alpha: Optional[float],
    risk_visual_patch_ratio: Optional[float],
    risk_threat_model: Optional[str],
    risk_instruction_prefix: Optional[str],
    risk_instruction_suffix: Optional[str],
    risk_apply_to_baseline: Optional[str],
    risk_attack_counts_as_ood: Optional[str],
    enable_risk_trigger: Optional[str],
    risk_window: Optional[int],
    risk_threshold: Optional[float],
    risk_min_unsafe_events: Optional[int],
    risk_attack_sampling_mode: Optional[str],
    risk_attack_hash_salt: Optional[str],
    safety_shield_profile_set: Optional[str],
    safety_shield_checkpoint: Optional[str],
    safety_shield_device: Optional[str],
    safety_shield_tau: Optional[float],
    safety_shield_mdsp_mode: Optional[str],
    safety_shield_refusal_risk_threshold: Optional[float],
    safety_shield_verify_projected_action: Optional[str],
    safety_shield_z_lift: Optional[float],
    safety_shield_xy_scale: Optional[float],
    safety_shield_max_translation_norm: Optional[float],
    apply_safety_shield: bool,
) -> List[str]:
    p = _profile_to_kwargs(profile)
    rp = _recovery_profile_kwargs(recovery_profile_set)
    rw = str(reward_window) if reward_window is not None else rp["reward_window"]
    re = str(reward_eps) if reward_eps is not None else "0.0"
    cw = str(confirm_windows) if confirm_windows is not None else "3"
    tcs = str(trigger_cooldown_steps) if trigger_cooldown_steps is not None else "30"
    meps = str(margin_eps) if margin_eps is not None else "0.0"
    mps = str(min_plateau_steps) if min_plateau_steps is not None else "24"
    wu = str(warmup_steps) if warmup_steps is not None else rp["warmup_steps"]
    rb = str(recovery_burst_steps) if recovery_burst_steps is not None else rp["recovery_burst_steps"]
    rb_clean = (
        str(recovery_burst_steps_clean)
        if recovery_burst_steps_clean is not None
        else rb
    )
    rb_ood = (
        str(recovery_burst_steps_ood)
        if recovery_burst_steps_ood is not None
        else rb
    )
    ras = str(recovery_action_scale) if recovery_action_scale is not None else rp["recovery_action_scale"]
    ras_clean = (
        str(recovery_action_scale_clean)
        if recovery_action_scale_clean is not None
        else ras
    )
    ras_ood = (
        str(recovery_action_scale_ood)
        if recovery_action_scale_ood is not None
        else ras
    )
    rclip = str(recovery_clip) if recovery_clip is not None else rp["recovery_clip"]
    rclip_clean = (
        str(recovery_clip_clean)
        if recovery_clip_clean is not None
        else rclip
    )
    rclip_ood = (
        str(recovery_clip_ood)
        if recovery_clip_ood is not None
        else rclip
    )
    mre = str(max_recovery_events_per_episode) if max_recovery_events_per_episode is not None else rp["max_recovery_events_per_episode"]
    mre_clean = (
        str(max_recovery_events_clean)
        if max_recovery_events_clean is not None
        else mre
    )
    mre_ood = (
        str(max_recovery_events_ood)
        if max_recovery_events_ood is not None
        else mre
    )
    msr = str(min_steps_between_recoveries) if min_steps_between_recoveries is not None else rp["min_steps_between_recoveries"]
    drs = str(disable_recovery_after_first_success).lower() if disable_recovery_after_first_success is not None else rp["disable_recovery_after_first_success"]
    roow = str(recovery_only_when_ood).lower() if recovery_only_when_ood is not None else None
    dbg = str(debug_integrity).lower() if debug_integrity is not None else None
    rba = str(recovery_blend_alpha) if recovery_blend_alpha is not None else None
    shr = str(shadow_recovery).lower() if shadow_recovery is not None else None
    rdw = str(recovery_drop_window) if recovery_drop_window is not None else None
    rdt = str(recovery_drop_tol) if recovery_drop_tol is not None else None
    bort = (
        str(blackout_on_recovery_trigger).lower()
        if blackout_on_recovery_trigger is not None
        else None
    )
    rge = str(recovery_guard_enabled).lower() if recovery_guard_enabled is not None else None
    rgdw = str(recovery_guard_delta_window) if recovery_guard_delta_window is not None else None
    rga = (
        str(recovery_guard_abort_on_risk_spike).lower()
        if recovery_guard_abort_on_risk_spike is not None
        else None
    )
    rws = str(recovery_watch_steps) if recovery_watch_steps is not None else None
    rvs = str(recovery_verify_steps) if recovery_verify_steps is not None else None
    rse = str(risk_suite_enabled).lower() if risk_suite_enabled is not None else None
    rach = str(risk_attack_channel) if risk_attack_channel is not None else None
    rap = str(risk_attack_probability) if risk_attack_probability is not None else None
    rasv = str(risk_attack_severity) if risk_attack_severity is not None else None
    rva = str(risk_visual_alpha) if risk_visual_alpha is not None else None
    rvr = str(risk_visual_patch_ratio) if risk_visual_patch_ratio is not None else None
    rtm = str(risk_threat_model) if risk_threat_model is not None else None
    rip = str(risk_instruction_prefix) if risk_instruction_prefix is not None else None
    ris = str(risk_instruction_suffix) if risk_instruction_suffix is not None else None
    rasm = str(risk_attack_sampling_mode) if risk_attack_sampling_mode is not None else None
    rhs = str(risk_attack_hash_salt) if risk_attack_hash_salt is not None else None
    rab = _is_truthy(risk_apply_to_baseline)
    racao = (
        str(risk_attack_counts_as_ood).lower()
        if risk_attack_counts_as_ood is not None
        else None
    )
    ert = str(enable_risk_trigger).lower() if enable_risk_trigger is not None else None
    rwnd = str(risk_window) if risk_window is not None else None
    rthr = str(risk_threshold) if risk_threshold is not None else None
    rmin = str(risk_min_unsafe_events) if risk_min_unsafe_events is not None else None
    shield_profile = (
        _safety_shield_profile_kwargs(str(safety_shield_profile_set).strip())
        if str(safety_shield_profile_set or "").strip()
        else {}
    )
    ssckpt = str(safety_shield_checkpoint).strip() if safety_shield_checkpoint is not None else ""
    ssdev = str(safety_shield_device).strip() if safety_shield_device is not None else None
    sstau = (
        str(safety_shield_tau).strip()
        if safety_shield_tau is not None
        else shield_profile.get("safety_shield_tau")
    )
    ssmdsp = (
        str(safety_shield_mdsp_mode).strip()
        if safety_shield_mdsp_mode is not None
        else shield_profile.get("safety_shield_mdsp_mode")
    )
    ssref = (
        str(safety_shield_refusal_risk_threshold).strip()
        if safety_shield_refusal_risk_threshold is not None
        else shield_profile.get("safety_shield_refusal_risk_threshold")
    )
    ssverify = (
        str(safety_shield_verify_projected_action).lower().strip()
        if safety_shield_verify_projected_action is not None
        else shield_profile.get("safety_shield_verify_projected_action")
    )
    ssz = (
        str(safety_shield_z_lift).strip()
        if safety_shield_z_lift is not None
        else shield_profile.get("safety_shield_z_lift")
    )
    ssxy = (
        str(safety_shield_xy_scale).strip()
        if safety_shield_xy_scale is not None
        else shield_profile.get("safety_shield_xy_scale")
    )
    ssmax = (
        str(safety_shield_max_translation_norm).strip()
        if safety_shield_max_translation_norm is not None
        else shield_profile.get("safety_shield_max_translation_norm")
    )

    def _append_risk_args(dst: List[str]) -> None:
        if rse is not None:
            dst.append(f"+roep.risk_suite_enabled={rse}")
        if rach is not None:
            dst.append(f"+roep.risk_attack_channel={_quote(rach)}")
        if rap is not None:
            dst.append(f"+roep.risk_attack_probability={rap}")
        if rasv is not None:
            dst.append(f"+roep.risk_attack_severity={rasv}")
        if rva is not None:
            dst.append(f"+roep.risk_visual_alpha={rva}")
        if rvr is not None:
            dst.append(f"+roep.risk_visual_patch_ratio={rvr}")
        if rtm is not None:
            dst.append(f"+roep.risk_threat_model={_quote(rtm)}")
        if rip is not None:
            dst.append(f"+roep.risk_instruction_prefix={_quote(rip)}")
        if ris is not None:
            dst.append(f"+roep.risk_instruction_suffix={_quote(ris)}")
        if racao is not None:
            dst.append(f"+roep.risk_attack_counts_as_ood={racao}")
        if ert is not None:
            dst.append(f"+roep.enable_risk_trigger={ert}")
        if rwnd is not None:
            dst.append(f"+roep.risk_window={rwnd}")
        if rthr is not None:
            dst.append(f"+roep.risk_threshold={rthr}")
        if rmin is not None:
            dst.append(f"+roep.risk_min_unsafe_events={rmin}")
        if rasm is not None:
            dst.append(f"+roep.risk_attack_sampling_mode={_quote(rasm)}")
        if rhs is not None:
            dst.append(f"+roep.risk_attack_hash_salt={_quote(rhs)}")

    def _append_safety_shield_args(dst: List[str]) -> None:
        if not ssckpt:
            return
        dst.append(f"+roep.safety_shield_checkpoint={_quote(ssckpt)}")
        if ssdev:
            dst.append(f"+roep.safety_shield_device={_quote(ssdev)}")
        if sstau is not None:
            dst.append(f"+roep.safety_shield_tau={sstau}")
        if ssmdsp:
            dst.append(f"+roep.safety_shield_mdsp_mode={_quote(ssmdsp)}")
        if ssref is not None:
            dst.append(f"+roep.safety_shield_refusal_risk_threshold={ssref}")
        if ssverify is not None:
            dst.append(f"+roep.safety_shield_verify_projected_action={ssverify}")
        if ssz is not None:
            dst.append(f"+roep.safety_shield_z_lift={ssz}")
        if ssxy is not None:
            dst.append(f"+roep.safety_shield_xy_scale={ssxy}")
        if ssmax is not None:
            dst.append(f"+roep.safety_shield_max_translation_norm={ssmax}")

    if condition in ("baseline", "baseline_clone"):
        base_args = [
            "+roep.enabled=true",
            f"+roep.seed={seed}",
            f"+roep.run_id={_quote(run_id)}",
            f"+roep.out_dir={_quote(str(out_dir))}",
            "+roep.occlusion_severity=0.0",
            "+roep.depth_dropout_prob=0.0",
            "+roep.gaussian_blur_sigma=0.0",
            "+roep.obs_blackout_steps=0",
            "+roep.enable_recovery=false",
        ]
        if dbg is not None:
            base_args.append(f"+roep.debug_integrity={dbg}")
        if rab:
            _append_risk_args(base_args)
        if apply_safety_shield:
            _append_safety_shield_args(base_args)
        return base_args

    args = [
        "+roep.enabled=true",
        f"+roep.seed={seed}",
        f"+roep.run_id={_quote(run_id)}",
        f"+roep.out_dir={_quote(str(out_dir))}",
        f"+roep.occlusion_severity={p['occlusion']}",
        f"+roep.depth_dropout_prob={p['depth']}",
        f"+roep.gaussian_blur_sigma={p['blur']}",
        f"+roep.obs_blackout_steps={p['blackout']}",
    ]

    if condition in ("perturb", "perturb_clone"):
        args += ["+roep.enable_recovery=false"]
    elif condition in ("recovery", "full"):
        if condition == "recovery":
            args = [
                "+roep.enabled=true",
                f"+roep.seed={seed}",
                f"+roep.run_id={_quote(run_id)}",
                f"+roep.out_dir={_quote(str(out_dir))}",
                "+roep.occlusion_severity=0.0",
                "+roep.depth_dropout_prob=0.0",
                "+roep.gaussian_blur_sigma=0.0",
                "+roep.obs_blackout_steps=0",
            ]
        recovery_enabled_value = "false" if apply_safety_shield else "true"
        args += [
            f"+roep.enable_recovery={recovery_enabled_value}",
            f"+roep.detector={detector}",
            f"+roep.reward_window={rw}",
            f"+roep.reward_eps={re}",
            f"+roep.confirm_windows={cw}",
            f"+roep.trigger_cooldown_steps={tcs}",
            f"+roep.margin_eps={meps}",
            f"+roep.min_plateau_steps={mps}",
            f"+roep.recovery_burst_steps={rb}",
            f"+roep.recovery_burst_steps_clean={rb_clean}",
            f"+roep.recovery_burst_steps_ood={rb_ood}",
            f"+roep.warmup_steps={wu}",
            f"+roep.recovery_action_scale={ras}",
            f"+roep.recovery_action_scale_clean={ras_clean}",
            f"+roep.recovery_action_scale_ood={ras_ood}",
            f"+roep.recovery_clip={rclip}",
            f"+roep.recovery_clip_clean={rclip_clean}",
            f"+roep.recovery_clip_ood={rclip_ood}",
            f"+roep.max_recovery_events_per_episode={mre}",
            f"+roep.max_recovery_events_clean={mre_clean}",
            f"+roep.max_recovery_events_ood={mre_ood}",
            f"+roep.min_steps_between_recoveries={msr}",
            f"+roep.disable_recovery_after_first_success={drs}",
            f"\"+roep.primitive_sequence='{primitive_sequence}'\"",
        ]
        if roow is not None:
            args.append(f"+roep.recovery_only_when_ood={roow}")
        if rba is not None:
            args.append(f"+roep.recovery_blend_alpha={rba}")
        if shr is not None:
            args.append(f"+roep.shadow_recovery={shr}")
        if rdw is not None:
            args.append(f"+roep.recovery_drop_window={rdw}")
        if rdt is not None:
            args.append(f"+roep.recovery_drop_tol={rdt}")
        if bort is not None:
            args.append(f"+roep.blackout_on_recovery_trigger={bort}")
        if rge is not None:
            args.append(f"+roep.recovery_guard_enabled={rge}")
        if rgdw is not None:
            args.append(f"+roep.recovery_guard_delta_window={rgdw}")
        if rga is not None:
            args.append(f"+roep.recovery_guard_abort_on_risk_spike={rga}")
        if rws is not None:
            args.append(f"+roep.recovery_watch_steps={rws}")
        if rvs is not None:
            args.append(f"+roep.recovery_verify_steps={rvs}")
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if dbg is not None:
        args.append(f"+roep.debug_integrity={dbg}")
    _append_risk_args(args)
    if apply_safety_shield:
        _append_safety_shield_args(args)

    return args


def _build_runs(
    logdir: Path,
    task: str,
    seeds: List[int],
    profiles: List[str],
    namespace: str,
    conditions: List[str],
) -> List[Tuple[str, str]]:
    runs: List[Tuple[str, str]] = []
    root = logdir / "multi" / "PERACT_BC" / "seed0"
    ns = str(namespace).strip()
    suffix = f"__{ns}" if ns else ""
    want = set(conditions)
    for seed in seeds:
        if "baseline" in want:
            rid = f"baseline_s{seed}"
            out = root / f"roepbench_{rid}{suffix}"
            runs.append((rid, "baseline|clean|%d|%s" % (seed, out)))
        if "baseline_clone" in want:
            rid = f"baseline_clone_s{seed}"
            out = root / f"roepbench_{rid}{suffix}"
            runs.append((rid, "baseline_clone|clean|%d|%s" % (seed, out)))
        if "recovery" in want:
            rid = f"recovery_no_ood_s{seed}"
            out = root / f"roepbench_{rid}{suffix}"
            runs.append((rid, "recovery|clean|%d|%s" % (seed, out)))
        # Perturb/full profiles
        for profile in profiles:
            if "perturb" in want:
                rid = f"perturb_{profile}_s{seed}"
                out = root / f"roepbench_{rid}{suffix}"
                runs.append((rid, "perturb|%s|%d|%s" % (profile, seed, out)))
            if "perturb_clone" in want:
                rid = f"perturb_clone_{profile}_s{seed}"
                out = root / f"roepbench_{rid}{suffix}"
                runs.append((rid, "perturb_clone|%s|%d|%s" % (profile, seed, out)))

            if "full" in want:
                rid = f"full_{profile}_s{seed}"
                out = root / f"roepbench_{rid}{suffix}"
                runs.append((rid, "full|%s|%d|%s" % (profile, seed, out)))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description="ROEP experiment launcher for PerAct eval")
    ap.add_argument("--peract_dir", type=Path, default=Path("external/peract"))
    ap.add_argument("--logdir", type=Path, default=Path("external/peract/ckpts"))
    ap.add_argument("--demo_path", type=Path, required=True)
    ap.add_argument("--task", type=str, default="open_drawer")
    ap.add_argument("--weight", type=int, default=600000)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--from_eps", type=int, default=0)
    ap.add_argument("--train_seed", type=int, default=0, help="PerAct checkpoint seed folder (usually 0)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--profiles", type=str, default="mild,medium,strong")
    ap.add_argument(
        "--conditions",
        type=str,
        default="baseline,recovery,perturb,full",
        help="Subset of conditions. Comma-separated from: baseline,baseline_clone,recovery,perturb,perturb_clone,full",
    )
    ap.add_argument(
        "--recovery_profile_set",
        type=str,
        default="conservative_v1",
        help="Preset recovery profile: conservative_v1|aggressive_v1|short_horizon_v1",
    )
    ap.add_argument(
        "--detector_variant",
        type=str,
        default="reward_plateau",
        choices=[
            "reward_plateau",
            "plateau_with_confirm",
            "plateau_strict",
            "plateau_with_margin",
            "plateau_with_confirm_v2",
            "plateau_strict_v2",
            "plateau_margin_v2",
            "low_dim",
        ],
    )
    ap.add_argument("--reward_window", type=int, default=None)
    ap.add_argument("--reward_eps", type=float, default=None)
    ap.add_argument("--confirm_windows", type=int, default=None)
    ap.add_argument("--trigger_cooldown_steps", type=int, default=None)
    ap.add_argument("--margin_eps", type=float, default=None)
    ap.add_argument("--min_plateau_steps", type=int, default=None)
    ap.add_argument("--warmup_steps", type=int, default=None)
    ap.add_argument("--recovery_burst_steps", type=int, default=None)
    ap.add_argument("--recovery_burst_steps_clean", type=int, default=None)
    ap.add_argument("--recovery_burst_steps_ood", type=int, default=None)
    ap.add_argument("--recovery_action_scale", type=float, default=None)
    ap.add_argument("--recovery_action_scale_clean", type=float, default=None)
    ap.add_argument("--recovery_action_scale_ood", type=float, default=None)
    ap.add_argument("--recovery_clip", type=float, default=None)
    ap.add_argument("--recovery_clip_clean", type=float, default=None)
    ap.add_argument("--recovery_clip_ood", type=float, default=None)
    ap.add_argument("--max_recovery_events_per_episode", type=int, default=None)
    ap.add_argument("--max_recovery_events_clean", type=int, default=None)
    ap.add_argument("--max_recovery_events_ood", type=int, default=None)
    ap.add_argument("--min_steps_between_recoveries", type=int, default=None)
    ap.add_argument("--disable_recovery_after_first_success", type=str, default=None)
    ap.add_argument(
        "--recovery_only_when_ood",
        type=str,
        default=None,
        help="If true, disable recovery in clean (non-OOD) runs.",
    )
    ap.add_argument(
        "--debug_integrity",
        type=str,
        default=None,
        help="If true, enable integrity diagnostics fields in episode logs.",
    )
    ap.add_argument("--recovery_blend_alpha", type=float, default=None)
    ap.add_argument("--shadow_recovery", type=str, default=None)
    ap.add_argument("--recovery_drop_window", type=int, default=None)
    ap.add_argument("--recovery_drop_tol", type=float, default=None)
    ap.add_argument(
        "--blackout_on_recovery_trigger",
        type=str,
        default=None,
        help="If true, obs_blackout_steps is applied only when recovery triggers.",
    )
    ap.add_argument(
        "--recovery_guard_enabled",
        type=str,
        default=None,
        help="Enable recovery guard to auto-abort recovery on unsafe spike.",
    )
    ap.add_argument(
        "--recovery_guard_delta_window",
        type=int,
        default=None,
        help="Guard window in steps after trigger to watch for risk spike.",
    )
    ap.add_argument(
        "--recovery_guard_abort_on_risk_spike",
        type=str,
        default=None,
        help="If true, abort recovery burst on unsafe spike within guard window.",
    )
    ap.add_argument(
        "--recovery_watch_steps",
        type=int,
        default=None,
        help="WATCH state steps before RECOVER state transition.",
    )
    ap.add_argument(
        "--recovery_verify_steps",
        type=int,
        default=None,
        help="VERIFY state steps after RECOVER burst ends.",
    )
    ap.add_argument(
        "--primitive_sequence",
        type=str,
        default="random_burst,safe_retract_like,regrasp_like",
        help="Comma-separated primitive sequence passed to +roep.primitive_sequence",
    )
    ap.add_argument(
        "--risk_suite_enabled",
        type=str,
        default=None,
        help="Enable threat-model attack suite (true|false).",
    )
    ap.add_argument(
        "--risk_attack_channel",
        type=str,
        default=None,
        help="Attack channel: none|instruction|visual|conceptual|hybrid",
    )
    ap.add_argument("--risk_attack_probability", type=float, default=None)
    ap.add_argument("--risk_attack_severity", type=float, default=None)
    ap.add_argument("--risk_visual_alpha", type=float, default=None)
    ap.add_argument("--risk_visual_patch_ratio", type=float, default=None)
    ap.add_argument("--risk_threat_model", type=str, default=None)
    ap.add_argument("--risk_instruction_prefix", type=str, default=None)
    ap.add_argument("--risk_instruction_suffix", type=str, default=None)
    ap.add_argument(
        "--risk_attack_sampling_mode",
        type=str,
        default=None,
        help="Attack sampling mode: bernoulli|deterministic_hash",
    )
    ap.add_argument(
        "--risk_attack_hash_salt",
        type=str,
        default=None,
        help="Hash salt used when risk_attack_sampling_mode=deterministic_hash",
    )
    ap.add_argument(
        "--risk_attack_counts_as_ood",
        type=str,
        default=None,
        help="If true, mark attack-applied episodes as OOD for recovery gating.",
    )
    ap.add_argument(
        "--enable_risk_trigger",
        type=str,
        default=None,
        help="Enable risk-trigger pathway in addition to detector trigger (true|false).",
    )
    ap.add_argument("--risk_window", type=int, default=None)
    ap.add_argument("--risk_threshold", type=float, default=None)
    ap.add_argument("--risk_min_unsafe_events", type=int, default=None)
    ap.add_argument(
        "--risk_apply_to_baseline",
        type=str,
        default="false",
        help="If true, apply risk-suite args to baseline/baseline_clone too.",
    )
    ap.add_argument(
        "--safety_shield_checkpoint",
        type=str,
        default="",
        help="Path to safety shield checkpoint (.pth). Empty disables shield.",
    )
    ap.add_argument(
        "--safety_shield_profile_set",
        type=str,
        default="paper_tunedfix_v1",
        help="Named safety-shield preset. Default keeps paper-tunedfix values when shield is enabled.",
    )
    ap.add_argument(
        "--safety_shield_device",
        type=str,
        default="cpu",
        help="Safety shield device (cpu|cuda).",
    )
    ap.add_argument(
        "--safety_shield_tau",
        type=float,
        default=None,
        help="Optional tau override for safety shield calibration threshold.",
    )
    ap.add_argument(
        "--safety_shield_mdsp_mode",
        type=str,
        default="lift_z_scale_xy",
        help="Safety shield action projection mode.",
    )
    ap.add_argument(
        "--safety_shield_refusal_risk_threshold",
        type=float,
        default=0.995,
        help="Safe refusal risk threshold.",
    )
    ap.add_argument(
        "--safety_shield_verify_projected_action",
        type=str,
        default="false",
        help="Whether shield verifies projected action (true|false).",
    )
    ap.add_argument(
        "--safety_shield_z_lift",
        type=float,
        default=0.05,
        help="MDSP z lift value.",
    )
    ap.add_argument(
        "--safety_shield_xy_scale",
        type=float,
        default=0.5,
        help="MDSP xy scale value.",
    )
    ap.add_argument(
        "--safety_shield_max_translation_norm",
        type=float,
        default=0.25,
        help="MDSP translation norm clamp.",
    )
    ap.add_argument(
        "--safety_shield_apply_conditions",
        type=str,
        default="recovery,full",
        help="Comma-separated conditions to apply shield, e.g. recovery,full.",
    )
    ap.add_argument("--max_consecutive_failures", type=int, default=3)
    ap.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Output namespace suffix to avoid overwriting existing run dirs. Default: auto timestamp.",
    )
    ap.add_argument("--execute", action="store_true", help="Actually run commands. Default: print only")
    args = ap.parse_args()

    peract_dir = args.peract_dir.resolve()
    logdir = args.logdir.resolve()
    demo_path = args.demo_path.resolve()

    seeds = _parse_int_list(args.seeds)
    profiles = _parse_str_list(args.profiles)
    conditions = _parse_str_list(args.conditions)
    valid_conditions = {
        "baseline",
        "baseline_clone",
        "recovery",
        "perturb",
        "perturb_clone",
        "full",
    }
    bad = [c for c in conditions if c not in valid_conditions]
    if bad:
        raise ValueError(f"Unknown condition(s): {bad}. Valid: {sorted(valid_conditions)}")
    safety_shield_apply_conditions = _parse_str_list(args.safety_shield_apply_conditions)
    bad_shield_conditions = [
        c for c in safety_shield_apply_conditions if c not in valid_conditions
    ]
    if bad_shield_conditions:
        raise ValueError(
            f"Unknown safety_shield_apply_condition(s): {bad_shield_conditions}. "
            f"Valid: {sorted(valid_conditions)}"
        )
    namespace = args.namespace or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs = _build_runs(logdir, args.task, seeds, profiles, namespace=namespace, conditions=conditions)

    print(f"[INFO] namespace={namespace}")

    commands: List[str] = []
    for run_id, encoded in runs:
        condition, profile, seed_str, out_dir_str = encoded.split("|", 3)
        seed = int(seed_str)
        out_dir = Path(out_dir_str)

        base = _build_base_cmd(
            peract_dir=peract_dir,
            gpu=args.gpu,
            task=args.task,
            logdir=logdir,
            weight=args.weight,
            episodes=args.episodes,
            from_eps=args.from_eps,
            demo_path=demo_path,
            train_seed=args.train_seed,
        )
        cond = _build_condition_args(
            condition=condition,
            profile=profile,
            seed=seed,
            out_dir=out_dir,
            run_id=run_id,
            recovery_profile_set=args.recovery_profile_set,
            detector=args.detector_variant,
            reward_window=args.reward_window,
            reward_eps=args.reward_eps,
            confirm_windows=args.confirm_windows,
            trigger_cooldown_steps=args.trigger_cooldown_steps,
            margin_eps=args.margin_eps,
            min_plateau_steps=args.min_plateau_steps,
            warmup_steps=args.warmup_steps,
            recovery_burst_steps=args.recovery_burst_steps,
            recovery_burst_steps_clean=args.recovery_burst_steps_clean,
            recovery_burst_steps_ood=args.recovery_burst_steps_ood,
            recovery_action_scale=args.recovery_action_scale,
            recovery_action_scale_clean=args.recovery_action_scale_clean,
            recovery_action_scale_ood=args.recovery_action_scale_ood,
            recovery_clip=args.recovery_clip,
            recovery_clip_clean=args.recovery_clip_clean,
            recovery_clip_ood=args.recovery_clip_ood,
            max_recovery_events_per_episode=args.max_recovery_events_per_episode,
            max_recovery_events_clean=args.max_recovery_events_clean,
            max_recovery_events_ood=args.max_recovery_events_ood,
            min_steps_between_recoveries=args.min_steps_between_recoveries,
            disable_recovery_after_first_success=args.disable_recovery_after_first_success,
            recovery_only_when_ood=args.recovery_only_when_ood,
            debug_integrity=args.debug_integrity,
            recovery_blend_alpha=args.recovery_blend_alpha,
            shadow_recovery=args.shadow_recovery,
            recovery_drop_window=args.recovery_drop_window,
            recovery_drop_tol=args.recovery_drop_tol,
            blackout_on_recovery_trigger=args.blackout_on_recovery_trigger,
            recovery_guard_enabled=args.recovery_guard_enabled,
            recovery_guard_delta_window=args.recovery_guard_delta_window,
            recovery_guard_abort_on_risk_spike=args.recovery_guard_abort_on_risk_spike,
            recovery_watch_steps=args.recovery_watch_steps,
            recovery_verify_steps=args.recovery_verify_steps,
            primitive_sequence=args.primitive_sequence,
            risk_suite_enabled=args.risk_suite_enabled,
            risk_attack_channel=args.risk_attack_channel,
            risk_attack_probability=args.risk_attack_probability,
            risk_attack_severity=args.risk_attack_severity,
            risk_visual_alpha=args.risk_visual_alpha,
            risk_visual_patch_ratio=args.risk_visual_patch_ratio,
            risk_threat_model=args.risk_threat_model,
            risk_instruction_prefix=args.risk_instruction_prefix,
            risk_instruction_suffix=args.risk_instruction_suffix,
            risk_apply_to_baseline=args.risk_apply_to_baseline,
            risk_attack_counts_as_ood=args.risk_attack_counts_as_ood,
            enable_risk_trigger=args.enable_risk_trigger,
            risk_window=args.risk_window,
            risk_threshold=args.risk_threshold,
            risk_min_unsafe_events=args.risk_min_unsafe_events,
            risk_attack_sampling_mode=args.risk_attack_sampling_mode,
            risk_attack_hash_salt=args.risk_attack_hash_salt,
            safety_shield_profile_set=args.safety_shield_profile_set,
            safety_shield_checkpoint=args.safety_shield_checkpoint,
            safety_shield_device=args.safety_shield_device,
            safety_shield_tau=args.safety_shield_tau,
            safety_shield_mdsp_mode=args.safety_shield_mdsp_mode,
            safety_shield_refusal_risk_threshold=args.safety_shield_refusal_risk_threshold,
            safety_shield_verify_projected_action=args.safety_shield_verify_projected_action,
            safety_shield_z_lift=args.safety_shield_z_lift,
            safety_shield_xy_scale=args.safety_shield_xy_scale,
            safety_shield_max_translation_norm=args.safety_shield_max_translation_norm,
            apply_safety_shield=(condition in safety_shield_apply_conditions),
        )

        cmd = " ".join(base + cond)
        commands.append(cmd)

    consecutive_failures = 0
    for i, cmd in enumerate(commands, start=1):
        print(f"[{i:03d}/{len(commands):03d}] {cmd}")
        if args.execute:
            result = subprocess.run(cmd, shell=True, check=False)
            if result.returncode != 0:
                consecutive_failures += 1
                print(
                    f"[WARN] command failed (rc={result.returncode}); consecutive_failures={consecutive_failures}"
                )
                if consecutive_failures >= int(args.max_consecutive_failures):
                    raise RuntimeError(
                        f"Aborting: reached max_consecutive_failures={args.max_consecutive_failures}"
                    )
            else:
                consecutive_failures = 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
