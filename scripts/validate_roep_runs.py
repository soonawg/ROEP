#!/usr/bin/env python3
"""Validate ROEP run folders before aggregation."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

REQUIRED_JSONL_KEYS = [
    "run_id",
    "episode_idx",
    "task",
    "seed",
    "success",
    "num_steps",
    "total_reward",
    "policy",
    "ood_active",
    "intervention_count",
    "unsafe_action_rate",
]

REQUIRED_JSONL_INTEGRITY_KEYS = [
    "debug_integrity",
    "recovery_enabled_effective",
    "ood_gate_open_steps",
    "detector_candidate_count",
    "detector_trigger_count",
    "recovery_action_applied_steps",
    "recovery_disabled_reason",
]

REQUIRED_JSONL_BLACKOUT_KEYS = [
    "blackout_on_recovery_trigger",
    "trigger_induced_blackout_steps",
]

REQUIRED_JSONL_STATE_MACHINE_KEYS = [
    "recovery_state",
    "recovery_watch_trigger_count",
    "recovery_risk_trigger_count",
    "recovery_state_transition_count",
    "recovery_trigger_reason",
]

REQUIRED_JSONL_PORTABLE_UAR_KEYS = [
    "unsafe_action_count_invalid",
    "unsafe_action_count_ik",
    "unsafe_action_count_path",
    "unsafe_action_rate_proxy",
    "unsafe_action_rate_exception_all",
    "unsafe_action_rate_total",
    "uar_scope",
]

REQUIRED_JSONL_LABEL_V2_KEYS = [
    "action_outcome_label",
    "safe_refusal",
    "unsafe_attempt",
    "invalid_command",
    "label_protocol_version",
]

REQUIRED_JSONL_HARM_DECOMP_KEYS = [
    "success",
    "action_outcome_label",
]

REQUIRED_SUMMARY_HARM_DECOMP_KEYS = [
    "unsafe_attempt_rate",
    "safe_refusal_rate",
    "invalid_command_rate",
    "timeout_rate",
]

REQUIRED_JSONL_THREAT_KEYS = [
    "risk_suite_enabled",
    "risk_attack_counts_as_ood",
    "threat_model",
    "attack_channel",
    "attack_variant",
    "attack_sampling_mode",
    "attack_assignment_id",
    "attack_applied",
    "attack_probability",
    "attack_severity",
    "attack_original_lang_goal",
    "attack_attacked_lang_goal",
    "attack_surface_instruction",
    "attack_surface_visual",
    "attack_surface_conceptual",
    "attack_visual_overlay_pixels",
    "attack_visual_applied_steps",
]

REQUIRED_SUMMARY_KEYS = [
    "success_rate",
    "sr_ood",
    "rsr",
    "mean_ic",
    "mean_ttr_steps",
    "uar",
]

REQUIRED_SUMMARY_THREAT_KEYS = [
    "threat_model",
    "attack_sampling_mode",
    "attack_applied_rate",
    "attack_instruction_rate",
    "attack_visual_rate",
    "attack_conceptual_rate",
    "mean_attack_visual_overlay_pixels",
    "mean_attack_visual_applied_steps",
    "asr_unsafe_attempt",
]

REQUIRED_SUMMARY_INTEGRITY_KEYS = [
    "mean_ood_gate_open_steps",
    "mean_detector_candidate_count",
    "mean_detector_trigger_count",
    "mean_recovery_action_applied_steps",
]

REQUIRED_SUMMARY_BLACKOUT_KEYS = [
    "mean_trigger_induced_blackout_steps",
]

REQUIRED_SUMMARY_STATE_MACHINE_KEYS = [
    "mean_recovery_watch_trigger_count",
    "mean_recovery_risk_trigger_count",
    "mean_recovery_state_transition_count",
]

REQUIRED_SUMMARY_PORTABLE_UAR_KEYS = [
    "uar",
    "uar_proxy",
    "uar_exception_all",
    "uar_total",
    "uar_scope",
]

REQUIRED_SUMMARY_LABEL_V2_KEYS = [
    "safe_refusal_rate",
    "unsafe_attempt_rate",
    "invalid_command_rate",
    "timeout_rate",
    "label_protocol_version",
]

REQUIRED_MANIFEST_KEYS = [
    "run_id",
    "episode_idx",
    "task",
    "seed",
    "eval_demo_seed",
    "lang_goal",
    "ood_active",
    "ood_profile_id",
    "pairing_key",
]

REQUIRED_MANIFEST_THREAT_KEYS = [
    "threat_model",
    "attack_channel",
    "attack_applied",
    "attack_variant",
]


def _parse_name(name: str) -> Dict[str, str]:
    if name.startswith("roepbench_"):
        name = name[len("roepbench_") :]
    # Optional namespace suffix pattern: <base_name>__<namespace>
    if "__" in name:
        name = name.split("__", 1)[0]
    out = {"condition": "unknown", "profile": "unknown", "seed": "-1"}
    if name.startswith("baseline_s"):
        out["condition"] = "baseline"
        out["profile"] = "clean"
        out["seed"] = name.rsplit("_s", 1)[-1]
        return out
    if name.startswith("baseline_clone_s"):
        out["condition"] = "baseline_clone"
        out["profile"] = "clean"
        out["seed"] = name.rsplit("_s", 1)[-1]
        return out
    if name.startswith("perturb_clone_") and "_s" in name:
        out["condition"] = "perturb_clone"
        prefix, seed = name.rsplit("_s", 1)
        out["profile"] = prefix.replace("perturb_clone_", "", 1)
        out["seed"] = seed
        return out
    if name.startswith("recovery_") and "_s" in name:
        out["condition"] = "recovery"
        prefix, seed = name.rsplit("_s", 1)
        out["profile"] = prefix.replace("recovery_", "", 1)
        out["seed"] = seed
        return out
    if name.startswith("perturb_") and "_s" in name:
        out["condition"] = "perturb"
        prefix, seed = name.rsplit("_s", 1)
        out["profile"] = prefix.replace("perturb_", "", 1)
        out["seed"] = seed
        return out
    if name.startswith("full_") and "_s" in name:
        out["condition"] = "full"
        prefix, seed = name.rsplit("_s", 1)
        out["profile"] = prefix.replace("full_", "", 1)
        out["seed"] = seed
        return out
    return out


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[0] if rows else {}


def _read_csv_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y")


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _pair_key(row: Dict) -> str:
    pk = row.get("pairing_key")
    if pk is not None and str(pk) != "":
        return str(pk)
    return f"{row.get('task')}|{row.get('seed')}|{row.get('episode_idx')}"


def _validate_dir(
    d: Path,
    expected_episodes: int,
    require_extended_schema: bool,
    require_episode_manifest: bool,
) -> List[str]:
    errs = []
    s_path = d / "summary.csv"
    j_path = d / "episode_logs.jsonl"
    m_path = d / "episode_manifest.csv"

    if not s_path.exists():
        return [f"{d.name}: missing summary.csv"]
    if not j_path.exists():
        return [f"{d.name}: missing episode_logs.jsonl"]

    summary = _read_summary(s_path)
    rows = _read_jsonl(j_path)

    if len(rows) != expected_episodes:
        errs.append(f"{d.name}: expected {expected_episodes} episodes, got {len(rows)}")

    if require_extended_schema:
        for k in REQUIRED_SUMMARY_KEYS:
            if k not in summary:
                errs.append(f"{d.name}: summary missing key '{k}'")

    if require_extended_schema:
        for i, row in enumerate(rows):
            for k in REQUIRED_JSONL_KEYS:
                if k not in row:
                    errs.append(f"{d.name}: jsonl row#{i} missing key '{k}'")

    # pairing keys should be unique within run
    keys = set()
    for row in rows:
        key = _pair_key(row)
        if key in keys:
            errs.append(f"{d.name}: duplicate pair key {key}")
        keys.add(key)

    if require_episode_manifest:
        if not m_path.exists():
            errs.append(f"{d.name}: missing episode_manifest.csv")
        else:
            manifest_rows = _read_csv_rows(m_path)
            if len(manifest_rows) != len(rows):
                errs.append(
                    f"{d.name}: episode_manifest rows mismatch "
                    f"(manifest={len(manifest_rows)} vs jsonl={len(rows)})"
                )
            for k in REQUIRED_MANIFEST_KEYS:
                if manifest_rows and (k not in manifest_rows[0]):
                    errs.append(f"{d.name}: manifest missing key '{k}'")

            jsonl_keys = {_pair_key(r) for r in rows}
            manifest_keys = {_pair_key(r) for r in manifest_rows}
            missing_in_manifest = sorted(list(jsonl_keys - manifest_keys))
            missing_in_jsonl = sorted(list(manifest_keys - jsonl_keys))
            if missing_in_manifest:
                errs.append(
                    f"{d.name}: manifest missing {len(missing_in_manifest)} pairing keys"
                )
            if missing_in_jsonl:
                errs.append(
                    f"{d.name}: manifest has extra {len(missing_in_jsonl)} pairing keys"
                )

    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate ROEP run outputs")
    ap.add_argument("--seed0_dir", type=Path, default=Path("external/peract/ckpts/multi/PERACT_BC/seed0"))
    ap.add_argument("--expected_episodes", type=int, default=10)
    ap.add_argument("--glob", type=str, default="roepbench*_v2")
    ap.add_argument("--require_extended_schema", action="store_true")
    ap.add_argument("--require_pair_sets", action="store_true")
    ap.add_argument(
        "--pair_profiles",
        type=str,
        default="mild,medium,strong",
        help=(
            "Comma-separated perturb/full profiles required when --require_pair_sets is set. "
            "Recovery clean pair baseline/recovery is always required."
        ),
    )
    ap.add_argument("--require_integrity_signals", action="store_true")
    ap.add_argument("--require_blackout_fields", action="store_true")
    ap.add_argument("--require_state_machine_signals", action="store_true")
    ap.add_argument("--require_portable_uar", action="store_true")
    ap.add_argument("--require_label_protocol_v2", action="store_true")
    ap.add_argument("--require_harm_decomposition_fields", action="store_true")
    ap.add_argument("--require_threat_model_fields", action="store_true")
    ap.add_argument(
        "--portable_uar_allowed_scopes",
        type=str,
        default="portable_partial,portable_full",
        help="Comma-separated allowed uar_scope values when --require_portable_uar is set.",
    )
    ap.add_argument("--require_recovery_activity_for_full", action="store_true")
    ap.add_argument("--min_full_recovery_applied_steps", type=float, default=0.1)
    ap.add_argument("--require_episode_manifest", action="store_true")
    ap.add_argument(
        "--require_attack_parity",
        action="store_true",
        help=(
            "Require attack_applied_rate parity between paired runs "
            "(baseline/recovery, perturb/full) within each seed."
        ),
    )
    ap.add_argument(
        "--max_attack_parity_gap",
        type=float,
        default=0.05,
        help="Maximum allowed absolute gap in attack_applied_rate for parity checks.",
    )
    ap.add_argument("--warn_low_support_denom", type=int, default=5)
    args = ap.parse_args()

    dirs = sorted([p for p in args.seed0_dir.glob(args.glob) if p.is_dir()])
    if not dirs:
        print(f"No dirs matched '{args.glob}' in {args.seed0_dir}")
        return 1

    all_errs: List[str] = []
    attack_parity_by_seed: Dict[str, Dict[tuple, float]] = {}
    allowed_uar_scopes = {
        s.strip() for s in str(args.portable_uar_allowed_scopes).split(",") if s.strip()
    }
    for d in dirs:
        all_errs.extend(
            _validate_dir(
                d,
                args.expected_episodes,
                args.require_extended_schema,
                args.require_episode_manifest,
            )
        )
        parsed = _parse_name(d.name)
        rows = _read_jsonl(d / "episode_logs.jsonl")
        summary = _read_summary(d / "summary.csv")
        if args.require_attack_parity:
            atk_rate = _to_float(summary.get("attack_applied_rate", None), default=-1.0)
            if atk_rate < 0.0:
                atk_rate = (
                    sum(1 for r in rows if _to_bool(r.get("attack_applied", False)))
                    / float(max(1, len(rows)))
                )
            seed_key = str(parsed["seed"])
            attack_parity_by_seed.setdefault(seed_key, {})[
                (parsed["condition"], parsed["profile"])
            ] = float(atk_rate)
        if args.require_integrity_signals:
            for k in REQUIRED_SUMMARY_INTEGRITY_KEYS:
                if k not in summary:
                    all_errs.append(f"{d.name}: summary missing integrity key '{k}'")
            for i, row in enumerate(rows):
                for k in REQUIRED_JSONL_INTEGRITY_KEYS:
                    if k not in row:
                        all_errs.append(f"{d.name}: jsonl row#{i} missing integrity key '{k}'")
        if args.require_blackout_fields:
            for k in REQUIRED_SUMMARY_BLACKOUT_KEYS:
                if k not in summary:
                    all_errs.append(f"{d.name}: summary missing blackout key '{k}'")
            for i, row in enumerate(rows):
                for k in REQUIRED_JSONL_BLACKOUT_KEYS:
                    if k not in row:
                        all_errs.append(f"{d.name}: jsonl row#{i} missing blackout key '{k}'")
        if args.require_state_machine_signals:
            for k in REQUIRED_SUMMARY_STATE_MACHINE_KEYS:
                if k not in summary:
                    all_errs.append(f"{d.name}: summary missing state-machine key '{k}'")
            valid_states = {"NORMAL", "WATCH", "RECOVER", "VERIFY", "LOCKOUT"}
            for i, row in enumerate(rows):
                for k in REQUIRED_JSONL_STATE_MACHINE_KEYS:
                    if k not in row:
                        all_errs.append(
                            f"{d.name}: jsonl row#{i} missing state-machine key '{k}'"
                        )
                s = str(row.get("recovery_state", "")).upper().strip()
                if s and (s not in valid_states):
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} invalid recovery_state '{row.get('recovery_state')}'"
                    )
        if args.require_portable_uar:
            for k in REQUIRED_SUMMARY_PORTABLE_UAR_KEYS:
                if k not in summary:
                    all_errs.append(f"{d.name}: summary missing portable UAR key '{k}'")

            scope = str(summary.get("uar_scope", "")).strip()
            if scope == "":
                all_errs.append(f"{d.name}: summary missing uar_scope value")
            elif allowed_uar_scopes and scope not in allowed_uar_scopes:
                all_errs.append(
                    f"{d.name}: summary uar_scope '{scope}' not in allowed {sorted(allowed_uar_scopes)}"
                )

            try:
                s_proxy = float(summary.get("uar_proxy", 0.0))
                s_exc = float(summary.get("uar_exception_all", 0.0))
                s_total = float(summary.get("uar_total", 0.0))
                if s_total + 1e-12 < s_proxy:
                    all_errs.append(
                        f"{d.name}: summary uar_total({s_total:.6f}) < uar_proxy({s_proxy:.6f})"
                    )
                if s_total + 1e-12 < s_exc:
                    all_errs.append(
                        f"{d.name}: summary uar_total({s_total:.6f}) < uar_exception_all({s_exc:.6f})"
                    )
            except Exception:
                all_errs.append(f"{d.name}: summary portable UAR numeric parse failed")

            for i, row in enumerate(rows):
                for k in REQUIRED_JSONL_PORTABLE_UAR_KEYS:
                    if k not in row:
                        all_errs.append(f"{d.name}: jsonl row#{i} missing portable UAR key '{k}'")

                try:
                    r_proxy = float(row.get("unsafe_action_rate_proxy", 0.0))
                    r_exc = float(row.get("unsafe_action_rate_exception_all", 0.0))
                    r_total = float(row.get("unsafe_action_rate_total", 0.0))
                    if r_total + 1e-12 < r_proxy:
                        all_errs.append(
                            f"{d.name}: jsonl row#{i} unsafe_action_rate_total({r_total:.6f}) < "
                            f"unsafe_action_rate_proxy({r_proxy:.6f})"
                        )
                    if r_total + 1e-12 < r_exc:
                        all_errs.append(
                            f"{d.name}: jsonl row#{i} unsafe_action_rate_total({r_total:.6f}) < "
                            f"unsafe_action_rate_exception_all({r_exc:.6f})"
                        )
                except Exception:
                    all_errs.append(f"{d.name}: jsonl row#{i} portable UAR numeric parse failed")

                row_scope = str(row.get("uar_scope", "")).strip()
                if row_scope == "":
                    all_errs.append(f"{d.name}: jsonl row#{i} missing uar_scope value")
                elif allowed_uar_scopes and row_scope not in allowed_uar_scopes:
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} uar_scope '{row_scope}' "
                        f"not in allowed {sorted(allowed_uar_scopes)}"
                    )
        if args.require_label_protocol_v2:
            for k in REQUIRED_SUMMARY_LABEL_V2_KEYS:
                if k not in summary:
                    all_errs.append(f"{d.name}: summary missing label-v2 key '{k}'")

            summary_lpv = str(summary.get("label_protocol_version", "")).strip()
            if summary_lpv != "v2":
                all_errs.append(
                    f"{d.name}: summary label_protocol_version must be 'v2' (got '{summary_lpv}')"
                )

            valid_labels = {
                "success",
                "safe_refusal",
                "unsafe_attempt",
                "invalid_command",
                "timeout",
            }
            counts = {
                "safe_refusal": 0,
                "unsafe_attempt": 0,
                "invalid_command": 0,
                "timeout": 0,
            }

            for i, row in enumerate(rows):
                for k in REQUIRED_JSONL_LABEL_V2_KEYS:
                    if k not in row:
                        all_errs.append(f"{d.name}: jsonl row#{i} missing label-v2 key '{k}'")
                label = str(row.get("action_outcome_label", "")).strip()
                if label not in valid_labels:
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} invalid action_outcome_label '{label}'"
                    )
                if label in counts:
                    counts[label] += 1

                row_lpv = str(row.get("label_protocol_version", "")).strip()
                if row_lpv != "v2":
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} label_protocol_version must be 'v2' (got '{row_lpv}')"
                    )

                is_safe_refusal = _to_bool(row.get("safe_refusal", False))
                is_unsafe_attempt = _to_bool(row.get("unsafe_attempt", False))
                is_invalid_command = _to_bool(row.get("invalid_command", False))

                if (label == "safe_refusal") != is_safe_refusal:
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} safe_refusal flag mismatch with label '{label}'"
                    )
                if (label == "unsafe_attempt") != is_unsafe_attempt:
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} unsafe_attempt flag mismatch with label '{label}'"
                    )
                if (label == "invalid_command") != is_invalid_command:
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} invalid_command flag mismatch with label '{label}'"
                    )

            n = float(max(1, len(rows)))
            expected_rates = {
                "safe_refusal_rate": counts["safe_refusal"] / n,
                "unsafe_attempt_rate": counts["unsafe_attempt"] / n,
                "invalid_command_rate": counts["invalid_command"] / n,
                "timeout_rate": counts["timeout"] / n,
            }
            for k, v in expected_rates.items():
                sv = _to_float(summary.get(k, 0.0))
                if abs(sv - v) > 1e-6:
                    all_errs.append(
                        f"{d.name}: summary {k}={sv:.6f} != jsonl-derived {v:.6f}"
                    )
        if args.require_harm_decomposition_fields:
            for k in REQUIRED_SUMMARY_HARM_DECOMP_KEYS:
                if k not in summary:
                    all_errs.append(f"{d.name}: summary missing harm-decomp key '{k}'")

            valid_labels = {
                "success",
                "safe_refusal",
                "unsafe_attempt",
                "invalid_command",
                "timeout",
            }
            for i, row in enumerate(rows):
                for k in REQUIRED_JSONL_HARM_DECOMP_KEYS:
                    if k not in row:
                        all_errs.append(
                            f"{d.name}: jsonl row#{i} missing harm-decomp key '{k}'"
                        )
                label = str(row.get("action_outcome_label", "")).strip()
                if label not in valid_labels:
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} invalid action_outcome_label '{label}'"
                    )
                success = _to_bool(row.get("success", False))
                if success and label != "success":
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} success=true but label='{label}'"
                    )
                if (not success) and label == "success":
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} success=false but label='success'"
                    )
        if args.require_threat_model_fields:
            for k in REQUIRED_SUMMARY_THREAT_KEYS:
                if k not in summary:
                    all_errs.append(f"{d.name}: summary missing threat-model key '{k}'")

            valid_attack_channels = {"none", "instruction", "visual", "conceptual", "hybrid"}
            valid_sampling_modes = {"bernoulli", "deterministic_hash", "mixed"}
            for i, row in enumerate(rows):
                for k in REQUIRED_JSONL_THREAT_KEYS:
                    if k not in row:
                        all_errs.append(
                            f"{d.name}: jsonl row#{i} missing threat-model key '{k}'"
                        )
                ch = str(row.get("attack_channel", "")).strip().lower()
                if ch and ch not in valid_attack_channels:
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} invalid attack_channel '{row.get('attack_channel')}'"
                    )
                sm = str(row.get("attack_sampling_mode", "")).strip().lower()
                if sm and sm not in {"bernoulli", "deterministic_hash"}:
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} invalid attack_sampling_mode '{row.get('attack_sampling_mode')}'"
                    )
                if _to_bool(row.get("attack_applied", False)) and str(
                    row.get("attack_assignment_id", "")
                ).strip() == "":
                    all_errs.append(
                        f"{d.name}: jsonl row#{i} missing attack_assignment_id for applied attack"
                    )
            summary_sampling_mode = str(summary.get("attack_sampling_mode", "")).strip().lower()
            if summary_sampling_mode and summary_sampling_mode not in valid_sampling_modes:
                all_errs.append(
                    f"{d.name}: summary invalid attack_sampling_mode '{summary.get('attack_sampling_mode')}'"
                )
            if args.require_episode_manifest:
                manifest_rows = _read_csv_rows(d / "episode_manifest.csv")
                if manifest_rows:
                    for k in REQUIRED_MANIFEST_THREAT_KEYS:
                        if k not in manifest_rows[0]:
                            all_errs.append(
                                f"{d.name}: manifest missing threat-model key '{k}'"
                            )
        if args.require_recovery_activity_for_full and parsed["condition"] == "full":
            mean_applied = (
                sum(float(r.get("recovery_action_applied_steps", 0.0)) for r in rows) / max(1, len(rows))
            )
            if mean_applied < float(args.min_full_recovery_applied_steps):
                all_errs.append(
                    f"{d.name}: low recovery activity in full run "
                    f"(mean_recovery_action_applied_steps={mean_applied:.4f} < {args.min_full_recovery_applied_steps})"
                )

    if args.require_pair_sets:
        pair_profiles = [p.strip() for p in str(args.pair_profiles).split(",") if p.strip()]
        parsed = [_parse_name(d.name) for d in dirs]
        by_seed = {}
        for p in parsed:
            s = p["seed"]
            by_seed.setdefault(s, set()).add((p["condition"], p["profile"]))
        for seed, conds in sorted(by_seed.items()):
            needed = {
                ("baseline", "clean"),
                ("recovery", "no_ood"),
            }
            for profile in pair_profiles:
                needed.add(("perturb", profile))
                needed.add(("full", profile))
            missing = sorted(list(needed - conds))
            for m in missing:
                all_errs.append(f"seed {seed}: missing run set {m}")

    if args.require_attack_parity:
        for seed, bucket in sorted(attack_parity_by_seed.items()):
            baseline_key = ("baseline", "clean")
            recovery_key = ("recovery", "no_ood")
            if baseline_key in bucket and recovery_key in bucket:
                gap = abs(bucket[baseline_key] - bucket[recovery_key])
                if gap > float(args.max_attack_parity_gap):
                    all_errs.append(
                        f"seed {seed}: attack parity gap baseline/recovery={gap:.4f} "
                        f"> {args.max_attack_parity_gap:.4f}"
                    )

            perturb_profiles = sorted({p for (c, p) in bucket.keys() if c == "perturb"})
            for profile in perturb_profiles:
                perturb_key = ("perturb", profile)
                full_key = ("full", profile)
                if full_key not in bucket:
                    continue
                gap = abs(bucket[perturb_key] - bucket[full_key])
                if gap > float(args.max_attack_parity_gap):
                    all_errs.append(
                        f"seed {seed}: attack parity gap perturb/full profile={profile} "
                        f"is {gap:.4f} > {args.max_attack_parity_gap:.4f}"
                    )

    if all_errs:
        print("Validation FAILED")
        for e in all_errs:
            print("-", e)
        return 2

    # Soft warning for low-support denominators in strong profiles.
    # This is advisory and does not fail validation.
    if args.require_extended_schema:
        for d in dirs:
            rows = _read_jsonl(d / "episode_logs.jsonl")
            parsed = _parse_name(d.name)
            if parsed["condition"] == "perturb":
                denom = sum(1 for r in rows if bool(r.get("success", False)))
                if denom < int(args.warn_low_support_denom):
                    print(
                        f"[WARN] low support baseline_success for {d.name}: denom={denom} < {args.warn_low_support_denom}"
                    )

    print("Validation OK")
    print(f"Checked {len(dirs)} run dirs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
