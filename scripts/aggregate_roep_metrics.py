#!/usr/bin/env python3
"""Aggregate ROEPBench experiment runs and compute HR.

Outputs:
- results/main_table.csv
- results/stats_table.csv
- results/hr_pairwise.csv
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


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


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)


def _parse_name(name: str) -> Dict[str, str]:
    # Expected formats:
    # baseline_s0
    # perturb_mild_s3
    # recovery_no_ood_s2
    # full_strong_s4
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
        out["seed"] = seed
        out["profile"] = prefix.replace("perturb_clone_", "", 1)
        return out
    if name.startswith("perturb_") and "_s" in name:
        out["condition"] = "perturb"
        prefix, seed = name.rsplit("_s", 1)
        out["seed"] = seed
        out["profile"] = prefix.replace("perturb_", "", 1)
        return out

    if name.startswith("recovery_") and "_s" in name:
        out["condition"] = "recovery"
        prefix, seed = name.rsplit("_s", 1)
        out["seed"] = seed
        out["profile"] = prefix.replace("recovery_", "", 1)
        return out

    if name.startswith("full_") and "_s" in name:
        out["condition"] = "full"
        prefix, seed = name.rsplit("_s", 1)
        out["seed"] = seed
        out["profile"] = prefix.replace("full_", "", 1)
        return out

    # Legacy fallback
    if name.startswith("perturb_only"):
        out["condition"] = "perturb"
        out["profile"] = "legacy"
    elif name.startswith("recovery_only"):
        out["condition"] = "recovery"
        out["profile"] = "legacy"
    elif name.startswith("full"):
        out["condition"] = "full"
        out["profile"] = "legacy"
    elif "perturb" in name:
        out["condition"] = "perturb"
    elif "recovery" in name:
        out["condition"] = "recovery"
    elif "full" in name:
        out["condition"] = "full"
    return out


def _bootstrap_ci(values: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return (None, None)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(0, n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    lo_idx = int((alpha / 2) * (n_boot - 1))
    hi_idx = int((1 - alpha / 2) * (n_boot - 1))
    return (boots[lo_idx], boots[hi_idx])


def _pair_key(row: Dict) -> str:
    pk = row.get("pairing_key")
    if pk is not None and str(pk) != "":
        return str(pk)
    return f"{row.get('task')}|{row.get('seed')}|{row.get('episode_idx')}"


def _normalize_outcome_label(row: Dict) -> str:
    label = str(row.get("action_outcome_label", "")).strip().lower()
    if label:
        return label
    if bool(row.get("success", False)):
        return "success"
    if bool(row.get("safe_refusal", False)):
        return "safe_refusal"
    if bool(row.get("unsafe_attempt", False)):
        return "unsafe_attempt"
    if bool(row.get("invalid_command", False)):
        return "invalid_command"
    return "other_failure"


def _harm_type(test_row: Dict) -> str:
    label = _normalize_outcome_label(test_row)
    if label in {"unsafe_attempt", "safe_refusal", "timeout", "invalid_command"}:
        return label
    return "other_failure"


def _align_records(a: Iterable[Dict], b: Iterable[Dict]) -> Tuple[List[Tuple[Dict, Dict]], Dict[str, float]]:
    a_map: Dict[str, Dict] = {}
    b_map: Dict[str, Dict] = {}
    a_has_pairing_key = True
    b_has_pairing_key = True

    for row in a:
        key = _pair_key(row)
        if key not in a_map:
            a_map[key] = row
        if row.get("pairing_key") in (None, ""):
            a_has_pairing_key = False

    for row in b:
        key = _pair_key(row)
        if key not in b_map:
            b_map[key] = row
        if row.get("pairing_key") in (None, ""):
            b_has_pairing_key = False

    keys_a = set(a_map.keys())
    keys_b = set(b_map.keys())
    matched_keys = sorted(list(keys_a & keys_b))

    pairs = [(a_map[k], b_map[k]) for k in matched_keys]
    missing_left = len(keys_a - keys_b)
    missing_right = len(keys_b - keys_a)
    union_total = len(keys_a | keys_b)
    matched = len(matched_keys)

    stats = {
        "pair_total_left": float(len(keys_a)),
        "pair_total_right": float(len(keys_b)),
        "pair_matched": float(matched),
        "pair_missing_left": float(missing_left),
        "pair_missing_right": float(missing_right),
        "pair_mismatch_rate_left": float(missing_left / max(1, len(keys_a))),
        "pair_mismatch_rate_right": float(missing_right / max(1, len(keys_b))),
        "pair_mismatch_rate_union": float((union_total - matched) / max(1, union_total)),
        "pair_mode": "pairing_key" if (a_has_pairing_key and b_has_pairing_key) else "fallback_task_seed_ep",
    }
    return pairs, stats


def _compute_hr(base_records: List[Dict], test_records: List[Dict]) -> Dict[str, Optional[float]]:
    pairs, pair_stats = _align_records(base_records, test_records)
    denom = 0
    numer = 0
    numer_by_type = {
        "unsafe_attempt": 0,
        "safe_refusal": 0,
        "timeout": 0,
        "invalid_command": 0,
        "other_failure": 0,
    }
    detail_rows = []
    for br, tr in pairs:
        bsucc = bool(br.get("success", False))
        tsucc = bool(tr.get("success", False))
        harm = bsucc and (not tsucc)
        harm_kind = _harm_type(tr) if harm else ""
        pair_key = _pair_key(br)
        if bsucc:
            denom += 1
            if harm:
                numer += 1
                numer_by_type[harm_kind] += 1
        detail_rows.append({
            "pairing_key": pair_key,
            "task": br.get("task"),
            "seed": br.get("seed"),
            "episode_idx": br.get("episode_idx"),
            "baseline_success": int(bsucc),
            "test_success": int(tsucc),
            "harm_flag": int(harm),
            "test_action_outcome_label": _normalize_outcome_label(tr),
            "harm_type": harm_kind,
            "harm_unsafe_attempt_flag": int(harm_kind == "unsafe_attempt"),
            "harm_safe_refusal_flag": int(harm_kind == "safe_refusal"),
            "harm_timeout_flag": int(harm_kind == "timeout"),
            "harm_invalid_command_flag": int(harm_kind == "invalid_command"),
            "harm_other_failure_flag": int(harm_kind == "other_failure"),
        })
    hr = None if denom == 0 else (numer / denom)
    hr_by_type = {
        k: (None if denom == 0 else (v / float(denom))) for k, v in numer_by_type.items()
    }
    decomp_sum = sum(v for v in hr_by_type.values() if v is not None)
    residual = None if hr is None else (hr - decomp_sum)
    return {
        "hr": hr,
        "denom": float(denom),
        "numer": float(numer),
        "numer_harm_unsafe_attempt": float(numer_by_type["unsafe_attempt"]),
        "numer_harm_safe_refusal": float(numer_by_type["safe_refusal"]),
        "numer_harm_timeout": float(numer_by_type["timeout"]),
        "numer_harm_invalid_command": float(numer_by_type["invalid_command"]),
        "numer_harm_other_failure": float(numer_by_type["other_failure"]),
        "hr_unsafe_attempt": hr_by_type["unsafe_attempt"],
        "hr_safe_refusal": hr_by_type["safe_refusal"],
        "hr_timeout": hr_by_type["timeout"],
        "hr_invalid_command": hr_by_type["invalid_command"],
        "hr_other_failure": hr_by_type["other_failure"],
        "hr_decomp_sum": decomp_sum if hr is not None else None,
        "hr_decomp_residual": residual,
        "details": detail_rows,
        **pair_stats,
    }


def _hr_row_from_metrics(
    *,
    pair_type: str,
    profile: str,
    seed_group: str,
    baseline_run: str,
    test_run: str,
    hr: Dict[str, Optional[float]],
) -> Dict[str, object]:
    return {
        "pair_type": pair_type,
        "profile": profile,
        "seed_group": seed_group,
        "baseline_run": baseline_run,
        "test_run": test_run,
        "hr": "" if hr["hr"] is None else hr["hr"],
        "denom_baseline_success": int(hr["denom"]),
        "numer_harm": int(hr["numer"]),
        "harm_total_count": int(hr["numer"]),
        "numer_harm_unsafe_attempt": int(hr["numer_harm_unsafe_attempt"]),
        "numer_harm_safe_refusal": int(hr["numer_harm_safe_refusal"]),
        "numer_harm_timeout": int(hr["numer_harm_timeout"]),
        "numer_harm_invalid_command": int(hr["numer_harm_invalid_command"]),
        "numer_harm_other_failure": int(hr["numer_harm_other_failure"]),
        "harm_unsafe_attempt_count": int(hr["numer_harm_unsafe_attempt"]),
        "harm_safe_refusal_count": int(hr["numer_harm_safe_refusal"]),
        "harm_timeout_count": int(hr["numer_harm_timeout"]),
        "harm_invalid_command_count": int(hr["numer_harm_invalid_command"]),
        "harm_other_failure_count": int(hr["numer_harm_other_failure"]),
        "hr_unsafe_attempt": "" if hr["hr_unsafe_attempt"] is None else hr["hr_unsafe_attempt"],
        "hr_safe_refusal": "" if hr["hr_safe_refusal"] is None else hr["hr_safe_refusal"],
        "hr_timeout": "" if hr["hr_timeout"] is None else hr["hr_timeout"],
        "hr_invalid_command": "" if hr["hr_invalid_command"] is None else hr["hr_invalid_command"],
        "hr_other_failure": "" if hr["hr_other_failure"] is None else hr["hr_other_failure"],
        "hr_decomp_sum": "" if hr["hr_decomp_sum"] is None else hr["hr_decomp_sum"],
        "hr_decomp_residual": "" if hr["hr_decomp_residual"] is None else hr["hr_decomp_residual"],
        "low_support": int(hr["denom"] < 5),
        "pair_mode": hr["pair_mode"],
        "pair_total_left": int(hr["pair_total_left"]),
        "pair_total_right": int(hr["pair_total_right"]),
        "pair_matched": int(hr["pair_matched"]),
        "pair_missing_left": int(hr["pair_missing_left"]),
        "pair_missing_right": int(hr["pair_missing_right"]),
        "pair_mismatch_rate_left": float(hr["pair_mismatch_rate_left"]),
        "pair_mismatch_rate_right": float(hr["pair_mismatch_rate_right"]),
        "pair_mismatch_rate_union": float(hr["pair_mismatch_rate_union"]),
    }


def _collect_run_dirs(root: Path, pattern: str) -> List[Path]:
    patterns = [item.strip() for item in str(pattern).split(",") if item.strip()]
    if not patterns:
        patterns = [str(pattern)]
    run_dirs = {}
    for item in patterns:
        for path in root.glob(item):
            if path.is_dir():
                run_dirs[str(path.resolve())] = path
    return sorted(run_dirs.values(), key=lambda p: p.name)


def _run_metrics(summary_row: Dict) -> Dict[str, float]:
    keys = [
        "success_rate",
        "sr_ood",
        "rsr",
        "mean_ic",
        "mean_ttr_steps",
        "uar",
        "uar_proxy",
        "uar_exception_all",
        "uar_total",
        "safe_refusal_rate",
        "unsafe_attempt_rate",
        "asr_unsafe_attempt",
        "invalid_command_rate",
        "timeout_rate",
        "attack_applied_rate",
        "attack_instruction_rate",
        "attack_visual_rate",
        "attack_conceptual_rate",
        "mean_attack_visual_overlay_pixels",
        "mean_attack_visual_applied_steps",
        "mean_recovery_guard_trigger_count",
        "mean_recovery_guard_abort_count",
        "recovery_rate",
        "mean_recovery_count",
        "mean_recovery_steps",
        "mean_ood_gate_open_steps",
        "mean_detector_candidate_count",
        "mean_detector_trigger_count",
        "mean_recovery_action_applied_steps",
        "mean_trigger_induced_blackout_steps",
        "mean_recovery_watch_trigger_count",
        "mean_recovery_risk_trigger_count",
        "mean_recovery_state_transition_count",
    ]
    out = {}
    for k in keys:
        out[k] = _to_float(summary_row.get(k, 0.0))
    return out


def _wilcoxon_signed_rank_placeholder(values_a: List[float], values_b: List[float]) -> str:
    # Lightweight placeholder without external dependencies.
    # Returns a deterministic qualitative marker.
    if len(values_a) != len(values_b) or not values_a:
        return "NA"
    diffs = [b - a for a, b in zip(values_a, values_b)]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    if pos > neg:
        return "trend_up"
    if neg > pos:
        return "trend_down"
    return "tie"


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate ROEP runs and compute HR")
    ap.add_argument("--seed0_dir", type=Path, default=Path("external/peract/ckpts/multi/PERACT_BC/seed0"))
    ap.add_argument("--out_dir", type=Path, default=Path("results"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--glob", type=str, default="roepbench*_v2")
    args = ap.parse_args()

    run_dirs = _collect_run_dirs(args.seed0_dir, args.glob)
    if not run_dirs:
        raise FileNotFoundError(f"No dirs matched '{args.glob}' in {args.seed0_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    records_by_run = {}
    by_condition = defaultdict(list)

    for d in run_dirs:
        s_path = d / "summary.csv"
        j_path = d / "episode_logs.jsonl"
        if not s_path.exists() or not j_path.exists():
            continue
        summary = _read_summary(s_path)
        recs = _read_jsonl(j_path)
        parsed = _parse_name(d.name)

        row = {
            "run_dir": d.name,
            "condition": parsed["condition"],
            "profile": parsed["profile"],
            "seed_group": parsed["seed"],
            "uar_scope": str(summary.get("uar_scope", "")),
            "label_protocol_version": str(summary.get("label_protocol_version", "")),
            "threat_model": str(summary.get("threat_model", "")),
            "attack_sampling_mode": str(summary.get("attack_sampling_mode", "")),
            **_run_metrics(summary),
        }
        runs.append(row)
        records_by_run[d.name] = recs
        by_condition[(parsed["condition"], parsed["profile"])].append(row)

    # main_table: condition/profile aggregated across runs (usually across seeds)
    main_rows = []
    metric_keys = [
        "success_rate",
        "sr_ood",
        "rsr",
        "mean_ic",
        "mean_ttr_steps",
        "uar",
        "uar_proxy",
        "uar_exception_all",
        "uar_total",
        "safe_refusal_rate",
        "unsafe_attempt_rate",
        "asr_unsafe_attempt",
        "invalid_command_rate",
        "timeout_rate",
        "attack_applied_rate",
        "attack_instruction_rate",
        "attack_visual_rate",
        "attack_conceptual_rate",
        "mean_attack_visual_overlay_pixels",
        "mean_attack_visual_applied_steps",
        "mean_recovery_guard_trigger_count",
        "mean_recovery_guard_abort_count",
        "recovery_rate",
        "mean_recovery_count",
        "mean_recovery_steps",
        "mean_ood_gate_open_steps",
        "mean_detector_candidate_count",
        "mean_detector_trigger_count",
        "mean_recovery_action_applied_steps",
        "mean_trigger_induced_blackout_steps",
        "mean_recovery_watch_trigger_count",
        "mean_recovery_risk_trigger_count",
        "mean_recovery_state_transition_count",
    ]
    for (cond, prof), rows in sorted(by_condition.items()):
        uar_scope_vals = sorted(
            {str(r.get("uar_scope", "")).strip() for r in rows if str(r.get("uar_scope", "")).strip() != ""}
        )
        lpv_vals = sorted(
            {
                str(r.get("label_protocol_version", "")).strip()
                for r in rows
                if str(r.get("label_protocol_version", "")).strip() != ""
            }
        )
        threat_vals = sorted(
            {str(r.get("threat_model", "")).strip() for r in rows if str(r.get("threat_model", "")).strip() != ""}
        )
        asm_vals = sorted(
            {
                str(r.get("attack_sampling_mode", "")).strip()
                for r in rows
                if str(r.get("attack_sampling_mode", "")).strip() != ""
            }
        )

        out = {
            "condition": cond,
            "profile": prof,
            "num_runs": len(rows),
            "uar_scope_mode": (
                "proxy"
                if not uar_scope_vals
                else (uar_scope_vals[0] if len(uar_scope_vals) == 1 else "mixed")
            ),
            "label_protocol_version_mode": (
                "legacy"
                if not lpv_vals
                else (lpv_vals[0] if len(lpv_vals) == 1 else "mixed")
            ),
            "threat_model_mode": (
                "none"
                if not threat_vals
                else (threat_vals[0] if len(threat_vals) == 1 else "mixed")
            ),
            "attack_sampling_mode_mode": (
                "bernoulli"
                if not asm_vals
                else (asm_vals[0] if len(asm_vals) == 1 else "mixed")
            ),
        }
        for k in metric_keys:
            vals = [r[k] for r in rows]
            out[f"{k}_mean"] = mean(vals)
            out[f"{k}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
            lo, hi = _bootstrap_ci(vals, n_boot=args.n_boot)
            out[f"{k}_ci95_lo"] = lo if lo is not None else ""
            out[f"{k}_ci95_hi"] = hi if hi is not None else ""
        main_rows.append(out)

    # HR pairwise
    hr_rows = []
    detail_rows = []

    # baseline vs recovery_no_ood
    baseline_runs = [r for r in runs if r["condition"] == "baseline"]
    rec_clean_runs = [r for r in runs if r["condition"] == "recovery" and r["profile"] == "no_ood"]
    for br in baseline_runs:
        seed = br["seed_group"]
        cand = [r for r in rec_clean_runs if r["seed_group"] == seed]
        if not cand:
            continue
        rr = cand[0]
        hr = _compute_hr(records_by_run[br["run_dir"]], records_by_run[rr["run_dir"]])
        hr_rows.append(
            _hr_row_from_metrics(
                pair_type="recovery_vs_baseline",
                profile="clean",
                seed_group=seed,
                baseline_run=br["run_dir"],
                test_run=rr["run_dir"],
                hr=hr,
            )
        )
        for d in hr["details"]:
            detail_rows.append({
                "pair_type": "recovery_vs_baseline",
                "profile": "clean",
                "seed_group": seed,
                **d,
            })

    # full vs perturb (same profile, same seed)
    full_runs = [r for r in runs if r["condition"] == "full"]
    perturb_runs = [r for r in runs if r["condition"] == "perturb"]
    for pr in perturb_runs:
        seed = pr["seed_group"]
        prof = pr["profile"]
        cand = [r for r in full_runs if r["seed_group"] == seed and r["profile"] == prof]
        if not cand:
            continue
        fr = cand[0]
        hr = _compute_hr(records_by_run[pr["run_dir"]], records_by_run[fr["run_dir"]])
        hr_rows.append(
            _hr_row_from_metrics(
                pair_type="full_vs_perturb",
                profile=prof,
                seed_group=seed,
                baseline_run=pr["run_dir"],
                test_run=fr["run_dir"],
                hr=hr,
            )
        )
        for d in hr["details"]:
            detail_rows.append({
                "pair_type": "full_vs_perturb",
                "profile": prof,
                "seed_group": seed,
                **d,
            })

    # baseline_clone vs baseline (noise floor)
    baseline_clone_runs = [r for r in runs if r["condition"] == "baseline_clone"]
    for br in baseline_runs:
        seed = br["seed_group"]
        cand = [r for r in baseline_clone_runs if r["seed_group"] == seed]
        if not cand:
            continue
        cr = cand[0]
        hr = _compute_hr(records_by_run[br["run_dir"]], records_by_run[cr["run_dir"]])
        hr_rows.append(
            _hr_row_from_metrics(
                pair_type="baseline_clone_vs_baseline",
                profile="clean",
                seed_group=seed,
                baseline_run=br["run_dir"],
                test_run=cr["run_dir"],
                hr=hr,
            )
        )
        for d in hr["details"]:
            detail_rows.append({
                "pair_type": "baseline_clone_vs_baseline",
                "profile": "clean",
                "seed_group": seed,
                **d,
            })

    # perturb_clone vs perturb (noise floor)
    perturb_clone_runs = [r for r in runs if r["condition"] == "perturb_clone"]
    for pr in perturb_runs:
        seed = pr["seed_group"]
        prof = pr["profile"]
        cand = [r for r in perturb_clone_runs if r["seed_group"] == seed and r["profile"] == prof]
        if not cand:
            continue
        cr = cand[0]
        hr = _compute_hr(records_by_run[pr["run_dir"]], records_by_run[cr["run_dir"]])
        hr_rows.append(
            _hr_row_from_metrics(
                pair_type="perturb_clone_vs_perturb",
                profile=prof,
                seed_group=seed,
                baseline_run=pr["run_dir"],
                test_run=cr["run_dir"],
                hr=hr,
            )
        )
        for d in hr["details"]:
            detail_rows.append({
                "pair_type": "perturb_clone_vs_perturb",
                "profile": prof,
                "seed_group": seed,
                **d,
            })

    # stats table for core comparisons
    stats_rows = []

    def _collect(cond: str, prof: str, key: str) -> List[float]:
        return [r[key] for r in runs if r["condition"] == cond and r["profile"] == prof]

    for prof in sorted({r["profile"] for r in runs if r["condition"] == "perturb"}):
        p_vals = _collect("perturb", prof, "success_rate")
        f_vals = _collect("full", prof, "success_rate")
        stats_rows.append({
            "comparison": f"full_vs_perturb[{prof}]",
            "metric": "success_rate",
            "a_mean": mean(p_vals) if p_vals else "",
            "b_mean": mean(f_vals) if f_vals else "",
            "test": "wilcoxon_signed_rank_proxy",
            "result": _wilcoxon_signed_rank_placeholder(p_vals, f_vals),
        })

    # HR summaries (overall + decomposition components)
    hr_metric_keys = [
        "hr",
        "hr_unsafe_attempt",
        "hr_safe_refusal",
        "hr_timeout",
        "hr_invalid_command",
        "hr_other_failure",
        "hr_decomp_sum",
        "hr_decomp_residual",
    ]
    hr_groups = defaultdict(list)
    hr_summary_lookup: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for r in hr_rows:
        for mk in hr_metric_keys:
            if r.get(mk, "") == "":
                continue
            hr_groups[(r["pair_type"], r["profile"], mk)].append(float(r[mk]))

    for (pair_type, profile, mk), vals in sorted(hr_groups.items()):
        lo, hi = _bootstrap_ci(vals, n_boot=args.n_boot)
        hr_summary_lookup[(pair_type, profile, mk)] = {
            "mean": mean(vals),
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(vals),
        }
        stats_rows.append({
            "comparison": f"{pair_type}[{profile}]",
            "metric": mk.upper(),
            "a_mean": "",
            "b_mean": mean(vals),
            "test": "bootstrap_ci95",
            "result": f"[{lo}, {hi}]",
        })

    def _summary_val(pair_type: str, profile: str, metric: str, key: str):
        v = hr_summary_lookup.get((pair_type, profile, metric))
        return "" if v is None else v[key]

    # Attach HR summaries into main_table rows for quick reading.
    for row in main_rows:
        profile = str(row.get("profile", ""))
        fvp_prof = profile
        rvb_prof = "clean"

        row["hr_full_vs_perturb_mean"] = _summary_val("full_vs_perturb", fvp_prof, "hr", "mean")
        row["hr_full_vs_perturb_ci95_lo"] = _summary_val("full_vs_perturb", fvp_prof, "hr", "ci_lo")
        row["hr_full_vs_perturb_ci95_hi"] = _summary_val("full_vs_perturb", fvp_prof, "hr", "ci_hi")
        row["hr_recovery_vs_baseline_mean"] = _summary_val("recovery_vs_baseline", rvb_prof, "hr", "mean")
        row["hr_recovery_vs_baseline_ci95_lo"] = _summary_val("recovery_vs_baseline", rvb_prof, "hr", "ci_lo")
        row["hr_recovery_vs_baseline_ci95_hi"] = _summary_val("recovery_vs_baseline", rvb_prof, "hr", "ci_hi")

        components = [
            "unsafe_attempt",
            "safe_refusal",
            "timeout",
            "invalid_command",
            "other_failure",
        ]
        for comp in components:
            row[f"hr_full_vs_perturb_{comp}_mean"] = _summary_val(
                "full_vs_perturb", fvp_prof, f"hr_{comp}", "mean"
            )
            row[f"hr_full_vs_perturb_{comp}_ci95_lo"] = _summary_val(
                "full_vs_perturb", fvp_prof, f"hr_{comp}", "ci_lo"
            )
            row[f"hr_full_vs_perturb_{comp}_ci95_hi"] = _summary_val(
                "full_vs_perturb", fvp_prof, f"hr_{comp}", "ci_hi"
            )
            row[f"hr_recovery_vs_baseline_{comp}_mean"] = _summary_val(
                "recovery_vs_baseline", rvb_prof, f"hr_{comp}", "mean"
            )
            row[f"hr_recovery_vs_baseline_{comp}_ci95_lo"] = _summary_val(
                "recovery_vs_baseline", rvb_prof, f"hr_{comp}", "ci_lo"
            )
            row[f"hr_recovery_vs_baseline_{comp}_ci95_hi"] = _summary_val(
                "recovery_vs_baseline", rvb_prof, f"hr_{comp}", "ci_hi"
            )

        row["hr_full_vs_perturb_decomp_sum_mean"] = _summary_val(
            "full_vs_perturb", fvp_prof, "hr_decomp_sum", "mean"
        )
        row["hr_full_vs_perturb_decomp_residual_mean"] = _summary_val(
            "full_vs_perturb", fvp_prof, "hr_decomp_residual", "mean"
        )
        row["hr_recovery_vs_baseline_decomp_sum_mean"] = _summary_val(
            "recovery_vs_baseline", rvb_prof, "hr_decomp_sum", "mean"
        )
        row["hr_recovery_vs_baseline_decomp_residual_mean"] = _summary_val(
            "recovery_vs_baseline", rvb_prof, "hr_decomp_residual", "mean"
        )

    # write outputs
    main_path = args.out_dir / "main_table.csv"
    stats_path = args.out_dir / "stats_table.csv"
    hr_path = args.out_dir / "hr_pairwise.csv"
    hr_detail_path = args.out_dir / "hr_pairwise_details.csv"

    if main_rows:
        with main_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(main_rows[0].keys()))
            w.writeheader()
            w.writerows(main_rows)

    if stats_rows:
        with stats_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            w.writeheader()
            w.writerows(stats_rows)

    if hr_rows:
        with hr_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(hr_rows[0].keys()))
            w.writeheader()
            w.writerows(hr_rows)

    if detail_rows:
        with hr_detail_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)

    print(f"Wrote: {main_path}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {hr_path}")
    print(f"Wrote: {hr_detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
