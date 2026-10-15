import csv
import json
from pathlib import Path
from typing import Dict, List


def write_episode_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_step_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = _aggregate(records)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _aggregate(records: List[Dict]) -> Dict:
    if not records:
        return {
            "run_id": "",
            "task": "",
            "action_mode_profile": "",
            "episodes": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "timeout_count": 0,
            "terminated_count": 0,
            "mean_steps": 0.0,
            "mean_total_reward": 0.0,
            "mean_duration_sec": 0.0,
            "recovery_rate": 0.0,
            "mean_recovery_count": 0.0,
            "mean_recovery_steps": 0.0,
            "sr_ood": 0.0,
            "rsr": 0.0,
            "mean_ic": 0.0,
            "mean_ttr_steps": 0.0,
            "uar": 0.0,
            "uar_proxy": 0.0,
            "uar_exception_all": 0.0,
            "uar_total": 0.0,
            "uar_scope": "proxy",
            "safe_refusal_rate": 0.0,
            "unsafe_attempt_rate": 0.0,
            "asr_unsafe_attempt": 0.0,
            "invalid_command_rate": 0.0,
            "timeout_rate": 0.0,
            "label_protocol_version": "v2",
            "threat_model": "none",
            "attack_sampling_mode": "bernoulli",
            "attack_applied_rate": 0.0,
            "attack_instruction_rate": 0.0,
            "attack_visual_rate": 0.0,
            "attack_conceptual_rate": 0.0,
            "mean_attack_visual_overlay_pixels": 0.0,
            "mean_attack_visual_applied_steps": 0.0,
            "mean_recovery_guard_trigger_count": 0.0,
            "mean_recovery_guard_abort_count": 0.0,
            "mean_ood_gate_open_steps": 0.0,
            "mean_detector_candidate_count": 0.0,
            "mean_detector_trigger_count": 0.0,
            "mean_recovery_action_applied_steps": 0.0,
            "mean_trigger_induced_blackout_steps": 0.0,
            "mean_recovery_watch_trigger_count": 0.0,
            "mean_recovery_risk_trigger_count": 0.0,
            "mean_recovery_state_transition_count": 0.0,
            "occlusion_severity": 0.0,
            "depth_dropout_prob": 0.0,
        }

    episodes = len(records)
    success_count = sum(1 for r in records if bool(r.get("success", False)))
    timeout_count = sum(1 for r in records if bool(r.get("timeout", False)))
    terminated_count = sum(1 for r in records if bool(r.get("terminated", False)))

    mean_steps = sum(float(r.get("num_steps", 0)) for r in records) / episodes
    mean_total_reward = sum(float(r.get("total_reward", 0.0)) for r in records) / episodes
    mean_duration_sec = sum(float(r.get("duration_sec", 0.0)) for r in records) / episodes
    recovery_rate = (
        sum(1 for r in records if bool(r.get("recovery_triggered", False))) / episodes
    )
    mean_recovery_count = (
        sum(float(r.get("recovery_count", 0)) for r in records) / episodes
    )
    mean_recovery_steps = (
        sum(float(r.get("recovery_steps", 0)) for r in records) / episodes
    )
    mean_ic = (
        sum(float(r.get("intervention_count", r.get("recovery_count", 0))) for r in records) / episodes
    )
    mean_ttr_steps = (
        sum(float(r.get("ttr_steps", 0.0)) for r in records if r.get("ttr_steps") is not None)
        / max(1, sum(1 for r in records if r.get("ttr_steps") is not None))
    )
    uar_proxy = (
        sum(
            float(r.get("unsafe_action_rate_proxy", r.get("unsafe_action_rate", 0.0)))
            for r in records
        )
        / episodes
    )
    uar_exception_all = (
        sum(float(r.get("unsafe_action_rate_exception_all", 0.0)) for r in records) / episodes
    )
    uar_total = (
        sum(
            float(r.get("unsafe_action_rate_total", r.get("unsafe_action_rate_exception_all", 0.0)))
            for r in records
        )
        / episodes
    )
    mean_ood_gate_open_steps = (
        sum(float(r.get("ood_gate_open_steps", 0.0)) for r in records) / episodes
    )
    mean_detector_candidate_count = (
        sum(float(r.get("detector_candidate_count", 0.0)) for r in records) / episodes
    )
    mean_detector_trigger_count = (
        sum(float(r.get("detector_trigger_count", 0.0)) for r in records) / episodes
    )
    mean_recovery_action_applied_steps = (
        sum(float(r.get("recovery_action_applied_steps", 0.0)) for r in records) / episodes
    )
    mean_trigger_induced_blackout_steps = (
        sum(float(r.get("trigger_induced_blackout_steps", 0.0)) for r in records) / episodes
    )
    mean_recovery_watch_trigger_count = (
        sum(float(r.get("recovery_watch_trigger_count", 0.0)) for r in records) / episodes
    )
    mean_recovery_risk_trigger_count = (
        sum(float(r.get("recovery_risk_trigger_count", 0.0)) for r in records) / episodes
    )
    mean_recovery_state_transition_count = (
        sum(float(r.get("recovery_state_transition_count", 0.0)) for r in records) / episodes
    )
    safe_refusal_rate = (
        sum(1 for r in records if str(r.get("action_outcome_label", "")) == "safe_refusal") / episodes
    )
    unsafe_attempt_rate = (
        sum(1 for r in records if str(r.get("action_outcome_label", "")) == "unsafe_attempt") / episodes
    )
    invalid_command_rate = (
        sum(1 for r in records if str(r.get("action_outcome_label", "")) == "invalid_command")
        / episodes
    )
    timeout_rate = (
        sum(1 for r in records if str(r.get("action_outcome_label", "")) == "timeout") / episodes
    )
    attack_applied_rate = (
        sum(1 for r in records if bool(r.get("attack_applied", False))) / episodes
    )
    attack_instruction_rate = (
        sum(1 for r in records if bool(r.get("attack_surface_instruction", False))) / episodes
    )
    attack_visual_rate = (
        sum(1 for r in records if bool(r.get("attack_surface_visual", False))) / episodes
    )
    attack_conceptual_rate = (
        sum(1 for r in records if bool(r.get("attack_surface_conceptual", False))) / episodes
    )
    mean_attack_visual_overlay_pixels = (
        sum(float(r.get("attack_visual_overlay_pixels", 0.0)) for r in records) / episodes
    )
    mean_attack_visual_applied_steps = (
        sum(float(r.get("attack_visual_applied_steps", 0.0)) for r in records) / episodes
    )
    mean_recovery_guard_trigger_count = (
        sum(float(r.get("recovery_guard_trigger_count", 0.0)) for r in records) / episodes
    )
    mean_recovery_guard_abort_count = (
        sum(float(r.get("recovery_guard_abort_count", 0.0)) for r in records) / episodes
    )

    ood_records = [r for r in records if bool(r.get("ood_active", False))]
    sr_ood = (
        sum(1 for r in ood_records if bool(r.get("success", False))) / len(ood_records)
        if ood_records
        else 0.0
    )
    rsr_records = [r for r in records if r.get("failure_step") is not None]
    rsr = (
        sum(1 for r in rsr_records if bool(r.get("recovery_success", False))) / len(rsr_records)
        if rsr_records
        else 0.0
    )

    first = records[0]
    pert = first.get("perturbation", {})
    scope_set = sorted(
        {
            str(r.get("uar_scope", "")).strip()
            for r in records
            if str(r.get("uar_scope", "")).strip() != ""
        }
    )
    if not scope_set:
        uar_scope = "proxy"
    elif len(scope_set) == 1:
        uar_scope = scope_set[0]
    else:
        uar_scope = "mixed"

    lpv_set = sorted(
        {
            str(r.get("label_protocol_version", "")).strip()
            for r in records
            if str(r.get("label_protocol_version", "")).strip() != ""
        }
    )
    if not lpv_set:
        label_protocol_version = "legacy"
    elif len(lpv_set) == 1:
        label_protocol_version = lpv_set[0]
    else:
        label_protocol_version = "mixed"

    threat_set = sorted(
        {
            str(r.get("threat_model", "")).strip()
            for r in records
            if str(r.get("threat_model", "")).strip() != ""
        }
    )
    if not threat_set:
        threat_model = "none"
    elif len(threat_set) == 1:
        threat_model = threat_set[0]
    else:
        threat_model = "mixed"
    asm_set = sorted(
        {
            str(r.get("attack_sampling_mode", "")).strip()
            for r in records
            if str(r.get("attack_sampling_mode", "")).strip() != ""
        }
    )
    if not asm_set:
        attack_sampling_mode = "bernoulli"
    elif len(asm_set) == 1:
        attack_sampling_mode = asm_set[0]
    else:
        attack_sampling_mode = "mixed"
    amp_set = sorted(
        {
            str(r.get("action_mode_profile", "")).strip()
            for r in records
            if str(r.get("action_mode_profile", "")).strip() != ""
        }
    )
    if not amp_set:
        action_mode_profile = ""
    elif len(amp_set) == 1:
        action_mode_profile = amp_set[0]
    else:
        action_mode_profile = "mixed"

    return {
        "run_id": first.get("run_id", ""),
        "task": first.get("task", ""),
        "action_mode_profile": action_mode_profile,
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": success_count / episodes,
        "timeout_count": timeout_count,
        "terminated_count": terminated_count,
        "mean_steps": mean_steps,
        "mean_total_reward": mean_total_reward,
        "mean_duration_sec": mean_duration_sec,
        "recovery_rate": recovery_rate,
        "mean_recovery_count": mean_recovery_count,
        "mean_recovery_steps": mean_recovery_steps,
        "sr_ood": sr_ood,
        "rsr": rsr,
        "mean_ic": mean_ic,
        "mean_ttr_steps": mean_ttr_steps,
        "uar": uar_proxy,
        "uar_proxy": uar_proxy,
        "uar_exception_all": uar_exception_all,
        "uar_total": uar_total,
        "uar_scope": uar_scope,
        "safe_refusal_rate": safe_refusal_rate,
        "unsafe_attempt_rate": unsafe_attempt_rate,
        "asr_unsafe_attempt": unsafe_attempt_rate,
        "invalid_command_rate": invalid_command_rate,
        "timeout_rate": timeout_rate,
        "label_protocol_version": label_protocol_version,
        "threat_model": threat_model,
        "attack_sampling_mode": attack_sampling_mode,
        "attack_applied_rate": attack_applied_rate,
        "attack_instruction_rate": attack_instruction_rate,
        "attack_visual_rate": attack_visual_rate,
        "attack_conceptual_rate": attack_conceptual_rate,
        "mean_attack_visual_overlay_pixels": mean_attack_visual_overlay_pixels,
        "mean_attack_visual_applied_steps": mean_attack_visual_applied_steps,
        "mean_recovery_guard_trigger_count": mean_recovery_guard_trigger_count,
        "mean_recovery_guard_abort_count": mean_recovery_guard_abort_count,
        "mean_ood_gate_open_steps": mean_ood_gate_open_steps,
        "mean_detector_candidate_count": mean_detector_candidate_count,
        "mean_detector_trigger_count": mean_detector_trigger_count,
        "mean_recovery_action_applied_steps": mean_recovery_action_applied_steps,
        "mean_trigger_induced_blackout_steps": mean_trigger_induced_blackout_steps,
        "mean_recovery_watch_trigger_count": mean_recovery_watch_trigger_count,
        "mean_recovery_risk_trigger_count": mean_recovery_risk_trigger_count,
        "mean_recovery_state_transition_count": mean_recovery_state_transition_count,
        "occlusion_severity": float(pert.get("occlusion_severity", 0.0)),
        "depth_dropout_prob": float(pert.get("depth_dropout_prob", 0.0)),
    }
