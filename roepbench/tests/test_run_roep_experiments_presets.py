from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_run_roep_experiments_module():
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "scripts" / "run_roep_experiments.py"
    spec = importlib.util.spec_from_file_location("run_roep_experiments_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_tunedfix_safety_shield_profile_values() -> None:
    module = _load_run_roep_experiments_module()

    profile = module._safety_shield_profile_kwargs("paper_tunedfix_v1")

    assert profile == {
        "safety_shield_tau": "0.88",
        "safety_shield_mdsp_mode": "lift_z_scale_xy",
        "safety_shield_refusal_risk_threshold": "0.995",
        "safety_shield_verify_projected_action": "false",
        "safety_shield_z_lift": "0.05",
        "safety_shield_xy_scale": "0.8",
        "safety_shield_max_translation_norm": "1.05",
    }


def test_build_condition_args_uses_paper_tunedfix_defaults_when_shield_enabled() -> None:
    module = _load_run_roep_experiments_module()

    args = module._build_condition_args(
        condition="full",
        profile="medium",
        seed=0,
        out_dir=Path("/tmp/roepbench_out"),
        run_id="full_medium_s0",
        recovery_profile_set="conservative_v1",
        detector="reward_plateau",
        reward_window=None,
        reward_eps=None,
        confirm_windows=None,
        trigger_cooldown_steps=None,
        margin_eps=None,
        min_plateau_steps=None,
        warmup_steps=None,
        recovery_burst_steps=None,
        recovery_burst_steps_clean=None,
        recovery_burst_steps_ood=None,
        recovery_action_scale=None,
        recovery_action_scale_clean=None,
        recovery_action_scale_ood=None,
        recovery_clip=None,
        recovery_clip_clean=None,
        recovery_clip_ood=None,
        max_recovery_events_per_episode=None,
        max_recovery_events_clean=None,
        max_recovery_events_ood=None,
        min_steps_between_recoveries=None,
        disable_recovery_after_first_success=None,
        recovery_only_when_ood=None,
        debug_integrity=None,
        recovery_blend_alpha=None,
        shadow_recovery=None,
        recovery_drop_window=None,
        recovery_drop_tol=None,
        blackout_on_recovery_trigger=None,
        recovery_guard_enabled=None,
        recovery_guard_delta_window=None,
        recovery_guard_abort_on_risk_spike=None,
        recovery_watch_steps=None,
        recovery_verify_steps=None,
        primitive_sequence="random_burst,safe_retract_like,regrasp_like",
        risk_suite_enabled=None,
        risk_attack_channel=None,
        risk_attack_probability=None,
        risk_attack_severity=None,
        risk_visual_alpha=None,
        risk_visual_patch_ratio=None,
        risk_threat_model=None,
        risk_instruction_prefix=None,
        risk_instruction_suffix=None,
        risk_apply_to_baseline=None,
        risk_attack_counts_as_ood=None,
        enable_risk_trigger=None,
        risk_window=None,
        risk_threshold=None,
        risk_min_unsafe_events=None,
        risk_attack_sampling_mode=None,
        risk_attack_hash_salt=None,
        safety_shield_profile_set="paper_tunedfix_v1",
        safety_shield_checkpoint="/tmp/fake_shield.pth",
        safety_shield_device="cuda",
        safety_shield_tau=None,
        safety_shield_mdsp_mode=None,
        safety_shield_refusal_risk_threshold=None,
        safety_shield_verify_projected_action=None,
        safety_shield_z_lift=None,
        safety_shield_xy_scale=None,
        safety_shield_max_translation_norm=None,
        apply_safety_shield=True,
    )

    joined = " ".join(args)
    assert "+roep.safety_shield_tau=0.88" in joined
    assert "+roep.safety_shield_mdsp_mode=lift_z_scale_xy" in joined
    assert "+roep.safety_shield_refusal_risk_threshold=0.995" in joined
    assert "+roep.safety_shield_verify_projected_action=false" in joined
    assert "+roep.safety_shield_z_lift=0.05" in joined
    assert "+roep.safety_shield_xy_scale=0.8" in joined
    assert "+roep.safety_shield_max_translation_norm=1.05" in joined
