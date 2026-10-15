from __future__ import annotations

import json
from pathlib import Path


def test_run_external_libero_pipeline_avoids_unsafe_array_expansion_under_set_u():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_external_libero_pipeline.sh"
    text = script_path.read_text(encoding="utf-8")

    # Under `set -u`, quoted associative-array expansions like "$ARR[$key]" are unsafe.
    assert '"$PERTURB_STATUS[$model]"' not in text
    assert '"$NOISE_STATUS[$model]"' not in text
    assert '"$RECOVERY_STATUS[$model]"' not in text

    assert '${PERTURB_STATUS[$model]:-not_run}' in text
    assert '${NOISE_STATUS[$model]:-not_run}' in text
    assert '${RECOVERY_STATUS[$model]:-not_run}' in text


def test_run_external_libero_pipeline_uses_profile_aware_recovery_validation():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_external_libero_pipeline.sh"
    text = script_path.read_text(encoding="utf-8")

    assert "validate_recovery_phase_ns()" in text
    assert '--pair_profiles "$RESEARCH_PROFILES"' in text


def test_run_external_libero_pipeline_exports_libero_root_on_pythonpath():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_external_libero_pipeline.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'LIBERO_ROOT="${ROEP_LIBERO_ROOT:-${HOME}/LIBERO}"' in text
    assert 'prepend_pythonpath "$PROJECT_ROOT/roepbench"' in text
    assert 'prepend_pythonpath "$LIBERO_ROOT"' in text


def test_protocol_first_runner_exposes_expansion_mode():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/libero/run_protocol_first_track.py"
    text = script_path.read_text(encoding="utf-8")

    assert '"recovery_recalibration_expansion"' in text
    assert '"protocol_first_recovery_recalibration_expansion_v1"' in text


def test_protocol_first_runner_exposes_support_close_namespace_controls():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/libero/run_protocol_first_track.py"
    text = script_path.read_text(encoding="utf-8")

    assert "--support_close_namespace_tag" in text
    assert "--reuse_existing_phase_outputs" in text
    assert "--budget_episodes" in text
    assert "--budget_seeds" in text
    assert "v1_paperstage" in text


def test_protocol_first_runner_propagates_selected_config_path():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/libero/run_protocol_first_track.py"
    text = script_path.read_text(encoding="utf-8")

    assert "config_path: Path" in text
    assert '"CONFIG_PATH": str(config_path)' in text
    assert 'in {"", "."}' in text
    assert "path.is_dir()" in text


def test_runtime_service_exposes_vla_adapter_builder_mapping():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_model_runtime_service.py"
    text = script_path.read_text(encoding="utf-8")

    assert "build_vla_adapter" in text
    assert '"vla_adapter": build_vla_adapter' in text


def test_runtime_service_exposes_xvla_builder_mapping():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_model_runtime_service.py"
    text = script_path.read_text(encoding="utf-8")

    assert "build_xvla" in text
    assert '"xvla": build_xvla' in text


def test_export_external_model_env_sets_xvla_bridge_first_defaults():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/export_external_model_env.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'ROEP_EXTERNAL_XVLA_PYTHON:-${HOME}/miniconda3/envs/roep_smolvla_libero/bin/python' in text
    assert 'ROEP_EXTERNAL_XVLA_RUNTIME_BRIDGE:-1' in text
    assert 'ROEP_XVLA_CKPT:-lerobot/xvla-libero' in text
    assert 'ROEP_XVLA_IMPL:-roepbench.model_plugins.xvla_plugin:build_adapter' in text
    assert 'ROEP_EXTERNAL_XVLA_SHIELD_CKPT' in text
    assert 'ROEP_EXTERNAL_XVLA_SHIELD_TAU' in text


def test_run_external_libero_pipeline_defaults_include_xvla_mainline():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_external_libero_pipeline.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'MODELS_CSV="${MODELS_CSV:-openvla_7b,smolvla_base,xvla}"' in text
    assert 'MODEL_ORDER=(openvla_7b smolvla_base xvla pi0_fast spatialvla vla_adapter)' in text
    assert '["xvla"]="$NS_XVLA_SMOKE"' in text
    assert '["xvla"]="$XVLA_PY"' in text
    assert '["xvla"]="$XVLA_SHIELD_CKPT"' in text
    assert '["xvla"]="$XVLA_SHIELD_TAU"' in text
    assert 'xvla) script="$PROJECT_ROOT/scripts/libero/run_xvla_official_canary.sh" ;;' in text
    assert "resolve_model_shields.py" in text
    assert 'export ROEP_EXTERNAL_XVLA_SHIELD_SOURCE' not in text


def test_libero_research_v2_main_roster_is_openvla_smolvla_xvla():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs/libero_research_v2.yaml"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["model_roster"]["main_models_csv"] == "openvla_7b,smolvla_base,xvla"
    models = {row["name"]: row for row in payload["models"]}
    assert models["xvla"]["enabled"] is True
    assert models["xvla"]["paper_candidate"] is True
    assert models["spatialvla"]["enabled"] is False
    assert models["vla_adapter"]["enabled"] is False


def test_libero_research_object_v1_main_roster_is_openvla_smolvla_xvla():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs/libero_research_object_v1.yaml"
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload["model_roster"]["main_models_csv"] == "openvla_7b,smolvla_base,xvla"
    models = {row["name"]: row for row in payload["models"]}
    assert models["xvla"]["enabled"] is True
    assert models["spatialvla"]["enabled"] is False
    assert models["vla_adapter"]["enabled"] is False


def test_runtime_service_exposes_pi0_fast_builder_mapping():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_model_runtime_service.py"
    text = script_path.read_text(encoding="utf-8")

    assert "build_pi0_fast" in text
    assert '"pi0_fast": build_pi0_fast' in text


def test_runtime_service_exposes_pi05_libero_builder_mapping():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_model_runtime_service.py"
    text = script_path.read_text(encoding="utf-8")

    assert "build_pi05_libero" in text
    assert '"pi05_libero": build_pi05_libero' in text


def test_runtime_service_exposes_lap_libero_builder_mapping():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_model_runtime_service.py"
    text = script_path.read_text(encoding="utf-8")

    assert "build_lap_libero" in text
    assert '"lap_libero": build_lap_libero' in text


def test_run_external_libero_pipeline_falls_back_to_cache_namespace_when_smoke_disabled():
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts/run_external_libero_pipeline.sh"
    text = script_path.read_text(encoding="utf-8")

    assert 'local control_ns=""' in text
    assert 'if [[ "$RUN_SMOKE" == "1" ]]; then' in text
    assert '--control_namespace "$control_ns"' in text
