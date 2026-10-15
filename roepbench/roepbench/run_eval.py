import argparse
import subprocess
import sys
from pathlib import Path

from .backends import BackendNotReadyError, EpisodeRunConfig, get_backend
from .logging_utils import write_episode_jsonl, write_step_jsonl, write_summary_csv


def _find_xvfb_script() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "scripts" / "start_xvfb.sh",  # ./roepbench/scripts
        here.parents[2] / "scripts" / "start_xvfb.sh",  # ./scripts
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find scripts/start_xvfb.sh")


def _start_xvfb_if_requested(use_xvfb: bool) -> None:
    if not use_xvfb:
        return
    script = _find_xvfb_script()
    subprocess.run(["bash", str(script)], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROEPBench Gate-1 evaluation runner")
    parser.add_argument(
        "--benchmark_backend",
        type=str,
        default="rlbench",
        choices=["rlbench", "libero"],
        help="Benchmark backend. RLBench is legacy; LIBERO is the main paper-track scaffold.",
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--libero_suite", type=str, default="")
    parser.add_argument("--libero_task_name", type=str, default="")
    parser.add_argument("--libero_benchmark_name", type=str, default="")
    parser.add_argument("--libero_task_id", type=str, default="")
    parser.add_argument("--libero_init_states_path", type=str, default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    parser.add_argument("--use_xvfb", action="store_true")
    parser.add_argument("--occlusion_severity", type=float, default=0.0)
    parser.add_argument("--depth_dropout_prob", type=float, default=0.0)
    parser.add_argument("--enable_recovery", dest="enable_recovery", action="store_true")
    parser.add_argument("--disable_recovery", dest="enable_recovery", action="store_false")
    parser.set_defaults(enable_recovery=True)
    parser.add_argument(
        "--detector",
        type=str,
        choices=[
            "low_dim",
            "reward_plateau",
            "plateau_with_confirm",
            "plateau_strict",
            "plateau_with_margin",
            "plateau_with_confirm_v2",
            "plateau_strict_v2",
            "plateau_margin_v2",
        ],
        default="reward_plateau",
    )
    parser.add_argument("--reward_window", type=int, default=25)
    parser.add_argument("--reward_eps", type=float, default=0.0)
    parser.add_argument("--confirm_windows", type=int, default=3)
    parser.add_argument("--trigger_cooldown_steps", type=int, default=30)
    parser.add_argument("--margin_eps", type=float, default=0.0)
    parser.add_argument("--min_plateau_steps", type=int, default=24)
    parser.add_argument("--stagnation_window", type=int, default=15)
    parser.add_argument("--stagnation_eps", type=float, default=1e-4)
    parser.add_argument("--recovery_burst_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--safety_shield_checkpoint", type=str, default=None)
    parser.add_argument(
        "--safety_shield_device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--safety_shield_tau", type=float, default=None)
    parser.add_argument("--safety_shield_mdsp_mode", type=str, default="lift_z_scale_xy")
    parser.add_argument("--safety_shield_action_low", type=float, default=-1.0)
    parser.add_argument("--safety_shield_action_high", type=float, default=1.0)
    parser.add_argument(
        "--safety_shield_refusal_risk_threshold",
        type=float,
        default=0.995,
    )
    parser.add_argument("--safety_shield_verify_projected_action", action="store_true")
    parser.add_argument("--safety_shield_z_lift", type=float, default=0.05)
    parser.add_argument("--safety_shield_xy_scale", type=float, default=0.5)
    parser.add_argument("--safety_shield_max_translation_norm", type=float, default=0.25)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    try:
        _start_xvfb_if_requested(args.use_xvfb)

        backend = get_backend(args.benchmark_backend)
        records, step_records = backend.run_policy_episodes(
            EpisodeRunConfig(
                benchmark_backend=args.benchmark_backend,
                task=args.task,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                headless=args.headless,
                run_id=run_id,
                occlusion_severity=args.occlusion_severity,
                depth_dropout_prob=args.depth_dropout_prob,
                enable_recovery=args.enable_recovery,
                detector=args.detector,
                reward_window=args.reward_window,
                reward_eps=args.reward_eps,
                confirm_windows=args.confirm_windows,
                trigger_cooldown_steps=args.trigger_cooldown_steps,
                margin_eps=args.margin_eps,
                min_plateau_steps=args.min_plateau_steps,
                stagnation_window=args.stagnation_window,
                stagnation_eps=args.stagnation_eps,
                recovery_burst_steps=args.recovery_burst_steps,
                warmup_steps=args.warmup_steps,
                safety_shield_checkpoint=args.safety_shield_checkpoint,
                safety_shield_device=args.safety_shield_device,
                safety_shield_tau=args.safety_shield_tau,
                safety_shield_mdsp_mode=args.safety_shield_mdsp_mode,
                safety_shield_action_low=args.safety_shield_action_low,
                safety_shield_action_high=args.safety_shield_action_high,
                safety_shield_refusal_risk_threshold=args.safety_shield_refusal_risk_threshold,
                safety_shield_verify_projected_action=args.safety_shield_verify_projected_action,
                safety_shield_z_lift=args.safety_shield_z_lift,
                safety_shield_xy_scale=args.safety_shield_xy_scale,
                safety_shield_max_translation_norm=args.safety_shield_max_translation_norm,
                libero_suite=args.libero_suite,
                libero_task_name=args.libero_task_name,
                libero_benchmark_name=args.libero_benchmark_name,
                libero_task_id=args.libero_task_id,
                libero_init_states_path=args.libero_init_states_path,
            )
        )
    except BackendNotReadyError as exc:
        records = [
            {
                "run_id": run_id,
                "episode_idx": 0,
                "task": args.task,
                "seed": args.seed,
                "success": False,
                "success_source": "error",
                "terminated": False,
                "timeout": False,
                "num_steps": 0,
                "total_reward": 0.0,
                "duration_sec": 0.0,
                "policy": "none",
                "benchmark_backend": args.benchmark_backend,
                "error": f"BackendNotReadyError: {exc}",
            }
        ]
        step_records = []
    except Exception as exc:
        # Always emit output artifacts for easier debugging.
        records = [
            {
                "run_id": run_id,
                "episode_idx": 0,
                "task": args.task,
                "seed": args.seed,
                "success": False,
                "success_source": "error",
                "terminated": False,
                "timeout": False,
                "num_steps": 0,
                "total_reward": 0.0,
                "duration_sec": 0.0,
                "policy": "none",
                "benchmark_backend": args.benchmark_backend,
                "perturbation": {
                    "occlusion_severity": float(args.occlusion_severity),
                    "depth_dropout_prob": float(args.depth_dropout_prob),
                    "occluded_pixels": 0,
                    "dropped_depth_pixels": 0,
                },
                "recovery_triggered": False,
                "recovery_count": 0,
                "recovery_steps": 0,
                "recovery_first_step": None,
                "detector_type": str(args.detector),
                "reward_window": int(args.reward_window),
                "reward_eps": float(args.reward_eps),
                "confirm_windows": int(args.confirm_windows),
                "trigger_cooldown_steps": int(args.trigger_cooldown_steps),
                "margin_eps": float(args.margin_eps),
                "min_plateau_steps": int(args.min_plateau_steps),
                "stagnation_eps": float(args.stagnation_eps) if args.detector == "low_dim" else None,
                "stagnation_window": int(args.stagnation_window) if args.detector == "low_dim" else None,
                "recovery_mode": "random_burst",
                "observation": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]
        step_records = []

    episode_path = out_dir / "episode_logs.jsonl"
    step_path = out_dir / "step_logs.jsonl"
    summary_path = out_dir / "summary.csv"
    write_episode_jsonl(episode_path, records)
    write_step_jsonl(step_path, step_records)
    write_summary_csv(summary_path, records)

    if not episode_path.exists() or not summary_path.exists() or not step_path.exists():
        raise RuntimeError("Failed to generate required output files.")

    print(f"Wrote {episode_path}")
    print(f"Wrote {step_path}")
    print(f"Wrote {summary_path}")
    if records and records[0].get("error"):
        print("Run completed with errors. Check episode_logs.jsonl", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
