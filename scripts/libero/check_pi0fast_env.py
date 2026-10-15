#!/usr/bin/env python3
"""Check that the Pi0Fast model env is import-ready."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

MODULES = [
    "torch",
    "transformers",
    "lerobot",
    "lerobot.policies.pi0_fast",
]


def _check_cuda(require_cuda: bool) -> int:
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        message = f"torch import failed before CUDA probe: {type(exc).__name__}: {exc}"
        print(f"[error] {message}" if require_cuda else f"[warn] {message}")
        return 1 if require_cuda else 0

    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    print(f"[info] torch.cuda.is_available: {available}")
    print(f"[info] torch.cuda.device_count: {device_count}")

    if not require_cuda:
        return 0
    if not available:
        print("[error] CUDA is required for Pi0Fast runs but is not available in this execution context.")
        return 1
    try:
        _ = torch.zeros(1, device="cuda")
        print(f"[ok] CUDA tensor allocation succeeded: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[error] CUDA tensor allocation failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


def _maybe_reexec() -> int:
    if os.environ.get("_ROEP_PI0FAST_CHECK_REEXEC") == "1":
        return 0
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        conda_prefix = str(Path(sys.executable).resolve().parents[1])
    if not conda_prefix:
        return 0
    env = os.environ.copy()
    libdir = os.path.join(conda_prefix, "lib")
    env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    preload = [
        os.path.join(libdir, "libstdc++.so.6"),
        os.path.join(libdir, "libgcc_s.so.1"),
    ]
    preload = [path for path in preload if os.path.exists(path)]
    if preload:
        joined = ":".join(preload)
        if env.get("LD_PRELOAD"):
            joined = f"{joined}:{env['LD_PRELOAD']}"
        env["LD_PRELOAD"] = joined
    env["NUMBA_DISABLE_JIT"] = env.get("NUMBA_DISABLE_JIT", "1")
    env["_ROEP_PI0FAST_CHECK_REEXEC"] = "1"
    return subprocess.run([sys.executable, __file__, *sys.argv[1:]], env=env, check=False).returncode


def _smoke_load(checkpoint: str) -> int:
    try:
        import torch  # type: ignore
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[error] pi0_fast smoke imports failed: {type(exc).__name__}: {exc}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        policy = PI0FastPolicy.from_pretrained(str(checkpoint)).to(device).eval()
    except Exception as exc:  # noqa: BLE001
        print(f"[error] PI0Fast smoke load failed: {type(exc).__name__}: {exc}")
        return 1
    print(f"[ok] PI0Fast smoke load succeeded: {checkpoint}")
    print(f"[info] policy device: {device}")
    print(f"[info] policy class: {type(policy).__name__}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Check that the Pi0Fast model env is import-ready.")
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--smoke-load", action="store_true")
    args = ap.parse_args()

    if os.environ.get("_ROEP_PI0FAST_CHECK_REEXEC") != "1":
        rc = _maybe_reexec()
        if rc != 0:
            return rc
        if os.environ.get("_ROEP_PI0FAST_CHECK_REEXEC") != "1":
            return 0

    ok = True
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"[ok] import {name}: {getattr(mod, '__file__', '<builtin>')}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[error] import {name} failed: {type(exc).__name__}: {exc}")

    ok = (_check_cuda(require_cuda=bool(args.require_cuda)) == 0) and ok
    if args.smoke_load:
        checkpoint = str(args.checkpoint or os.environ.get("ROEP_PI0_FAST_CKPT", "")).strip()
        if checkpoint == "":
            print("[error] --smoke-load requires --checkpoint or ROEP_PI0_FAST_CKPT")
            return 1
        ok = (_smoke_load(checkpoint) == 0) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
