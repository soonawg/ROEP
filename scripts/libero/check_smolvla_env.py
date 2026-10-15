#!/usr/bin/env python3
"""Check that the SmolVLA model env is import-ready."""

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys

MODULES = [
    "torch",
    "transformers",
    "lerobot",
    "lerobot.envs.libero",
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
        print("[error] CUDA is required for SmolVLA runs but is not available in this execution context.")
        return 1

    try:
        _ = torch.zeros(1, device="cuda")
        print(f"[ok] CUDA tensor allocation succeeded: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[error] CUDA tensor allocation failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Check that the SmolVLA model env is import-ready.")
    ap.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when CUDA compute is unavailable in the current execution context.",
    )
    args = ap.parse_args()

    if os.environ.get("_ROEP_SMOLVLA_CHECK_REEXEC") != "1":
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if not conda_prefix:
            conda_prefix = str(Path(sys.executable).resolve().parents[1])
        if conda_prefix:
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
            env["_ROEP_SMOLVLA_CHECK_REEXEC"] = "1"
            return subprocess.run([sys.executable, __file__], env=env, check=False).returncode

    ok = True
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"[ok] import {name}: {getattr(mod, '__file__', '<builtin>')}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[error] import {name} failed: {type(exc).__name__}: {exc}")
    ok = (_check_cuda(require_cuda=bool(args.require_cuda)) == 0) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
