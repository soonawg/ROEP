#!/usr/bin/env python3
"""Check that the OpenVLA model env is import-ready and version-coherent."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
from typing import Dict, Iterable, Tuple


REQUIRED_MODULES: Tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "tokenizers",
    "timm",
    "bitsandbytes",
    "libero",
)

EXPECTED_VERSIONS: Dict[str, str] = {
    "torch": "2.2.0",
    "torchvision": "0.17.0",
    "torchaudio": "2.2.0",
    "transformers": "4.40.1",
    "tokenizers": "0.19.1",
    "timm": "0.9.10",
}

PRISMATIC_MODULES: Tuple[str, ...] = (
    "prismatic",
    "prismatic.vla",
)


def _dist_name(module_name: str) -> str:
    if module_name == "torchvision":
        return "torchvision"
    if module_name == "torchaudio":
        return "torchaudio"
    if module_name == "bitsandbytes":
        return "bitsandbytes"
    return module_name


def _version_for(module_name: str) -> str:
    try:
        return metadata.version(_dist_name(module_name))
    except metadata.PackageNotFoundError:
        return ""


def _check_required_modules(allow_version_drift: bool) -> bool:
    ok = True
    for name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(name)
            location = getattr(mod, "__file__", "<builtin>")
            version = _version_for(name) or getattr(mod, "__version__", "")
            print(f"[ok] import {name}: {location}")
            if version:
                print(f"[ok] version {name}: {version}")
            expected = EXPECTED_VERSIONS.get(name)
            if expected and not allow_version_drift and version and version != expected:
                ok = False
                print(f"[error] version mismatch for {name}: expected {expected}, got {version}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[error] import {name} failed: {type(exc).__name__}: {exc}")
    return ok


def _check_torchvision_ops() -> bool:
    try:
        torchvision = importlib.import_module("torchvision")
        ops = importlib.import_module("torchvision.ops")
        _ = getattr(ops, "nms")
        print(f"[ok] torchvision nms op available: {getattr(torchvision, '__file__', '<builtin>')}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[error] torchvision nms probe failed: {type(exc).__name__}: {exc}")
        return False


def _check_modules(names: Iterable[str], strict: bool) -> bool:
    ok = True
    for name in names:
        try:
            mod = importlib.import_module(name)
            print(f"[ok] import {name}: {getattr(mod, '__file__', '<builtin>')}")
        except Exception as exc:  # noqa: BLE001
            if strict:
                ok = False
                print(f"[error] import {name} failed: {type(exc).__name__}: {exc}")
            else:
                print(f"[warn] import {name} failed: {type(exc).__name__}: {exc}")
    return ok


def _check_cuda(require_cuda: bool) -> bool:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001
        if require_cuda:
            print(f"[error] torch import failed before CUDA probe: {type(exc).__name__}: {exc}")
            return False
        print(f"[warn] torch import failed before CUDA probe: {type(exc).__name__}: {exc}")
        return True

    ok = True
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        available = False
        ok = False
        print(f"[error] torch.cuda.is_available probe failed: {type(exc).__name__}: {exc}")
    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        device_count = -1
        ok = False
        print(f"[error] torch.cuda.device_count probe failed: {type(exc).__name__}: {exc}")

    print(f"[info] torch.cuda.is_available: {available}")
    print(f"[info] torch.cuda.device_count: {device_count}")

    if available:
        try:
            _ = torch.zeros(1, device="cuda")
            print(f"[ok] CUDA tensor allocation succeeded: {torch.cuda.get_device_name(0)}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[error] CUDA tensor allocation failed: {type(exc).__name__}: {exc}")
    elif require_cuda:
        ok = False
        print("[error] CUDA is required for OpenVLA runs but is not available in this execution context.")
    else:
        print("[warn] CUDA is not available; GPU-backed OpenVLA runs will not work.")

    return ok if require_cuda else True


def main() -> int:
    ap = argparse.ArgumentParser(description="Check that the OpenVLA model env is coherent.")
    ap.add_argument(
        "--allow-prismatic-warning",
        action="store_true",
        help="Warn instead of failing when prismatic imports are broken.",
    )
    ap.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="Warn instead of failing when pinned OpenVLA package versions drift.",
    )
    ap.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail when CUDA compute is unavailable in the current execution context.",
    )
    args = ap.parse_args()

    ok = True
    ok = _check_required_modules(allow_version_drift=bool(args.allow_version_drift)) and ok
    ok = _check_torchvision_ops() and ok
    ok = _check_modules(PRISMATIC_MODULES, strict=not bool(args.allow_prismatic_warning)) and ok
    ok = _check_cuda(require_cuda=bool(args.require_cuda)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
