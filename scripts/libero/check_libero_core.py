#!/usr/bin/env python3
"""Check that the LIBERO core env is import-ready."""

import importlib
import sys

MODULES = [
    "libero",
    "libero.libero",
    "libero.libero.envs",
]


def main() -> int:
    ok = True
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"[ok] import {name}: {getattr(mod, '__file__', '<builtin>')}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[error] import {name} failed: {type(exc).__name__}: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
