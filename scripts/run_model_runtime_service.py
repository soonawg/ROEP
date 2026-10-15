#!/usr/bin/env python3
"""Model runtime service for cross-environment inference.

This process runs in a model-specific environment and receives prediction
requests from the evaluator through a local authenticated Unix socket.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = PROJECT_ROOT / "roepbench"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from roepbench.model_adapters.base import AdapterNotReadyError  # noqa: E402
from roepbench.model_plugins.lap_libero_plugin import build_adapter as build_lap_libero  # noqa: E402
from roepbench.model_plugins.octo_base_plugin import build_adapter as build_octo_base  # noqa: E402
from roepbench.model_plugins.openvla_7b_plugin import build_adapter as build_openvla_7b  # noqa: E402
from roepbench.model_plugins.pi05_libero_plugin import build_adapter as build_pi05_libero  # noqa: E402
from roepbench.model_plugins.pi0_fast_plugin import build_adapter as build_pi0_fast  # noqa: E402
from roepbench.model_plugins.rdt_170m_plugin import build_adapter as build_rdt_170m  # noqa: E402
from roepbench.model_plugins.smolvla_base_plugin import build_adapter as build_smolvla_base  # noqa: E402
from roepbench.model_plugins.vla_adapter_plugin import build_adapter as build_vla_adapter  # noqa: E402
from roepbench.model_plugins.xvla_plugin import build_adapter as build_xvla  # noqa: E402


BLOCKED_EXIT_CODE = 42

BUILDERS: Dict[str, Callable[[str, str, str], Any]] = {
    "lap_libero": build_lap_libero,
    "octo_base": build_octo_base,
    "openvla_7b": build_openvla_7b,
    "pi05_libero": build_pi05_libero,
    "pi0_fast": build_pi0_fast,
    "rdt_170m": build_rdt_170m,
    "smolvla_base": build_smolvla_base,
    "vla_adapter": build_vla_adapter,
    "xvla": build_xvla,
    # Backward-compatible aliases.
    "lap": build_lap_libero,
    "pi05": build_pi05_libero,
    "octo_small": build_octo_base,
    "openvla": build_openvla_7b,
    "vit_bc_lang": build_rdt_170m,
}


def _clear_recursive_bridge_flags() -> None:
    os.environ["ROEP_EXTERNAL_RUNTIME_BRIDGE"] = "0"
    os.environ["ROEP_EXTERNAL_RUNTIME_CHILD"] = "1"
    os.environ.pop("ROEP_LAP_LIBERO_RUNTIME_HOOK", None)
    os.environ.pop("ROEP_OCTO_BASE_RUNTIME_HOOK", None)
    os.environ.pop("ROEP_OPENVLA_7B_RUNTIME_HOOK", None)
    os.environ.pop("ROEP_RDT_170M_RUNTIME_HOOK", None)
    os.environ.pop("ROEP_SMOLVLA_BASE_RUNTIME_HOOK", None)


def _build_adapter(model: str, checkpoint: str, device: str) -> Any:
    builder = BUILDERS.get(model)
    if builder is None:
        raise AdapterNotReadyError(f"Unsupported model for runtime service: {model}")
    _clear_recursive_bridge_flags()
    return builder(checkpoint_path=checkpoint, device=device, model_name=model)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run model inference runtime service.")
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--host", default="127.0.0.1", type=str)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--authkey_b64", required=True, type=str)
    args = ap.parse_args()

    authkey = base64.b64decode(args.authkey_b64.encode("ascii"))

    listener = Listener(address=(args.host, int(args.port)), family="AF_INET", authkey=authkey)
    conn = None
    try:
        conn = listener.accept()
        try:
            adapter = _build_adapter(args.model, args.checkpoint, args.device)
        except AdapterNotReadyError as exc:
            conn.send({"type": "blocked", "error": str(exc)})
            return BLOCKED_EXIT_CODE
        except Exception as exc:  # noqa: BLE001
            conn.send(
                {
                    "type": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            return 1

        conn.send({"type": "ready"})
        while True:
            try:
                req = conn.recv()
            except EOFError:
                break
            if not isinstance(req, dict):
                conn.send({"ok": False, "error": f"Invalid request type: {type(req)}"})
                continue

            op = str(req.get("op", "")).strip().lower()
            if op == "stop":
                conn.send({"ok": True, "stopped": True})
                break
            if op == "ping":
                conn.send({"ok": True, "type": "pong"})
                continue
            if op != "predict":
                conn.send({"ok": False, "error": f"Unsupported op: {op}"})
                continue

            obs_dict = req.get("obs_dict", {})
            lang_goal = req.get("lang_goal", None)
            try:
                action = adapter.predict_action(obs_dict, lang_goal)
                action_np = np.asarray(action, dtype=np.float32).reshape(-1)
                conn.send({"ok": True, "action": action_np})
            except AdapterNotReadyError as exc:
                conn.send({"ok": False, "blocked": True, "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                conn.send(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
        return 0
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        try:
            listener.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
