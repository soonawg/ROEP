"""Runtime bridge client for model/env isolation.

This module lets the evaluator run in an RLBench-ready environment while model
inference runs in a model-specific environment.
"""

from __future__ import annotations

import atexit
import base64
import os
import secrets
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..model_adapters.base import AdapterNotReadyError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = PROJECT_ROOT / "roepbench"
SERVICE_SCRIPT = PROJECT_ROOT / "scripts" / "run_model_runtime_service.py"

_CLIENTS: Dict[Tuple[str, str, str, str], "RuntimeBridgeClient"] = {}


def _parse_bool(value: Any, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if text == "":
        return bool(default)
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _model_tokens(model_key: str) -> Tuple[str, ...]:
    mapping = {
        "lap_libero": ("LAP_LIBERO", "LAP"),
        "octo_base": ("OCTO_BASE", "OCTO_SMALL"),
        "openvla_7b": ("OPENVLA_7B", "OPENVLA"),
        "pi05_libero": ("PI05_LIBERO", "PI05"),
        "pi0_fast": ("PI0_FAST", "PI0"),
        "rdt_170m": ("RDT_170M", "VIT_BC_LANG"),
        "smolvla_base": ("SMOLVLA_BASE",),
        "spatialvla": ("SPATIALVLA",),
        "vla_adapter": ("VLA_ADAPTER",),
        "xvla": ("XVLA",),
    }
    return mapping.get(model_key, tuple())


def _bridge_enabled(model_key: str) -> bool:
    global_default = _parse_bool(os.environ.get("ROEP_EXTERNAL_RUNTIME_BRIDGE", "0"), default=False)
    for token in _model_tokens(model_key):
        key = f"ROEP_EXTERNAL_{token}_RUNTIME_BRIDGE"
        raw = os.environ.get(key, "")
        if str(raw).strip() != "":
            return _parse_bool(raw, default=global_default)
    return global_default


def _resolve_model_python(model_key: str) -> str:
    for token in _model_tokens(model_key):
        key = f"ROEP_EXTERNAL_{token}_PYTHON"
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    return ""


def _collect_model_env_nvidia_lib_dirs(model_python: str) -> list[str]:
    """Collect CUDA runtime lib dirs shipped inside the target model env.

    This is used to avoid version mismatches with system `/usr/local/cuda*`
    libraries when JAX/Torch wheels expect pip/conda-bundled NVIDIA libs.
    """
    try:
        py_path = Path(model_python).resolve()
    except Exception:
        return []
    # .../env/bin/python -> .../env
    env_root = py_path.parent.parent
    if not env_root.exists():
        return []

    site_pkgs = sorted(env_root.glob("lib/python*/site-packages"))
    if not site_pkgs:
        return []

    rels = (
        "nvidia/cublas/lib",
        "nvidia/cuda_cupti/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cudnn/lib",
        "nvidia/cufft/lib",
        "nvidia/cusolver/lib",
        "nvidia/cusparse/lib",
        "nvidia/nccl/lib",
        "nvidia/nvjitlink/lib",
    )
    out: list[str] = []
    for sp in site_pkgs:
        for rel in rels:
            p = sp / rel
            if p.exists():
                out.append(str(p))
    # de-dup while preserving order
    dedup: list[str] = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    return dedup


class RuntimeBridgeClient:
    def __init__(self, *, model_key: str, checkpoint_path: str, device: str, model_python: str):
        self.model_key = str(model_key)
        self.checkpoint_path = str(checkpoint_path)
        self.device = str(device)
        self.model_python = str(model_python)

        self._startup_timeout = float(os.environ.get("ROEP_RUNTIME_BRIDGE_STARTUP_TIMEOUT", "180"))
        self._request_timeout = float(os.environ.get("ROEP_RUNTIME_BRIDGE_REQUEST_TIMEOUT", "30"))

        self._authkey = secrets.token_bytes(16)
        self._host = str(os.environ.get("ROEP_RUNTIME_BRIDGE_HOST", "127.0.0.1"))
        try:
            self._port = self._reserve_local_port(self._host)
        except PermissionError as exc:
            raise AdapterNotReadyError(
                "Runtime bridge socket creation is not permitted in current environment. "
                "Disable bridge mode or run in an environment that allows local IPC sockets."
            ) from exc
        self._conn: Optional[Client] = None
        self._proc: Optional[subprocess.Popen] = None
        self._start()

    @staticmethod
    def _reserve_local_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return int(s.getsockname()[1])

    def _build_child_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["ROEP_EXTERNAL_RUNTIME_BRIDGE"] = "0"
        env["ROEP_EXTERNAL_RUNTIME_CHILD"] = "1"
        # Prevent recursive hook selection in child-side native plugins.
        env.pop("ROEP_LAP_LIBERO_RUNTIME_HOOK", None)
        env.pop("ROEP_OCTO_BASE_RUNTIME_HOOK", None)
        env.pop("ROEP_OPENVLA_7B_RUNTIME_HOOK", None)
        env.pop("ROEP_RDT_170M_RUNTIME_HOOK", None)
        env.pop("ROEP_SMOLVLA_BASE_RUNTIME_HOOK", None)

        py_path = str(env.get("PYTHONPATH", "")).strip()
        root_path = str(PKG_ROOT)
        if py_path:
            env["PYTHONPATH"] = f"{root_path}:{py_path}"
        else:
            env["PYTHONPATH"] = root_path

        # Prefer model-env bundled NVIDIA libs over host `/usr/local/cuda*`.
        # This avoids ABI/version skew (e.g., jaxlib cuda wheel + system CUDA).
        model_lib_dirs = _collect_model_env_nvidia_lib_dirs(self.model_python)
        if model_lib_dirs:
            existing = [
                p
                for p in str(env.get("LD_LIBRARY_PATH", "")).split(":")
                if p and not p.startswith("/usr/local/cuda")
            ]
            env["LD_LIBRARY_PATH"] = ":".join(model_lib_dirs + existing)
            # Prevent accidental fallback to host toolkit paths.
            env.pop("CUDA_HOME", None)
            env.pop("CUDA_PATH", None)
        return env

    def _start(self) -> None:
        if not SERVICE_SCRIPT.exists():
            raise AdapterNotReadyError(f"Runtime bridge service script missing: {SERVICE_SCRIPT}")

        cmd = [
            self.model_python,
            str(SERVICE_SCRIPT),
            "--model",
            self.model_key,
            "--checkpoint",
            self.checkpoint_path,
            "--device",
            self.device,
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--authkey_b64",
            base64.b64encode(self._authkey).decode("ascii"),
        ]
        env = self._build_child_env()
        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,
        )
        self._conn = self._connect_with_timeout()
        self._wait_ready()

    def _connect_with_timeout(self) -> Client:
        assert self._proc is not None
        deadline = time.time() + self._startup_timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise AdapterNotReadyError(
                    f"Runtime bridge child exited early (rc={self._proc.returncode}) for model={self.model_key}"
                )
            try:
                return Client((self._host, self._port), family="AF_INET", authkey=self._authkey)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.2)
        raise AdapterNotReadyError(
            f"Runtime bridge connect timeout for model={self.model_key}; last_error={last_err}"
        )

    def _wait_ready(self) -> None:
        assert self._conn is not None
        if not self._conn.poll(self._startup_timeout):
            raise AdapterNotReadyError(
                f"Runtime bridge startup timeout waiting for ready signal (model={self.model_key})."
            )
        msg = self._conn.recv()
        kind = str(msg.get("type", "")).strip().lower() if isinstance(msg, dict) else ""
        if kind == "ready":
            return
        if kind == "blocked":
            raise AdapterNotReadyError(str(msg.get("error", "Runtime bridge blocked.")))
        raise AdapterNotReadyError(f"Runtime bridge failed during startup: {msg}")

    def predict_action(self, obs_dict: Dict[str, Any], lang_goal: Optional[str] = None) -> np.ndarray:
        if self._conn is None:
            raise RuntimeError("Runtime bridge connection is not initialized.")
        req = {
            "op": "predict",
            "obs_dict": obs_dict,
            "lang_goal": lang_goal,
        }
        self._conn.send(req)
        if not self._conn.poll(self._request_timeout):
            raise RuntimeError(
                f"Runtime bridge request timeout (>{self._request_timeout}s) model={self.model_key}"
            )
        resp = self._conn.recv()
        if not isinstance(resp, dict):
            raise RuntimeError(f"Runtime bridge protocol error: invalid response type: {type(resp)}")
        if bool(resp.get("ok", False)):
            return np.asarray(resp.get("action", []), dtype=np.float32).reshape(-1)
        if bool(resp.get("blocked", False)):
            raise AdapterNotReadyError(str(resp.get("error", "Runtime bridge blocked.")))
        error = str(resp.get("error", "Runtime bridge prediction failed."))
        trace = str(resp.get("traceback", "")).strip()
        if trace:
            raise RuntimeError(f"{error}\n{trace}")
        raise RuntimeError(error)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send({"op": "stop"})
                if self._conn.poll(2.0):
                    _ = self._conn.recv()
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=3.0)
                    except Exception:
                        self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _get_or_create_client(model_key: str, checkpoint_path: str, device: str, model_python: str) -> RuntimeBridgeClient:
    key = (str(model_key), str(model_python), str(checkpoint_path), str(device))
    client = _CLIENTS.get(key)
    if client is None:
        client = RuntimeBridgeClient(
            model_key=model_key,
            checkpoint_path=checkpoint_path,
            device=device,
            model_python=model_python,
        )
        _CLIENTS[key] = client
    return client


def _close_all_clients() -> None:
    for client in list(_CLIENTS.values()):
        try:
            client.close()
        except Exception:
            pass
    _CLIENTS.clear()


atexit.register(_close_all_clients)


def maybe_build_runtime_bridge_hook(
    *,
    model_key: str,
    checkpoint_path: str,
    device: str,
) -> Optional[Callable[[Dict[str, Any], Optional[str]], np.ndarray]]:
    """Return a prediction callable backed by runtime bridge if enabled.

    Returns:
        callable(obs_dict, lang_goal) -> np.ndarray when bridge mode is enabled.
        None when bridge mode is disabled.
    """
    if str(os.environ.get("ROEP_EXTERNAL_RUNTIME_CHILD", "0")).strip() == "1":
        return None
    if not _bridge_enabled(model_key):
        return None

    model_python = _resolve_model_python(model_key)
    if model_python == "":
        raise AdapterNotReadyError(
            f"Runtime bridge is enabled for {model_key}, but model launcher is missing. "
            f"Set ROEP_EXTERNAL_{_model_tokens(model_key)[0]}_PYTHON."
        )
    if not Path(model_python).exists():
        raise AdapterNotReadyError(f"Runtime bridge model launcher does not exist: {model_python}")

    client = _get_or_create_client(
        model_key=model_key,
        checkpoint_path=checkpoint_path,
        device=device,
        model_python=model_python,
    )
    return client.predict_action


class _RuntimeBridgePluginAdapter:
    """Adapter facade that forwards prediction through the runtime bridge."""

    def __init__(self, *, checkpoint_path: str, device: str, model_name: str):
        self.model_name = str(model_name)
        hook = maybe_build_runtime_bridge_hook(
            model_key=self.model_name,
            checkpoint_path=str(checkpoint_path),
            device=str(device),
        )
        if hook is None:
            raise AdapterNotReadyError(
                f"{self.model_name}: runtime bridge is disabled. "
                "Set ROEP_EXTERNAL_RUNTIME_BRIDGE=1 or "
                f"ROEP_EXTERNAL_{_model_tokens(self.model_name)[0]}_RUNTIME_BRIDGE=1, "
                "and provide the matching model-specific python launcher."
            )
        self._predict = hook

    def predict_action(self, obs_dict: Dict[str, Any], lang_goal: Optional[str] = None) -> np.ndarray:
        return self._predict(obs_dict, lang_goal)


def build_adapter(checkpoint_path: str, device: str, model_name: str) -> Any:
    """Factory used by adapter loaders for bridge-backed external models."""
    return _RuntimeBridgePluginAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )
