from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = PROJECT_ROOT / "roepbench"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from roepbench.model_adapters.base import AdapterNotReadyError
from roepbench.model_plugins.runtime_bridge import RuntimeBridgeClient, build_adapter


class _FakeConn:
    def __init__(self, response: dict) -> None:
        self._response = response

    def send(self, req: dict) -> None:
        self.last_request = req

    def poll(self, timeout: float) -> bool:
        return True

    def recv(self) -> dict:
        return self._response


def test_runtime_bridge_build_adapter_requires_bridge_enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROEP_EXTERNAL_RUNTIME_BRIDGE", raising=False)
    monkeypatch.delenv("ROEP_EXTERNAL_VLA_ADAPTER_RUNTIME_BRIDGE", raising=False)
    monkeypatch.delenv("ROEP_EXTERNAL_VLA_ADAPTER_PYTHON", raising=False)

    with pytest.raises(AdapterNotReadyError, match="runtime bridge is disabled"):
        build_adapter(
            checkpoint_path="/tmp/nonexistent",
            device="cuda",
            model_name="vla_adapter",
        )


def test_runtime_bridge_predict_includes_child_traceback() -> None:
    client = RuntimeBridgeClient.__new__(RuntimeBridgeClient)
    client._conn = _FakeConn(
        {
            "ok": False,
            "error": "RuntimeError: boom",
            "traceback": "Traceback (most recent call last):\n  ...",
        }
    )
    client._request_timeout = 1.0
    client.model_key = "vla_adapter"

    with pytest.raises(RuntimeError, match="Traceback"):
        client.predict_action({}, None)
