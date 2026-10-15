"""Goal-conditioned text embedding hook for RDT-170M.

Usage:
1) Build an index JSON that maps goal text to embedding file path.
2) Export:
   ROEP_RDT_170M_TEXT_EMBED_HOOK=roepbench.model_plugins.rdt_text_embed_hook:embed_text
   ROEP_RDT_170M_TEXT_EMBED_INDEX=/abs/path/to/embed_index.json

Index format (JSON object):
{
  "open the middle drawer": "/abs/path/to/open_middle.pt",
  "open the top drawer": "/abs/path/to/open_top.pt",
  "__default__": "/abs/path/to/default.pt"
}

Each embedding file can be:
- torch tensor (recommended)
- dict containing key "embeddings"
- numpy-compatible array payload
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_INDEX_CACHE: Optional[Dict[str, str]] = None
_EMBED_CACHE: Dict[str, Any] = {}


def _normalize_goal(goal: str) -> str:
    return " ".join(str(goal).strip().lower().split())


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    return re.sub(r"_+", "_", value).strip("_") or "na"


def _load_index() -> Dict[str, str]:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    index_path_raw = str(os.environ.get("ROEP_RDT_170M_TEXT_EMBED_INDEX", "")).strip()
    if index_path_raw == "":
        raise RuntimeError(
            "ROEP_RDT_170M_TEXT_EMBED_INDEX is required for rdt_text_embed_hook. "
            "Provide a JSON mapping {goal_text: embedding_file_path}."
        )
    index_path = Path(index_path_raw).expanduser().resolve()
    if not index_path.exists():
        raise RuntimeError(f"Text embedding index not found: {index_path}")

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid embed index format (expected JSON object): {index_path}")

    out: Dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, str):
            continue
        p = Path(value)
        if not p.is_absolute():
            p = (index_path.parent / p).resolve()
        out[key] = str(p)
    _INDEX_CACHE = out
    return out


def _resolve_embed_file(goal: str) -> Path:
    index = _load_index()
    goal_raw = str(goal).strip()
    goal_norm = _normalize_goal(goal_raw)
    candidates = [
        goal_raw,
        goal_norm,
        _slug(goal_raw),
        _slug(goal_norm),
        "__default__",
        "default",
    ]
    for key in candidates:
        if key in index:
            p = Path(index[key]).expanduser().resolve()
            if p.exists():
                return p
    raise RuntimeError(
        "No text embedding file mapped for goal. "
        f"goal='{goal_raw}', normalized='{goal_norm}'. "
        "Add an explicit key or '__default__' to ROEP_RDT_170M_TEXT_EMBED_INDEX."
    )


def _load_embedding_from_file(path: Path) -> Any:
    cache_key = str(path)
    if cache_key in _EMBED_CACHE:
        return _EMBED_CACHE[cache_key]

    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "embeddings" in obj:
        obj = obj["embeddings"]
    tensor = torch.as_tensor(obj)
    _EMBED_CACHE[cache_key] = tensor
    return tensor


def embed_text(goal: str, device: str = "cuda") -> Any:
    """Return goal-conditioned text embedding tensor for RDT policy.step()."""
    embed_file = _resolve_embed_file(goal)
    emb = _load_embedding_from_file(embed_file)
    if hasattr(emb, "to"):
        return emb.to(device)
    return torch.as_tensor(emb, device=device)
