from __future__ import annotations

import importlib.util


def resolve_torch_device(preferred: str) -> str:
    normalized = str(preferred).strip().lower() or "auto"
    if normalized == "auto":
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if normalized.startswith("cuda"):
        return preferred if _cuda_available() else "cpu"

    if normalized == "mps":
        return "mps" if _mps_available() else "cpu"

    return preferred


def _cuda_available() -> bool:
    torch = _load_torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    torch = _load_torch()
    if torch is None:
        return False
    try:
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None)
        return bool(mps and mps.is_available())
    except Exception:
        return False


def _load_torch():
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch
    except Exception:
        return None
    return torch
