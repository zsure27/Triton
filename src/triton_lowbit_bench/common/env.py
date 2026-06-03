"""Environment inspection helpers."""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Dict, Tuple

import torch


def collect_environment() -> Dict[str, object]:
    """Collect Python, PyTorch, CUDA, Triton and GPU information.

    Returns:
        Dictionary suitable for printing or storing with benchmark metadata.
    """
    info: Dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        triton = importlib.import_module("triton")
        info["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - environment dependent.
        info["triton"] = f"unavailable: {exc}"

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = f"{capability[0]}.{capability[1]}"
    else:
        info["gpu_name"] = "unavailable"
        info["gpu_capability"] = "unavailable"
    return info


def get_gpu_capability() -> Tuple[int, int]:
    """Return current CUDA device capability.

    Returns:
        Tuple ``(major, minor)`` for device 0.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    return torch.cuda.get_device_capability(0)


def has_mxfp_tools() -> bool:
    """Check whether ``triton.tools.mxfp`` can be imported.

    Returns:
        ``True`` when MXFP helper classes are available.
    """
    try:
        importlib.import_module("triton.tools.mxfp")
        return True
    except Exception:
        return False


def print_environment(prefix: str = "Environment") -> None:
    """Print environment information in a compact format.

    Args:
        prefix: Heading printed before the key-value lines.
    """
    print("=" * 88)
    print(prefix)
    print("=" * 88)
    for key, value in collect_environment().items():
        print(f"{key}: {value}")
    print("=" * 88)
