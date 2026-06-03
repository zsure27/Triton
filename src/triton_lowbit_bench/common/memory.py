"""Memory accounting helpers."""

from __future__ import annotations

import torch


def tensor_memory_mb(tensor: torch.Tensor) -> float:
    """Return tensor storage size in MiB.

    Args:
        tensor: A torch tensor of any shape and dtype.

    Returns:
        Number of MiB occupied by ``tensor.numel() * tensor.element_size()``.
    """
    return float(tensor.numel() * tensor.element_size() / 1024 / 1024)


def avoided_dequant_memory_mb(k_size: int, n_size: int, dtype: torch.dtype) -> float:
    """Return memory avoided by not materializing a full dequantized weight.

    Args:
        k_size: Input feature dimension ``K``.
        n_size: Output feature dimension ``N``.
        dtype: Dtype of the materialized dequantized weight.

    Returns:
        Size in MiB for a dense ``[K, N]`` tensor with ``dtype``.
    """
    element_size = torch.empty((), dtype=dtype).element_size()
    return float(k_size * n_size * element_size / 1024 / 1024)
