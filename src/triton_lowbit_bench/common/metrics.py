"""Correctness and finite-value metrics for benchmark outputs."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as functional


def tensor_finite_status(tensor: torch.Tensor) -> Dict[str, bool]:
    """Return NaN/Inf status for a tensor.

    Args:
        tensor: Any torch tensor. Benchmark outputs are usually shaped
            ``[M, N]`` and stored in ``float16`` or ``float32``.

    Returns:
        A dictionary with ``has_nan`` and ``has_inf`` boolean flags.
    """
    value = tensor.detach()
    return {
        "has_nan": bool(torch.isnan(value).any().item()),
        "has_inf": bool(torch.isinf(value).any().item()),
    }


def compute_error_metrics(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> Dict[str, float]:
    """Compute standard error metrics against a reference output.

    Args:
        output: Test output tensor, usually shaped ``[M, N]``.
        reference: Reference tensor with the same shape as ``output``.

    Returns:
        Dictionary containing ``max_abs_error``, ``mean_abs_error``,
        ``relative_l2_error`` and ``cosine_similarity``.
    """
    output_f = output.detach().float()
    reference_f = reference.detach().float()

    output_status = tensor_finite_status(output_f)
    reference_status = tensor_finite_status(reference_f)
    if output_status["has_nan"] or reference_status["has_nan"]:
        return {
            "max_abs_error": float("nan"),
            "mean_abs_error": float("nan"),
            "relative_l2_error": float("nan"),
            "cosine_similarity": float("nan"),
        }
    if output_status["has_inf"] or reference_status["has_inf"]:
        return {
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "relative_l2_error": float("inf"),
            "cosine_similarity": float("nan"),
        }

    diff = output_f - reference_f
    reference_norm = torch.clamp(torch.linalg.vector_norm(reference_f), min=1e-8)
    relative_l2 = torch.linalg.vector_norm(diff) / reference_norm
    cosine = functional.cosine_similarity(
        output_f.flatten(),
        reference_f.flatten(),
        dim=0,
        eps=1e-8,
    )
    return {
        "max_abs_error": float(diff.abs().max().item()),
        "mean_abs_error": float(diff.abs().mean().item()),
        "relative_l2_error": float(relative_l2.item()),
        "cosine_similarity": float(cosine.item()),
    }
