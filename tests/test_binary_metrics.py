"""Tests for correctness metrics and finite checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triton_lowbit_bench.common.metrics import (  # noqa: E402
    compute_error_metrics,
    tensor_finite_status,
)


def test_identical_outputs_have_cosine_one() -> None:
    """Identical tensors should have zero error and cosine similarity near 1."""
    output = torch.randn(4, 8)
    metrics = compute_error_metrics(output, output.clone())
    finite = tensor_finite_status(output)
    assert metrics["max_abs_error"] == pytest.approx(0.0)
    assert metrics["mean_abs_error"] == pytest.approx(0.0)
    assert metrics["relative_l2_error"] == pytest.approx(0.0)
    assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert finite["has_nan"] is False
    assert finite["has_inf"] is False
