"""Environment-gated tests for triton.tools.mxfp reference objects."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triton_lowbit_bench.quantization.mxfp4 import manual_pack_fp4_along_k  # noqa: E402


def test_mxfp4_tensor_reference_to_float_and_pack_shape() -> None:
    """Create MXFP4Tensor/MXScaleTensor and verify float conversion and packing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; skip MXFP4 reference test.")
    try:
        mxfp = importlib.import_module("triton.tools.mxfp")
    except Exception as exc:
        pytest.skip(f"triton.tools.mxfp unavailable: {exc}")

    weight_mx = mxfp.MXFP4Tensor(size=(32, 16), device="cuda").random()
    scale_data = torch.full((1, 16), 127, device="cuda", dtype=torch.uint8)
    scale_mx = mxfp.MXScaleTensor(data=scale_data)

    weight_float = weight_mx.to(torch.float32)
    scale_float = scale_mx.to(torch.float32)
    packed, original_shape = manual_pack_fp4_along_k(weight_mx.data)

    assert weight_float.shape == (32, 16)
    assert scale_float.shape == (1, 16)
    assert original_shape == (32, 16)
    assert packed.shape == (16, 16)
