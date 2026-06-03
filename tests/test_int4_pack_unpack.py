"""Tests for INT4 pack/unpack helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from triton_lowbit_bench.quantization.int4 import (  # noqa: E402
    pack_int4_along_k,
    unpack_int4_along_k,
)


def test_int4_pack_unpack_roundtrip() -> None:
    """Pack random unsigned INT4 codes and verify exact unpack recovery."""
    codes = torch.randint(0, 16, (17, 13), dtype=torch.uint8)
    packed, original_shape = pack_int4_along_k(codes)
    restored = unpack_int4_along_k(packed, original_shape)
    assert restored.shape == codes.shape
    assert torch.equal(restored, codes)
