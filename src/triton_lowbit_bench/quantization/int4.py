"""INT4 量化、打包、解包与反量化工具。

本工程采用与 WSL 原始 Demo1/Demo2 一致的 INT4 表示：

- 真实 signed INT4 数值范围：-8..7。
- 保存到 uint8 时使用 ``q_u = q_signed + 8``，因此 unsigned code 0..15 对应 signed -8..7。
- packed 格式沿 K 维打包：偶数 k 放低 4 bit，奇数 k 放高 4 bit。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def signed_code_to_int4(codes: torch.Tensor) -> torch.Tensor:
    """把 unsigned INT4 code 还原为 signed INT4。

    Args:
        codes: uint8/int tensor，元素范围为 ``[0, 15]``。

    Returns:
        int8 tensor，数值范围为 ``[-8, 7]``。
    """
    return (codes.to(torch.int16) - 8).to(torch.int8)


def quantize_int4_per_tensor(
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """对二维权重矩阵做 per-tensor 对称 INT4 量化。

    Args:
        weight: 浮点权重矩阵，形状为 ``[K, N]``。
        eps: scale 下限，避免全零权重导致除零。

    Returns:
        字典，包含 ``codes``、``scale`` 和 ``original_shape``。
        ``codes`` 是 uint8，值域 0..15；真实 signed INT4 值为 ``codes - 8``。
    """
    if weight.dim() != 2:
        raise ValueError("weight must be shaped [K, N].")

    # 对称量化：用 max(abs(W)) / 7 作为 scale，把浮点权重映射到 [-8, 7]。
    # 为了与原始 WSL demo 保持一致，保存时不是 two's-complement，而是 q_u = q_signed + 8。
    weight_f32 = weight.detach().float()
    max_abs = torch.clamp(weight_f32.abs().max(), min=eps)
    scale = (max_abs / 7.0).to(torch.float32).reshape(1)
    q_signed = torch.round(weight_f32 / scale).clamp(-8, 7).to(torch.int8)
    codes = (q_signed.to(torch.int16) + 8).to(torch.uint8).contiguous()
    return {
        "codes": codes,
        "scale": scale.contiguous(),
        "original_shape": torch.tensor(weight.shape, dtype=torch.int64),
    }


def pack_int4_along_k(codes: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """沿 K 维把两个 INT4 code 打包成一个 uint8。

    Args:
        codes: uint8 tensor，形状为 ``[K, N]``，每个元素只使用低 4 bit。

    Returns:
        ``(packed, original_shape)``，packed 形状为 ``[ceil(K / 2), N]``。
    """
    if codes.dim() != 2:
        raise ValueError("codes must be shaped [K, N].")
    if codes.dtype != torch.uint8:
        codes = codes.to(torch.uint8)

    k_size, n_size = codes.shape
    if k_size % 2 != 0:
        # signed 0 对应 unsigned code 8；padding 不应改变数学结果。
        padding = torch.full((1, n_size), 8, device=codes.device, dtype=torch.uint8)
        codes = torch.cat([codes, padding], dim=0)

    # INT4 每个权重占 4 bit。偶数 k 放低 4 bit，奇数 k 放高 4 bit，
    # 因此 2 个 INT4 权重刚好打包到 1 个 uint8。
    low = codes[0::2, :] & 0x0F
    high = (codes[1::2, :] & 0x0F) << 4
    packed = (low | high).contiguous()
    return packed, (k_size, n_size)


def unpack_int4_along_k(
    packed: torch.Tensor,
    original_shape: Tuple[int, int],
) -> torch.Tensor:
    """把 packed uint8 权重还原成 unsigned INT4 code。

    Args:
        packed: packed tensor，形状为 ``[ceil(K / 2), N]``。
        original_shape: 原始未打包形状 ``(K, N)``。

    Returns:
        uint8 code tensor，形状为 ``[K, N]``，值域 0..15。
    """
    k_size, n_size = original_shape
    if packed.dim() != 2:
        raise ValueError("packed must be shaped [ceil(K / 2), N].")
    if packed.shape[1] != n_size:
        raise ValueError("packed N dimension does not match original_shape.")

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.empty(
        (packed.shape[0] * 2, n_size),
        device=packed.device,
        dtype=torch.uint8,
    )
    unpacked[0::2, :] = low
    unpacked[1::2, :] = high
    return unpacked[:k_size, :].contiguous()


def dequant_int4(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """把 INT4 code 反量化为 FP16 权重。

    Args:
        codes: uint8 code tensor，形状为 ``[K, N]``。
        scale: 标量量化 scale。

    Returns:
        FP16 dense 权重矩阵，形状为 ``[K, N]``。
    """
    # naive 路线会显式生成 W_dequant。这样虽然便于写 PyTorch 代码，
    # 但会把低比特权重重新扩展成完整 FP16 矩阵，带来显存写回和带宽开销。
    signed = signed_code_to_int4(codes).float()
    return (signed * scale.float()).to(torch.float16).contiguous()


def make_int4_weight(weight: torch.Tensor) -> Dict[str, object]:
    """从 dense FP16 权重生成 packed INT4 权重包。"""
    quantized = quantize_int4_per_tensor(weight)
    packed, original_shape = pack_int4_along_k(quantized["codes"])
    dequant = dequant_int4(quantized["codes"], quantized["scale"])
    return {
        "packed_weight": packed,
        "scale": quantized["scale"],
        "codes": quantized["codes"],
        "dequant_weight": dequant,
        "original_shape": original_shape,
    }


def unpack_dequant_from_packed(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_shape: Tuple[int, int],
) -> torch.Tensor:
    """naive 路线使用：先解包 packed INT4，再反量化为 FP16。"""
    codes = unpack_int4_along_k(packed_weight, original_shape)
    return dequant_int4(codes, scale)
