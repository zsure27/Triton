"""MXFP4 数据构造、FP4 打包和参考反量化工具。"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def manual_pack_fp4_along_k(
    fp4_codes: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """沿 K 维把两个 FP4 code 打包成一个 uint8。

    Args:
        fp4_codes: uint8 tensor，形状为 ``[K, N]``，每个元素是 0..15 的 FP4 code。

    Returns:
        ``(packed, original_shape)``，packed 形状为 ``[ceil(K / 2), N]``。
    """
    if fp4_codes.dim() != 2:
        raise ValueError("fp4_codes must be shaped [K, N].")
    if fp4_codes.dtype != torch.uint8:
        fp4_codes = fp4_codes.to(torch.uint8)

    k_size, n_size = fp4_codes.shape
    if k_size % 2 != 0:
        pad = torch.zeros((1, n_size), device=fp4_codes.device, dtype=torch.uint8)
        fp4_codes = torch.cat([fp4_codes, pad], dim=0)

    # 与 INT4 类似，两个 FP4 code 合成一个 uint8。沿 K 维打包后，
    # Triton kernel 可以按 W[k, n] 的访问模式局部解码。
    low = fp4_codes[0::2, :] & 0x0F
    high = (fp4_codes[1::2, :] & 0x0F) << 4
    return (low | high).contiguous(), (k_size, n_size)


def decode_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """把 MXFP4 E2M1-like code 解码为 float32 数值。"""
    mag_code = codes & 0x07
    values = torch.empty_like(codes, dtype=torch.float32)
    lookup = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=codes.device,
        dtype=torch.float32,
    )
    values.copy_(lookup[mag_code.long()])
    return torch.where(codes >= 8, -values, values)


def decode_e8m0_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """把 E8M0 scale code 解码为 float32 scale：``2 ** (code - 127)``。"""
    return torch.exp2(scale_codes.float() - 127.0)


def make_mxfp4_weight(
    k_size: int,
    n_size: int,
    vec_size: int = 32,
    device: str = "cuda",
    fixed_scale: bool = True,
) -> Dict[str, object]:
    """使用 ``triton.tools.mxfp`` 构造 MXFP4 权重包。

    Args:
        k_size: 输入特征维 K。
        n_size: 输出特征维 N。
        vec_size: MX block 沿 K 维的大小，通常为 32。
        device: 设备，例如 ``cuda``。
        fixed_scale: True 时使用 scale_code=127，也就是 scale=1。

    Returns:
        包含 MXFP4Tensor、MXScaleTensor、packed FP4 权重和参考 FP16 反量化权重的字典。
    """
    if k_size % vec_size != 0:
        raise ValueError("K must be divisible by vec_size=32 for MXFP4.")

    from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

    weight_mx = MXFP4Tensor(size=(k_size, n_size), device=device).random()
    if fixed_scale:
        # 默认使用 fixed_scale=True。scale_code=127 对应 scale=1，
        # 可以避免随机 scale 过大导致 FP16 matmul 出现 Inf/NaN。
        scale_data = torch.full(
            (k_size // vec_size, n_size),
            127,
            device=device,
            dtype=torch.uint8,
        )
        scale_mx = MXScaleTensor(data=scale_data)
    else:
        # 随机 scale 可能产生很大的 2^(code - 127)，在 RTX 3060 的 FP16 路径上容易溢出。
        # 因此 random scale 只建议作为额外压力测试，不作为默认正确性验证路径。
        scale_mx = MXScaleTensor(
            size=(k_size // vec_size, n_size),
            device=device,
        ).random()

    fp4_codes = weight_mx.data.contiguous()
    packed_weight, original_shape = manual_pack_fp4_along_k(fp4_codes)
    scale_fp32 = scale_mx.to(torch.float32).contiguous()
    scale_broadcast = scale_fp32.repeat_interleave(vec_size, dim=0)
    weight_fp4_value = weight_mx.to(torch.float32)
    weight_dequant = (weight_fp4_value * scale_broadcast).to(torch.float16)

    return {
        "W_mx": weight_mx,
        "S_mx": scale_mx,
        "fp4_codes": fp4_codes,
        "W_packed": packed_weight,
        "scale_u8": scale_mx.data.contiguous(),
        "scale_fp32": scale_fp32,
        "W_dequant_fp16": weight_dequant.contiguous(),
        "original_shape": original_shape,
        "vec_size": vec_size,
        "fixed_scale": fixed_scale,
    }


def full_dequant_mxfp4(qpack: Dict[str, object]) -> torch.Tensor:
    """Full naive 路线使用：每次调用都重新解码 FP4 并应用 block scale。"""
    vec_size = int(qpack["vec_size"])
    weight_fp4_value = qpack["W_mx"].to(torch.float32)
    scale_fp32 = qpack["S_mx"].to(torch.float32)
    scale_broadcast = scale_fp32.repeat_interleave(vec_size, dim=0)
    return (weight_fp4_value * scale_broadcast).to(torch.float16).contiguous()
