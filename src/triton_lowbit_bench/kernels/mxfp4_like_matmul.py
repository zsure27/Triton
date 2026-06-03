"""Triton MXFP-like 软件融合 matmul kernel。"""

from __future__ import annotations

from typing import Dict

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on environment.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _mxfp4_like_linear_kernel(
        x_ptr,
        packed_w_ptr,
        scale_u8_ptr,
        y_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        VEC_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=x_mask,
                other=0.0,
            )

            # packed_w 形状为 [ceil(K / 2), N]，每个字节包含两个 FP4 code。
            # kernel 只解码当前 tile 需要的 code，避免生成完整 W_dequant。
            packed_k = offs_k // 2
            w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            raw = tl.load(
                packed_w_ptr + packed_k[:, None] * N + offs_n[None, :],
                mask=w_mask,
                other=0,
            ).to(tl.int32)
            low = raw & 0x0F
            high = (raw >> 4) & 0x0F
            is_odd = (offs_k & 1)[:, None]
            code = tl.where(is_odd == 0, low, high)

            # MXFP4 E2M1-like 解码表。最高 bit 表示符号，低 3 bit 表示幅值。
            # 这是软件融合下限验证，不是 Blackwell 原生 block-scaled Tensor Core 路径。
            mag_code = code & 0x07
            mag = tl.where(
                mag_code == 0,
                0.0,
                tl.where(
                    mag_code == 1,
                    0.5,
                    tl.where(
                        mag_code == 2,
                        1.0,
                        tl.where(
                            mag_code == 3,
                            1.5,
                            tl.where(
                                mag_code == 4,
                                2.0,
                                tl.where(
                                    mag_code == 5,
                                    3.0,
                                    tl.where(mag_code == 6, 4.0, 6.0),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            fp4_value = tl.where(code >= 8, -mag, mag)

            # E8M0 scale 解码：scale = 2 ** (scale_code - 127)。
            # 默认 fixed_scale=True 时 scale_code=127，因此 scale=1。
            scale_k = offs_k // VEC_SIZE
            scale_code = tl.load(
                scale_u8_ptr + scale_k[:, None] * N + offs_n[None, :],
                mask=w_mask,
                other=127,
            )
            scale_value = tl.exp2(scale_code.to(tl.float32) - 127.0)
            w = (fp4_value.to(tl.float32) * scale_value).to(tl.float16)
            acc += tl.dot(x, w, out_dtype=tl.float32)

        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            y_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(tl.float16),
            mask=y_mask,
        )


def triton_mxfp4_like_linear(
    x: torch.Tensor,
    qpack: Dict[str, object],
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """运行 MXFP-like Triton fused Linear。

    Args:
        x: FP16 激活矩阵，形状为 ``[M, K]``。
        qpack: ``make_mxfp4_weight`` 返回的 MXFP4 权重包。
        block_m: M 维 tile 大小。
        block_n: N 维 tile 大小。
        block_k: K 维 tile 大小。

    Returns:
        FP16 输出矩阵，形状为 ``[M, N]``。
    """
    if triton is None:
        raise RuntimeError("Triton is not available in this environment.")
    if not x.is_cuda:
        raise RuntimeError("triton_mxfp4_like_linear requires CUDA tensors.")

    k_size, n_size = qpack["original_shape"]
    m_size = x.shape[0]
    y = torch.empty((m_size, n_size), device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    _mxfp4_like_linear_kernel[grid](
        x,
        qpack["W_packed"],
        qpack["scale_u8"],
        y,
        m_size,
        k_size,
        n_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        VEC_SIZE=qpack["vec_size"],
        num_warps=4,
    )
    return y
