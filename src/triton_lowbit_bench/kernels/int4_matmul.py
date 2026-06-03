"""Triton fused INT4 Linear kernel。"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on environment.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _int4_matmul_kernel(
        x_ptr,
        packed_w_ptr,
        scale_ptr,
        y_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)

            # 读取当前 tile 的激活 X，形状逻辑为 [BLOCK_M, BLOCK_K]。
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=x_mask,
                other=0.0,
            )

            # packed W 的形状是 [ceil(K / 2), N]。
            # 因为两个 K 位置共享一个 uint8，所以实际读取行号是 offs_k // 2。
            packed_k = offs_k // 2
            w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            raw = tl.load(
                packed_w_ptr + packed_k[:, None] * N + offs_n[None, :],
                mask=w_mask,
                other=0x88,
            ).to(tl.int32)

            # INT4 每个权重占 4 bit：偶数 k 在低 4 bit，奇数 k 在高 4 bit。
            # 原始 WSL demo 使用 q_u = q_signed + 8，因此 unsigned code 8 表示 signed 0。
            # Triton 只对当前 tile 做局部解包和 scale 还原，不生成完整 W_dequant。
            low = raw & 0x0F
            high = (raw >> 4) & 0x0F
            is_odd = (offs_k & 1)[:, None]
            code = tl.where(is_odd == 0, low, high)
            signed = code.to(tl.int16) - 8
            w_dequant = (signed.to(tl.float32) * scale).to(tl.float16)

            # 解包后的权重片段立即参与 dot，避免中间矩阵写回显存。
            acc += tl.dot(x, w_dequant, out_dtype=tl.float32)

        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            y_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(tl.float16),
            mask=y_mask,
        )


def triton_int4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_shape: Tuple[int, int],
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """运行 Triton fused INT4 Linear。

    Args:
        x: FP16 激活矩阵，形状为 ``[M, K]``。
        packed_weight: packed INT4 权重，形状为 ``[ceil(K / 2), N]``。
        scale: 标量反量化 scale。
        original_shape: dense 权重原始形状 ``(K, N)``。
        block_m: M 维 tile 大小。
        block_n: N 维 tile 大小。
        block_k: K 维 tile 大小。

    Returns:
        FP16 输出矩阵，形状为 ``[M, N]``。
    """
    if triton is None:
        raise RuntimeError("Triton is not available in this environment.")
    if not x.is_cuda:
        raise RuntimeError("triton_int4_linear requires CUDA tensors.")

    k_size, n_size = original_shape
    m_size = x.shape[0]
    y = torch.empty((m_size, n_size), device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    _int4_matmul_kernel[grid](
        x,
        packed_weight,
        scale,
        y,
        m_size,
        k_size,
        n_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return y
