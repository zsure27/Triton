"""Demo1: INT4 Linear three-route performance validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from triton_lowbit_bench.common.benchmark import benchmark_cuda
from triton_lowbit_bench.common.env import print_environment
from triton_lowbit_bench.common.io_utils import (
    default_output_dir,
    parse_cases,
    plot_latency_bars,
    save_csv,
)
from triton_lowbit_bench.common.memory import (
    avoided_dequant_memory_mb,
    tensor_memory_mb,
)
from triton_lowbit_bench.common.metrics import (
    compute_error_metrics,
    tensor_finite_status,
)
from triton_lowbit_bench.kernels.int4_matmul import triton_int4_linear as _triton
from triton_lowbit_bench.quantization.int4 import (
    make_int4_weight,
    unpack_dequant_from_packed,
)


def fp16_linear(x: torch.Tensor, weight_fp16: torch.Tensor) -> torch.Tensor:
    """路线 A：运行标准 PyTorch FP16 Linear。

    Args:
        x: FP16 activation shaped ``[M, K]``.
        weight_fp16: Dense FP16 weight shaped ``[K, N]``.

    Returns:
        FP16 output shaped ``[M, N]``.
    """
    return x @ weight_fp16


def naive_int4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_shape: Tuple[int, int],
) -> torch.Tensor:
    """路线 B：运行 PyTorch naive INT4 Linear。

    Args:
        x: FP16 activation shaped ``[M, K]``.
        packed_weight: Packed INT4 weight shaped ``[ceil(K / 2), N]``.
        scale: Scalar INT4 dequant scale.
        original_shape: Dense weight shape ``(K, N)``.

    Returns:
        FP16 output shaped ``[M, N]``.
    """
    # naive 路线会完整执行 packed W -> unpack -> dequant -> torch.matmul。
    # 这里显式生成 W_dequant，用来展示低比特存储不会自动带来加速。
    weight_dequant = unpack_dequant_from_packed(
        packed_weight,
        scale,
        original_shape,
    )
    return x @ weight_dequant


def triton_int4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_shape: Tuple[int, int],
) -> torch.Tensor:
    """路线 C：运行 Triton fused INT4 Linear。

    Args:
        x: FP16 activation shaped ``[M, K]``.
        packed_weight: Packed INT4 weight shaped ``[ceil(K / 2), N]``.
        scale: Scalar INT4 dequant scale.
        original_shape: Dense weight shape ``(K, N)``.

    Returns:
        FP16 output shaped ``[M, N]``.
    """
    return _triton(x, packed_weight, scale, original_shape)


def _route_row(
    route: str,
    latency_ms: float,
    output: torch.Tensor,
    reference: torch.Tensor,
    extra: Dict[str, object],
) -> Dict[str, object]:
    """Build a CSV row with correctness and finite-value metrics.

    Args:
        route: Route name.
        latency_ms: Average latency in milliseconds.
        output: Route output shaped ``[M, N]``.
        reference: Reference output shaped ``[M, N]``.
        extra: Additional metadata.

    Returns:
        Result dictionary for one route.
    """
    metrics = compute_error_metrics(output, reference)
    finite = tensor_finite_status(output)
    row: Dict[str, object] = {
        "route": route,
        "latency_ms": latency_ms,
        **metrics,
        **finite,
    }
    row.update(extra)
    return row


def run_case(
    m_size: int,
    k_size: int,
    n_size: int,
    warmup: int,
    repeat: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Run one Demo1 matrix-size case.

    Args:
        m_size: Batch/token dimension ``M``.
        k_size: Input feature dimension ``K``.
        n_size: Output feature dimension ``N``.
        warmup: Warmup iterations.
        repeat: Timed repetitions.
        seed: Random seed.

    Returns:
        Three route result rows.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn((m_size, k_size), device="cuda", dtype=torch.float16)
    weight_fp16 = torch.randn((k_size, n_size), device="cuda", dtype=torch.float16)
    qpack = make_int4_weight(weight_fp16)

    y_fp16 = fp16_linear(x, weight_fp16)
    y_naive = naive_int4_linear(
        x,
        qpack["packed_weight"],
        qpack["scale"],
        qpack["original_shape"],
    )
    y_triton = triton_int4_linear(
        x,
        qpack["packed_weight"],
        qpack["scale"],
        qpack["original_shape"],
    )

    fp16_ms = benchmark_cuda(
        lambda: fp16_linear(x, weight_fp16),
        warmup=warmup,
        repeat=repeat,
    )
    naive_ms = benchmark_cuda(
        lambda: naive_int4_linear(
            x,
            qpack["packed_weight"],
            qpack["scale"],
            qpack["original_shape"],
        ),
        warmup=warmup,
        repeat=repeat,
    )
    triton_ms = benchmark_cuda(
        lambda: triton_int4_linear(
            x,
            qpack["packed_weight"],
            qpack["scale"],
            qpack["original_shape"],
        ),
        warmup=warmup,
        repeat=repeat,
    )

    common = {
        "M": m_size,
        "K": k_size,
        "N": n_size,
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "fp16_weight_mb": tensor_memory_mb(weight_fp16),
        "packed_int4_weight_mb": tensor_memory_mb(qpack["packed_weight"]),
        "avoided_w_dequant_mb": avoided_dequant_memory_mb(
            k_size,
            n_size,
            torch.float16,
        ),
    }
    rows = [
        _route_row("pytorch_fp16_linear", fp16_ms, y_fp16, y_fp16, common),
        _route_row("pytorch_naive_int4", naive_ms, y_naive, y_fp16, common),
        _route_row("triton_fused_int4", triton_ms, y_triton, y_fp16, common),
    ]
    triton_vs_naive = compute_error_metrics(y_triton, y_naive)
    rows[-1].update(
        {
            "speedup_vs_naive_int4": naive_ms / triton_ms,
            "speedup_vs_fp16": fp16_ms / triton_ms,
            "relerr_vs_naive": triton_vs_naive["relative_l2_error"],
            "cossim_vs_naive": triton_vs_naive["cosine_similarity"],
        }
    )
    rows[1].update(
        {
            "speedup_vs_naive_int4": 1.0,
            "speedup_vs_fp16": fp16_ms / naive_ms,
            "relerr_vs_naive": 0.0,
            "cossim_vs_naive": 1.0,
        }
    )
    rows[0].update(
        {
            "speedup_vs_naive_int4": naive_ms / fp16_ms,
            "speedup_vs_fp16": 1.0,
            "relerr_vs_naive": "",
            "cossim_vs_naive": "",
        }
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Demo1.

    Returns:
        Configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-name",
        default="demo1_int4_results.csv",
    )
    return parser


def main() -> None:
    """Run Demo1 from the command line."""
    args = build_parser().parse_args()
    print_environment("Demo1 INT4 three routes")
    if not torch.cuda.is_available():
        print("CUDA unavailable; Demo1 requires a CUDA GPU. Skip.")
        return

    rows: List[Dict[str, object]] = []
    for case in parse_cases(args.m, args.k, args.n):
        print(f"Running Demo1 case M={case[0]}, K={case[1]}, N={case[2]}")
        rows.extend(run_case(*case, args.warmup, args.repeat, args.seed))
        torch.cuda.empty_cache()

    output_path = args.output_dir / args.output_name
    save_csv(rows, output_path)
    plot_latency_bars(
        rows,
        ["latency_ms"],
        args.figure_dir / "demo1_int4_latency.png",
        "Demo1 INT4 route latency",
    )


if __name__ == "__main__":
    main()


