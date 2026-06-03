"""Demo3：纯 MXFP4 Linear 软件部署验证。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from triton_lowbit_bench.common.benchmark import benchmark_cuda
from triton_lowbit_bench.common.env import (
    get_gpu_capability,
    has_mxfp_tools,
    print_environment,
)
from triton_lowbit_bench.common.io_utils import (
    default_output_dir,
    parse_cases,
    plot_latency_bars,
    save_csv,
)
from triton_lowbit_bench.common.memory import tensor_memory_mb
from triton_lowbit_bench.common.metrics import (
    compute_error_metrics,
    tensor_finite_status,
)
from triton_lowbit_bench.kernels.mxfp4_like_matmul import (
    triton_mxfp4_like_linear,
)
from triton_lowbit_bench.quantization.mxfp4 import (
    full_dequant_mxfp4,
    make_mxfp4_weight,
)


def fp16_linear(x: torch.Tensor, weight_fp16: torch.Tensor) -> torch.Tensor:
    """Run FP16 reference matmul.

    Args:
        x: FP16 activation shaped ``[M, K]``.
        weight_fp16: Dense FP16 weight shaped ``[K, N]``.

    Returns:
        FP16 output shaped ``[M, N]``.
    """
    return x @ weight_fp16


def predequant_mxfp4_linear(
    x: torch.Tensor,
    qpack: Dict[str, object],
) -> torch.Tensor:
    """Run route B: pre-dequant MXFP4 matmul only.

    Args:
        x: FP16 activation shaped ``[M, K]``.
        qpack: MXFP4 weight package.

    Returns:
        FP16 output shaped ``[M, N]``.
    """
    return x @ qpack["W_dequant_fp16"]


def full_naive_mxfp4_linear(
    x: torch.Tensor,
    qpack: Dict[str, object],
) -> torch.Tensor:
    """Run route C: full naive MXFP4 decode, scale and matmul.

    Args:
        x: FP16 activation shaped ``[M, K]``.
        qpack: MXFP4 weight package.

    Returns:
        FP16 output shaped ``[M, N]``.
    """
    # Full naive 路线每次都完整执行 FP4 解码、block scale 应用、
    # W_dequant 生成和 matmul，用来暴露未融合实现的真实软件开销。
    weight_dequant = full_dequant_mxfp4(qpack)
    return x @ weight_dequant


def probe_native_mxfp_support() -> str:
    """Return current native MXFP hardware support status.

    Returns:
        Text status. RTX 3060 capability=8.6 is below Blackwell-class
        capability and does not support native block-scaled Tensor Core path.
    """
    major, minor = get_gpu_capability()
    print(f"Current GPU capability={major}.{minor}")
    if major < 10:
        print(
            "RTX 3060 capability=8.6 does not support native NVIDIA "
            "Blackwell block-scaled Tensor Core path."
        )
        print(
            "Demo3 only validates MXFP-like software fused lower-bound "
            "behavior, not native MXFP4 hardware upper-bound performance."
        )
        return "not_supported_on_current_gpu_cc_less_than_10"
    return "potentially_supported_cc10_or_higher"


def _row(
    route: str,
    latency_ms: float,
    output: torch.Tensor,
    reference: torch.Tensor,
    extra: Dict[str, object],
) -> Dict[str, object]:
    """Build a Demo3 CSV row.

    Args:
        route: Route name.
        latency_ms: Average CUDA Event latency in milliseconds.
        output: Route output shaped ``[M, N]``.
        reference: Reference output shaped ``[M, N]``.
        extra: Additional metadata.

    Returns:
        Result row.
    """
    row: Dict[str, object] = {
        "route": route,
        "latency_ms": latency_ms,
        **compute_error_metrics(output, reference),
        **tensor_finite_status(output),
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
    fixed_scale: bool,
) -> List[Dict[str, object]]:
    """Run one Demo3 MXFP4 benchmark case.

    Args:
        m_size: Batch/token dimension ``M``.
        k_size: Input feature dimension ``K``.
        n_size: Output feature dimension ``N``.
        warmup: Warmup iterations.
        repeat: Timed repetitions.
        seed: Random seed.
        fixed_scale: Use scale code 127 when ``True``.

    Returns:
        Four route result rows.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn((m_size, k_size), device="cuda", dtype=torch.float16)
    qpack = make_mxfp4_weight(
        k_size,
        n_size,
        vec_size=32,
        device="cuda",
        fixed_scale=fixed_scale,
    )
    native_status = probe_native_mxfp_support()

    y_fp16 = fp16_linear(x, qpack["W_dequant_fp16"])
    y_pre = predequant_mxfp4_linear(x, qpack)
    y_full = full_naive_mxfp4_linear(x, qpack)
    y_triton = triton_mxfp4_like_linear(x, qpack)

    fp16_ms = benchmark_cuda(lambda: fp16_linear(x, qpack["W_dequant_fp16"]), warmup, repeat)
    pre_ms = benchmark_cuda(lambda: predequant_mxfp4_linear(x, qpack), warmup, repeat)
    full_ms = benchmark_cuda(lambda: full_naive_mxfp4_linear(x, qpack), warmup, repeat)
    triton_ms = benchmark_cuda(lambda: triton_mxfp4_like_linear(x, qpack), warmup, repeat)

    common = {
        "M": m_size,
        "K": k_size,
        "N": n_size,
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "fixed_scale": fixed_scale,
        "native_mxfp_status": native_status,
        "fp16_weight_mb": tensor_memory_mb(qpack["W_dequant_fp16"]),
        "packed_mxfp4_weight_mb": tensor_memory_mb(qpack["W_packed"]),
        "scale_u8_mb": tensor_memory_mb(qpack["scale_u8"]),
        "avoided_w_dequant_mb": tensor_memory_mb(qpack["W_dequant_fp16"]),
    }
    rows = [
        _row("fp16_reference_matmul", fp16_ms, y_fp16, y_fp16, common),
        _row("predequant_mxfp4_matmul", pre_ms, y_pre, y_fp16, common),
        _row("full_naive_mxfp4", full_ms, y_full, y_fp16, common),
        _row("triton_mxfp_like_fused", triton_ms, y_triton, y_full, common),
    ]
    triton_vs_full = compute_error_metrics(y_triton, y_full)
    triton_vs_pre = compute_error_metrics(y_triton, y_pre)
    for row in rows:
        latency = float(row["latency_ms"])
        row["speedup_triton_vs_full_naive"] = full_ms / triton_ms
        row["speedup_triton_vs_predequant"] = pre_ms / triton_ms
        row["time_ratio_triton_over_fp16"] = triton_ms / fp16_ms
        row["speedup_vs_fp16"] = fp16_ms / latency
        row["triton_vs_full_relerr"] = triton_vs_full["relative_l2_error"]
        row["triton_vs_full_cossim"] = triton_vs_full["cosine_similarity"]
        row["triton_vs_predequant_relerr"] = triton_vs_pre["relative_l2_error"]
        row["triton_vs_predequant_cossim"] = triton_vs_pre["cosine_similarity"]
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Demo3.

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
    parser.add_argument("--fixed-scale", action="store_true", default=True)
    parser.add_argument(
        "--random-scale",
        action="store_false",
        dest="fixed_scale",
        help="Use random scales; may overflow and produce Inf/NaN.",
    )
    parser.add_argument(
        "--output-name",
        default="demo3_mxfp4_results.csv",
    )
    return parser


def main() -> None:
    """Run Demo3 from the command line."""
    args = build_parser().parse_args()
    print_environment("Demo3 pure MXFP4 software validation")
    if not torch.cuda.is_available():
        print("CUDA unavailable; Demo3 requires a CUDA GPU. Skip.")
        return
    if not has_mxfp_tools():
        print("triton.tools.mxfp unavailable; Demo3 is skipped in this environment.")
        return

    rows: List[Dict[str, object]] = []
    for case in parse_cases(args.m, args.k, args.n):
        print(f"Running Demo3 case M={case[0]}, K={case[1]}, N={case[2]}")
        rows.extend(
            run_case(
                case[0],
                case[1],
                case[2],
                args.warmup,
                args.repeat,
                args.seed,
                args.fixed_scale,
            )
        )
        torch.cuda.empty_cache()

    output_path = args.output_dir / args.output_name
    save_csv(rows, output_path)
    plot_latency_bars(
        rows,
        ["latency_ms"],
        args.figure_dir / "demo3_mxfp4_latency.png",
        "Demo3 MXFP4 route latency",
    )


if __name__ == "__main__":
    main()



