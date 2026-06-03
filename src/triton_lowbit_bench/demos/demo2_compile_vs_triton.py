"""Demo2：torch.compile 自动优化与手写 Triton INT4 对比。"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

from triton_lowbit_bench.common.benchmark import measure_first_call_and_steady
from triton_lowbit_bench.common.env import print_environment
from triton_lowbit_bench.common.io_utils import (
    default_output_dir,
    parse_cases,
    plot_latency_bars,
    save_csv,
)
from triton_lowbit_bench.common.metrics import (
    compute_error_metrics,
    tensor_finite_status,
)
from triton_lowbit_bench.demos.demo1_int4_three_routes import (
    fp16_linear,
    naive_int4_linear,
    triton_int4_linear,
)
from triton_lowbit_bench.quantization.int4 import make_int4_weight


def _compile_function(
    func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    mode: str,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """使用 ``torch.compile`` 编译 naive INT4 函数。

    Args:
        func: Function taking ``x``, ``packed_weight`` and ``scale``.
        mode: ``torch.compile`` mode, for example ``default`` or
            ``reduce-overhead``.

    Returns:
        Compiled callable.
    """
    # 使用 importlib 导入 torch._dynamo，避免在函数内部直接 import 导致局部变量解析问题。
    dynamo = importlib.import_module("torch._dynamo")
    dynamo.reset()
    return torch.compile(func, mode=mode)


def _row(
    route: str,
    first_call_ms: float,
    steady_ms: float,
    output: torch.Tensor,
    reference: torch.Tensor,
    extra: Dict[str, object],
) -> Dict[str, object]:
    """Build a Demo2 CSV row.

    Args:
        route: Route name.
        first_call_ms: First call latency, including compile/JIT wall time.
        steady_ms: Steady-state CUDA Event latency.
        output: Route output shaped ``[M, N]``.
        reference: Reference output shaped ``[M, N]``.
        extra: Additional metadata.

    Returns:
        Result row.
    """
    row: Dict[str, object] = {
        "route": route,
        "first_call_ms": first_call_ms,
        "steady_ms": steady_ms,
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
) -> List[Dict[str, object]]:
    """Run one Demo2 benchmark case.

    Args:
        m_size: Batch/token dimension ``M``.
        k_size: Input feature dimension ``K``.
        n_size: Output feature dimension ``N``.
        warmup: Warmup iterations.
        repeat: Timed repetitions.
        seed: Random seed.

    Returns:
        Result rows for FP16, eager naive, compiled naive and manual Triton.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn((m_size, k_size), device="cuda", dtype=torch.float16)
    weight_fp16 = torch.randn((k_size, n_size), device="cuda", dtype=torch.float16)
    qpack = make_int4_weight(weight_fp16)
    packed = qpack["packed_weight"]
    scale = qpack["scale"]
    original_shape = qpack["original_shape"]

    def naive_bound(
        x_arg: torch.Tensor,
        packed_arg: torch.Tensor,
        scale_arg: torch.Tensor,
    ) -> torch.Tensor:
        """Naive INT4 function with original shape captured."""
        return naive_int4_linear(x_arg, packed_arg, scale_arg, original_shape)

    eager_func = lambda: naive_bound(x, packed, scale)
    fp16_func = lambda: fp16_linear(x, weight_fp16)
    triton_func = lambda: triton_int4_linear(x, packed, scale, original_shape)

    y_naive = eager_func()
    y_fp16 = fp16_func()
    y_triton = triton_func()

    fp16_first, fp16_steady = measure_first_call_and_steady(
        fp16_func,
        warmup=warmup,
        repeat=repeat,
    )
    eager_first, eager_steady = measure_first_call_and_steady(
        eager_func,
        warmup=warmup,
        repeat=repeat,
    )

    rows: List[Dict[str, object]] = []
    common = {
        "M": m_size,
        "K": k_size,
        "N": n_size,
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
    }
    rows.append(_row("fp16_reference", fp16_first, fp16_steady, y_fp16, y_fp16, common))
    rows.append(_row("eager_naive_int4", eager_first, eager_steady, y_naive, y_naive, common))

    for mode in ("default", "reduce-overhead"):
        try:
            compiled = _compile_function(naive_bound, mode=mode)
            compiled_func = lambda compiled=compiled: compiled(x, packed, scale)
            first_ms, steady_ms = measure_first_call_and_steady(
                compiled_func,
                warmup=warmup,
                repeat=repeat,
            )
            y_compiled = compiled_func()
            rows.append(
                _row(
                    f"torch_compile_{mode}",
                    first_ms,
                    steady_ms,
                    y_compiled,
                    y_naive,
                    common,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "route": f"torch_compile_{mode}",
                    "first_call_ms": "",
                    "steady_ms": "",
                    "max_abs_error": "",
                    "mean_abs_error": "",
                    "relative_l2_error": "",
                    "cosine_similarity": "",
                    "has_nan": "",
                    "has_inf": "",
                    "status": f"skipped: {exc}",
                    **common,
                }
            )

    triton_first, triton_steady = measure_first_call_and_steady(
        triton_func,
        warmup=warmup,
        repeat=repeat,
    )
    rows.append(
        _row(
            "manual_triton_fused_int4",
            triton_first,
            triton_steady,
            y_triton,
            y_naive,
            common,
        )
    )

    for row in rows:
        steady = row.get("steady_ms")
        if isinstance(steady, (float, int)) and steady:
            row["speedup_vs_eager"] = eager_steady / float(steady)
            row["speedup_vs_fp16"] = fp16_steady / float(steady)
        else:
            row["speedup_vs_eager"] = ""
            row["speedup_vs_fp16"] = ""
        if row["route"] in {"fp16_reference", "eager_naive_int4"}:
            ref = y_naive if row["route"] == "eager_naive_int4" else y_fp16
            metrics = compute_error_metrics(ref, y_naive)
        else:
            metrics = {
                "relative_l2_error": row.get("relative_l2_error", ""),
                "cosine_similarity": row.get("cosine_similarity", ""),
            }
        row["relerr_vs_naive"] = metrics["relative_l2_error"]
        row["cossim_vs_naive"] = metrics["cosine_similarity"]

    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Demo2.

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
        default="demo2_compile_results.csv",
    )
    return parser


def main() -> None:
    """Run Demo2 from the command line."""
    args = build_parser().parse_args()
    print_environment("Demo2 torch.compile vs Triton")
    if not torch.cuda.is_available():
        print("CUDA unavailable; Demo2 requires a CUDA GPU. Skip.")
        return

    rows: List[Dict[str, object]] = []
    for case in parse_cases(args.m, args.k, args.n):
        print(f"Running Demo2 case M={case[0]}, K={case[1]}, N={case[2]}")
        rows.extend(run_case(*case, args.warmup, args.repeat, args.seed))
        torch.cuda.empty_cache()

    output_path = args.output_dir / args.output_name
    save_csv(rows, output_path)
    plot_latency_bars(
        rows,
        ["steady_ms"],
        args.figure_dir / "demo2_compile_latency.png",
        "Demo2 steady-state latency",
    )


if __name__ == "__main__":
    main()



