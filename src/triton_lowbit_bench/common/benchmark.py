"""CUDA benchmark helpers based on torch.cuda.Event."""

from __future__ import annotations

import time
from typing import Callable, Tuple

import torch


def benchmark_cuda(
    func: Callable[[], torch.Tensor],
    warmup: int = 20,
    repeat: int = 100,
) -> float:
    """Measure steady-state CUDA latency with ``torch.cuda.Event``.

    Args:
        func: Zero-argument callable returning a CUDA tensor.
        warmup: Number of warmup calls before timing.
        repeat: Number of timed repetitions.

    Returns:
        Average latency in milliseconds.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for torch.cuda.Event benchmark.")

    with torch.no_grad():
        for _ in range(warmup):
            _ = func()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(repeat):
            _ = func()
        end_event.record()
        torch.cuda.synchronize()

    return float(start_event.elapsed_time(end_event) / max(repeat, 1))


def measure_first_call_and_steady(
    func: Callable[[], torch.Tensor],
    warmup: int = 20,
    repeat: int = 100,
) -> Tuple[float, float]:
    """Measure first-call wall time and steady CUDA latency.

    Args:
        func: Zero-argument callable returning a CUDA tensor.
        warmup: Number of warmup calls used for steady-state measurement.
        repeat: Number of timed repetitions for steady-state measurement.

    Returns:
        ``(first_call_ms, steady_ms)``. ``first_call_ms`` intentionally uses
        wall-clock timing because ``torch.compile`` spends CPU time compiling
        before GPU kernels are launched; steady latency uses CUDA events.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark.")

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = func()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - start) * 1000.0
    steady_ms = benchmark_cuda(func, warmup=warmup, repeat=repeat)
    return float(first_call_ms), float(steady_ms)
