"""Run demos that are supported by the current Python environment."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def can_import(module: str) -> bool:
    """Return whether a module can be imported without raising.

    Args:
        module: Dotted module name.

    Returns:
        ``True`` when import succeeds.
    """
    return importlib.util.find_spec(module) is not None


def has_cuda() -> bool:
    """Return whether PyTorch CUDA appears available.

    Returns:
        ``True`` when torch imports and ``torch.cuda.is_available()`` is true.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception as exc:
        print(f"PyTorch unavailable, skip CUDA demos: {exc}")
        return False


def run_demo(module: str, output_name: str) -> None:
    """Run one demo module and keep going if it fails.

    Args:
        module: Python module path.
        output_name: CSV file name.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        module,
        "--output-dir",
        "results",
        "--output-name",
        output_name,
    ]
    print("\nRunning:", " ".join(command))
    try:
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Demo failed but run_all will continue: {module}: {exc}")


def main() -> None:
    """Run all demos supported by the current environment."""
    if not has_cuda():
        print("CUDA is unavailable; no GPU benchmark demo will run.")
        return

    run_demo(
        "triton_lowbit_bench.demos.demo1_int4_three_routes",
        "demo1_int4_results.csv",
    )
    run_demo(
        "triton_lowbit_bench.demos.demo2_compile_vs_triton",
        "demo2_compile_results.csv",
    )

    if can_import("triton.tools.mxfp"):
        run_demo(
            "triton_lowbit_bench.demos.demo3_pure_mxfp4_linear",
            "demo3_mxfp4_results.csv",
        )
    else:
        print("triton.tools.mxfp unavailable; skip Demo3.")


if __name__ == "__main__":
    main()
