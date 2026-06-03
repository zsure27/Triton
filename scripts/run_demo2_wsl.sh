#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "请确认已经进入 WSL 的 Triton 3.1 环境，例如：conda activate triton"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m triton_lowbit_bench.demos.demo2_compile_vs_triton \
  --output-name demo2_compile_vs_triton_results_unified.csv
