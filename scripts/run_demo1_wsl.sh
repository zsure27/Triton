#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "请确认已经进入 WSL 的 Triton 3.1 环境，例如：conda activate triton"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m triton_lowbit_bench.demos.demo1_int4_three_routes \
  --output-name demo1_int4_three_routes_results.csv
