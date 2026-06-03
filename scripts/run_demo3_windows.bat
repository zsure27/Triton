@echo off
setlocal

cd /d "%~dp0\.."

echo 请确认 Windows Conda 环境 triton_mxfp 可用。
call conda activate triton_mxfp

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
python -m triton_lowbit_bench.demos.demo3_pure_mxfp4_linear ^
  --output-name demo3_mxfp4_results.csv

endlocal
