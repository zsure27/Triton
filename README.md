[README.md](https://github.com/user-attachments/files/28547764/README.md)
# Triton 低比特量化 Linear Benchmark

## 项目简介

本项目主题为 **Triton Low-bit Quantization Benchmark：Demo1-Demo3 低比特量化算子学习与性能验证工程**。

本工程用于整理、复现和对比三个低比特 Linear 算子实验，重点验证 Triton 在低比特量化部署中的实际作用。当前项目并不是完整的 Transformer、LLM 或端到端模型部署工程，而是聚焦于 Transformer/LLM 中最核心的 Linear 层，比较低比特权重格式、PyTorch naive 实现、`torch.compile` 自动优化和手写 Triton Kernel 的性能边界。

本项目主要回答以下问题：

1. 低比特权重压缩后，是否一定能带来推理加速。
2. PyTorch naive 低比特实现为什么可能比 FP16 更慢。
3. `torch.compile` 自动优化能否替代手写 Triton Kernel。
4. Triton 在 INT4、MXFP4 等低比特格式中能否减少中间张量和显存访问。
5. 当前 RTX 3060 环境下能够验证哪些低比特部署能力，哪些能力需要更新硬件支持。

---

## 项目内容

本工程统一整理了三个实验 Demo。

### Demo1：INT4 Linear 三路线性能验证

对比三种 Linear 实现方式：

1. PyTorch FP16 Linear
2. PyTorch naive INT4
3. Triton fused INT4 Kernel

目标是验证 INT4 权重从“存储压缩”到“实际推理加速”的过程。

### Demo2：torch.compile 自动优化 vs 手写 Triton Kernel

对比四种实现方式：

1. PyTorch eager naive INT4
2. `torch.compile default`
3. `torch.compile reduce-overhead`
4. manual Triton fused INT4

目标是验证自动编译优化是否能够替代手写 Triton Kernel。

### Demo3：纯 MXFP4 Linear 软件部署验证

对比四种实现方式：

1. FP16 reference matmul
2. Pre-dequant MXFP4
3. Full naive MXFP4
4. Triton MXFP-like fused Kernel

目标是验证当前 GPU 环境下，MXFP4 数据格式和软件融合 Kernel 的部署能力边界。

---

## 项目目录结构

```text
TritonLowbitBench/
├── README.md
├── LICENSE
├── requirements_wsl_triton31.txt
├── requirements_windows_mxfp.txt
│
├── src/
│   └── triton_lowbit_bench/
│       ├── common/
│       ├── quantization/
│       ├── kernels/
│       └── demos/
│
├── scripts/
├── results/
├── docs/
└── tests/
```

---

## 主要目录说明

### 根目录文件

* `README.md`
  项目总说明，介绍实验目标、环境配置、目录结构、运行方式和结果指标含义。

* `LICENSE`
  项目许可证，采用 MIT License。

* `requirements_wsl_triton31.txt`
  Demo1 和 Demo2 在 WSL Triton 3.1 环境中的依赖参考。PyTorch CUDA wheel 需要按照 PyTorch 官方 CUDA 12.1 源安装。

* `requirements_windows_mxfp.txt`
  Demo3 在 Windows 原生 `triton_mxfp` 环境中的依赖参考。

* `.gitignore`
  用于忽略 Python 缓存、IDE 元数据、临时图表、日志文件和实验中间文件。

---

### `src/triton_lowbit_bench/common/`

该目录存放通用工具函数。

* `benchmark.py`
  统一的性能测试工具。
  其中 `benchmark_cuda()` 使用 `torch.cuda.Event` 统计稳定运行时间；`measure_first_call_and_steady()` 用于 Demo2，记录首次调用时间和稳定运行时间。

* `env.py`
  环境检测工具，用于打印 Python、PyTorch、CUDA、Triton、GPU 名称和 GPU capability。
  同时提供 `has_mxfp_tools()`，用于判断当前环境是否能导入 `triton.tools.mxfp`。

* `io_utils.py`
  结果路径、CSV 保存和可选绘图工具。
  Windows 环境优先保存到 `D:\TritonLowbitBench\results`，WSL 环境优先保存到 `/mnt/d/TritonLowbitBench/results`。

* `memory.py`
  显存估算工具，用于计算 FP16 权重、packed 低比特权重、避免生成的 `W_dequant` 等占用。

* `metrics.py`
  正确性指标工具，用于计算：

  * `max_abs_error`
  * `mean_abs_error`
  * `relative_l2_error`
  * `cosine_similarity`
  * NaN / Inf 检查结果

---

### `src/triton_lowbit_bench/quantization/`

该目录存放量化、打包、解包和反量化相关代码。

* `int4.py`
  INT4 量化、打包、解包和反量化函数。
  采用 `q_u = q_signed + 8` 的表示方式，将 signed INT4 映射到 unsigned code。

* `mxfp4.py`
  MXFP4 权重构造、FP4 code 沿 K 维打包、E2M1-like 解码、E8M0 scale 解码，以及 Full naive MXFP4 反量化辅助函数。

---

### `src/triton_lowbit_bench/kernels/`

该目录存放手写 Triton Kernel。

* `int4_matmul.py`
  Triton fused INT4 Linear Kernel。
  该 Kernel 在 GPU 内部完成 packed INT4 读取、低/高 4 bit 解包、signed INT4 恢复、scale 应用和 `tl.dot` 矩阵乘法。

* `mxfp4_like_matmul.py`
  Triton MXFP-like fused Kernel。
  该 Kernel 在 GPU 内部完成 FP4 code 解码、block scale 应用和矩阵乘法。
  需要注意，该实现属于软件融合路径，不是原生 MXFP4 Tensor Core 加速。

---

### `src/triton_lowbit_bench/demos/`

该目录存放三个 Demo 的统一入口脚本。

* `demo1_int4_three_routes.py`
  Demo1 统一入口。
  对比 PyTorch FP16 Linear、PyTorch naive INT4 和 Triton fused INT4。
  支持参数：

  * `--m`
  * `--k`
  * `--n`
  * `--warmup`
  * `--repeat`
  * `--output-dir`
  * `--seed`

* `demo2_compile_vs_triton.py`
  Demo2 统一入口。
  复用 Demo1 中的 INT4 量化函数和 Triton Kernel，对比 eager naive、`torch.compile default`、`torch.compile reduce-overhead` 和 manual Triton。
  记录：

  * `first_call_ms`
  * `steady_ms`
  * `speedup`
  * 正确性指标

* `demo3_pure_mxfp4_linear.py`
  Demo3 统一入口。
  对比 FP16 reference、Pre-dequant MXFP4、Full naive MXFP4 和 Triton MXFP-like fused。
  该 Demo 需要在 Windows 原生 `triton_mxfp` 环境中完整运行。

---

### `scripts/`

该目录存放运行脚本。

* `run_demo1_wsl.sh`
  在 WSL 环境中运行 Demo1，默认输出 `demo1_int4_three_routes_results.csv`。

* `run_demo2_wsl.sh`
  在 WSL 环境中运行 Demo2，默认输出 `demo2_compile_vs_triton_results_unified.csv`。

* `run_demo3_windows.bat`
  在 Windows 原生环境中激活 `triton_mxfp` 并运行 Demo3。

* `run_all_available.py`
  自动检测当前 Python 环境是否支持 CUDA，是否能导入 `triton.tools.mxfp`。
  程序会根据当前环境自动判断能够运行哪些 Demo。若某个 Demo 的环境不满足要求，脚本会给出提示，而不会直接崩溃退出。

---

### `results/`

该目录用于保存实验结果。

* `demo1_route_a_results.csv`
  Demo1 Route A：PyTorch FP16 Linear 结果。

* `demo1_route_b_results.csv`
  Demo1 Route B：PyTorch naive INT4 结果。

* `demo1_route_c_results.csv`
  Demo1 Route C：Triton fused INT4 结果。

* `demo1_int4_three_routes_results.csv`
  Demo1 三条路线合并后的统一结果表。

* `demo2_compile_vs_triton_results.csv`
  Demo2 原始运行结果。

* `demo2_compile_results.csv`
  Demo2 统一命名结果表。

* `demo3_pure_mxfp4_results_fixed.csv`
  Demo3 fixed scale 原始运行结果。

* `demo3_mxfp4_results.csv`
  Demo3 统一命名结果表。

---

### `docs/`

该目录存放实验说明文档。

* `Demo1_INT4三路线说明.md`
  说明 Demo1 三条路线的数据流、naive INT4 为什么慢，以及 Triton fused INT4 为什么能够减少访存开销。

* `Demo2_compile与Triton说明.md`
  说明 `torch.compile default`、`torch.compile reduce-overhead` 和手写 Triton Kernel 的区别。

* `Demo3_MXFP4软件融合说明.md`
  说明 MXFP4Tensor、MXScaleTensor、FP4 + scale、Pre-dequant、Full naive 和 Triton MXFP-like 的区别。

* `实验结论摘要.md`
  组会分享的实验结论摘要。

---

### `tests/`

该目录存放基本测试脚本。

* `test_int4_pack_unpack.py`
  检查随机 unsigned INT4 code 打包后再解包是否能够完全恢复。

* `test_binary_metrics.py`
  检查相同输出的误差是否为 0，cosine similarity 是否接近 1，并确认无 NaN / Inf。

* `test_mxfp4_reference.py`
  仅在能够导入 `triton.tools.mxfp` 且 CUDA 可用时运行。
  用于检查 MXFP4Tensor、MXScaleTensor 和 packed tensor shape 是否正常。

---

## INT4 数据表示说明

Demo1 和 Demo2 使用 INT4 权重格式。其 signed INT4 与 unsigned code 的关系如下：

```text
signed INT4:    -8 ... 7
unsigned code:   0 ... 15

量化映射:
q_u = q_signed + 8

反向恢复:
q_signed = q_u - 8
```

也就是说，unsigned code 8 表示 signed 0。

在本项目中，两个 INT4 code 会沿 K 维打包到一个 `uint8` 中：

```text
偶数 k: 放入低 4 bit
奇数 k: 放入高 4 bit
```

这种打包方式可以将 FP16 权重压缩为原来的四分之一，但如果推理时仍然完整解包并生成 `W_dequant`，则可能产生较大的中间张量和额外显存访问。

---

## 为什么使用 Triton

低比特量化首先减少的是权重存储，但权重变小并不意味着推理一定更快。

在 naive 低比特实现中，通常需要执行以下步骤：

1. 读取 packed 低比特权重。
2. 解包为 INT4 或 FP4 code。
3. 应用 scale，生成完整的 `W_dequant`。
4. 再调用普通矩阵乘法。

这种实现方式会显式生成中间张量，例如 `q_unpacked` 和 `W_dequant`，从而增加显存读写和数据搬运开销。

Triton 的价值在于：可以用 Python 编写直接运行在 GPU 上的自定义 Kernel，将低比特权重读取、解包、scale 应用和矩阵乘法融合在同一个计算过程中。这样可以避免完整 `W_dequant` 写回显存，减少中间张量和访存开销。

因此，Triton 更适合作为面向低比特新型数据格式的自定义算子开发与性能验证平台，而不是一个一键自动加速工具。

---

## Demo 总览

| Demo  | 实验目的                                 | 对比路线                                                                         | 已知结论                                                                 |
| ----- | ------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Demo1 | 验证 INT4 权重压缩能否转化为推理加速                | FP16 Linear / PyTorch naive INT4 / Triton fused INT4                         | Triton fused INT4 相对 naive INT4 加速约 6.55x-25.24x                     |
| Demo2 | 验证 `torch.compile` 自动优化能否替代手写 Triton | eager naive INT4 / compile default / compile reduce-overhead / manual Triton | `torch.compile` 有一定优化，但手写 Triton 在已有测试规模下更快                          |
| Demo3 | 验证当前 GPU 下纯 MXFP4 的软件部署能力            | FP16 / Pre-dequant / Full naive / Triton MXFP-like fused                     | Triton MXFP-like 相对 Full naive 加速约 44.5x-91.9x，但在 RTX 3060 上仍慢于 FP16 |

---

## 环境说明

### WSL 环境：Demo1 / Demo2

Demo1 和 Demo2 使用 WSL 环境运行。

环境配置如下：

* 系统：Ubuntu / WSL2
* Python：3.10.20
* PyTorch：2.5.1+cu121
* Triton：3.1.0
* CUDA：12.1
* GPU：NVIDIA GeForce RTX 3060 Laptop GPU

该环境主要用于：

1. INT4 量化实验。
2. PyTorch naive INT4 与 Triton fused INT4 对比。
3. `torch.compile` 与手写 Triton Kernel 对比。

---

### Windows 原生环境：Demo3

Demo3 使用 Windows 原生 `triton_mxfp` 环境运行。

环境配置如下：

* Python：3.11 或当前 `triton_mxfp` 环境实际版本
* PyTorch：2.9.1+cu128
* CUDA：12.8
* triton-windows：3.7.0.post26
* GPU：NVIDIA GeForce RTX 3060 Laptop GPU

该环境已验证可以正常导入：

```python
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
```

需要注意的是，RTX 3060 Laptop GPU 的 capability 为 8.6，不支持 NVIDIA Blackwell 原生 block-scaled Tensor Core path。因此 Demo3 只能表示 MXFP-like 软件融合下限，不代表原生 MXFP4 硬件加速上限。

---

## 安装参考

### WSL 环境

```bash
conda activate triton

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install triton==3.1.0 numpy matplotlib pandas pyyaml

python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
```

---

### Windows 原生环境

```bat
conda create -n triton_mxfp python=3.11 -y
conda activate triton_mxfp

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install triton-windows==3.7.0.post26 numpy matplotlib pandas pyyaml

python -c "from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor; print('mxfp ok')"
```

---

## 运行方式

### WSL 运行 Demo1 和 Demo2

```bash
bash scripts/run_demo1_wsl.sh
bash scripts/run_demo2_wsl.sh
```

---

### Windows 运行 Demo3

```bat
scripts\run_demo3_windows.bat
```

---

### Python 直接运行

在 Windows PowerShell 中运行：

```powershell
$env:PYTHONPATH="$PWD\src;$env:PYTHONPATH"

python -m triton_lowbit_bench.demos.demo1_int4_three_routes --m 1 --k 1024 --n 1024
python -m triton_lowbit_bench.demos.demo2_compile_vs_triton --m 1 --k 1024 --n 1024
python -m triton_lowbit_bench.demos.demo3_pure_mxfp4_linear --m 1 --k 1024 --n 1024
```

在 WSL 中运行时，可使用：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"

python -m triton_lowbit_bench.demos.demo1_int4_three_routes --m 1 --k 1024 --n 1024
python -m triton_lowbit_bench.demos.demo2_compile_vs_triton --m 1 --k 1024 --n 1024
```

---

## 结果保存路径

程序会根据运行环境自动保存结果：

* Windows：`D:\TritonLowbitBench\results`
* WSL：`/mnt/d/TritonLowbitBench/results`
* 其他情况：当前项目下的 `results/`

---

## 结果指标说明

* `latency_ms`
  使用 `torch.cuda.Event` 统计的稳定运行时间，单位为毫秒。

* `first_call_ms`
  Demo2 中的首次调用时间，包含 `torch.compile` 编译或 Triton JIT 开销。

* `steady_ms`
  Demo2 中排除首次编译后的稳定运行时间。

* `speedup`
  加速比，计算方式为：基线耗时 / 当前路线耗时。

* `relative_l2_error`
  相对 L2 误差，用于衡量输出结果与参考输出之间的差异。

* `cosine_similarity`
  输出向量余弦相似度。越接近 1，说明两个输出整体方向越一致。

* `has_nan` / `has_inf`
  检查输出是否出现 NaN 或 Inf，用于判断计算是否发生数值异常。

* `packed_*_weight_mb`
  低比特打包权重占用的显存大小。

* `avoided_w_dequant_mb`
  Triton fused 路线避免显式生成的完整 FP16 反量化权重大小。

---

## 已知实验结论

### Demo1 结论

Demo1 表明，单纯将权重压缩为 INT4 并不一定带来推理加速。PyTorch naive INT4 由于需要显式解包和生成完整 `W_dequant`，会产生额外中间张量和显存访问开销。

Triton fused INT4 将 INT4 解包、scale 还原和矩阵乘法融合在同一个 Kernel 内部，相比 PyTorch naive INT4 获得约 6.55x-25.24x 的加速。

---

### Demo2 结论

Demo2 表明，`torch.compile` 对 PyTorch naive INT4 流程具有一定优化效果，可以减少部分 eager 模式下的执行开销。

但是，对于 packed INT4 这种需要显式控制解包、scale 应用和矩阵乘法融合的数据格式，手写 Triton Kernel 在已有测试规模下仍然更快、更可控。因此，`torch.compile` 更适合作为自动优化基线，而不是完全替代手写低比特 Kernel。

---

### Demo3 结论

Demo3 表明，Triton MXFP-like fused 相比 Full naive MXFP4 获得约 44.5x-91.9x 的加速，说明将 FP4 解码、block scale 应用和矩阵乘法融合到同一个 Kernel 中，可以显著减少完整 `W_dequant` 的生成开销。

但在 RTX 3060 上，Triton MXFP-like 仍慢于 FP16 reference matmul。这是因为当前硬件不支持原生 MXFP4 block-scaled Tensor Core，因此该实验只代表当前硬件下 MXFP-like 软件融合下限，不代表原生 MXFP4 硬件加速上限。

---

## 当前项目边界

本项目当前处于 Linear 算子级验证阶段。

本项目不包含：

1. 完整 Transformer Block 部署。
2. 完整 LLM 推理。
3. 原生 MXFP4 Tensor Core 性能测试。
4. Squeeze10-LLM 的完整 mixed-bit 权重部署。
5. 真实模型权重加载与端到端推理。

当前实验主要服务于后续低比特网络部署研究的基础算子验证。

---

## 后续工作

后续可以从以下方向继续扩展：

1. 从 Linear 算子扩展到 Transformer Block。
2. 做 CNN 1x1 Conv 低比特替换。
3. 做 Mamba 投影层低比特替换。
4. 实现 Squeeze10-like 1-bit + 4-bit mixed-bit 自定义 Kernel。
5. 根据真实 mixed-bit 权重存储格式设计专用数据布局。
6. 在 NVIDIA Blackwell 或 AMD CDNA4 等支持原生 block-scaled matmul 的硬件上测试 MXFP4 上限。
7. 进一步比较 `torch.compile`、手写 Triton Kernel 和官方低比特算子库在不同网络结构中的适用边界。

---

## 学习资料

* Triton Paper
  https://dl.acm.org/doi/epdf/10.1145/3315508.3329973

* Triton GitHub
  https://github.com/triton-lang/triton

* Triton 中文文档
  https://triton-lang.cn/main/index.html

* Triton Block Scaled MatMul 教程
  https://triton-lang.cn/main/getting-started/tutorials/10-block-scaled-matmul.html

* LMDeploy Triton Kernels
  https://github.com/InternLM/lmdeploy/tree/main/lmdeploy/pytorch/kernels

* Triton-Windows PyPI
  https://pypi.org/project/triton-windows/

---

## 总结

本项目通过 Demo1、Demo2 和 Demo3，从 INT4 到 MXFP4 逐步验证了 Triton 在低比特量化 Linear 算子中的作用。

核心结论如下：

1. 低比特量化首先带来存储压缩，但不一定自动带来推理加速。
2. naive 低比特实现容易被解包、反量化和中间张量访存开销抵消收益。
3. Triton 可以将低比特解包、scale 应用和矩阵乘法融合到同一个 GPU Kernel 中，从而减少中间数据搬运。
4. `torch.compile` 可以作为自动优化基线，但对于 packed 低比特数据格式，手写 Triton Kernel 仍然更灵活、更可控。
5. 当前 RTX 3060 可以验证 MXFP-like 软件融合下限，但真正的 MXFP4 原生加速上限需要在支持 block-scaled matmul 的新硬件上测试。
