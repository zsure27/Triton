# Demo2：torch.compile 与手写 Triton 对比

本实验把 `torch.compile` 作为自动图优化基线，用来观察它能否把 naive INT4 的 PyTorch 图优化到接近手写 Triton 的水平。

比较路线：

- PyTorch eager naive INT4：不编译，直接执行 unpack、dequant 和 matmul。
- `torch.compile default`：默认编译模式。
- `torch.compile reduce-overhead`：偏向减少 Python/调度开销的编译模式。
- manual Triton fused INT4：手写 packed INT4 fused kernel。

Demo2 会分别记录：

- `first_call_ms`：首次调用耗时，包含 `torch.compile` 编译或 Triton JIT 的一次性开销。
- `steady_ms`：稳定运行耗时，使用 `torch.cuda.Event` 统计。

结论要点：`torch.compile` 适合作为自动优化基线，但 packed INT4 这类格式需要明确控制字节读取、低/高 4 bit 解包、scale 应用和 matmul 融合，因此手写 Triton 更适合作为低比特自定义算子的性能验证工具。
