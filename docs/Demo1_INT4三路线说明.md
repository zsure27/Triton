# Demo1：INT4 Linear 三路线说明

本实验比较同一个 Linear 算子在三种实现方式下的性能和误差：

- 路线 A：PyTorch FP16 Linear，作为常规稠密矩阵乘法参考。
- 路线 B：PyTorch naive INT4，流程是 `packed W -> unpack -> dequant -> torch.matmul`。
- 路线 C：Triton fused INT4，在 kernel 内局部解包、乘 scale，并直接参与矩阵乘。

naive INT4 的关键问题是：虽然权重以 INT4 存储节省显存，但运行时会显式生成完整 `W_dequant[K, N]`。这一步会产生额外的显存写回和读写带宽，所以低比特存储并不等价于自动加速。

Triton fused INT4 的优势是把解包、scale 还原和矩阵乘融合在同一个 kernel 内。kernel 只解码当前 tile 需要的权重片段，不把完整 `W_dequant` 写回显存，因此能显著降低中间张量开销。

已有实验结论：Triton fused INT4 相对 PyTorch naive INT4 加速约 6.55x 到 25.24x。
