# Demo3：MXFP4 软件融合说明

Demo3 使用 `triton.tools.mxfp` 中的 `MXFP4Tensor` 和 `MXScaleTensor` 构造纯 MXFP4 权重，用软件方式验证 FP4 解码、block scale 应用和矩阵乘融合的性能下限。

比较路线：

- FP16 reference matmul：使用提前反量化后的 FP16 权重做参考矩阵乘。
- Pre-dequant MXFP4：提前得到 `W_dequant_fp16`，benchmark 时只测 matmul。
- Full naive MXFP4：每次调用都执行 FP4 解码、scale broadcast、反量化和 matmul。
- Triton MXFP-like fused：在 Triton kernel 内局部解码 FP4、应用 block scale 并累加。

默认设置 `fixed_scale=True`，也就是 `scale_code=127`，对应 scale=1。这样可以避免随机 scale 过大导致 FP16 路径出现 Inf/NaN。如果改成随机 scale，需要把它当成数值压力测试，而不是默认正确性验证路径。

硬件边界必须明确：RTX 3060 Laptop GPU 的 capability 是 8.6，不支持 NVIDIA Blackwell 原生 block-scaled Tensor Core path。因此本 demo 不能被表述为“原生 MXFP4 硬件加速”，只能表述为 MXFP-like 软件融合下限验证。
