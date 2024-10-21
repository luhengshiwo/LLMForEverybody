FlashAttention V2在减少计算量和内存访问的同时，保持了算法的精度和效率，实现了更快的Attention计算。这些优化使得V2版本在A100 GPU上前向传播的速度提升了大约2倍，达到了理论计算峰值的50%-73%。

## 1. FlashAttention v1

> $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

FlashAttention不需要在全局内存上实现 X 和 A 矩阵，而是将上述公式中的整个计算融合到单个 CUDA 内核中。这要求我们设计一种算法来仔细管理片上内存(on-chip memory)（如流算法），因为 NVIDIA GPU 的共享内存(SRAM)很小。

对于矩阵乘法等经典算法，使用平铺(tiling)来确保片上内存不超过硬件限制。这种平铺方法是有效的原因是：加法是关联的，允许将整个矩阵乘法分解为许多平铺矩阵乘法的总和。

![alt text](<assest/FlashAttention v2相比于v1有哪些更新？/0.png>)

## 2. FlashAttention v2的更新

![alt text](<assest/FlashAttention v2相比于v1有哪些更新？/1.png>)

- 减少了non-matmul FLOPs的数量（消除了原先频繁rescale）。虽然non-matmul FLOPs仅占总FLOPs的一小部分，但它们的执行时间较长，这是因为GPU有专用的矩阵乘法计算单元，其吞吐量高达非矩阵乘法吞吐量的16倍。因此，减少non-matmul FLOPs并尽可能多地执行matmul FLOPs非常重要。

- 提出了在序列长度维度上并行化。该方法在输入序列很长（此时batch size通常很小）的情况下增加了GPU利用率。即使对于单个head，也在不同的thread block之间进行并行计算。

- 在一个attention计算块内，将工作分配在一个thread block的不同warp上，以减少通信和共享内存读/写。

## 3. 第一个更新详解

减少了non-matmul FLOPs的数量（消除了原先频繁rescale）。虽然non-matmul FLOPs仅占总FLOPs的一小部分，但它们的执行时间较长，这是因为GPU有专用的矩阵乘法计算单元，其吞吐量高达非矩阵乘法吞吐量的16倍。因此，减少non-matmul FLOPs并尽可能多地执行matmul FLOPs非常重要。

### non-matmul vs. GEMM

***non matrix multiply***

非矩阵乘法：指的是在矩阵乘法之外的操作，如加法、乘法、除法等, 例如`rescale`操作，使用的是GPU的通用硬件：CUDA Core.

***General Matrix Multiply***

通用矩阵乘法：指的是矩阵乘法的一种实现方式，使用的是GPU的专用硬件：Tensor Core.

### CUDA Core vs. Tensor Core

非矩阵乘法操作的执行时间较长，因为GPU有专用的矩阵乘法计算单元Tensor Core，其吞吐量高达非矩阵乘法吞吐量的16倍。
下图是A100的介绍，可以看到CUDA Core单精度TF16的计算能力是19.5TFLOPS，而Tensor Core的计算能力是312TFLOPS，相差16倍之多。

![alt text](<assest/FlashAttention v2相比于v1有哪些更新？/3.png>)

### Rescale操作

![alt text](<assest/FlashAttention v2相比于v1有哪些更新？/2.png>)

V2版本调整了算法，减少了非矩阵乘法操作的浮点运算次数，同时保持输出不变。在原始的FlashAttention（即V1版本）中，每个block的每次迭代都需要执行rescale操作，这涉及到除法运算。

而在V2中，这种rescale操作被延后到循环的最后才执行一次，每次计算可以减少一次除法运算。这样的调整是因为只要每次迭代确保分子部分被scale为正确值以及分母部分计算正确即可。这种优化减少了计算量，提高了效率


## 4. 第二个更新详解
提出了在序列长度维度上并行化。该方法在输入序列很长（此时batch size通常很小）的情况下增加了GPU利用率。即使对于单个head，也在不同的thread block之间进行并行计算。

### GPU硬件
  - Streaming Processor（SP）：是最基本的处理单元，从fermi架构开始被叫做CUDA core。
  - Streaming MultiProcessor（SM）：一个SM由多个CUDA core（SP）组成，每个SM在不同GPU架构上有不同数量的CUDA core，例如Pascal架构中一个SM有128个CUDA core。

### GPU软件
  - thread：一个CUDA并行程序由多个thread来执行，thread是最基本的执行单元（the basic unit of execution）；
  - warp：一个warp通常包含32个thread。每个warp中的thread可以同时执行相同的指令，从而实现SIMT（单指令多线程）并行。warp是SM中最小的调度单位（the smallest scheduling unit on an SM），一个SM可以同时处理多个warp；
  - thread block：一个thread block可以包含多个warp，同一个block中的thread可以同步，也可以通过shared memory进行通信。thread block是GPU执行的最小单位（the smallest unit of execution on the GPU）。一个warp中的threads必然在同一个block中，如果block所含thread数量不是warp大小的整数倍，那么多出的那个warp中会剩余一些inactive的thread。也就是说，即使warp的thread数量不足，硬件也会为warp凑足thread，只不过这些thread是inactive状态，但也会消耗SM资源。

V2 相对于 V1 的第二个主要更新是增加了序列长度维度的并行计算，这样做的目的是提高 GPU 的 SM（Streaming Multiprocessor）利用率，尤其是在处理长序列数据时。在 V1 中，计算是先按批次和头数并行执行，然后在序列长度上串行执行。这意味着当序列长度较长时，可能无法充分利用所有可用的 SM，因为每个 block 只能处理序列的一个片段。

在 V2 中，通过在序列长度维度上进行并行化，可以更有效地分配计算任务到更多的 block，从而更充分地利用 GPU 资源。具体来说，V2 通过增加 num_m_block 的概念，将 Q 矩阵在序列长度方向上进一步划分为多个小块，每一块由不同的 block 来处理。这样，每个 block 可以独立地计算它所负责的输出部分，减少了不同 block 之间的依赖和通信开销。

> 这边有点像continous batching的思路

## 5. 第三个更新详解

![alt text](<assest/FlashAttention v2相比于v1有哪些更新？/4.png>)

在V2中，通过调整循环顺序，将Q作为外循环，K和V作为内循环，每个线程块（thread block）负责计算输出矩阵O的一部分。这种设计允许每个线程块独立进行计算，减少了线程块之间的依赖和通信需求。同时，V2版本在前向传播中进一步减少了非矩阵乘法操作的浮点运算，以充分利用GPU上的专用计算单元，如Nvidia GPU上的Tensor Cores，从而最大化GPU的吞吐量。
此外，V2版本在反向传播中也进行了优化，采用了类似的分块策略来优化计算和内存访问，提高效率和性能。通过这种方式，FlashAttention V2能够实现更高的并行性，减少不必要的计算和内存访问，从而提升整体的计算性能。

## 参考

<div id="refer-anchor-1"></div>

[1] [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

[2] [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

[3] [FlashAttention2详解（性能比FlashAttention提升200%](https://cloud.tencent.com/developer/article/2353093)

[4] [GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)