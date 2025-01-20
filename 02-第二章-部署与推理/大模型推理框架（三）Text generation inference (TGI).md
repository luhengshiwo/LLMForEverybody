`Text Generation Inference（TGI）`[1](#refer-anchor-1)是一个由Hugging Face开发的用于部署和提供大型语言模型（LLMs）的框架。它是一个生产级别的工具包，专门设计用于在本地机器上以服务的形式运行大型语言模型。TGI使用Rust和Python编写，提供了一个端点来调用模型，使得文本生成任务更加高效和灵活.

![alt text](<assest/大模型推理框架（三）Text generation inference (TGI)/0.png>)

## 1. 加速推理技术

1. Tensor Parallelism

张量并行使用了矩阵乘法可以并行计算的特性，将模型的参数划分为多个部分，每个部分在不同的设备上进行计算，最后将结果进行汇总。下面，我们分别看FFN和Self-Attention的张量并行实现。

`MLP`的主要构建块都是完全连接的 nn.Linear，后跟非线性激活 GeLU。

按照 Megatron的论文符号，我们可以将其点积部分写为 Y = GeLU(XA)，其中 X 和 Y 是输入和输出向量，A 是权重矩阵。

如果我们以矩阵形式查看计算，很容易看出矩阵乘法如何在多个 GPU 之间拆分：

![alt text](<assest/大模型推理框架（三）Text generation inference (TGI)/1.png>)

如果我们将权重矩阵 A 按列拆分到 N 个 GPU 上，并行执行矩阵乘法 XA_1 到 XA_n，那么我们将得到 N 个输出向量 Y_1、Y_2、...、Y_n，这些向量可以独立输入到 GeLU 中：

![alt text](<assest/大模型推理框架（三）Text generation inference (TGI)/2.png>)

利用这一原理，我们可以更新任意深度的 MLP，而无需 GPU 之间进行任何同步，直到最后，我们才需要重建输出向量。

Megatron-LM 论文作者为此提供了一个有用的例子：

![alt text](<assest/大模型推理框架（三）Text generation inference (TGI)/3.png>)

`Self-Attention` 的张量并行更简单，因为self-attention天然的是多头注意力机制，可以将每个头的计算分配到不同的 GPU 上。

![alt text](<assest/大模型推理框架（三）Text generation inference (TGI)/4.png>)

在上图中，我们可以用2个GPU并行的计算self-attention，其中每个GPU计算一个头的注意力机制。那原则上，有几个头就可以用几个GPU并行计算。

2. Continuous batching

在传统的批处理方法中，一批请求必须全部完成处理后才能一起返回结果。这就意味着较短请求需要等待较长请求处理完成，导致了GPU资源的浪费和推理延迟的增加。而Continuous Batching技术允许模型在处理完当前迭代后，如果有请求已经处理完成，则可以立即返回该请求的结果，而不需要等待整个批次的请求都处理完成，这样可以显著提高硬件资源的利用率并减少空闲时间。

![alt text](<assest/大模型推理框架（三）Text generation inference (TGI)/5.png>)

此外，Continuous Batching还能够解决不同请求计算量不同导致的资源浪费问题，通过迭代级别的调度动态调整批处理大小，适应不同请求的复杂程度，有效降低高复杂度请求的等待时间。

值得注意的是，实现Continuous Batching需要考虑一些关键问题，如对Early-finished Requests、Late-joining Requests的处理，以及如何处理不同长度的请求Batching。OCRA提出的两个设计思路：Iteration-level Batching和Selective Batching，就是为了解决这些问题。

3. Flash Attention

Flash Attention 是一种新型的注意力机制算法，由斯坦福大学和纽约州立大学布法罗分校的科研团队共同开发，旨在解决传统 Transformer 模型在处理长序列数据时面临的时间和内存复杂度高的问题。该算法的核心思想是减少 GPU 高带宽内存（HBM）和 GPU 片上 SRAM 之间的内存读写次数，通过分块计算（tiling）和重计算（recomputation）技术，显著降低了对 HBM 的访问频率，从而提升了运行速度并减少了内存使用。

![alt text](<assest/大模型推理框架（三）Text generation inference (TGI)/6.png>)

Flash Attention 通过 IO 感知的设计理念，优化了内存访问模式，使得 Transformer 模型在长序列处理上更加高效，为构建更长上下文的高质量模型提供了可能。

4. PagedAttention

vLLM引入了 PagedAttention，这是一种注意力算法，其灵感来自操作系统中虚拟内存和分页的经典思想。与传统的注意力算法不同，PagedAttention 允许在非连续的内存空间中存储连续的键和值。具体来说，PagedAttention 将每个序列的 KV 缓存划分为块，每个块包含固定数量 token 的键和值。在注意力计算过程中，PagedAttention 内核会高效地识别和获取这些块。

![alt text](<assest/大模型推理框架（三）Text generation inference (TGI)/7.gif>)

## 2. TGI的特性

1. **简单的启动器**：TGI提供了一个简单的启动器，可以轻松服务最流行的LLMs。

2. **生产就绪**：TGI集成了分布式追踪（使用Open Telemetry）和Prometheus指标，满足生产环境的需求。

3. **张量并行**：通过在多个GPU上进行张量并行计算，TGI能够显著加快推理速度。

4. **令牌流式传输**：使用服务器发送事件（SSE）实现令牌的流式传输。

5. **连续批处理**：对传入请求进行连续批处理，提高总体吞吐量。

6. **优化的推理代码**：针对最流行的架构，TGI使用Flash Attention和Paged Attention等技术优化了Transformers代码。

7. **多种量化支持**：支持bitsandbytes、GPT-Q、EETQ、AWQ、Marlin和fp8等多种量化方法。

8. **安全加载权重**：使用Safetensors进行权重加载，提高安全性。

9. **水印技术**：集成了"A Watermark for Large Language Models"的水印技术。

10. **灵活的生成控制**：支持logits warper（温度缩放、top-p、top-k、重复惩罚等）、停止序列和对数概率输出。

11. **推测生成**：实现了约2倍的延迟优化。

12. **引导/JSON输出**：支持指定输出格式，加速推理并确保输出符合特定规范。

13. **自定义提示生成**：通过提供自定义提示来指导模型的输出，从而轻松生成文本。

14. **微调支持**：利用微调模型执行特定任务，以实现更高的精度和性能。

15. **硬件支持**：除了NVIDIA GPU，TGI还支持AMD GPU、Intel GPU、Inferentia、Gaudi和Google TPU等多种硬件。

## 参考

<div id="refer-anchor-1"></div>

[1] [text-generation-inference](https://huggingface.co/docs/text-generation-inference/index)

[2] [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

## 欢迎关注我的GitHub和微信公众号[真-忒修斯之船]，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！