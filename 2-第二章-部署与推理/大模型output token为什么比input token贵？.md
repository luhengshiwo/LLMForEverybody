## 导入

近年来，许多商业大模型的价格不断下降，但你是否注意到一个关键细节：output token 的价格通常比 input token 高出几倍。这背后究竟是什么原因呢？在这篇博客中，我们将深入探讨大模型生成 token 的机制，并揭示为什么 output token 的成本更高。

![请添加图片描述](https://i-blog.csdnimg.cn/direct/36772da353dd4d12b04a177dd908d864.png)

## 大模型的输出机制
在 KV cache（键值缓存）的支持下，大模型生成 token 分为两个阶段：

### 预填充（prefill）
在预填充阶段，模型会并行处理输入的 prompt（即 input token），生成 KV cache。这一步骤包括一次完整的前向传播（forward），并输出第一个 token。这个过程的时间主要由 input token 决定，因为它们需要进行一次全面的计算来初始化整个生成过程。

### 解码（decoding）
解码阶段是逐个生成下一个 token 的过程。在这一步中，output token 的数量决定了需要进行多少次前向传播。虽然每次前向传播由于 KV cache 的存在而更快，但这仍然需要模型多次计算，逐步生成每一个后续的 token。

## 影响因素
大模型的生成过程受多种因素影响，这些因素不仅决定了生成 token 的效率，还影响了成本：

### Input Token
1. **预填充时间**：input token 决定了预填充阶段的时间，即第一个 token 生成的时间，这个过程只计算一次。
2. **显存占用下限**：input token 决定了 KV cache 占用显存的最低限度，因为它们需要为后续的 token 生成提供基础。
3. **生成速度基线**：input token 还决定了从第二个 token 开始生成的速度基线，影响后续的计算效率。

### Output Token
1. **解码时间长度**：output token 决定了解码阶段的时间长度。由于 KV cache 的存在，每次计算比第一个 token 快得多，但仍需多次前向传播。
2. **显存占用上限**：output token 决定了 KV cache 占用显存的最高限度，因为每生成一个新的 token，缓存中存储的信息量也在增加。

## 哪个开销更大？
一次完整的前向传播(input token数量)与多次利用 KV cache 的前向传播(output token数量)，哪个开销更大呢？

![请添加图片描述](https://i-blog.csdnimg.cn/direct/e76239e525c743d091994be740f631e5.png)

### 计算与通信瓶颈
尽管 KV cache 能有效减少每次计算的量，但由于通信带宽的更新速度未能跟上计算能力的提升，显卡对 I/O 的敏感度更高。每个 output token 的生成仍需多次前向传播，加之显卡 I/O 速度的限制，使得每个 output token 的开销更大。这也解释了为什么 output token 的价格比 input token 高。

## 结语
小知识+1，可以和朋友们吹牛了！

## 参考
- [1] [GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)
