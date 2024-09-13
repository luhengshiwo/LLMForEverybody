vLLM: 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文翻译）

## 写在最前面
在学习vLLM和PagedAttention的过程中，我发现了很多非常优质的资料。其中最有价值应该是这篇vLLM的官方博客：https://blog.vllm.ai/2023/06/20/vllm.html

文章内容详实，我认为它写得非常好，因此尝试将其翻译成中文，希望这能帮助到更多需要相关信息的人。

由于译者水平有限，翻译过程中难免会有错误，还请大家多多包涵。如果有任何问题，欢迎在评论区指出，我会尽快修改。

## 正文开始：

LLM 有望从根本上改变我们在所有行业中使用 AI 的方式。然而，真正的要为模型搭建服务是很有挑战性的，即使在昂贵的硬件上也可能出奇地慢。今天(2023年6月20日，译者注)我们很高兴推出vLLM，这是一个用于快速LLM推理和服务的开源库。vLLM 采用了 PagedAttention，这是我们的新注意力算法，可以有效地管理注意力键keys和值values。配备 PagedAttention 的 vLLM 重新定义了 LLM 服务SOTA水平：它比 HuggingFace Transformers 的吞吐量高出 24 倍，而无需进行任何模型架构更改。

vLLM 是UC Berkeley开发的，并在过去两个月内部署在 [Chatbot Arena and Vicuna Demo](https://lmarena.ai/) 中。它是让 LLM 服务变得经济的核心技术，让即使是像 LMSYS 这样计算资源有限的小型研究团队也可以负担得起。现在只需在我们的 [GitHub 仓库](https://github.com/vllm-project/vllm)中使用一个命令即可试用 vLLM。

## 超越SOTA的性能

我们将 vLLM 的吞吐量Throughput与最受欢迎的 LLM 库 [HuggingFace Transformers (HF)](https://huggingface.co/docs/transformers/main_classes/text_generation) 和之前最先进的 [HuggingFace text-generation-inference (TGI)](https://github.com/huggingface/text-generation-inference) 进行了比较。我们在两种设置下进行评估：NVIDIA A10G GPU 上的 LLaMA-7B 和 NVIDIA A100 GPU (40GB) 上的 LLaMA-13B。我们从 ShareGPT 数据集中抽取请求的输入/输出长度。在我们的实验中，vLLM 的吞吐量比 HF 高出 24 倍，比 TGI 高出 3.5 倍。

![alt text](<assest/vLLM 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文版翻译）/0.png>)

上图为每个请求要求一个输出时的服务吞吐量。vLLM 的吞吐量比 HF 高 14 倍至 24 倍，比 TGI 高 2.2 倍至 2.5 倍。

![alt text](<assest/vLLM 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文版翻译）/1.png>)

上图为每个请求要求三个并行输出时的服务吞吐量。vLLM 的吞吐量比 HF 高 8.5 倍至 15 倍，比 TGI 高 3.3 倍至 3.5 倍。


##  秘诀：PagedAttention

在 vLLM 中，我们发现 LLM 服务的性能瓶颈在于内存（显存）。在自回归解码autoregressive decoding过程中，LLM 的所有输入tokens都会产生其注意键和值张量attention key and value tensors，这些张量保存在 GPU 内存中以生成下一个tokens。这些缓存的键和值张量通常称为 KV 缓存(KV cache)。KV 缓存有如下特点:

- 大：LLaMA-13B 中单个序列占用高达 1.7GB 的空间。
- 动态：其大小取决于序列长度，而序列长度变化很大且不可预测。因此，有效管理 KV 缓存是一项重大挑战。我们发现现有系统由于碎片化和过度预留而浪费了 60% - 80% 的内存。

为了解决这个问题，我们引入了 PagedAttention，这是一种注意力算法，其灵感来自操作系统中虚拟内存和分页的经典思想。与传统的注意力算法不同，PagedAttention 允许在非连续的内存空间中存储连续的键和值。具体来说，PagedAttention 将每个序列的 KV 缓存划分为块，每个块包含固定数量 token 的键和值。在注意力计算过程中，PagedAttention 内核会高效地识别和获取这些块。

![alt text](<assest/vLLM 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文版翻译）/2.gif>)

PagedAttention：KV Cache 被划分为块。块在内存空间中不需要连续。

由于块blocks在内存中不需要连续，因此我们可以像在操作系统的虚拟内存中一样以更灵活的方式管理键和值keys & values：可以将块视为页面pages，将tokens视为字节bytes，将序列sequences视为进程processes。序列的连续 ***逻辑块*** ***logical blocks*** 通过 **块表** **block table** 映射到非连续 **物理块** ***physical blocks***。物理块在生成新tokens时按需分配。

>译者注：面试要考的！

![alt text](<assest/vLLM 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文版翻译）/3.gif>)

使用 PagedAttention 的请求的示例生成过程。


在 PagedAttention 中，内存浪费仅发生在序列的最后一个块中。实际上，这会导致接近最佳的内存使用率，浪费率仅为 4% 以下。内存效率的提高被证明是非常有益的：它允许系统将更多序列批量处理在一起，提高 GPU 利用率，从而显著提高吞吐量，如上图性能结果所示。

PagedAttention 还有一个关键优势：高效的内存共享。例如，在并行采样中，同一个提示词prompt会生成多个输出序列。在这种情况下，输出序列之间可以共享该提示词prompt的计算和内存。

![alt text](<assest/vLLM 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文版翻译）/4.gif>)

并行采样的示例

PagedAttention 通过其块表自然地实现了内存共享。与进程共享物理页面的方式类似，PagedAttention 中的不同序列可以通过将其逻辑块映射到同一物理块来共享块。为了确保安全共享，PagedAttention 会跟踪物理块的引用计数并实现写时复制机制。

![alt text](<assest/vLLM 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文版翻译）/5.gif>)

对多个输出进行采样的请求的示例生成过程

PageAttention 的内存共享功能大大降低了复杂采样算法（例如parallel sampling和beam search）的内存开销，最多可减少 55% 的内存使用量。这可以转化为高达 2.2 倍的吞吐量提升。这使得此类采样方法在 LLM 服务中变得实用。

PagedAttention 是 vLLM 背后的核心技术，vLLM 是我们的 LLM 推理和服务引擎，支持多种模型，具有高性能和易于使用的界面。有关 vLLM 和 PagedAttention 的更多技术细节，请查看我们的 [GitHub 仓库](https://github.com/vllm-project/vllm)并继续关注我们的论文。


## LMSYS Vicuna and Chatbot Arena的幕后功臣

今年4月（2023年4月。译者注），LMSYS 开发了流行的 Vicuna 聊天机器人模型并将其公开。从那时起，Vicuna 已在 [Chatbot Arena](https://lmarena.ai/) 中为数百万用户提供服务。最初，LMSYS FastChat 采用基于 HF Transformers 的[服务后端](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/model_worker.py)来为聊天demo提供服务。随着demo越来越受欢迎，峰值流量增加了几倍，使 HF 后端成为一个重大瓶颈。LMSYS 和 vLLM 团队合作并很快开发了 FastChat-vLLM 集成，以使用 vLLM 作为[新的后端](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/vllm_worker.py)来支持不断增长的需求（高达 5 倍的流量）。在 LMSYS 早期的[内部微基准测试](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/test_throughput.py)中，vLLM 服务后端可以实现**比初始 HF 后端高出 30 倍的吞吐量**。

自2023年4月中旬以来，Vicuna、Koala 和 LLaMA 等最受欢迎的模型均已成功使用 FastChat-vLLM 集成提供服务 - 通过使用 FastChat 作为多模型聊天服务前端和 vLLM 作为推理后端，LMSYS 能够利用有限数量的大学赞助 GPU 为数百万用户提供高吞吐量和低延迟的 Vicuna。LMSYS 正在将 vLLM 的使用范围扩展到更广泛的模型，包括 Databricks Dolly、LAION 的 OpenAsssiant 和 Stability AI 的 stableLM。对更多[模型的支持](https://docs.vllm.ai/en/latest/models/supported_models.html)正在开发中，即将推出。

![alt text](<assest/vLLM 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文版翻译）/6.png>)

4 月至 5 月期间，Chatbot Arena 中 FastChat-vLLM 集成处理的请求。事实上，超过一半的 Chatbot Arena 请求使用 vLLM 作为推理后端。

使用 vLLM 还显著降低了运营成本。借助 vLLM，LMSYS 能够将用于处理上述流量的 GPU 数量减少 50%。vLLM 每天平均处理 30K 个请求，峰值为 60K，这充分证明了 vLLM 的稳健性。


## 开始使用 vLLM

使用以下命令安装 vLLM（更多信息请查看我们的[安装指南](https://docs.vllm.ai/en/latest/getting_started/installation.html)）：

```shell
$ pip install vllm
```

vLLM 既可用于离线推理，也可用于在线服务。要使用 vLLM 进行离线推理，你可以导入 vLLM 并在 Python 脚本中使用 LLM 类：

```python
from vllm import LLM

prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
llm = LLM(model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
outputs = llm.generate(prompts)  # Generate texts from the prompts.
```

要使用 vLLM 进行在线服务，你可以通过以下方式启动与 OpenAI API 兼容的服务器：

```shell
$ python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.3
```

你可以使用与 OpenAI API 相同的格式查询服务器：

```shell
$ curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "lmsys/vicuna-7b-v1.3",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

```

有关使用 vLLM 的更多方法，请查看[快速入门指南](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)。


## 欢迎关注我的GitHub和微信公众号，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！
