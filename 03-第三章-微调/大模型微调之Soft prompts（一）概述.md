LoRA不是唯一选择，Soft Prompts微调大模型的奥秘（一）综述

## 1. Prompting
训练大型预训练语言模型非常耗时且计算密集。随着模型规模的增长，人们越来越关注更高效的训练方法，比如Prompting。Prompting通过包含一段描述任务或展示任务示例的文本提示，为特定的下游任务调整一个冻结的预训练模型。有了Prompting，你可以避免为每个下游任务完全训练一个单独的模型，而是使用同一个冻结的预训练模型。这样做轻松得多，因为你可以用同一个模型处理多个不同的任务，而且训练和存储一小套提示参数比训练所有模型参数要高效得多。

## 2. Soft prompts
Soft Prompts（软提示）与Hard Prompts（硬提示）相对。软提示是可学习的连续向量，可以通过梯度优化方法针对特定数据集进行优化。这种方法不需要人工设计，可以自动优化以适应不同任务，计算效率高，支持多任务学习。然而，软提示不可读，无法解释为何选择这些向量。

软提示的工作原理是在模型输入层增加可学习的投影层，将原始输入映射到提示信息所表示的语义空间中。投影层中的参数通过训练数据学习得到，使得提示信息能够更好地适应任务需求。

## 3. 主流方法

### 3.1. Prompt Tuning

Prompt Tuning[1](#refer-anchor-1)的核心思想在于prompt tokens有自己的参数，这些参数可以独立更新。这意味着你可以保持预训练模型的参数不变，只更新prompt tokens的嵌入向量的梯度。这样的结果与传统的训练整个模型的方法相当，并且随着模型大小的增加，Prompt Tuning的性能也会提升。

![alt text](<assest/大模型微调之Soft prompts（一）概述/0.png>)

### 3.2 Prefix-Tuning

Prefix-Tuning[2](#refer-anchor-2)是Prompt Tuning的一种变体，它通过在模型输入的前缀位置添加可学习的提示向量来实现。这种方法的优势在于可以在不改变模型结构的情况下，为不同的任务提供不同的提示。

Prefix-Tuning和Prompt Tuning最主要区别在于，Prefix-Tuning的前缀参数被插入到模型的所有层中，而Prompt Tuning只将提示参数添加到模型的embedding层。

![alt text](<assest/大模型微调之Soft prompts（一）概述/1.png>)

### 3.3 P-Tuning

P-tuning[3](#refer-anchor-3)主要是为自然语言理解（NLU）任务设计的，它是Soft prompts的另一种变体。P-tuning 添加了一个可训练的嵌入张量，这个张量可以被优化以找到更好的提示，并且它使用一个提示编码器（一个双向长短期记忆网络或LSTM）来优化提示参数。

P-tuning的特点是将Decoder架构的模型变得适应Encoder架构的任务，如NLU任务。

![alt text](<assest/大模型微调之Soft prompts（一）概述/2.png>)

### 3.4 Multitask prompt tuning

多任务提示调整（MPT）[4](#refer-anchor-4)是一种从数据中学习单一提示的方法，该提示可以用于多种任务类型，并可以共享以适应不同的目标任务。与之相对的其他现有方法则为每个任务学习一个单独的软提示，这些提示需要被检索或聚合以适应目标任务。

简而言之：MPT先学习一个通用的提示，然后再根据具体任务进行调整。

![alt text](<assest/大模型微调之Soft prompts（一）概述/3.png>)

## 参考

<div id="refer-anchor-1"></div>

[1] [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)

<div id="refer-anchor-2"></div>

[2] [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

<div id="refer-anchor-3"></div>

[3] [P-tuning: A Simple Method for Prompt Tuning](https://arxiv.org/abs/2103.10385)

<div id="refer-anchor-4"></div>

[4] [Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2303.02861)

## 欢迎关注我的GitHub和微信公众号[真-忒修斯之船]，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！