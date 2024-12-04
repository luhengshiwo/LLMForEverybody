LoRA不是唯一选择，Soft Prompts微调大模型的奥秘（三）Prefix-Tuning

同样在2021年（和Prompt Tuning同一年），Prefix-Tuning被提出，它通过在模型输入的前缀位置添加可学习的提示向量来实现。这种方法的优势在于可以在不改变模型结构的情况下，为不同的任务提供不同的提示。

## 技术解读

![alt text](<assest/大模型微调之Soft prompts（三）Prefix-tuning/0.png>)

Prefix Tuning是一种用于自然语言生成任务的参数高效微调技术。它的核心思想是在模型的输入序列前添加一系列连续的任务特定向量，这些向量被称为前缀（prefix）。这些前缀向量是可训练的，而预训练语言模型（PLM）的其他参数则保持固定。

在Prefix Tuning中，模型的输入不仅包括原始的任务输入，还包括这些前缀向量。在自回归语言模型（如GPT-2）中，前缀被添加到输入序列的开始处，形成新的输入序列。在编码器-解码器架构模型（如BART）中，前缀不仅添加到输入序列的开始处，还添加到解码器的输入中。这样，模型在处理输入序列时，每一层的输入都会包含这些额外的前缀向量，从而适配下游任务。

Prefix Tuning的一个关键优势是它的参数效率。它只需要更新一小部分参数，即前缀向量，而不需要更新整个模型的参数。这大大减少了所需的计算资源和存储需求。此外，由于只更新前缀向量，Prefix Tuning可以更容易地适应多个任务，而不需要为每个任务训练和存储一个完整的模型副本。

在实际应用中，Prefix Tuning已被证明在多种自然语言处理任务中有效，包括文本生成、摘要生成等。它通过在模型的输入中添加可训练的前缀向量，使得模型能够在不改变原有参数的情况下，更好地适应特定的下游任务.

## 直观解释

Prefix-Tuning和Prompt Tuning最主要区别在于，Prefix-Tuning的前缀参数被插入到模型的所有层中，而Prompt Tuning只将提示参数添加到模型的embedding层。


## 参考

<div id="refer-anchor-1"></div>

[1] [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)


## 欢迎关注我的GitHub和微信公众号[真-忒修斯之船]，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！