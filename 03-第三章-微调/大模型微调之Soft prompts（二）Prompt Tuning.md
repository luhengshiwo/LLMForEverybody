微调不要只知道LoRA，微调大模型之Soft Prompts（二）Prompt Tuning

在2021年，大型语言模型的概念尚未完全清晰，人们对其的认识还处于探索阶段。在众多研究焦点中，Casual Language Model（仅解码器模型）只是其中之一。当时，GPT-3的问世引起了广泛关注。紧接着，在同年4月，谷歌推出了Prompt Tuning方法，最初在T5模型上进行实验——T5采用的是编码器-解码器架构。随后，Prompt Tuning在其他下游任务上也显示出了其有效性。


## 技术解读

![alt text](<assest/大模型微调之Soft prompts（二）Prompt-tuning/0.png>)

Prompt Tuning是一种高效的微调方法，它通过在模型输入中添加特定的文本提示（prompt）来适配下游任务，而不需要对预训练模型的参数进行全面的更新。这种方法的核心在于，它通过优化输入提示的参数来调整模型的行为，使得模型能够更好地适应新的任务，而预训练模型的主体参数保持不变。

Prompt Tuning的关键优势在于它的参数效率。相比于传统的微调方法，Prompt Tuning只需要更新一小部分参数，即与提示相关的参数，这样可以显著减少所需的计算资源和训练时间
。此外，Prompt Tuning还可以提高模型的泛化能力，因为它允许模型在没有大量标注数据的情况下适应新任务。

Prompt Tuning的过程通常包括以下几个步骤：

1. 选择一个预训练模型作为基础；
2. 为特定任务设计或选择一个提示模板；
3. 将提示模板与输入数据结合，形成新的输入序列；
4. 在预训练模型上进行训练，只更新提示模板的参数；
5. 使用测试数据集评估模型的性能。


## 直观解释

Prompt Tuning的核心思想在于prompt tokens有自己的参数，这些参数可以独立更新。这意味着你可以保持预训练模型的参数不变，只更新prompt tokens的嵌入向量的梯度。这样的结果与传统的训练整个模型的方法相当，并且随着模型大小的增加，Prompt Tuning的性能也会提升。



## 参考

<div id="refer-anchor-1"></div>

[1] [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)

## 欢迎关注我的GitHub和微信公众号[真-忒修斯之船]，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！