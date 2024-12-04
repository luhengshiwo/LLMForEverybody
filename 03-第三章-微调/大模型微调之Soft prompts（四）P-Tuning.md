LoRA不是唯一选择，Soft Prompts微调大模型的奥秘（四）P-Tuning

P—Tuning是为了解决NLU任务而设计的Soft prompts方法，P-tuning添加了一个可训练的嵌入张量，这个张量可以被优化以找到更好的提示，并且它使用一个提示编码器（例如BiLSTM+MLP）来优化提示参数。

## 技术解读

![alt text](<assest/大模型微调之Soft prompts（四）P-tuning/0.png>)

P-tuning有两个版本，P-tuning v1（2021年）和P-tuning v2（2023年）。P-tuning v1通过使用一个prompt encoder（例如BiLSTM+MLP）来优化prompt参数，但这种方法在一些复杂的自然语言理解（NLU）任务上效果不佳，且只在较大的模型规模上表现良好。

为了解决这些问题，P-tuning v2被提出。P-tuning v2在v1的基础上进行了改进，它不仅在输入层，而且在模型的每一层都加入了可训练的连续提示，这样可以更好地适应各种任务，包括复杂的NLU任务。P-tuning v2通过多任务学习策略和深度提示编码器来优化提示参数，使得它在不同规模的模型上都能取得与全参数微调相媲美的性能。

## 直观解释

P-Tuning最大的特点是将通过引入BiLSTM/MLP的方法，使得模型可以更好的完成NLU任务。这种方法的优势在于可以在不改变模型结构的情况下，为decoder架构的模型提供encoder架构的特性。

## 参考

<div id="refer-anchor-1"></div>

[1] [P-tuning: A Simple Method for Prompt Tuning](https://arxiv.org/abs/2103.10385)

## 欢迎关注我的GitHub和微信公众号[真-忒修斯之船]，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！