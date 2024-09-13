## 导入

如果你看GPT系列的论文，你学习到的self-attention是**Multi-Head Attention**(MHA)即多头注意力机制，
MHA 包含h个Query、Key 和 Value 矩阵，所有注意力头(head)的 Key 和 Value 矩阵权重不共享。

这个机制已经能很好的捕捉信息了，为什么会继续发展出MQA和GQA？

很多文章上来就是这三种attention机制的数学公式差别，但没有说为什么有了MHA，还需要MQA，甚至GQA。本文简单阐述原因，给大家一个直觉式的理解。

![alt text](<assest/为什么会发展出Multi Query Attention和Group Query Attention/1.PNG>)

## KV Cache
随着大模型的参数量越来越大，推理速度也受到了严峻的挑战。于是人们采用了KV Cache，想用空间换时间。

> kv cache的文章

## MQA

而增加了空间后，显存又是一个问题，于是人们尝试在attention机制里面共享keys和values来减少KV cache的内容。
这就有了Multi-Query Attention(MQA)，即query的数量还是多个，而keys和values只有一个，所有的query共享一组。这样KV Cache就变小了。

## GQA

但MQA的缺点就是损失了精度，所以研究人员又想了一个折中方案：不是所有的query共享一组KV，而是一个group的guery共享一组KV，这样既降低了KV cache，又能满足精度。这就有了Group-Query Attention。


## 参考

<div id="refer-anchor-1"></div>

[1] [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints]( https://arxiv.org/pdf/2305.13245)

[2] [GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)



