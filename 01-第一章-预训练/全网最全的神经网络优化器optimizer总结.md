（也许是）全网最全的神经网络优化器optimizer总结

前一段时间，我想搞清楚优化器的发展脉络，试图了解从梯度下降到现在最常用的AdamW的发展。但搜索了很多资料，都没找到一个全面的总结。所以我决定自己整理一份，希望能帮助到大家。

optimizer负责在训练过程中更新模型的参数, 目的是通过调整参数来最小化损失函数，即模型预测和实际数据之间的差异. 

![alt text](assest/神经网络的优化器（一）概述/00.png)

## 文章链接

[神经网络的优化器（一）综述](神经网络的优化器（一）概述.md)

[神经网络的优化器（二）SGD](神经网络的优化器（二）SGD.md)

[神经网络的优化器（三）Momentum](神经网络的优化器（三）Momentum.md)

[神经网络的优化器（四）ASGD](神经网络的优化器（四）ASGD.md)

[神经网络的优化器（五）Rprop](神经网络的优化器（五）Rprop.md)

[神经网络的优化器（六）AdaGrad](神经网络的优化器（六）AdaGrad.md)

[神经网络的优化器（七）AdaDeleta](神经网络的优化器（七）AdaDeleta.md)

[神经网络的优化器（八）RMSprop](神经网络的优化器（八）RMSprop.md)

[神经网络的优化器（九）Adam](神经网络的优化器（九）Adam.md)

[神经网络的优化器（十）Nadam](神经网络的优化器（十）Nadam.md)

[神经网络的优化器（十一）AdamW](神经网络的优化器（十一）AdamW.md)

[神经网络的优化器（十二）RAdam](神经网络的优化器（十二）RAdam.md)

从1951年Herbert Robbins和Sutton Monro在其题为“随机近似方法”的文章中提出SGD，到2017年出现的AdamW成为最主流的选择，优化器的发展经历了70多年的时间。本系列从时间的角度出发，对神经网络的优化器进行梳理，希望能够帮助大家更好地理解优化器的发展历程。

## 欢迎关注我的GitHub和微信公众号，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！