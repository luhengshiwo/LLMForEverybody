每天3分钟，彻底弄懂神经网络的优化器（七）AdaDelta

## 1. AdaDelta算法的提出
AdaDelta算法是由Matthew D. Zeiler在2012年提出的。这一算法的详细描述和原理可以在论文《ADADELTA: An Adaptive Learning Rate Method》[1](#refer-anchor-7)中找到。AdaDelta算法旨在解决AdaGrad算法中学习率单调递减的问题，通过限制累积梯度的窗口大小来调整学习率，使得算法在训练过程中能够自适应地调整每个参数的学习率，而不需要手动设置。这种方法对噪声梯度信息、不同的模型结构、各种数据模式以及超参数选择都表现出了较强的鲁棒性。


## 2. AdaDelta算法的原理

Adadelta（AdaDelta）是一种自适应学习率的优化算法，它解决了Adagrad算法中学习率递减导致的问题。Adadelta算法通过限制累积梯度的窗口大小，并且不需要设置全局学习率，因为它会根据之前的参数更新量来自适应地调整学习率。

Adadelta的更新规则如下：
1. 初始化两个状态变量：累积平方梯度的指数加权平均变量 `s` 和累积更新量的指数加权平均变量 `delta`。
2. 在每次迭代中，计算梯度 `g`。
3. 更新累积平方梯度的指数加权平均 `s`：
   $s = \rho \cdot s + (1 - \rho) \cdot g^2$
4. 计算参数更新量 `delta_p`：
   $\delta_p = -\frac{\sqrt{\delta + \epsilon}}{\sqrt{s + \epsilon}} \cdot g$
5. 更新参数 `w`：
   $w = w + \delta_p$
6. 更新累积更新量的指数加权平均 `delta`：
   $\delta = \rho \cdot \delta + (1 - \rho) \cdot \delta_p^2$

其中，`ρ` 是用于计算平方梯度的指数加权平均的系数（通常设为0.9），`ε` 是一个很小的数（如 `1e-6`），用于增加数值计算的稳定性。

## 3. AdaDelta算法的主要特点

Adadelta算法的优点包括：
- 自适应学习率，不需要手动设置。
- 适合处理稀疏数据。
- 加速模型的收敛过程。

缺点可能包括：
- 对超参数 `ρ` 和 `ε` 敏感。
- 在某些情况下可能导致不稳定的学习过程。

![alt text](assest/神经网络的优化器（七）AdaDelta/0.png)

## 参考

[1] [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)

## 欢迎关注我的GitHub和微信公众号，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！