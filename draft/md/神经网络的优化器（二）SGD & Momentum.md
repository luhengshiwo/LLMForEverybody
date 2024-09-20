神经网络的优化器（二）SGD & Momentum

本文从最初的SGD开始，介绍SGD的算法细节，以及其在深度神经网络中的劣势，并引入Momentum算法，解决SGD的一些问题。


## 1. SGD
随机梯度下降（Stochastic Gradient Descent，SGD）是一种用于优化可微分目标函数的迭代方法，它是梯度下降优化的随机近似。SGD的提出可以追溯到1951年，由Herbert Robbins和Sutton Monro在他们的论文《A Stochastic Approximation Method》[1]中首次描述了随机近似方法，这可以看作是SGD的前身。随后，J. Kiefer和J. Wolfowitz在1952年发表了论文《Stochastic Estimation of the Maximum of a Regression Function》[2]，这篇论文更接近于机器学习领域中SGD的现代理解。

随机梯度下降（SGD）的更新公式是梯度下降法的一种变体，它用于优化目标函数，特别是在处理大规模数据集时。SGD 在每次迭代中只使用一个或一小批样本来计算梯度，然后更新模型参数。这样做可以减少每次迭代的计算成本，并有助于算法逃离局部最小值。

- 00

SGD 的参数更新公式如下：

$$\theta_{t+1} = \theta_t - \eta_t \nabla J(\theta_t; x^{(i)}, y^{(i)})$$

其中：
- $\theta_t$表示在时间步 t的模型参数（可以是权重和偏置）；
- $\eta_t$表示学习率，用于控制参数更新的步长；
- $\nabla J(\theta_t; x^{(i)}, y^{(i)})$表示损失函数 J 对参数 $\theta_t$ 在样本 $(x^{(i)}, y^{(i)})$ 处的梯度；
- $\theta_{t+1}$是更新后的模型参数。

在实际应用中，学习率$\eta_t$可以是一个固定的值，也可以随着时间步逐渐减小（学习率衰减），以确保算法在训练过程中的稳定性和收敛性。

## 2. SGD的缺点-非凸优化问题

非凸优化问题是指目标函数具有多个局部最小值的情况，这种情况下，SGD可能会陷入局部最小值，并且很难跳出。这是因为SGD在每次迭代中只使用一个或一小批样本来计算梯度，这样可能导致梯度的方向不准确，从而影响参数更新的方向。

**鞍点**

- 0

在优化和机器学习领域，鞍点（Saddle Point）是指目标函数的临界点，在这个点上，某些方向的导数为正（上升），而另一些方向的导数为负（下降）。换句话说，鞍点既不是局部最小值也不是局部最大值，而是局部最小值和局部最大值的某种组合。

具体来说：

- 局部最小值：在该点的所有邻域内，函数值都大于或等于该点的函数值；
- 局部最大值：在该点的所有邻域内，函数值都小于或等于该点的函数值；
- 鞍点：在该点的某些邻域方向上，函数值大于或等于该点的函数值，而在另一些邻域方向上，函数值小于或等于该点的函数值。

在二维空间中，可以将鞍点想象成马鞍的形状，从马鞍的一侧走到顶部再走到另一侧，你会先经历一个上升过程（局部最小值的特征），然后是一个下降过程（局部最大值的特征）。

在机器学习中，尤其是在深度学习中，鞍点的存在可能会使基于梯度的优化算法（如梯度下降）遇到难题，因为梯度下降算法可能会在接近鞍点时停滞不前，因为梯度在该点为零，算法无法判断下一步应该向哪个方向移动。


## 3. 指数加权平均

在介绍Momentum算法之前，我们先来了解一下指数加权平均（Exponential Weighted Average）的概念。

指数加权平均（Exponentially Weighted Average，EWA）是一种统计方法，用于计算一组数值的加权平均，其中最近的数据点被赋予更高的权重。这种方法在信号处理、时间序列分析、机器学习等领域中非常有用，尤其是在需要对数据的最新变化做出快速反应时。


指数加权平均的计算公式如下：

$$ \text{EWA}_t = \beta \cdot \text{EWA}_{t-1} + (1 - \beta) \cdot x_t $$

其中：
- $ \text{EWA}_t $ 是在时间点 $t$ 的指数加权平均值。
- $\beta$ 是介于 0 和 1 之间的衰减系数（decay factor），决定了历史数据的权重。
- $x_t$ 是在时间点 $t$ 的观测值。
- $\text{EWA}_{t-1}$ 是前一时间点的指数加权平均值。

衰减系数 $\beta$ 的选择对指数加权平均的影响很大。如果 $\beta$ 接近 1，那么历史数据的影响会持续很长时间，平滑效果更强；如果 $\beta$ 接近 0，则新数据的影响更大，对变化的反应更快。

指数加权平均的一个特性是它对异常值（outliers）不太敏感，因为每个数据点的权重都会随着时间的推移而指数级减少。这使得它在处理含有噪声的数据时非常有用。

## 4. Momentum 

动量（Momentum）方法是一种在深度学习中广泛使用的优化策略，它通过引入动量项来加速梯度下降算法的收敛并提高其稳定性。动量方法的核心思想是模拟物理学中的动量概念，通过累积过去梯度的信息来调整参数更新的方向和幅度。动量通过指数加权平均的方式来计算。

- 1

动量方法的更新公式可以表示为：
$$
\begin{align}
v_t &= \gamma v_{t-1} + \eta_t \nabla J(\theta_t) \\
\theta_t &= \theta_{t-1} - v_t 
\end{align}
$$

其中：
- $ v_t$ 是时间步 $t$ 的动量项, 这个动量项是通过指数加权平均的方式计算得到的;
- $\gamma$ 是动量衰减系数，通常设置在 $[0,1)$ 之间，如 0.9 或 0.99;
- $\eta_t$ 是学习率;
- $\nabla J(\theta_t) $ 是在参数 $\theta_t$ 处的损失函数梯度。

动量方法的优势包括：
1. **加速收敛**：通过累积历史梯度，可以在相关方向上加速参数更新；
2. **抑制振荡**：有助于减少训练过程中的震荡，特别是在目标函数的平坦区域或接近最小值时；
3. **跳出局部最小值**：在某些情况下，动量可以帮助算法跳出局部最小值，从而找到更好的全局最小值。

## 5. 指数加权平均为什么叫“指数”？

指数加权平均（Exponentially Weighted Average，EWA）之所以被称为“指数”，是因为它在计算平均值时，给予不同时间点的数据以指数级衰减的权重。

在每次计算时，新数据 $x_t$ 被赋予的权重是 $(1 - \beta) $，而之前的指数加权平均 $\text{EWA}_{t-1}$ 被赋予的权重是 $\beta$。由于 $\beta$ 接近 1，所以越早的数据其权重会以 $\beta$ 的多次方的速度迅速减小，这就是“指数”名称的来源。


## 参考

[1] [A Stochastic Approximation Method](https://www.jstor.org/stable/2236626)

[2] [Stochastic Estimation of the Maximum of a Regression Function](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-23/issue-3/Stochastic-Estimation-of-the-Maximum-of-a-Regression-Function/10.1214/aoms/1177729392.full)

[3] [Some methods of speeding up the convergence of iteration methods](https://www.sciencedirect.com/science/article/abs/pii/0041555364901375)

## 欢迎关注我的GitHub和微信公众号，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！