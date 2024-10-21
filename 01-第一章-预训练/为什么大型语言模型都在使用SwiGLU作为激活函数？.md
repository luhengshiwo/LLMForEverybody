为什么大型语言模型都在使用 SwiGLU 作为激活函数？

激活函数是神经网络设计中的关键组成部分，它们通过引入非线性和控制梯度流动，使得网络能够学习和执行各种复杂的任务。而在大型语言模型中，SwiGLU 作为激活函数的使用已经成为了一种趋势。那么，为什么大型语言模型都在使用 SwiGLU 作为激活函数呢？

**阅读提示**：可以只看公式和代码，了解激活函数的数学表达和实现方式。


## 1. 最初的激活函数-Sigmoid

在生物学中，神经元的激活通常具有非线性和阈值特性，即神经元在接收到足够强的输入信号后才会激活并产生输出。在人工神经网络中，虽然sigmoid函数在深度学习中的使用已经减少，但它曾经是神经网络中最受欢迎的激活函数之一。sigmoid函数的S形曲线与生物神经元的激活方式有一定的相似性，因为它具有一个明显的阈值，当输入超过这个阈值时，输出会急剧增加。

![alt text](<assest/为什么大型语言模型都在使用 SwiGLU 作为激活函数？/0.png>)

Sigmoid函数是一种广泛应用于统计学和机器学习的激活函数，特别是在早期的神经网络模型中。它的形状呈S形，能够将输入值压缩到0和1之间，这使得它在某些情况下特别有用，比如将输出解释为概率或者在二分类问题中作为输出层的激活函数。

Sigmoid函数的数学表达式如下：

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

其中 e 是自然对数的底数（约等于2.71828）， x  是输入值。

![alt text](<assest/为什么大型语言模型都在使用 SwiGLU 作为激活函数？/1.png>)

### Sigmoid函数的特点包括：

1. **输出范围**：输出值始终在(0, 1)区间内，这使得sigmoid函数在输出概率或者用于二分类问题时非常有用。

2. **平滑连续**：sigmoid函数在整个定义域内都是光滑且连续的，这有助于在优化过程中计算梯度。

3. **中心对称**：sigmoid函数关于直线  y = x  对称，这意味着它能够处理正负输入值。

4. **梯度问题**：当输入值  x  非常大或非常小的时候，sigmoid函数的梯度接近于0，这可能导致梯度消失问题，从而使得在神经网络的深层中梯度难以传播。

5. **非零中心**：sigmoid函数的输出不是以0为中心的，这可能会导致网络在优化过程中花费更多的时间和资源来调整权重。

### Sigmoid函数的用途：

- **二分类**：在二分类问题的输出层，sigmoid函数可以将输出转换为概率值，用于表示属于某一类的概率。
- **逻辑回归**：在逻辑回归模型中，sigmoid函数用于将线性回归的输出转换为概率预测。
- **隐藏层**：虽然在现代深度学习模型中较少用于隐藏层，但在某些情况下，sigmoid仍然可以作为隐藏层的激活函数。

### 局限性：

- **梯度消失**：在深层网络中，由于sigmoid函数在输入值绝对值较大时梯度接近于0，容易导致梯度消失问题，从而阻碍深层网络的训练。
- **计算开销**：相比于某些激活函数（如ReLU），sigmoid函数的计算复杂度较高，因为它涉及到指数运算。

由于这些局限性，sigmoid函数在现代深度学习模型中逐渐被其他激活函数（如ReLU及其变种）所取代，尤其是在隐藏层中。然而，在需要输出概率预测的特定场景下，sigmoid函数仍然是一种重要的激活函数。

```python
import numpy as np
def sigmoid(x):
    """Sigmoid 激活函数

    参数:
    x -- 输入值

    返回:
    Sigmoid 激活后的值
    """
    return 1 / (1 + np.exp(-x))

```


## 2. 深度学习的功臣-ReLU

ReLU（Rectified Linear Unit）是一种简单而有效的激活函数，它在深度学习中得到了广泛的应用。ReLU函数的定义如下：

$$ f(x) = max(0, x) $$

这意味着当输入 x 大于0时，输出等于输入；当输入 x 小于或等于0时，输出为0。ReLU函数的图像是一条折线，当 x 为正时，函数的斜率为1，当 x 为负时，函数的斜率为0。

![alt text](<assest/为什么大型语言模型都在使用 SwiGLU 作为激活函数？/2.png>)

ReLU的特点包括：
- 稀疏激活：由于ReLU在负半轴的输出为0，这意味着在任何时候，大约有一半的神经元（假设输入是零均值的）不会被激活，这种稀疏性有助于减少计算量和防止过拟合。

- 非线性：ReLU引入了非线性，使得神经网络能够学习和模拟复杂的函数。

- 计算效率高：由于其简单的数学形式，ReLU的计算非常快速，这有助于加快神经网络的训练和推理速度。

- 缓解梯度消失问题：相比于sigmoid或tanh函数，ReLU在正区间的梯度恒定为1，这有助于缓解梯度消失问题，使得深层网络的训练变得更加可行。

```python
import numpy as np
def relu(x):
    """ReLU 激活函数

    参数:
    x -- 输入值

    返回:
    ReLU 激活后的值
    """
    return np.maximum(0, x)
```


## 3. 百花齐放-类ReLU激活函数

除了ReLU之外，还有许多类似的激活函数，它们在ReLU的基础上进行了一些改进，以解决ReLU存在的一些问题。这些类ReLU激活函数包括：

- **Leaky ReLU**：Leaky ReLU是对ReLU的改进，当输入为负时，Leaky ReLU引入了一个小的斜率，而不是直接输出0。这有助于解决ReLU在负区间的输出为0的问题。

- **Parametric ReLU (PReLU)**：PReLU是Leaky ReLU的一种扩展，它引入了一个可学习的参数，用于控制负区间的斜率。这使得神经网络能够自适应地学习负区间的激活函数。

- **Exponential Linear Unit (ELU)**：ELU是一种类似ReLU的激活函数，它在负区间引入了一个指数项，使得负区间的输出更加平滑。ELU在一些情况下可以提供更好的性能。

- **Scaled Exponential Linear Unit (SELU)**：SELU是ELU的一种变种，它引入了缩放参数，使得SELU在一定条件下能够保持输入的均值和方差不变，从而有助于稳定训练。

- **GELU**：GELU是一种基于高斯误差线性单元的激活函数，它在一些情况下能够提供更好的性能。GELU的数学形式是一个带有误差函数的平滑函数。

这些类ReLU激活函数在实践中都得到了广泛的应用，它们在一定程度上改进了ReLU的性能，并且在不同的场景下可能会有不同的表现。选择合适的激活函数取决于具体的任务和模型结构。


## 4. 实验最佳激活函数-Swish

Swish是由Google Brain提出的一种激活函数，它的数学表达式如下：
$$ f(x) = x \cdot \sigma(x) $$
其中 $\sigma(x)$ 是sigmoid函数。

![alt text](<assest/为什么大型语言模型都在使用 SwiGLU 作为激活函数？/3.png>)

Swish的名称可能来源于其形状与鱼的尾巴相似，给人一种平滑、流畅的联想，这与"swish"这个词的含义相吻合

Swish函数的特点包括：
- **非线性**：Swish引入了非线性，使得神经网络能够学习和模拟复杂的函数。
- **平滑性**：Swish函数在整个定义域内都是光滑且连续的，这有助于在优化过程中计算梯度。
- **自适应性**：Swish函数的输出取决于输入值，这使得它能够自适应地调整激活函数的形状。

Swish函数在一些实验中表现出了比ReLU更好的性能，尤其是在一些深度神经网络中。然而，Swish函数的计算复杂度较高，因为它涉及到sigmoid函数的计算。因此，在实际应用中，需要根据具体的任务和模型结构来选择合适的激活函数。

```python
import numpy as np
def swish(x,beta=1.0):
    """Swish 激活函数

    参数:
    x -- 输入值

    返回:
    Swish 激活后的值
    """
    return x * sigmoid(beta*x)

def sigmoid(x):
    """Sigmoid 函数

    参数:
    x -- 输入值

    返回:
    Sigmoid 函数的输出值
    """
    return 1 / (1 + np.exp(-x))
```


## 5. GLU 

GLU（Gated Linear Unit）其实不算是一种激活函数，而是一种神经网络层。它是一个线性变换后面接门控机制的结构。其中门控机制是一个sigmoid函数用来控制信息能够通过多少。，它结合了线性单元和门控机制，能够有效地学习输入数据的不同特征。GLU的数学表达式如下：

$$f(X) = (X ∗ W + b) ⊗ σ(X ∗ V + c)$$
其中 ⊗ 表示逐元素乘法，$X$ 是输入，$W$ 和 $V$ 是权重矩阵，$b$ 和 $c$ 是偏置项。

GLU的特点包括：
- **门控机制**：GLU引入了门控机制，通过sigmoid函数控制输入的线性变换，从而使得神经网络能够学习输入数据的不同特征。
- **非线性**：GLU引入了非线性，使得神经网络能够学习和模拟复杂的函数。
- **自适应性**：GLU函数的输出取决于输入值，这使得它能够自适应地调整激活函数的形状。

参考pytorch的GLU实现

![alt text](<assest/为什么大型语言模型都在使用 SwiGLU 作为激活函数？/4.png>)

```python
import numpy as np
def glu(x):
    """GLU 激活函数

    参数:
    x -- 输入数组，维度必须是偶数

    返回:
    GLU 激活后的数组
    """
    assert x.shape[-1] % 2 == 0, "输入数组的最后一个维度必须是偶数"
    half_dim = x.shape[-1] // 2
    return x[..., :half_dim] * sigmoid(x[..., half_dim:])

def sigmoid(x):
    """Sigmoid 函数

    参数:
    x -- 输入值

    返回:
    Sigmoid 函数的输出值
    """
    return 1 / (1 + np.exp(-x))
```

## 6. 实践出真知-SwiGLU

终于到了我们今天的主角-SwiGLU。SwiGLU是一种结合了Swish和GLU的激活函数，它结合了Swish的平滑性和GLU的门控机制，能够有效地学习输入数据的不同特征。SwiGLU的数学表达式如下：

$$ f(X) = (X ∗ W + b) ⊗ Swish(X ∗ V + c) $$

```python
import numpy as np
def SwiGLU(x):
    """SwiGLU 激活函数

    参数:
    x -- 输入数组，维度必须是偶数

    返回:
    SwiGLU 激活后的数组
    """
    assert x.shape[-1] % 2 == 0, "输入数组的最后一个维度必须是偶数"
    half_dim = x.shape[-1] // 2
    return x[..., :half_dim] * swish(x[..., half_dim:])

def swish(x,beta=1.0):
    """Swish 激活函数

    参数:
    x -- 输入值

    返回:
    Swish 激活后的值
    """
    return x * sigmoid(beta*x)

def sigmoid(x):
    """Sigmoid 函数

    参数:
    x -- 输入值

    返回:
    Sigmoid 函数的输出值
    """
    return 1 / (1 + np.exp(-x))

```

## 为什么用它？

SwiGLU激活函数因其在多个方面的优势而被广泛应用于大型语言模型中。它结合了Swish和GLU的特点，提供了一种有效的激活机制，具体来说：

1. **非线性能力**：SwiGLU通过Swish激活函数引入非线性，这使得模型能够学习和表示更复杂的数据模式 。

2. **门控特性**：GLU的门控机制允许模型动态地调整信息流，使得模型在处理长序列数据时能够更好地捕捉长距离依赖关系 。

3. **梯度稳定性**：SwiGLU在负输入区域提供非零的梯度，有助于缓解梯度消失问题，从而提高模型的训练稳定性 。

4. **可学习参数**：SwiGLU的参数可以通过训练学习，使得模型可以根据不同任务和数据集动态调整，增强了模型的灵活性和适应性 。

5. **计算效率**：相比于一些复杂的激活函数，SwiGLU在保持性能的同时，具有较高的计算效率，这对于大规模语言模型的训练和推理尤为重要 。

由于这些优势，SwiGLU在大型语言模型如LLAMA、OLMO和PALM中得到了应用 。它通过结合Swish的平滑性和GLU的门控机制，提供了一种有效的激活函数，以支持复杂和高效的深度学习模型训练 。


## 参考

<div id="refer-anchor-1"></div>

[1] [神经元的故事：有生命的电线细胞内部一瞥](https://www.visiblebody.com/zh/learn/nervous/neurons)

[2] [Sigmoid Function](https://botpenguin.com/glossary/sigmoid-function)

[3] [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)

[4] [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083)

[5] [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

## 欢迎关注我的GitHub和微信公众号，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！