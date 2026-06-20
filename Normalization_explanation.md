# 深度解析：主流归一化算法与残差结构对比与数学证明

本篇文档对深度学习与大语言模型中核心的归一化算法（**BatchNorm**, **LayerNorm**, **RMSNorm**）以及残差结构布局（**Pre-Norm**, **Post-Norm**）进行深入的数学分析与理论推导，用严密的公式证明其优势与特性。

---

## 一、 归一化算法公式与特性证明

归一化（Normalization）是现代神经网络训练的基石。其核心作用是通过调整特征的均值与方差，控制信号的幅值（Magnitude），防止梯度消失或梯度爆炸。

### 1. LayerNorm (LN) 与 RMSNorm (RMSN) 的定义

设单样本特征向量为 $x \in \mathbb{R}^{d}$ ，其各个分量为 $x_{i}$ （其中 $i \in \{1, 2, \dots, d\}$）。

#### LayerNorm 的数学公式
LayerNorm 计算特征向量的均值 $\mu$ 与方差 $\sigma^{2}$：

$$
\mu = \frac{1}{d} \sum_{j=1}^{d} x_{j}
$$

$$
\sigma^{2} = \frac{1}{d} \sum_{j=1}^{d} (x_{j} - \mu)^{2}
$$

归一化后的输出为：

$$
\text{LN}(x)_{i} = \frac{x_{i} - \mu}{\sqrt{\sigma^{2} + \epsilon}} \cdot \gamma_{i} + \beta_{i}
$$

其中 $\gamma$ 与 $\beta$ 分别为可学习的缩放与偏置向量，$\epsilon$ 为防止除以 0 的极小常数。

#### RMSNorm 的数学公式
RMSNorm 去掉了均值偏移的操作，仅除以均方根（Root Mean Square）：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{j=1}^{d} x_{j}^{2} + \epsilon}
$$

归一化后的输出为：

$$
\text{RMSNorm}(x)_{i} = \frac{x_{i}}{\text{RMS}(x)} \cdot \gamma_{i}
$$

RMSNorm 舍弃了可学习的偏置项 $\beta$ ，仅保留缩放因子 $\gamma$ 。

---

### 2. RMSNorm 梯度性质与正交性证明

为了分析 RMSNorm 在梯度传播中的稳定性，我们求损失函数 $L$ 对输入特征 $x_{i}$ 的梯度。
定义不带偏置和缩放因子的基础归一化输出为：

$$
y_{i} = \frac{x_{i}}{\text{RMS}(x)}
$$

其中（先忽略 $\epsilon$）：

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{k=1}^{d} x_{k}^{2}}
$$

根据多元复合函数求导的链式法则，损失函数 $L$ 对输入 $x_{i}$ 的偏导数为：

$$
\frac{\partial L}{\partial x_{i}} = \sum_{j=1}^{d} \frac{\partial L}{\partial y_{j}} \frac{\partial y_{j}}{\partial x_{i}}
$$

#### 第一步：求 RMS(x) 对输入 $x_{i}$ 的偏导数

$$
\frac{\partial \text{RMS}(x)}{\partial x_{i}} = \frac{\partial}{\partial x_{i}} \left( \frac{1}{d} \sum_{k=1}^{d} x_{k}^{2} \right)^{1/2} = \frac{1}{2} \left( \frac{1}{d} \sum_{k=1}^{d} x_{k}^{2} \right)^{-1/2} \cdot \frac{2 x_{i}}{d} = \frac{x_{i}}{d \cdot \text{RMS}(x)}
$$

#### 第二步：求 $\frac{\partial y_{j}}{\partial x_{i}}$ 的雅可比矩阵元素
我们需要分两种情况讨论：

**情况 1：对角元素（ $j = i$ ）**

$$
\frac{\partial y_{i}}{\partial x_{i}} = \frac{\partial}{\partial x_{i}} \left( \frac{x_{i}}{\text{RMS}(x)} \right) = \frac{\text{RMS}(x) \cdot 1 - x_{i} \cdot \frac{\partial \text{RMS}(x)}{\partial x_{i}}}{\text{RMS}(x)^{2}}
$$

代入第一步的结果：

$$
\frac{\partial y_{i}}{\partial x_{i}} = \frac{\text{RMS}(x) - \frac{x_{i}^{2}}{d \cdot \text{RMS}(x)}}{\text{RMS}(x)^{2}} = \frac{1}{\text{RMS}(x)} \left( 1 - \frac{x_{i}^{2}}{d \cdot \text{RMS}(x)^{2}} \right) = \frac{1}{\text{RMS}(x)} \left( 1 - \frac{y_{i}^{2}}{d} \right)
$$

**情况 2：非对角元素（ $j \neq i$ ）**

$$
\frac{\partial y_{j}}{\partial x_{i}} = \frac{\partial}{\partial x_{i}} \left( \frac{x_{j}}{\text{RMS}(x)} \right) = \frac{0 - x_{j} \cdot \frac{\partial \text{RMS}(x)}{\partial x_{i}}}{\text{RMS}(x)^{2}} = - \frac{x_{j} \cdot \frac{x_{i}}{d \cdot \text{RMS}(x)}}{\text{RMS}(x)^{2}} = - \frac{y_{i} y_{j}}{d \cdot \text{RMS}(x)}
$$

#### 第三步：代入链式法则求梯度公式
将两种情况的结果代入偏导数累加式：

$$
\frac{\partial L}{\partial x_{i}} = \frac{\partial L}{\partial y_{i}} \frac{1}{\text{RMS}(x)} \left( 1 - \frac{y_{i}^{2}}{d} \right) + \sum_{j \neq i} \frac{\partial L}{\partial y_{j}} \left( - \frac{y_{i} y_{j}}{d \cdot \text{RMS}(x)} \right)
$$

提取公因子 $\frac{1}{\text{RMS}(x)}$ 并展开：

$$
\frac{\partial L}{\partial x_{i}} = \frac{1}{\text{RMS}(x)} \left[ \frac{\partial L}{\partial y_{i}} - \frac{\partial L}{\partial y_{i}} \frac{y_{i}^{2}}{d} - \sum_{j \neq i} \frac{\partial L}{\partial y_{j}} \frac{y_{i} y_{j}}{d} \right]
$$

合并括号内的求和项：

$$
\frac{\partial L}{\partial x_{i}} = \frac{1}{\text{RMS}(x)} \left[ \frac{\partial L}{\partial y_{i}} - \frac{y_{i}}{d} \sum_{j=1}^{d} \frac{\partial L}{\partial y_{j}} y_{j} \right]
$$

将其写为更优雅的向量形式：

$$
\frac{\partial L}{\partial x} = \frac{1}{\text{RMS}(x)} \left[ \frac{\partial L}{\partial y} - \frac{y}{d} \left( \left( \frac{\partial L}{\partial y} \right)^{T} y \right) \right]
$$

#### 第四步：证明梯度与输出向量的正交性
我们计算梯度向量 $\frac{\partial L}{\partial x}$ 与归一化输出向量 $y$ 的内积：

$$
\left( \frac{\partial L}{\partial x} \right)^{T} y = \frac{1}{\text{RMS}(x)} \left[ \left( \frac{\partial L}{\partial y} \right)^{T} y - \frac{y^{T} y}{d} \left( \left( \frac{\partial L}{\partial y} \right)^{T} y \right) \right]
$$

注意到输出向量 $y$ 的平方和为：

$$
y^{T} y = \sum_{j=1}^{d} y_{j}^{2} = \sum_{j=1}^{d} \frac{x_{j}^{2}}{\text{RMS}(x)^{2}} = \frac{\sum_{j=1}^{d} x_{j}^{2}}{\frac{1}{d} \sum_{k=1}^{d} x_{k}^{2}} = d
$$

因此，括号内的系数 $\frac{y^{T} y}{d} = 1$。
带入上式：

$$
\left( \frac{\partial L}{\partial x} \right)^{T} y = \frac{1}{\text{RMS}(x)} \left[ \left( \frac{\partial L}{\partial y} \right)^{T} y - 1 \cdot \left( \left( \frac{\partial L}{\partial y} \right)^{T} y \right) \right] = 0
$$

**定理证明完成**。该性质表明：
> [!IMPORTANT]
> **正交梯度特性**：损失函数对输入的梯度向量 $\frac{\partial L}{\partial x}$ 始终与当前层的输出向量 $y$ 保持正交。这从数学上保证了网络在进行梯度更新时，只调整特征的激活方向，而不会在累积更新中导致特征向量的模长发生偏移。这解释了为什么只保留“重缩放（Re-scaling）”就足以确保训练稳定性。

---

### 3. 权重缩放不变性 (Weight Scaling Invariance) 证明

权重缩放不变性可以防止权重由于某些方向的剧烈更新而导致激活值尺度失控。

假设某一层计算为线性映射后接归一化层。输入向量为 $a$ ，权重矩阵为 $W$。
线性输出为：

$$
z = W a
$$

若我们将权重矩阵缩放常数 $c$ 倍（ $c > 0$ ），即 $W' = c W$ ，则线性输出变为：

$$
z' = W' a = c z
$$

我们计算 $z'$ 的均方根 $\text{RMS}(z')$：

$$
\text{RMS}(z') = \sqrt{\frac{1}{d} \sum_{j=1}^{d} (c z_{j})^{2}} = c \cdot \sqrt{\frac{1}{d} \sum_{j=1}^{d} z_{j}^{2}} = c \cdot \text{RMS}(z)
$$

则归一化层在缩放后的输出为：

$$
\text{RMSNorm}(z')_{i} = \frac{z'_{i}}{\text{RMS}(z')} \cdot \gamma_{i} = \frac{c z_{i}}{c \cdot \text{RMS}(z)} \cdot \gamma_{i} = \frac{z_{i}}{\text{RMS}(z)} \cdot \gamma_{i} = \text{RMSNorm}(z)_{i}
$$

**性质证明完成**：

$$
\text{RMSNorm}(c W a) = \text{RMSNorm}(W a)
$$

> [!TIP]
> **训练稳健性**：该性质表明，权重的任意尺度缩放均不会影响归一化层后续的输出，从而防止了参数尺度的发散，极大提升了模型训练的鲁棒性。

---

## 二、 Pre-Norm 与 Post-Norm 梯度流与特征幅值推导

我们以下通过数学公式推导，证明为什么 **Post-Norm 极易梯度消失**，以及为什么 **Pre-Norm 虽然梯度稳定，但会产生表征塌陷**。

### 1. Post-Norm 的梯度衰减证明

在 Post-Norm 架构中，残差块将归一化置于加法之后。以简化的一维模型为例：

$$
x_{l} = \text{Norm}(x_{l-1} + F_{l}(x_{l-1}))
$$

这里我们将 $\text{Norm}$ 简化为除以当前方差（标准差）因子 $\sigma_{l}$ 的形式：

$$
x_{l} = \frac{x_{l-1} + F_{l}(x_{l-1})}{\sigma_{l}}
$$

对输入 $x_{l-1}$ 求偏导（主要提取梯度主干，暂时忽略分母中 $\sigma_{l}$ 的微小偏导）：

$$
\frac{\partial x_{l}}{\partial x_{l-1}} \approx \frac{1}{\sigma_{l}} \left( I + \frac{\partial F_{l}}{\partial x_{l-1}} \right)
$$

对于一个 $L$ 层的深度网络，从最后一层 $x_{L}$ 到第一层输入 $x_{0}$ 的梯度为：

$$
\frac{\partial x_{L}}{\partial x_{0}} = \prod_{l=1}^{L} \frac{\partial x_{l}}{\partial x_{l-1}} \approx \left( \prod_{l=1}^{L} \frac{1}{\sigma_{l}} \right) \prod_{l=1}^{L} \left( I + \frac{\partial F_{l}}{\partial x_{l-1}} \right)
$$

在网络初始化时，各层残差输出与主干累加，使得合并后的方差大于 1，即归一化分母常数恒满足 $\sigma_{l} > 1$ 。
导致前置的连乘项：

$$
\prod_{l=1}^{L} \frac{1}{\sigma_{l}} = O \left( \left( \frac{1}{c} \right)^{L} \right) \quad (\text{其中 } c > 1)
$$

> [!WARNING]
> 当层数 $L$ 较大时，前置连乘项以几何级数（指数级）衰减并迅速趋近于 0。这导致深层梯度传回浅层时发生**严重的梯度消失**。因此，Post-Norm 架构如果不使用极小的 Warmup 学习率进行前期预热，极易直接训练崩溃。

---

### 2. Pre-Norm 的梯度稳定证明

在 Pre-Norm 架构中，归一化被放置在残差分支内部：

$$
x_{l} = x_{l-1} + F_{l}(\text{Norm}(x_{l-1}))
$$

我们将公式展开，可以发现第 $L$ 层的输出可以直接写成最初输入与各分支累加的和：

$$
x_{L} = x_{0} + \sum_{l=1}^{L} F_{l}(\text{Norm}(x_{l-1}))
$$

直接对 $x_{0}$ 求偏导：

$$
\frac{\partial x_{L}}{\partial x_{0}} = I + \sum_{l=1}^{L} \frac{\partial F_{l}(\text{Norm}(x_{l-1}))}{\partial x_{0}}
$$

> [!NOTE]
> **恒等梯度通路**：在上述雅共比矩阵中，右侧存在一个恒等的单位矩阵 $I$ 。即使所有的残差分支 $F_{l}$ 梯度由于饱和等原因全部衰减为 0，输入端的梯度流依然能通过主干通道无损地传回。这保证了 Pre-Norm 能够稳定地训练上百层的网络。

---

### 3. Pre-Norm 的表征塌陷证明

虽然 Pre-Norm 稳定了梯度，但也带来了深层特征退化的问题。
假定归一化后的残差分支输出的方差保持恒定，即：

$$
\text{Var}(F_{l}(\text{Norm}(x_{l-1}))) = \sigma^{2}
$$

由于残差分支的输出和主干上的特征在统计上可以近似为相互独立的，主干上特征的方差随着层数 $L$ 呈线性增长：

$$
\text{Var}(x_{L}) = \text{Var}(x_{0}) + \sum_{l=1}^{L} \text{Var}(F_{l}(\text{Norm}(x_{l-1}))) = \text{Var}(x_{0}) + L \cdot \sigma^{2}
$$

这导致主干网络上的特征模长（标准差尺度）以如下速度增长：

$$
\| x_{L} \| = O(\sqrt{L})
$$

当第 $L$ 层准备将其作为输入传给残差分支时，需要先进行归一化操作：

$$
\text{Norm}(x_{L-1}) \approx \frac{x_{L-1}}{\sqrt{L} \cdot \sigma}
$$

因此，第 $L$ 层残差分支的输出相对于主干特征的比例为：

$$
\frac{\| F_{L}(\text{Norm}(x_{L-1})) \|}{\| x_{L-1} \|} \approx \frac{O(1)}{O(\sqrt{L})} = O\left(\frac{1}{\sqrt{L}}\right)
$$

> [!CAUTION]
> **表征塌陷（Representation Collapse）**：当网络层数 $L \to \infty$ 时，深层残差分支相对主干的实质贡献占比趋近于 0。网络深层退化成了几乎不起作用的恒等映射，导致网络表达能力并没有随着深度线性增加。这就是 Pre-Norm 模型的容量上限在相同参数量下略逊于 Post-Norm 的理论根源。

### 4. Pre-Norm + RMSNorm 组合对梯度稳定与表征塌陷的探讨

在现代大语言模型中，**Pre-Norm + RMSNorm** 是一对极为经典的组合（如 LLaMA, Gemma）。这一组合能完美解决梯度稳定，但无法解决表征塌陷问题。

#### 梯度稳定性：完美解决
因为使用了 Pre-Norm 拓扑结构，其反向传播依然保有恒等梯度通路。而 RMSNorm 梯度的正交性（已在第一部分证明）：

$$
\left( \frac{\partial L}{\partial x} \right)^{T} y = 0
$$

保证了每一层更新时只调整特征的方向，不增加模长，进一步增强了梯度的稳健性。

#### 表征塌陷：完全没有解决
表征塌陷是由 Pre-Norm 的拓扑性质决定的，与使用何种归一化算法（LayerNorm 或 RMSNorm）无关。

当使用 RMSNorm 时，第 $L$ 层的主干方差依旧随着残差分支输出的独立性而累加：

$$
\text{Var}(x_{L}) = \text{Var}(x_{0}) + \sum_{l=1}^{L} \text{Var}(F_{l}(\text{RMSNorm}(x_{l-1}))) = \text{Var}(x_{0}) + L \cdot \sigma^{2}
$$

主干向量的模长依然呈 $\sqrt{L}$ 速度膨胀：

$$
\| x_{L} \| = O(\sqrt{L})
$$

由于 RMSNorm 将向量投影到超球面上，其均方根分母 $\text{RMS}(x_{L-1})$ 也管辖于 $O(\sqrt{L})$ 的量级：

$$
\text{RMS}(x_{L-1}) \approx \sqrt{\frac{1}{d} \sum_{j=1}^{d} x_{L-1, j}^{2}} = O(\sqrt{L})
$$

在将特征送入第 $L$ 层残差分支前，RMSNorm 进行缩放：

$$
\text{RMSNorm}(x_{L-1}) \approx \frac{x_{L-1}}{\text{RMS}(x_{L-1})} = \frac{x_{L-1}}{O(\sqrt{L})}
$$

这导致第 $L$ 层残差分支的输出 $F_{L}(\text{RMSNorm}(x_{L-1}))$ 的相对贡献比为：

$$
\frac{\| F_{L}(\text{RMSNorm}(x_{L-1})) \|}{\| x_{L-1} \|} \approx \frac{O(1)}{O(\sqrt{L})} = O\left(\frac{1}{\sqrt{L}}\right)
$$

当 $L \to \infty$ 时，深层残差分支对主干的贡献依然趋近于 0。因此，哪怕使用更快速的 RMSNorm，也无法从数学上避免 Pre-Norm 带来的表征塌陷。

### 5. 业界缓解与解决表征塌陷的方案

为了同时保留 Pre-Norm 的梯度稳定性和 Post-Norm 不退化的表征能力，研究人员提出了以下几种主流方案：

#### 方案一：DeepNorm（微软提出，彻底解决）
DeepNorm 在 Post-Norm 的基础上，通过在残差连接中引入常数缩放因子并修改权重初始化范围，使其既有 Pre-Norm 的稳定性，又有 Post-Norm 的高精度：

$$
x_{l} = \text{LayerNorm}( \alpha \cdot x_{l-1} + F_{l}(x_{l-1}) )
$$

通过让 $\alpha = (2L)^{1/4}$ 增大主干比例，并对分支的权重初始化乘以一个小于 1 的缩减系数 $\beta = (8L)^{-1/4}$ ，在维持极深网络稳定训练的同时，从数学上保证了深层特征不发生退化。

#### 方案二：可学习门控机制 (ReZero / SkipInit)
在残差连接处引入一个初始值为 0 的可学习标量参数 $\alpha_{l}$ ：

$$
x_{l} = x_{l-1} + \alpha_{l} \cdot F_{l}(x_{l-1})
$$

在训练初期，由于 $\alpha_{l} = 0$ ，网络退化为完美的恒等映射，梯度可以 100% 传回。随着训练的进行，网络会自适应调整 $\alpha_{l}$ 的大小，在特征幅值累积与分支实质贡献之间达到更好的动态平衡。

#### 方案三：门控激活函数与更大表征（SwiGLU 等）
现代大模型广泛使用门控多层感知机（如 SwiGLU / Gated MLP）来替代传统的 FFN。通过双通道非线性投影和乘积门控，每个分支能产生更强的梯度回传和更大的特征表达量，能在一定深度内有效缓解表征塌陷所带来的模型容量损失。

---

## 三、 主流归一化算法综合对比

下面对 BatchNorm、LayerNorm 和 RMSNorm 的归一化维度与核心机制进行横向对比：

| 算法名称　　　　　 | 归一化维度　　　　　　　　　　　　　　| 统计量计算公式　　　　　　　　　　　| 主要特点与优劣势　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| :-------------------| :--------------------------------------| :------------------------------------| :----------------------------------------------------------------------------------------------------------------------------------------------------|
| **BatchNorm (BN)** | 跨 Batch 维度，对每个通道独立归一化　 | 均值 $\mu_{c}$，标准差 $\sigma_{c}$ | **优势**：引入 Batch 噪声起正则化作用，加速 CNN 收敛。<br>**劣势**：受 Batch Size 限制大；不适用于 NLP 变长输入。　　　　　　　　　　　　　　　　　 |
| **LayerNorm (LN)** | 跨 Feature 维度，对每个样本独立归一化 | 均值 $\mu_{n}$，标准差 $\sigma_{n}$ | **优势**：不依赖 Batch Size，极度适用于序列长度可变的 NLP 任务。<br>**劣势**：需要二次遍历特征数据，在大模型中计算占比较高。　　　　　　　　　　　　|
| **RMSNorm (RMSN)** | 跨 Feature 维度，对每个样本独立归一化 | 均方根 $\text{RMS}_{n}$　　　　　　 | **优势**：不计算均值，减少 7% - 64% 的计算开销，降低显存带宽压力；效果等价于 LN。<br>**劣势**：对特征均值绝对值极其敏感的特定网络可能会有微弱影响。 |
