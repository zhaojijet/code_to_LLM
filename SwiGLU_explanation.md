# 深度解析：SwiGLU 激活函数与 FFN 结构演进及数学证明

本篇文档对大语言模型（LLM）中广泛采用的 **SwiGLU** 激活函数进行深入剖析，并结合手写板书 `swiglue.png` 中的内容，对比分析 **ReLU**、**GELU** 与 **SwiGLU** 的数学定义、导数性质、结构演进及优劣势。

---

## 一、 `swiglue.png` 板书内容结构解读

板书 `swiglue.png` 从三个维度梳理了激活函数从传统多层感知机（MLP）到现代大语言模型中的演进过程：

### 1. 激活函数的平滑演进（板书左上 & 右上区域）
- **ReLU 阶段**：传统激活函数 $f(x) = \max(0, x)$ ，在 $x < 0$ 时输出恒为 0，导数为 0 （即板书中的 `max(0, x)` 标识）。
- **GELU 阶段**：引入了平滑性，在 $x < 0$ 时存在一个极小的负值区域（即板书中标注的 `<0` 区域），从而允许在此区间传播微小的梯度。
- **GELU 与 Swish 的形状等价性**：板书右侧指出， $\text{Swish}(x) = x \cdot \sigma(\beta x)$ 与 $\text{GELU}(x) = x \cdot \Phi(x)$ 在图形上“基本一样”，可通过选择适当的缩放因子进行互相代替，且 SwiGLU 在实际实验中表现更优。

### 2. 门控激活机制 GLU（板书左下区域）
- **非门控结构（xLU）**：
  
  $$
  \text{xLU}(x) = \text{Act}(x \cdot W_1) \cdot W_2
  $$
  
- **门控结构（xGLU）**：
  
  $$
  \text{xGLU}(x) = (\text{Act}(x \cdot W_1) \otimes (x \cdot V)) \cdot W_2
  $$
  
  其中 $\otimes$ 代表逐元素相乘（Element-wise Product）。输入特征 $x$ 一分为二，一条通道进行非线性激活 $\text{Act}(x \cdot W_1)$ （作为门控信号），另一条通道进行线性映射 $x \cdot V$ ，两者相乘后经过 $W_2$ 投影输出。

### 3. 参数量与算力的折中（板书右下区域）
- 板书中提到 **“缩小 $W_1$ ，以换 $V$ ；算力相同但表达力更丰富”**。这解释了在引入门控投影分支 $V$ 后，为了保持 FFN 整体参数量与 FLOPs 相比于标准 FFN 不变，需要适当降低中间隐层的维度 $d_{\text{ff}}$ （通常缩减至原来的 $2/3$ ），从而在相同的算力预算下获取更强的表达能力。

---

## 二、 ReLU 与 GELU 的数学定义与推导

为了深刻理解门控机制，我们需要先推导基础激活函数 ReLU 和 GELU 的导数性质。

### 1. ReLU (Rectified Linear Unit)

#### 数学定义
ReLU 是最经典的分段线性激活函数：

$$
\text{ReLU}(x) = \max(0, x)
$$

#### 导数推导
当 $x \neq 0$ 时，其导数为：

$$
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}
$$

在 $x = 0$ 处不可导。在深度学习框架（如 PyTorch）的实际反向传播中，通常将次梯度定义为：

$$
\text{ReLU}'(0) = 0 \quad (\text{或 } 0.5)
$$

#### 优劣势分析
- **优势**：计算极其简单，只有比较和置零操作；在正区间导数恒为 1，不存在梯度饱和问题，极大地缓解了深层网络中的梯度消失。
- **劣势（死神经问题 Dying ReLU）**：若输入为负值，其输出和导数恒为 0。一旦某个神经元的权重更新使其输入在所有训练样本上均为负，该神经元将再也无法获得梯度更新，成为“死神经元”。

---

### 2. GELU (Gaussian Error Linear Unit)

#### 数学定义
GELU 是一种概率加权激活函数。它根据输入 $x$ 的值决定保留信息的概率，假设输入服从标准正态分布 $X \sim \mathcal{N}(0, 1)$ ：

$$
\text{GELU}(x) = x \cdot P(X \le x) = x \cdot \Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF）：

$$
\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-\frac{t^2}{2}} dt = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]
$$

代入 CDF 后，GELU 的精确表达式为：

$$
\text{GELU}(x) = \frac{1}{2} x \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]
$$

#### 导数推导
使用乘积求导法则，对 $f(x) = x \Phi(x)$ 求导：

$$
\text{GELU}'(x) = \Phi(x) + x \cdot \Phi'(x)
$$

根据 CDF 与概率密度函数（PDF）的关系， $\Phi'(x) = \phi(x)$ 是标准正态分布的 PDF：

$$
\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}
$$

代入可得 GELU 的一阶导数公式：

$$
\text{GELU}'(x) = \Phi(x) + \frac{x}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}
$$

当 $x \to +\infty$ 时， $\Phi(x) \to 1$ ， $e^{-\frac{x^2}{2}} \to 0$ ，因此 $\text{GELU}'(x) \to 1$。
当 $x \to -\infty$ 时， $\Phi(x) \to 0$ ， $x e^{-\frac{x^2}{2}} \to 0$ ，因此 $\text{GELU}'(x) \to 0$。

#### Sigmoid 近似公式与常数 $1.702$ 的推导证明
由于高斯误差函数 $\text{erf}(x)$ 的计算代价较高，早期常使用 Sigmoid 函数进行近似：

$$
\Phi(x) \approx \sigma(k \cdot x) = \frac{1}{1 + e^{-k \cdot x}}
$$

我们通过**匹配原点处（ $x = 0$ ）的导数**来求解缩放常数 $k$。

1. **计算正态 CDF $\Phi(x)$ 在 $x = 0$ 处的导数**：
   
   $$
   \Phi'(0) = \phi(0) = \frac{1}{\sqrt{2\pi}} \approx 0.398942
   $$
   
2. **计算近似函数 $\sigma(k \cdot x)$ 在 $x = 0$ 处的导数**：
   
   $$
   \frac{d}{dx} [\sigma(k \cdot x)] = k \cdot \sigma(k \cdot x)(1 - \sigma(k \cdot x))
   $$
   
   在 $x = 0$ 处， $\sigma(0) = 0.5$ ，代入得：
   
   $$
   \frac{d}{dx} [\sigma(k \cdot x)]\Big|_{x=0} = k \cdot 0.5 \cdot (1 - 0.5) = 0.25 k
   $$
   
3. **匹配两者的导数**：
   
   $$
   0.25 k = \frac{1}{\sqrt{2\pi}} \implies k = \frac{4}{\sqrt{2\pi}} \approx 1.595769
   $$
   
   利用该 $k$ 值，在原点附近的近似度非常高。

> [!NOTE]
> **为什么最终使用的是 1.702？**
> 匹配原点处的导数仅保证了在 $x = 0$ 处的局部切线重合。为了在整个实数定义域 $x \in (-\infty, \infty)$ 上使得最大绝对误差最小（即 Minimax 近似问题）：
> 
> $$
> \min_k \max_{x \in \mathbb{R}} | \Phi(x) - \sigma(k \cdot x) |
> $$
> 
> 经过数值优化求解，得到最优常数为 $k \approx 1.702$ 。此时，最大绝对误差小于 $0.0095$ （不足 $1\%$ ）。这也是经典近似公式 $\text{GELU}(x) \approx x \cdot \sigma(1.702 \cdot x)$ 的数学根源。

---

## 三、 Swish 与 SwiGLU 的数学定义与推导

板书右下侧提到，Swish 与 GELU 类似，且在实验数据中更优。我们在本节推导其公式和 FFN 结构的演进。

### 1. Swish / SiLU 激活函数

#### 数学定义
Swish 激活函数在 $\beta = 1$ 时也称为 **SiLU (Sigmoid Linear Unit)**：

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

#### 导数推导
根据乘积求导法则与 Sigmoid 导数性质 $\sigma'(x) = \sigma(x)(1 - \sigma(x))$，对 $\text{Swish}(x)$ 求一阶导数：

$$
\text{Swish}'(x) = \sigma(x) + x \cdot \sigma'(x)
$$

$$
\text{Swish}'(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))
$$

$$
\text{Swish}'(x) = \sigma(x) \left[ 1 + x(1 - \sigma(x)) \right]
$$

由于 $\text{Swish}(x) = x \sigma(x)$，上式也可以改写为：

$$
\text{Swish}'(x) = \sigma(x) + \text{Swish}(x)(1 - \sigma(x))
$$

或者利用 $\sigma(-x) = 1 - \sigma(x)$，写为：

$$
\text{Swish}'(x) = \sigma(x)(1 + x \cdot \sigma(-x))
$$

---

### 2. GLU (Gated Linear Unit) 门控线性单元

GLU 是由 Dauphin 等人在 2016 年提出的一种门控机制，其标准形式为：

$$
\text{GLU}(x) = (x \cdot W + b) \otimes \sigma(x \cdot V + c)
$$

其中 $W$ 和 $V$ 为线性映射矩阵， $b$ 和 $c$ 为偏置， $\otimes$ 为逐元素乘法（哈达玛积，Hadamard Product）。第二项 $\sigma(x \cdot V + c)$ 作为“门（Gate）”，控制第一项的信息通过比例。

---

### 3. SwiGLU 门控激活函数

Shazeer (2020) 提出了 GLU 的变体，即将门控分支的 Sigmoid 激活函数替换为其他的激活函数（如 ReLU, GELU, Swish）。
当采用无偏置的 Swish 时，其定义为 **SwiGLU**：

$$
\text{SwiGLU}(x) = \text{Swish}(x \cdot W_1) \otimes (x \cdot V)
$$

对应地，在大语言模型的 FFN（Feed-Forward Network）层中，SwiGLU 结构定义为：

$$
\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{Swish}(x \cdot W_1) \otimes (x \cdot V) \right) \cdot W_2
$$

这就是板书第 2 部分展示的公式。

---

### 4. 参数量与算力（FLOPs）对比证明

我们将传统的 FFN 与 SwiGLU FFN 进行严格的参数量与算力开销对比。设输入维度为 $d$，隐层维度为 $d_{\text{ff}}$。

#### 方案 A：传统 FFN (以 ReLU/GELU 为例，无偏置)

其前向传播计算为：

$$
\text{FFN}_{\text{std}}(x) = \text{Act}(x \cdot W_1) \cdot W_2
$$

其中 $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$， $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$。

1. **参数量 (Parameters)**：
   
   $$
   P_{\text{std}} = d \cdot d_{\text{ff}} + d_{\text{ff}} \cdot d = 2 \cdot d \cdot d_{\text{ff}}
   $$
   
2. **计算量 (FLOPs，以单 Token 计算)**：
   - 投影分支 $x \cdot W_1$：输入 $[1 \times d]$ 与权重 $[d \times d_{\text{ff}}]$ 矩阵相乘，计算量为 $2 \cdot d \cdot d_{\text{ff}}$ FLOPs。
   - 激活函数 $\text{Act}(\cdot)$：逐元素操作，计算量为 $O(d_{\text{ff}})$ （在大模型中通常忽略不计）。
   - 输出投影 $W_2$：输入 $[1 \times d_{\text{ff}}]$ 与权重 $[d_{\text{ff}} \times d]$ 矩阵相乘，计算量为 $2 \cdot d \cdot d_{\text{ff}}$ FLOPs。
   - **总计算量**：
     
     $$
     F_{\text{std}} \approx 4 \cdot d \cdot d_{\text{ff}}
     $$

#### 方案 B：SwiGLU FFN (无偏置)

其前向传播计算为：

$$
\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{Swish}(x \cdot W_1) \otimes (x \cdot V) \right) \cdot W_2
$$

其中 $W_1 \in \mathbb{R}^{d \times d_{\text{ff\_gated}}}$， $V \in \mathbb{R}^{d \times d_{\text{ff\_gated}}}$， $W_2 \in \mathbb{R}^{d_{\text{ff\_gated}} \times d}$。

1. **参数量 (Parameters)**：
   
   $$
   P_{\text{SwiGLU}} = d \cdot d_{\text{ff\_gated}} + d \cdot d_{\text{ff\_gated}} + d_{\text{ff\_gated}} \cdot d = 3 \cdot d \cdot d_{\text{ff\_gated}}
   $$
   
2. **计算量 (FLOPs，以单 Token 计算)**：
   - 投影分支 $x \cdot W_1$ 与 $x \cdot V$：两次矩阵相乘，共 $2 \cdot (2 \cdot d \cdot d_{\text{ff\_gated}}) = 4 \cdot d \cdot d_{\text{ff\_gated}}$ FLOPs。
   - 逐元素操作（Swish 激活与逐元素相乘 $\otimes$）：计算量为 $O(d_{\text{ff\_gated}})$ （忽略不计）。
   - 输出投影 $W_2$：计算量为 $2 \cdot d_{\text{ff\_gated}} \cdot d$ FLOPs。
   - **总计算量**：
     
     $$
     F_{\text{SwiGLU}} \approx 6 \cdot d \cdot d_{\text{ff\_gated}}
     $$

#### 结论：参数与算力等价转换 (缩小 $W_1$ 以换 $V$)

若要使 SwiGLU FFN 的参数量和计算量与传统 FFN 保持一致：

1. **参数量一致**：
   
   $$
   P_{\text{SwiGLU}} = P_{\text{std}} \implies 3 \cdot d \cdot d_{\text{ff\_gated}} = 2 \cdot d \cdot d_{\text{ff}} \implies d_{\text{ff\_gated}} = \frac{2}{3} d_{\text{ff}}
   $$
   
2. **计算量一致**：
   
   $$
   F_{\text{SwiGLU}} = F_{\text{std}} \implies 6 \cdot d \cdot d_{\text{ff\_gated}} = 4 \cdot d \cdot d_{\text{ff}} \implies d_{\text{ff\_gated}} = \frac{2}{3} d_{\text{ff}}
   $$
   

> [!TIP]
> **大模型最佳实践**：
> 在经典的 Standard Transformer 中，中间层维度通常设为 $d_{\text{ff}} = 4d$ 。
> 按照上面的推导，为了保持相同的计算开销，SwiGLU 结构的隐层维度应当设为：
> 
> $$
> d_{\text{ff\_gated}} = \frac{2}{3} \cdot 4d = \frac{8}{3}d
> $$
> 
> 比如在 LLaMA-7B 中，模型隐向量维度 $d = 4096$ 。按比例计算：
> 
> $$
> d_{\text{ff\_gated}} = \frac{8}{3} \times 4096 \approx 10922.67
> $$
> 
> LLaMA 实际为了显存对齐，将该维度向上取整到 256 的倍数，最终定为 **$11008$** （即 $43 \times 256$ ）。
> 这印证了板书中的结论：**“缩小 $W_1$ 以换 $V$”**。我们削减了单个分支的宽度（从 $4d$ 降到 $\approx 2.68d$ ），增加了用于控制信息流的门控投影矩阵 $V$ ，使得计算量与参数量完全等效，而门控机制赋予了模型更强的表达力。

---

## 四、 ReLU、GELU 与 SwiGLU 的综合对比与异同优劣势

本节将这几种核心激活函数进行多维度的横向对比，深入分析其数学特性与训练表现。

### 1. 异同与数学性质对比

- **单调性与平滑度**：
  - ReLU 在原点 $x = 0$ 处不连续且不可导，左半部分是死区（梯度恒等于零）。
  - GELU 和 Swish 在实数域上处处光滑可导，且都是非单调的（在负半轴有一个向下凹的小坑，即板书中特别标注的 `<0` 的负区）。
- **门控性质**：
  - ReLU 和 GELU/Swish 属于“自门控”（Self-gated），因为它们的输出形如 $x \cdot f(x)$ ，其中激活状态由自身作为自变量决定。
  - SwiGLU 则是“双输入分支门控”（Bilinear/Gate），它将输入 $x$ 的不同映射结果（ $x W_1$ 和 $x V$ ）进行相乘。这解耦了“激活强度计算”与“数据流通过门限”，让网络学会更加精细的特征筛选。

---

### 2. 优劣势多维度对比表

| 激活函数 | 核心计算公式 | 导数连续性 / 平滑度 | 优势 (Pros) | 劣势 (Cons) | 典型应用 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ReLU** | $\max(0, x)$ | 不连续（ $x=0$ 突变） | 1. 计算极快（仅阈值比较）<br>2. 正半轴导数恒为 1，无梯度饱和。 | 1. 存在“死神经”（Dying ReLU）问题。<br>2. 均值非零，导致层间均值偏移。 | 传统 CNN, ResNet, 早期 Transformer |
| **GELU** | $x \cdot \Phi(x)$ | 处处平滑连续 | 1. 缓解死神经问题，允许小负梯度传播。<br>2. 引入随机正则化思想，表征能力强。 | 1. 正确公式计算量大，通常需要 Sigmoid 近似。<br>2. 无门控矩阵解耦。 | BERT, GPT-3, ViT 等主流 Transformer 模型 |
| **Swish / SiLU** | $x \cdot \sigma(x)$ | 处处平滑连续 | 1. 与 GELU 曲线极度相似（可以通过参数缩放等效）。<br>2. 计算代价比高精度 GELU 略低。 | 1. 仍然是自门控，无法像 GLU 一样实现两个独立分支交互。 | EfficientNet, 早期 LLaMA 探索版 |
| **SwiGLU** | $\text{Swish}(x W_1) \otimes (x V)$ | 处处平滑连续 | 1. 引入双分支门控机制，表达能力极强。<br>2. 相同计算量 and 参数量预算下（即缩减隐层维度后），收敛速度与最终困惑度（PPL）显著优于前三者。 | 1. 每次前向计算涉及三个线性投影矩阵，显存带宽占用稍大。<br>2. 对隐层维度缩放设计较为复杂。 | LLaMA 系列, Mistral, DeepSeek 等主流 LLM 骨干网络 |

---

### 3. 为什么 SwiGLU 在大模型中表现最好？

1. **更平滑的梯度流**：
   与 ReLU 粗暴的硬截断相比，SwiGLU 的非线性激活部分（Swish）保留了对负输入的平滑小幅度响应。这避免了网络在训练初期由于大量死神经元导致的表示能力崩溃。
2. **多项式表达能力（Bilinear Gating）**：
   传统激活函数仅实现对单路特征的变换。SwiGLU 形式上等价于两路不同线性投影特征的乘积（双线性池化思想），这使得它能够以单层结构拟合特征之间的二阶交叉项。这种乘性交互机制使得参数的利用效率显著提升。
3. **自适应的软门控**：
   在门控机制中， $\text{Swish}(x W_1)$ 的输出范围在大约 $[-0.09, \infty)$ 。当门控分支响应为负时，它不是直接将其抹平，而是根据其相对重要性分配非常微弱的负号响应，这种自适应能力大幅提高了网络的泛化上限。


