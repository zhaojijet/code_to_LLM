# 深度解析：Attention Residuals 算法与 HC/mHC 架构对比及数学推导

本篇文档对大语言模型（LLM）架构中关于残差连接演进的最新前沿进展进行深入剖析。我们将结合板书 `attentionResidule.png` 中的结构图，详细分析 Moonshot AI 提出的 **Attention Residuals (AttnRes)** 算法，并与字节跳动提出的 **Hyper-Connections (HC)** 以及 DeepSeek 提出的 **Manifold-Constrained Hyper-Connections (mHC)** 进行多维度对比，提供严格的数学公式推导与稳定性证明。

---

## 一、 `attentionResidule.png` 板书结构解读

板书 `attentionResidule.png` 从左至右展示了 Transformer 中信息流残差连接从经典方案到动态路由方案的演进路径：

### 1. (a) Standard Residuals（标准残差连接）
* **结构流向**：Embedding -> Attention -> + -> MoE -> + -> Attention -> + -> MoE -> ...
* **特点**：每一层的输出通过固定的单位权重直接累加到残差主干（Residual Stream）上。即对于层 $l$ ，其输出为 $h_l = h_{l-1} + f(l-1)(h_{l-1})$ 。由于每一层的加权系数固定为 1，随着网络加深，早期的特征表示极易被后续层的海量叠加信号所“稀释”。

### 2. (b) Full Attention Residuals（全注意力残差连接）
* **结构流向**：Embedding 作为初始状态 $v_0$ 。在每一层（Attention 或 MoE）前，均有一个由圆圈 $\alpha$ 和小方块 $w$ 构成的 **AttnRes 算子**。
* **特点**：去除了所有传统残差的加号（ $+$ ）。每一层的输入 $x_l$ 不再是前一层的直接输出与残差之和，而是**通过 softmax 深度注意力机制，动态选择性地聚合之前所有层输出的加权和**。
* **算子原理**（见板书中部的放大虚线框 `AttnRes Op (α)`）：
  * 输入表示集合 $V = [v_0, v_1, \dots, v_{l-1}]$ 对应 Attention 的 Key 和 Value。
  * 小方块 $w$ 代表针对当前层 $l$ 的**可学习伪查询向量（Pseudo-Query）** $q_l$ 。
  * 经过与 Key 进行点积和 Softmax，生成注意力分布 $\alpha$，对 $V$ 进行加权汇聚，作为当前层的输入。
* **空间复杂度**：由于需要缓存并检索之前所有层的输出，在深度为 $L$ 、维度为 $d$ 且序列长度为 $S$ 的情况下，存储开销为 $O(Ld)$ （单 token），总路由路径为 $O(L^2)$ 。

### 3. (c) Block Attention Residuals（分块注意力残差连接）
* **结构流向**：将网络划分为若干个 Block（如 `Block n-1`, `Block n-2` 等）。
* **特点**：
  * **Block 内部**：依然保留传统残差的均匀累加（ $+$ ）以保证计算局部的高效性。
  * **Block 边界**：跨 Block 之间采用 Attention Residuals 进行动态聚合，即圆圈 $\alpha$ 只对前序各个 Block 的最终输出进行选择性聚合。
* **空间复杂度**：若网络共包含 $N$ 个 Block，跨 Block 的动态路由开销从 $O(Ld)$ 降至 $O(Nd)$ ，大大缓解了长文本与深层模型下的显存与计算瓶颈。

---

## 二、 经典 ResNet 设计与 Transformer 中 PreNorm vs PostNorm 的深层博弈

### 1. ResNet 的初衷与 Transformer 的困境
在经典残差网络（ResNet，He et al., 2015）中，残差连接的形式为：

$$
y = f(x) + x
$$

其设计初衷是通过直连通道（Identity Highway）将损失函数 $\mathcal{L}$ 的梯度无阻碍地传回底层：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left( I + \frac{\partial f}{\partial x} \right)
$$

其中的恒等映射项 $I$ 保证了即使残差分支的梯度极小，也能将底层梯度无损传回，从而解决了超深层 CNN 训练失效的问题。

然而，在 **Transformer** 架构中，自注意力机制（Self-Attention）和前馈网络（FFN）包含大量的点积计算与矩阵乘法，激活值和梯度的方差在连续累加中会呈指数级增长。为了稳定信号尺度，**必须引入归一化（LayerNorm / RMSNorm）**。如果不加归一化，模型在初始化阶段就会直接崩溃。

这就在 Transformer 中引入了 Normalization 摆放位置的博弈：

#### PostNorm (层后归一化，Transformer 早期标准)
其 LayerNorm 作用在相加之后：

$$
h_l = \text{LayerNorm}(h_{l-1} + f(h_{l-1}))
$$

* **优势**：没有稀释效应。每次相加后都通过 LayerNorm 强行将尺度拉回 $O(1)$ ，每一层对隐藏状态的相对修改幅度都在同一个数量级上。
* **致命缺陷（梯度回传阻断）**：因为 LayerNorm 位于残差主干（加法）的外侧，反向传播的梯度必须穿透每一层的 LayerNorm 雅可比矩阵（Jacobian）。随着层数 $L$ 的增加，传回底层的梯度会以 $O(1/L)$ 甚至指数级衰减，引起严重的梯度消失或爆炸。因此深层 PostNorm Transformer 极难收敛。

#### PreNorm (层前归一化，现代大模型标配)
将 LayerNorm 移入分支，使主干残差通道保持畅通：

$$
y_l = f_l(\text{RMSNorm}(h_{l-1}))
$$

$$
h_l = h_{l-1} + y_l
$$

这使得梯度能够无损传回，但代价是带来了 **PreNorm 稀释效应 (PreNorm Dilution)**。

---

### 2. PreNorm 稀释效应（PreNorm Dilution）数学证明
将 PreNorm 的递推公式展开，最终第 $L$ 层的隐藏状态为：

$$
h_L = h_0 + \sum_{l=1}^L y_l
$$

#### 隐藏状态的模长增长分析
在初始化或训练早期，假设各个子层输出 $y_l$ 在空间上大致正交且独立同分布，其方差为 $\sigma^2$ 。根据独立随机变量和的性质，隐藏状态的方差随深度线性增长：

$$
\text{Var}(h_L) = \text{Var}(h_0) + \sum_{l=1}^L \text{Var}(y_l) = \text{Var}(h_0) + L\sigma^2
$$

因此，隐藏状态 $h_L$ 的欧氏模长随层数呈平方根级增长：

$$
\|h_L\|_2 \approx O(\sqrt{L})
$$

在网络收敛阶段，由于各层输出协同，极端情况下模长甚至随层数呈线性增长：

$$
\|h_L\|_2 \approx O(L)
$$

#### 后续层贡献稀释的数学证明
由于下一层输入前必须经过归一化，以 RMSNorm 为例：

$$
\text{RMSNorm}(h_{l-1}) = \frac{h_{l-1}}{\sqrt{\frac{1}{d} \|h_{l-1}\|_2^2 + \epsilon}}
$$

代入第 $l$ 个子层的输出：

$$
y_l = f_l\left( \frac{h_{l-1}}{\|h_{l-1}\|_2 / \sqrt{d}} \right)
$$

由于归一化限制了输入方差为 1，在合理初始化下，每一层的输出模长是有界的：

$$
\mathbb{E}[\|y_l\|_2] \approx C_0
$$

我们分析第 $l$ 层对最终状态 $h_L$ 的实际相对贡献率 $\gamma_l$ ：

$$
\gamma_l = \frac{\|y_l\|_2}{\|h_L\|_2} \approx \frac{C_0}{O(\sqrt{L}) \text{ to } O(L)}
$$

由此可得：

$$
\lim_{L \to \infty} \gamma_l = 0
$$

**结论**：随着模型加深，后续层在累加时所面对的残差主干基数 $h_{l-1}$ 的模长变得极大，导致后续层对特征的实际修改能力无限趋近于 0。前序累加的状态如同巨大的洪流，后续微弱的新增信号根本无法动摇洪流的方向。

---

### 3. 不加 PreNorm（换回 PostNorm 或去掉 Norm）是否可行？
既然 PreNorm 存在严重的稀释效应，我们直接换回 PostNorm 或者完全不加归一化可以吗？
答案是**行不通**的：
* **换回 PostNorm**：对于层数较深（如 32 到 80 层）的大语言模型，PostNorm 在训练初期的几步就会直接导致梯度爆炸或数值溢出，模型完全无法收敛。
* **完全去掉 Norm**：由于没有归一化压制尺度，激活值会在注意力矩阵与前馈投影中以指数速度向外扩散，导致特征崩塌，模型甚至无法完成初始化前向传播。

### 4. AttnRes 与 mHC 的破局之道
为了突破“PostNorm 无法训练，PreNorm 被动稀释”的经典悖论，新一代架构（如 Kimi 的 AttnRes 和 DeepSeek 的 mHC）**抛弃了传统的算术相加连接，改用加权平均（凸组合）**：

* **Kimi AttnRes** 采用 Softmax 注意力对前序所有层进行混合：
  
  $$
  x_l = \sum_{i=0}^{l-1} \alpha_{l, i} v_i
  $$
  
  因为 $\sum \alpha = 1$ （凸组合性质），它强行让输入状态 $x_l$ 的模长限制在 $O(1)$ 范围内，既保留了类似 PreNorm 的极佳梯度传播路径，又彻底消除了 PreNorm 累加导致的稀释。
* **DeepSeek mHC** 通过双随机矩阵 $H_l$ 将多流通道混合的谱范数锁定为 1：
  
  $$
  X_{l+1} = H_l X_l + F_l(X_l)
  $$
  
  从而在多通道流之间实现了模长守恒的特征混合，既保留了恒等映射的梯度流，又控制了数值爆炸。

---

## 三、 Attention Residuals (AttnRes) 详解

Kimi 团队提出的 Attention Residuals (AttnRes) 正是为了打破上述 PreNorm 的无差别单位累加带来的退化，用**动态的 Softmax 门控**代替了原本的算术加法。

### 1. 算法数学定义
在 Full AttnRes 中，第 $l$ 个子层的输入 $x_l$ 不再等于 $h_{l-1}$ ，而是之前所有子层输出 $v_i$ 的动态注意力加权和：

$$
x_l = \sum_{i=0}^{l-1} \alpha_{l, i} \cdot v_i
$$

其中 $v_0$ 为 Embedding 输出，而对于 $i \ge 1$ ， $v_i$ 为第 $i$ 个子层的直接输出：

$$
v_i = f_i(\text{RMSNorm}(x_i))
$$

### 2. 注意力权重 $\alpha_{l, i}$ 的计算公式
对于第 $l$ 层，其计算对前序 $i$ 层的关注权重时，使用了一个**可学习的层专属伪查询向量（Pseudo-Query）** $q_l \in \mathbb{R}^d$ 。该向量不依赖于当前的 Token 输入，而是模型参数的一部分：

$$
e_{l, i} = \frac{q_l^T \cdot \text{RMSNorm}(v_i)}{\sqrt{d}}
$$

$$
\alpha_{l, i} = \text{softmax}_i (e_{l, i}) = \frac{\exp(e_{l, i})}{\sum_{j=0}^{l-1} \exp(e_{l, j})}
$$

### 3. 设计美学与关键点分析

#### 伪查询向量 $q_l$ 的解耦作用
普通的 Self-Attention 中 Query 是由当前输入 Token 线性投影出来的，因此其注意力取决于词汇和语义上下文。而 AttnRes 的 $q_l$ 是**与输入无关的可学习全局参数**。
这保证了其学到的是一种**拓扑级的信息路由图谱**。例如，网络可以学会“第 40 层永远需要高度关注第 2 层的基本表征，而忽略中间的杂音”。

#### 对 Key 进行 RMSNorm 的必要性（板书虚线框所示）
在计算点积前，必须对 $v_i$ 进行归一化： $k_i = \text{RMSNorm}(v_i)$ 。
若不对 $v_i$ 归一化，由于不同层的输出尺度天然存在差异，模长较大的层将在 Softmax 计算中占据绝对统治地位，导致注意力退化为单热（One-hot）选择，产生“路由坍缩”。归一化后，注意力机制能够纯粹基于特征方向的相似度进行路由分配。

#### 信息检索视角
传统 Transformer 像是一个栈（Stack），信息只能被动地向下传递和层层压紧；而 AttnRes 赋予了模型在深度方向上的**检索（Retrieval）**能力。高层可以像查询 Key-Value 数据库一样，精准地把深埋在底层的原始语义“捞”上来。

---

## 四、 HC 与 mHC 异同及优劣势对比分析

在 AttnRes 之前，学术界和工业界也曾尝试过通过拓扑结构的重构来解决深层残差网络的问题。其中最典型的是字节跳动的 **Hyper-Connections (HC)** 与 DeepSeek 的 **Manifold-Constrained Hyper-Connections (mHC)**。

### 1. 字节跳动 Hyper-Connections (HC)
为了扩宽残差主干的“宽度”，HC 将单条残差通道扩展为 $C$ 条平行的流（Streams）。
设第 $l$ 层的多流隐藏状态表示为矩阵 $X_l = [x_{l, 1}, x_{l, 2}, \dots, x_{l, C}]^T \in \mathbb{R}^{C \times d}$ 。
其演进方式为：

$$
X'_{l} = H_l \cdot X_l
$$

$$
X_{l+1} = X'_{l} + F_l(X_l)
$$

其中 $H_l \in \mathbb{R}^{C \times C}$ 是一个**完全可学习的无约束混合矩阵（Mixing Matrix）**，用于在不同的残差流之间进行线性混合与信息跨流交换。

#### 局限性与不稳定性证明
在传统的残差网络中，残差矩阵是恒等映射 $I$ 。而在 HC 中，残差主干的映射转为 $H_l$ 。
若不对 $H_l$ 施加限制，其 spectral norm （谱范数，即最大奇异值） $\|H_l\|_2$ 无法保证恒等于 1 。
我们来考察多层级联后的信号传播：
从输入 $X_0$ 传导到第 $L$ 层，忽略非线性子层 $F_l$ 的微扰，残差通道的线性传播算子为：

$$
\mathcal{P} = \prod_{l=1}^L H_l
$$

根据算子范数的次可乘性：

$$
\|\mathcal{P}\|_2 \le \prod_{l=1}^L \|H_l\|_2
$$

* **梯度爆炸/信号膨胀**：如果可学习矩阵的参数更新导致 $\|H_l\|_2 > 1$ （例如存在奇异值大于 1），随着网络层数 $L$ 的增加，信号模长将会以指数级增长：
  
  $$
  \|X_L\|_2 \approx O(\lambda_{\max}^L) \cdot \|X_0\|_2 \quad (\lambda_{\max} > 1)
  $$
  
  这在 DeepSeek 的实验中被证实会导致输出信号膨胀达 3000 倍以上，导致网络无法收敛。
* **梯度消失**：反之，若 $\|H_l\|_2 < 1$ ，残差主干上的信号将以指数速度衰减到 0，退化为无残差的深层网络，引起严重的梯度消失。

### 2. DeepSeek Manifold-Constrained Hyper-Connections (mHC)
为了解决原始 HC 破坏恒等映射（Identity Mapping）所带来的数值毁灭性灾难，DeepSeek 提出了 mHC。其核心思想是：将混合矩阵 $H_l$ **强制约束在 Birkhoff 多面体（Birkhoff Polytope）这一特定流形（Manifold）上**。

#### Birkhoff 多面体与双随机矩阵（Doubly Stochastic Matrix）
Birkhoff 多面体 $\mathcal{B}_C$ 是所有 $C \times C$ 维 **双随机矩阵**的集合。一个矩阵 $H$ 为双随机矩阵，必须同时满足以下三个条件：

1. 非负性：

$$
H_{ij} \ge 0 \quad (\forall i, j \in \{1, \dots, C\})
$$

2. 行和为 1：

$$
\sum_{j=1}^C H_{ij} = 1 \quad (\forall i \in \{1, \dots, C\})
$$

3. 列和为 1：

$$
\sum_{i=1}^C H_{ij} = 1 \quad (\forall j \in \{1, \dots, C\})
$$

#### mHC 稳定性的数学证明（凸组合与谱界限限制）

**定理**：任何双随机矩阵 $H \in \mathcal{B}_C$ 的谱范数（最大奇异值）恒满足 $\|H\|_2 = 1$ 。

**证明**：
根据 **Birkhoff-von Neumann 定理**，任何双随机矩阵 $H$ 都可以表示为有限个置换矩阵（Permutation Matrices） $P_k$ 的凸组合（Convex Combination）：

$$
H = \sum_{k} \theta_k P_k \quad \text{其中 } \theta_k \ge 0 \text{ 且 } \sum_k \theta_k = 1
$$

对于任何一个置换矩阵 $P_k$，由于它仅仅是对坐标轴进行重新排列，它是一个正交矩阵。对于任何向量 $z$，有：

$$
\|P_k z\|_2 = \|z\|_2 \implies \|P_k\|_2 = 1
$$

我们利用谱范数的三角不等式对凸组合进行展开：

$$
\|H\|_2 = \left\| \sum_{k} \theta_k P_k \right\|_2 \le \sum_{k} \theta_k \|P_k\|_2
$$

由于 $\|P_k\|_2 = 1$，上式简化为：

$$
\|H\|_2 \le \sum_{k} \theta_k \cdot 1 = 1
$$

由于非负矩阵的性质，它的最大奇异值必然等于 1。因此，我们严格证明了双随机约束能够确保：

$$
\|H_l\|_2 = 1
$$

这就将残差主干的多流线性传播算子 $\mathcal{P}$ 的谱范数严格锁定在有界范围内：

$$
\|\mathcal{P}\|_2 \le \prod_{l=1}^L \|H_l\|_2 = 1
$$

信号在多流通道间流转时，其模长既不会发生指数级爆炸，也不会迅速衰减。mHC 巧妙地在引入“多流混合”提高网络容量的同时，完美继承了经典残差连接的绝对稳定性。

#### mHC 的工程实现：Sinkhorn-Knopp 映射
在实际的 PyTorch / CUDA 代码中，为了将一个无约束的可学习实数矩阵 $W_l \in \mathbb{R}^{C \times C}$ 映射到 Birkhoff 多面体上，mHC 采用了 **Sinkhorn-Knopp 算法**。步骤如下：

1. 对 $W_l$ 进行逐元素指数化（或 Softmax 初始化），使其非负：

$$
A_{ij}^{(0)} = \exp(W_{l, ij})
$$

2. 迭代地进行行归一化与列归一化（通常进行 3~5 次循环即可高度逼近双随机矩阵）：

$$
\text{行归一化：} A_{ij}^{(2t+1)} = \frac{A_{ij}^{(2t)}}{\sum_{k=1}^C A_{ik}^{(2t)}}
$$

$$
\text{列归一化：} A_{ij}^{(2t+2)} = \frac{A_{ij}^{(2t+1)}}{\sum_{k=1}^C A_{kj}^{(2t+1)}}
$$

最后输出的 $H_l = A^{(2T)}$ 即为用于多流残差混合的双随机矩阵。

---

## 五、 HC、mHC 与 Attention Residuals (AttnRes) 的综合横向对比

| 维度 | 传统残差 (Standard) | 字节跳动 HC | DeepSeek mHC | Kimi AttnRes (Full) |
| :--- | :--- | :--- | :--- | :--- |
| **残差通道数** | 单通道主干 ($1$ Stream) | 多通道主干 ($C$ Streams) | 多通道主干 ($C$ Streams) | 动态全图层路由（等价于 $L$ 选 1 软路由） |
| **残差系数性质** | 固定常数 1 | 可学习，无约束矩阵 | 可学习，限制在 Birkhoff 多面体上 | 输入动态相关，Softmax 凸组合 |
| **数学稳定性** | 极高（恒等映射） | 极差（奇异值 $>1$ 导致爆炸，$<1$ 导致消失） | 极高（谱范数被数学限制恒等于 1） | 高（Softmax 概率之和为 1，自带归一化） |
| **训练收敛难度** | 容易 | 极难（大模型几乎无法收敛） | 容易（与传统残差相当） | 容易（梯度回传极为均匀） |
| **信息路由机制** | 静态无差别累加 | 静态可学习线性混合 | 静态可学习线性混合 | 动态输入自适应检索 |
| **空间复杂度 (单 Token)**| $O(1)$ | $O(C)$ | $O(C)$ | $O(L)$ （需缓存之前所有层的输出） |
| **工程优化策略** | 无需额外优化 | 无需额外优化 | 需要 CUDA 融合算子与重计算（Recomputation） | 需要 Block 划分与双阶段读取优化 |

### 异同点总结

#### 1. 相同点：打破静态“加法”藩篱
三者均认为标准的 $h_l = h_{l-1} + y_l$ 过于死板，试图通过引入多分支信息交换（HC/mHC 的多流混合，AttnRes 的全图层 Softmax 混合）来提升深层网络的表达上限。

#### 2. 不同点一：静态权重 vs 动态自适应
* **HC/mHC**：其多流混合矩阵 $H_l$ 在训练完成后是**固定不变**的。也就是说，对于任何不同的 Token 输入，多流残差流的交互和路由权重完全一致。这是一种**静态的拓扑增强**。
* **AttnRes**：其注意力权重 $\alpha_{l, i}$ 依赖于先序输出的 $\text{RMSNorm}(v_i)$。由于 $v_i$ 包含具体的 Token 特征，这意味着**不同的输入词、不同的语境，其残差的路由通路是实时变化、动态自适应的**。这是一种**动态的特征检索**。

#### 3. 不同点二：稳定性的实现路径
* **mHC**：依靠严格的**几何几何流形约束**（Birkhoff Polytope）将混合矩阵的谱范数限制为 1，从代数上锁定了信号的稳定性。
* **AttnRes**：依靠 **Softmax 的归一化性质**（所有层权重之和恒为 1，构成凸组合）以及 **RMSNorm 对 Key 尺度的缩放**，使得隐藏状态的模长在统计上保持稳定，避免了 PreNorm dilution。

---

## 六、 异同优劣势总结与架构选择建议

### 1. 字节跳动 HC 架构
* **劣势**：不推荐使用。缺乏数学约束导致奇异值容易偏离 1，在大模型或深层网络中表现出灾难性的梯度爆炸和数值不稳定。

### 2. DeepSeek mHC 架构
* **优势**：
  * **稳定性有数学保证**：谱范数恒等于 1，训练极其平稳。
  * **极高的计算效率**：由于 $C$ 一般取得较小（通常为 2 或 4），其相比于传统模型只增加了极小的矩阵乘法开销，且无需像 AttnRes 一样缓存历史所有层的激活，显存开销小。
  * **表达能力提升**：提供了比单残差主干更宽的“高速公路”，特征在多流间流转互补。
* **劣势**：
  * 混合是静态的，无法根据当前输入 Token 动态改变路由路径。
  * 需要复杂的 CUDA kernel 开发（如 Sinkhorn-Knopp 快速投影、反向传播重计算）以降低系统层面的额外延时。

### 3. Kimi AttnRes (及 Block AttnRes) 架构
* **优势**：
  * **动态自适应路由**：真正实现了“输入决定网络深度通路”，极高地提升了复杂长程逻辑推理（如代码生成、数学定理证明）的上下文提取能力。
  * **彻底解决稀释效应**：让早期的底层特征能被高层“无损检索”，在Scaling Law实验中被证实具有明显的“算力放大效应”（等效 1.25 倍计算量）。
* **劣势**：
  * **显存与存取开销大**：即使采用 Block 变体，由于每一层/块都需要检索前序表示，KV-like 缓存的读取仍然对硬件内存带宽（Memory Bandwidth）提出了巨大挑战。
  * 训练时反向传播梯度流的依赖关系更为复杂，对并行框架（如 3D Parallelism）的切分和通信提出了更高要求。
