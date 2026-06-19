# RoPE (Rotary Position Embedding) 算法详解

## 一、为什么需要位置编码？

### Transformer 的位置盲点

Transformer 的 Self-Attention 核心计算是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

这个运算对输入序列的**排列顺序完全不敏感**（Permutation Invariant）。也就是说，如果把 "我爱你" 和 "你爱我" 的 token 输入模型，在没有位置信息的情况下，attention 得分完全相同。

所以必须通过某种方式注入位置信息。

### 位置编码的演进

| 方法 | 代表 | 特点 |
|------|------|------|
| 绝对位置编码 | Sinusoidal (原始 Transformer) | 给每个位置加一个固定向量 |
| 可学习绝对位置编码 | GPT-2, BERT | 位置向量是可训练参数 |
| 相对位置编码 | T5 Relative Bias, ALiBi | 关注的是 token 间的距离 |
| **旋转位置编码 (RoPE)** | **RoFormer, LLaMA, Qwen** | **通过旋转注入相对位置** |

RoPE 的核心优势：它是一个**绝对编码的实现方式**，却能让 attention 的内积自动产生**相对位置**的效果。

---

## 二、RoPE 核心思想：用旋转编码位置

### 2.1 二维直觉

先考虑最简单的情况：向量只有 2 维 $[x_1, x_2]$ 。

要给位置 $m$ 处的向量编码位置信息，RoPE 的做法是：**将这个 2D 向量旋转 $m \cdot \theta$ 角度**。

$$
R(m, \theta) \cdot x = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} x_1 \cos(m\theta) - x_2 \sin(m\theta) \\ x_2 \cos(m\theta) + x_1 \sin(m\theta) \end{pmatrix}
$$

**为什么旋转能编码相对位置？** 因为旋转有一个优美的性质：

$$
\langle R(m,\theta) \cdot q, \; R(n,\theta) \cdot k \rangle = \langle R(m-n, \theta) \cdot q, \; k \rangle
$$

**证明：** 需要用到旋转矩阵的两个基本性质：

**性质 1：** 旋转矩阵是正交矩阵，转置 = 逆 = 反向旋转： $R(\alpha)^T = R(\alpha)^{-1} = R(-\alpha)$

**性质 2：** 旋转可叠加： $R(\alpha) \cdot R(\beta) = R(\alpha + \beta)$

推导过程：

$$
\langle R(m\theta) \cdot q, \; R(n\theta) \cdot k \rangle
$$

$$
= (R(m\theta) \cdot q)^T \cdot (R(n\theta) \cdot k) \quad \text{← 内积展开为转置乘法}
$$

$$
= q^T \cdot R(m\theta)^T \cdot R(n\theta) \cdot k \quad \text{← 转置分配律: } (AB)^T = B^T A^T
$$

$$
= q^T \cdot R(-m\theta) \cdot R(n\theta) \cdot k \quad \text{← 性质1: } R(m\theta)^T = R(-m\theta)
$$

$$
= q^T \cdot R((n-m)\theta) \cdot k \quad \text{← 性质2: } R(a) \cdot R(b) = R(a+b)
$$

最后一步，上式等价于 $\langle R((m-n)\theta) \cdot q, \; k \rangle$ ，因为：

$$
\langle R((m-n)\theta) \cdot q, \; k \rangle = q^T \cdot R((m-n)\theta)^T \cdot k = q^T \cdot R(-(m-n)\theta) \cdot k = q^T \cdot R((n-m)\theta) \cdot k \quad \blacksquare
$$

**直觉理解：** 两个向量分别旋转 $m\theta$ 和 $n\theta$ 后做内积，等于只旋转其中一个向量 $(m-n)\theta$ 后做内积 — 因为内积只关心两个向量之间的**夹角差**，而不是各自的绝对角度。

即：位置 $m$ 处的 query 和位置 $n$ 处的 key 做内积，等价于用相对距离 $(m-n)$ 旋转 query 后和原始 key 做内积。内积的结果**只依赖相对位置 $m-n$** ，而不是绝对位置 $m$ 和 $n$ 。

### 2.2 推广到高维

实际中 `head_dim = 64` 或 `128`。RoPE 的策略是把 $d$ 维向量拆成 $d/2$ 个二维子空间，每个子空间用不同的旋转频率：

```text
d 维向量 = [x₁, x₂, x₃, x₄, ..., x_{d-1}, x_d]
             ├──┘    ├──┘         ├────────┘
           子空间₁  子空间₂   ...  子空间_{d/2}
```

每个子空间 $i$ 的旋转频率：

$$
\theta_i = \theta_{\text{base}}^{-2i/d}
$$

其中 $\theta_{\text{base}} = 10000$ （默认值）。

不同频率的设计意图（以 `dim=64` 为例）：

```text
子空间 0 (i=0):   θ₀ = 10000^(0/64)   = 1.0      → 高频，变化快，捕获近距离关系
子空间 1 (i=1):   θ₁ = 10000^(-2/64)  ≈ 0.72     → 次高频
...
子空间 15 (i=15): θ₁₅ = 10000^(-30/64) ≈ 0.0032  → 次低频
子空间 31 (i=31): θ₃₁ = 10000^(-62/64) ≈ 0.00013 → 最低频，变化极慢，捕获长距离关系
```

这和傅里叶变换的思想类似：用不同频率的旋转组合来表示位置信息。

### 2.3 完整的 RoPE 数学表达

对位置 $m$ 处的 $d$ 维向量 $x$ ，RoPE 变换为：

$$
\text{RoPE}(x, m) = \begin{pmatrix} x_1 \cos(m\theta_0) - x_2 \sin(m\theta_0) \\ x_2 \cos(m\theta_0) + x_1 \sin(m\theta_0) \\ x_3 \cos(m\theta_1) - x_4 \sin(m\theta_1) \\ x_4 \cos(m\theta_1) + x_3 \sin(m\theta_1) \\ \vdots \\ x_{d-1} \cos(m\theta_{d/2-1}) - x_d \sin(m\theta_{d/2-1}) \\ x_d \cos(m\theta_{d/2-1}) + x_{d-1} \sin(m\theta_{d/2-1}) \end{pmatrix}
$$

---

## 三、代码逐行解析

### 3.1 `precompute_freqs_cis` — 预计算频率

```python
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # Step 1: 计算每个子空间的基础频率 θᵢ
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # torch.arange(0, dim, 2) = [0, 2, 4, ..., dim-2]  → 就是 2i
    # / dim  → 得到 2i/d
    # theta ** (...)  → 得到 θ^(2i/d)
    # 1.0 / (...)  → 得到 θ^(-2i/d) = θᵢ
    # freqs.shape = [dim//2]

    # Step 2: 生成位置索引
    t = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]

    # Step 3: 外积得到所有 (位置, 频率) 组合的角度
    freqs = torch.outer(t, freqs).float()
    # freqs[m][i] = m * θᵢ  即位置 m 在子空间 i 的旋转角度
    # freqs.shape = [seq_len, dim//2]

    # Step 4: 返回 cos 和 sin 值
    return torch.cos(freqs), torch.sin(freqs)
    # cos.shape = sin.shape = [seq_len, dim//2]
```

**形状演变图：**

以 `dim=64, seq_len=10` 为例：

```text
arange(0,64,2)     → shape [32]    → [0, 2, 4, ..., 62]
/ dim               → shape [32]    → [0/64, 2/64, ..., 62/64]
theta ** (...)      → shape [32]    → [θ^0, θ^(2/64), ..., θ^(62/64)]
1.0 / (...)         → shape [32]    → [θ^0, θ^(-2/64), ..., θ^(-62/64)]

arange(seq_len)     → shape [10]    → [0, 1, 2, ..., 9]

outer(t, freqs)     → shape [10, 32] → m·θᵢ 角度矩阵

cos(freqs)          → shape [10, 32] → 每个 (位置, 子空间) 的 cos 值
sin(freqs)          → shape [10, 32] → 每个 (位置, 子空间) 的 sin 值
```

### 3.2 `rotate_half` — 构造旋转的 "负交叉" 部分

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]   # 前半部分: [x₁, x₃, x₅, ...]
    x2 = x[..., x.shape[-1] // 2 :]   # 后半部分: [x₂, x₄, x₆, ...]
    return torch.cat((-x2, x1), dim=-1)  # [-x₂, -x₄, ..., x₁, x₃, ...]
```

> **注意：** 这里用的是 **"半拆分" (half-split)** 风格，不是 "交错配对" (interleave) 风格。两种方式数学上等价，只是维度排列不同。下面第四节会详细对比。

### 3.3 `apply_rotary_emb` — 应用旋转嵌入

```python
def apply_rotary_emb(x, cos, sin):
    # cos/sin 原始 shape: [seq_len, dim//2]
    # 需要扩展到和 x 一样的 shape: [batch, heads, seq_len, dim]

    cos = torch.cat([cos, cos], dim=-1)  # [seq_len, dim//2] → [seq_len, dim]
    sin = torch.cat([sin, sin], dim=-1)  # 复制一份，因为前半和后半都需要同样的 cos/sin

    cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]  → 广播到 batch 和 head 维度
    sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]

    return (x * cos) + (rotate_half(x) * sin)
```

**为什么 `x * cos + rotate_half(x) * sin` 等价于旋转矩阵？** 展开验证：

```text
x = [x₁, x₂, ..., x_{d/2}, x_{d/2+1}, ..., x_d]
     ├──── 前半 (x1) ────┤  ├──── 后半 (x2) ────┤

rotate_half(x) = [-x_{d/2+1}, ..., -x_d, x₁, ..., x_{d/2}]
                  ├──── -x2 ──────────┤  ├──── x1 ────────┤

cos = [cos(m·θ₀), cos(m·θ₁), ..., cos(m·θ₀), cos(m·θ₁), ...]
sin = [sin(m·θ₀), sin(m·θ₁), ..., sin(m·θ₀), sin(m·θ₁), ...]

x * cos = [x₁·cos(m·θ₀), ..., x_{d/2}·cos(m·θ_{d/2-1}),
           x_{d/2+1}·cos(m·θ₀), ..., x_d·cos(m·θ_{d/2-1})]

rotate_half(x) * sin = [-x_{d/2+1}·sin(m·θ₀), ..., -x_d·sin(m·θ_{d/2-1}),
                         x₁·sin(m·θ₀), ..., x_{d/2}·sin(m·θ_{d/2-1})]

结果的第 i 个元素 (i ≤ d/2):
    xᵢ·cos(m·θᵢ) - x_{i+d/2}·sin(m·θᵢ)    ← 正好是旋转公式！

结果的第 i+d/2 个元素:
    x_{i+d/2}·cos(m·θᵢ) + xᵢ·sin(m·θᵢ)    ← 正好是旋转公式！
```

在这种 **half-split** 排列中，子空间 $i$ 的两个配对维度是 $(i, \; i+d/2)$ 而不是 $(2i, \; 2i+1)$ 。

---

## 四、两种实现风格对比

### 风格 A：交错配对 (Interleaved) — LLaMA 原始实现

```text
向量排列:  [x₁, x₂, x₃, x₄, x₅, x₆, ...]
配对方式:   ├──┘  ├──┘  ├──┘
         子空间0 子空间1 子空间2

子空间 i 的配对: (x_{2i}, x_{2i+1})
```

```python
# 交错风格的实现 (LLaMA 风格)
def apply_rotary_emb_interleaved(xq, freqs_cis):
    # 将实数向量视为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]  # 广播
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq)
```

这利用了复数乘法 $(a + bi)(\cos\theta + i\sin\theta)$ 恰好就是旋转。

### 风格 B：半拆分 (Half-Split) — 本代码的实现

```text
向量排列:  [x₁, x₂, ..., x_{d/2}, x_{d/2+1}, ..., x_d]
配对方式:   ├── 前半 ──────────────┤  ├── 后半 ──────────┤
           子空间0的第1维 ...         子空间0的第2维 ...

子空间 i 的配对: (xᵢ, x_{i+d/2})
```

```python
# 半拆分风格的实现（本代码）
def apply_rotary_emb(x, cos, sin):
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)
```

### 两种风格等价吗？

**表达能力等价，但不能互换。** 两种风格的维度配对确实不同：

```text
交错配对:  子空间 i → (x_{2i}, x_{2i+1})
半拆分:    子空间 i → (x_i, x_{i+d/2})

以 dim=8 为例：
交错: (x₀,x₁) (x₂,x₃) (x₄,x₅) (x₆,x₇)
半拆: (x₀,x₄) (x₁,x₅) (x₂,x₆) (x₃,x₇)
```

这意味着同一个输入向量，两种风格旋转出来的结果是**不同的**。那为什么说"等价"？

**关键在于 $W_Q$ 和 $W_K$ 是可学习的。** 两种风格之间只差一个固定的维度排列（permutation）。设排列矩阵为 $P$ ，则：

$$
\text{HalfSplit-RoPE}(x) = \text{Interleaved-RoPE}(P \cdot x)
$$

而 attention 中实际输入 RoPE 的是 $W_Q \cdot h$ ，学习到的 $W_Q$ 可以自动"吸收"这个排列：

$$
W_Q^{\text{half-split}} = P \cdot W_Q^{\text{interleaved}}
$$

所以两种风格训练出的模型具有**相同的表达能力**，只是学到的 $W_Q, W_K$ 权重矩阵内部的维度排列不同。

> **⚠️ 注意：** 训练好的模型**不能**切换风格！切换等于随机打乱了维度配对，会导致输出乱码。必须训练和推理用同一种风格。

| 风格　　 | 代表模型　　　　 | 优势　　　　　　　　　　　　　　　　 |
| ----------| ------------------| --------------------------------------|
| 交错配对 | LLaMA, Mistral　 | 用复数乘法实现，代码最简洁　　　　　 |
| 半拆分　 | GPT-NeoX, 本代码 | 不依赖 `view_as_complex`，兼容性更好 |

---

## 五、长度外推问题

### 5.1 问题描述

假设模型在 $L_{\text{train}} = 4096$ 长度的序列上训练。推理时，如果输入序列长度 $L_{\text{test}} = 8192$ ，会发生什么？

```text
训练时见过的位置:  m ∈ [0, 4095]
推理时出现的位置:  m ∈ [0, 8191]

位置 5000 的旋转角度:  m · θᵢ = 5000 · θᵢ
```

对于高频子空间（ $\theta_0 \approx 1.0$ ），位置 5000 的角度 = 5000 弧度。模型训练时最大只见过 4095 弧度。虽然 cos/sin 是周期函数，但 attention 的内积模式在训练数据中只学过 $m \cdot \theta_i \in [0, 4095 \cdot \theta_i]$ 范围内的 pattern。

### 5.2 外推失败的根本原因

Attention 内积可以分解为频率通道的求和：

$$
q_m \cdot k_n = \sum_i f\big((m - n) \cdot \theta_i\big)
$$

- **训练时：** 最大相对距离 $= L_{\text{train}} - 1 = 4095$ ，所有频率通道的角度都在训练分布内
- **推理时：** 如果 $m - n > 4095$ ，某些频率通道会出现训练时没见过的角度 → attention 分数产生 OOD (out-of-distribution) 值 → 模型输出质量急剧下降

实验表明，vanilla RoPE 在超出训练长度后，PPL (perplexity) 会爆炸性增长。

---

## 六、长度外推解决方案

### 6.1 位置插值 (Position Interpolation, PI)

> 论文: *Extending Context Window of Large Language Models via Position Interpolation* (Meta, 2023)

**核心思想：** 不要外推，而是把新的位置范围**压缩**回训练范围内。

$$
\text{原始 RoPE:} \quad \text{angle} = m \cdot \theta_i
$$

$$
\text{PI:} \quad \text{angle} = \frac{m}{s} \cdot \theta_i, \quad s = \frac{L_{\text{target}}}{L_{\text{train}}}
$$

等价于将位置索引缩放 $m' = m / s$ 。

```python
# Position Interpolation 实现
def precompute_freqs_cis_PI(dim, seq_len, theta=10000.0,
                             scale_factor=1.0):  # scale_factor = L_target / L_train
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float() / scale_factor   # ← 唯一的改动：缩放位置索引
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)
```

**优缺点：**

| | |
|---|---|
| ✅ | 简单优雅，只改一行代码 |
| ✅ | 保证所有角度都在训练分布内 |
| ❌ | 压缩了位置分辨率：原来距离 1 的两个 token 现在在 RoPE 看来距离变成了 $1/s$ |
| ❌ | 需要少量微调 (fine-tuning) 才能恢复性能 |
| ❌ | 高频信息损失，近距离 token 的区分度下降 |

### 6.2 NTK-aware RoPE (Neural Tangent Kernel 感知)

> 来自 Reddit 用户 bloc97 的发现 (2023)

**核心思想：** PI 均匀缩放所有频率，这对高频维度伤害很大。NTK-aware 方法改为**只缩放低频，保留高频**。

```text
直觉：
- 高频子空间 (i 小) → 周期短，已经见过多个完整周期 → 天然能外推，不需要缩放
- 低频子空间 (i 大) → 周期长，训练时可能连一个周期都没走完 → 需要缩放

方法：修改 theta_base，让低频自动被拉伸
```

$$
\theta'_{\text{base}} = \theta_{\text{base}} \cdot s^{\frac{d}{d-2}}
$$

**为什么指数是 $d/(d-2)$ ？** 将新 theta 代入频率公式，展开看每个维度实际被缩放了多少：

$$
\theta'_i = (\theta \cdot s^{\frac{d}{d-2}})^{-2i/d} = \theta^{-2i/d} \cdot s^{-\frac{2i}{d-2}} = \theta_i \cdot s^{-\frac{2i}{d-2}}
$$

所以每个维度 $i$ 的缩放因子是 $s^{-2i/(d-2)}$ ，代入边界值（ $i$ 的范围是 $0$ 到 $d/2 - 1$ ）：

- $i = 0$ （最高频）： $s^{-2 \cdot 0 / (d-2)} = s^0 = 1$ → 完全不缩放 ✓
- $i = d/2 - 1$ （最低频）： $s^{-2(d/2-1)/(d-2)} = s^{-(d-2)/(d-2)} = s^{-1} = 1/s$ → 恰好等于完全 PI 缩放 ✓

$d/(d-2)$ 的精确意义：**它是让最低频维度恰好获得 $1/s$ 倍缩放（即完全等价于 PI）的那个指数**。虽然对于高维（如 $d=64$ ）， $d/(d-2) = 64/62 \approx 1.032$ 看似接近 1，但它的作用体现在**展开到每个频率维度后**产生从 1 到 $1/s$ 的精确缩放梯度。

```python
# NTK-aware RoPE 实现
def precompute_freqs_cis_NTK(dim, seq_len, theta=10000.0,
                              scale_factor=1.0):
    # 将 theta 放大，等效于拉伸低频
    theta_new = theta * (scale_factor ** (dim / (dim - 2)))
    freqs = 1.0 / (theta_new ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)
```

**效果对比：**

假设 `dim=64, theta=10000, scale_factor=2`（外推到 2 倍长度）：

| 频率 | 原始 RoPE | PI（均匀缩放） | NTK-aware |
|------|-----------|---------------|-----------|
| 高频 (i=0) | $\theta_0 = 1.0$ | $0.5$ | $\approx 0.98$ ← 几乎不动 |
| 中频 (i=16) | $\theta_{16} \approx 0.01$ | $0.005$ | $\approx 0.008$ ← 适度缩放 |
| 低频 (i=31) | $\theta_{31} \approx 0.0001$ | $0.00005$ | $\approx 0.00005$ ← 大幅缩放 |

**优缺点：**

| | |
|---|---|
| ✅ | 不需要微调即可获得一定效果 |
| ✅ | 保留了高频的位置分辨率 |
| ❌ | 效果不如 PI + 微调 |
| ❌ | scale_factor 需要预先确定 |

### 6.3 Dynamic NTK

> 改进自 NTK-aware，由 Emozilla 提出

**核心思想：** 不预先设定固定的 scale_factor，而是根据**当前实际推理的序列长度**动态计算。

```python
# Dynamic NTK 实现
def precompute_freqs_cis_dynamic_ntk(dim, seq_len, theta=10000.0,
                                      max_train_len=4096):
    if seq_len > max_train_len:
        # 动态计算缩放因子
        scale_factor = seq_len / max_train_len
        theta = theta * (scale_factor ** (dim / (dim - 2)))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)
```

| | |
|---|---|
| ✅ | 自适应，无需预先指定目标长度 |
| ✅ | 短序列时保持原始行为（不损失性能） |
| ✅ | 不需要微调 |

### 6.4 YaRN (Yet another RoPE extensioN)

> 论文: *YaRN: Efficient Context Window Extension of Large Language Models* (2023)

**核心思想：** 结合 PI 和 NTK 的优点，将频率维度分成三个区间，分别处理。

```text
频率维度分区：

├────── 高频区 ──────┼──── 过渡区 ────┼──── 低频区 ──────┤
│  不缩放 (保持原样)  │  渐进式插值    │  完全插值 (PI)   │
│  wavelength < α·L  │  过渡平滑      │  wavelength > β·L│
```

对于每个频率维度 $i$ ，计算其波长 $\lambda_i = 2\pi / \theta_i$ 。

**为什么要计算波长？** 因为波长的物理含义是“转完一整圈需要多少个 token 位置”，单位和训练长度 $L_{\text{train}}$ 一致（都是 token 数），可以直接比较：

```text
例: 频率 θ_i = 0.01 → 波长 λ_i = 2π/0.01 ≈ 628 个位置

若 L_train = 4096:  训练时转了 4096/628 ≈ 6.5 圈 → 充分学会了这个周期 → 可外推
若 L_train = 100:   训练时转了 100/628  ≈ 0.16 圈 → 连 1/6 圈都没转完 → 无法外推
```

频率 $\theta_i$ 本身的单位是“弧度/位置”，无法与 $L_{\text{train}}$ 直接比较。波长将其转换为同单位量，让分区判断变得直观：

- 如果 $\lambda_i < \alpha \cdot L_{\text{train}}$ （高频）：训练时已见过多个完整周期 → 完全不缩放（ $\gamma = 0$ ）
- 如果 $\lambda_i > \beta \cdot L_{\text{train}}$ （低频）：训练时连一个周期都没走完 → 完全使用 PI 缩放（ $\gamma = 1$ ）
- 过渡区：线性插值混合两种策略

$$
\theta'_i = (1 - \gamma) \cdot \theta_i + \gamma \cdot \frac{\theta_i}{s}
$$

```python
# YaRN 实现（简化版）
def precompute_freqs_cis_yarn(dim, seq_len, theta=10000.0,
                               scale_factor=2.0,
                               alpha=1, beta=32):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 计算每个频率维度的波长
    wavelengths = 2 * 3.14159 / freqs

    # 训练长度
    L_train = seq_len / scale_factor

    # 计算每个维度的插值比例 (0=不缩放, 1=完全PI)
    low_bound = L_train * alpha
    high_bound = L_train * beta

    # 线性映射到 [0, 1]
    gamma = (wavelengths - low_bound) / (high_bound - low_bound)
    gamma = gamma.clamp(0, 1)

    # 混合：(1-gamma) * 原始频率 + gamma * 缩放后频率
    freqs_scaled = freqs / scale_factor
    freqs = (1 - gamma) * freqs + gamma * freqs_scaled

    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)

    # YaRN 还会乘一个 attention 温度缩放因子
    # 来补偿插值带来的 attention 分布变化
    return torch.cos(freqs), torch.sin(freqs)
```

**完整数值计算示例：**

以 `dim=8, theta=10000, L_train=100, scale_factor=4`（扩展到 400）为例，共 $d/2 = 4$ 个子空间：

**Step 1：** 计算基础频率和波长

| $i$ | $\theta_i = 10000^{-2i/8}$ | $\lambda_i = 2\pi/\theta_i$ | 含义 |
|-----|---------------------------|---------------------------|------|
| 0 | 1.0 | 6.28 | 每 6 个位置转一圈 |
| 1 | 0.1 | 62.8 | 每 63 个位置转一圈 |
| 2 | 0.01 | 628 | 每 628 个位置转一圈 |
| 3 | 0.001 | 6283 | 每 6283 个位置转一圈 |

**Step 2：** 计算分区边界和 $\gamma$

$$
\text{low\_bound} = L_{\text{train}} \times \alpha = 100, \quad \text{high\_bound} = L_{\text{train}} \times \beta = 3200
$$

$$
\gamma_i = \text{clamp}\left(\frac{\lambda_i - 100}{3200 - 100}, \; 0, \; 1\right)
$$

| $i$ | $\lambda_i$ | $\gamma_i$ | 区间 |
|-----|------------|-----------|------|
| 0 | 6.28 | $(6.28-100)/3100 < 0$ → **0** | 🟢 高频区 |
| 1 | 62.8 | $(62.8-100)/3100 < 0$ → **0** | 🟢 高频区 |
| 2 | 628 | $(628-100)/3100$ = **0.170** | 🟡 过渡区 |
| 3 | 6283 | $(6283-100)/3100 = 1.99$ → **1.0** | 🔴 低频区 |

**Step 3：** 混合频率 $\theta'_i = (1-\gamma)\theta_i + \gamma \cdot \theta_i/s$

| $i$ | $\gamma$ | 计算过程 | $\theta'_i$ | 等效缩放 |
|-----|---------|---------|------------|----------|
| 0 | 0 | $1.0 \times 1.0 + 0 \times 0.25$ | 1.0 | 1.0x 不变 |
| 1 | 0 | $1.0 \times 0.1 + 0 \times 0.025$ | 0.1 | 1.0x 不变 |
| 2 | 0.170 | $0.83 \times 0.01 + 0.17 \times 0.0025$ | 0.008725 | 0.87x 轻微压缩 |
| 3 | 1.0 | $0 \times 0.001 + 1.0 \times 0.00025$ | 0.00025 | 0.25x 完全 PI |

**对比总览：**

| $i$ | 原始 | PI（全缩放 $\div 4$ ） | NTK | YaRN | YaRN 区间 |
|-----|------|---------------------|-----|------|----------|
| 0 | 1.0 | 0.25 | $\approx 0.98$ | 1.0 | 🟢 高频-保留 |
| 1 | 0.1 | 0.025 | $\approx 0.085$ | 0.1 | 🟢 高频-保留 |
| 2 | 0.01 | 0.0025 | $\approx 0.007$ | 0.008725 | 🟡 过渡-轻压 |
| 3 | 0.001 | 0.00025 | $\approx 0.0004$ | 0.00025 | 🔴 低频-全压 |

**优缺点：**

| | |
|---|---|
| ✅ | 高频保持分辨率（近距离 token 不受影响） |
| ✅ | 低频得到充分压缩（远距离不 OOD） |
| ✅ | 过渡区平滑，避免突变 |
| ✅ | 只需极少量微调（~400 步） |
| ✅ | 目前效果最好的方法之一 |
| ❌ | 超参数较多 (alpha, beta, scale_factor) |

### 6.5 扩大 theta_base（LLaMA 3 / Code LLaMA 风格）

> LLaMA 3 将 theta 从 10000 提高到 500000

**核心思想：** 最简单粗暴的方法 — 直接增大基础频率 theta，让所有频率都变低，从而自然支持更长序列。

```python
# 直接扩大 theta
cos, sin = precompute_freqs_cis(dim=128, seq_len=131072,
                                 theta=500000.0)  # 从 10000 → 500000
```

```text
theta=10000 时:
  最低频率 θ_{d/2-1} ≈ 0.00013  → 周期 ≈ 48000
  最大安全长度 ≈ 4K~8K

theta=500000 时:
  最低频率 θ_{d/2-1} ≈ 0.0000026 → 周期 ≈ 2,400,000
  最大安全长度 ≈ 128K+
```

| | |
|---|---|
| ✅ | 零代码改动，只改超参数 |
| ✅ | 需要在长序列数据上 pretrain/continue-pretrain |
| ❌ | 需要大量长序列训练数据 |
| ❌ | 所有频率都被拉低，可能损失短距离分辨率 |

---

## 七、各方法总结对比

| 方法 | 需要微调？ | 保留高频？ | 实现难度 | 代表应用 |
|------|-----------|-----------|---------|---------|
| Position Interpolation | 需要 | ❌ | 最简单 | CodeLLaMA |
| NTK-aware | 不一定 | ✅ | 简单 | 社区方案 |
| Dynamic NTK | 不一定 | ✅ | 简单 | Qwen 1 |
| YaRN | 极少量 | ✅ | 中等 | Qwen 2 |
| 扩大 theta | 预训练时 | ⚠️ | 最简单 | LLaMA 3 |
