# Mixture of Experts (MoE) 深度解析

> 参考资料：moe.png 手写笔记 + MokioMind `MokioModel.py` 源码
>
> 源码路径：`MokioMind/model/MokioModel.py`

---

## 目录

1. [MoE 是什么？—— FFN 的稀疏化升级](#一moe-是什么-ffn-的稀疏化升级)
2. [核心数学公式](#二核心数学公式)
3. [完整数字例子](#三完整数字例子)
4. [路由分配算法原理深析](#四路由分配算法原理深析)
5. [负载均衡：辅助损失完整推导](#五负载均衡辅助损失完整推导)
6. [训练 vs 推理：两套分发实现](#六训练-vs-推理两套分发实现)
7. [DeepSeek MoE 三大架构创新](#七deepseek-moe-三大架构创新)
8. [共享专家：路由的兜底机制](#八共享专家路由的兜底机制)
9. [整体架构总结](#九整体架构总结)

---

## 一、MoE 是什么？—— FFN 的稀疏化升级

### 标准 Transformer Block 回顾

标准 Transformer 每层由两步构成：

```text
u_t^l  = SelfAttn( h_{1:T}^{l-1} ) + h_t^{l-1}    ← Attention + 残差
h_t^l  = FFN( u_t^l ) + u_t^l                       ← FFN + 残差
```

**问题**：FFN 是所有 token 共用的同一个网络，限制了模型表达不同类型知识的能力。

### MoE 的核心思想

**用 N 个"专家 FFN"替换一个 FFN，每次只激活其中 K 个（K << N）。**

```text
                ┌──────────────────────────────────┐
                │         MoE Layer                 │
                │                                   │
   u_t^l ──→   │  Gate ──→ TopK 选 K 个专家         │  ──→ h_t^l
                │          ↓                        │
                │    Expert_i1(u_t) * w_1           │
                │  + Expert_i2(u_t) * w_2           │
                │  + Shared_Expert(u_t)             │
                └──────────────────────────────────┘
```

**MoE 的核心价值**：

| 维度 | 效果 |
|------|------|
| 参数量 | 线性增大（N 个专家）|
| 每 token 计算量 | 几乎不变（只算 K 个）|
| 模型容量 | 大幅提升 |
| 专家分工 | 自然涌现（语言专家、代码专家等）|

---

## 二、核心数学公式

moe.png 中的 DeepSeek MoE 完整定义：

### 2.1 输出计算

```text
h_t^l = Σ(i=1 to mN)  g_{i,t} · FFN_i( u_t^l )  +  u_t^l
         ↑ 所有专家求和，但绝大多数 g_{i,t} = 0（稀疏）
```

### 2.2 门控权重 g_{i,t}

```text
         ┌ s_{i,t}     if s_{i,t} ∈ TopK( {s_{j,t} | j=1..mN}, mK )
g_{i,t} = ┤
         └ 0,          otherwise
```

只有亲和度分数进入 Top-K 的专家才被激活，其余置 0。

### 2.3 亲和度分数 s_{i,t}

```text
s_{i,t} = Softmax_i( u_t^l · e_i )
           ↑ token 隐向量 u_t 与专家嵌入向量 e_i 做点积，再 softmax
```

> **参数说明**
> - `mN`：总专家数（m 是精细化倍数，N 是原始专家数）
> - `mK`：每 token 激活的专家数（对应代码中 `num_experts_per_tok`）
> - `e_i`：第 i 个专家的可学习嵌入向量（即 `MoEGate.weight[i]`）

---

## 三、完整数字例子

### 设定

```text
路由专家数 N = 4
每 token 激活数 K = 2
hidden_size = 4
```

### Step 1：计算亲和度 logits

```text
token "猫" 的隐向量 u = [0.3, -0.1, 0.8, 0.2]

专家嵌入矩阵 W_gate (4×4):
  e_0 = [ 0.1,  0.2, -0.3,  0.4]
  e_1 = [-0.2,  0.5,  0.1, -0.1]
  e_2 = [ 0.6, -0.1,  0.9,  0.3]   ← 与 u 方向最近
  e_3 = [ 0.2,  0.3,  0.4,  0.1]

logits = u @ W_gate.T：
  logit_0 = 0.3*0.1 + (-0.1)*0.2 + 0.8*(-0.3) + 0.2*0.4 = -0.09
  logit_1 = 0.3*(-0.2) + (-0.1)*0.5 + 0.8*0.1 + 0.2*(-0.1) = -0.03
  logit_2 = 0.3*0.6 + (-0.1)*(-0.1) + 0.8*0.9 + 0.2*0.3  = 1.07  ← 最高
  logit_3 = 0.3*0.2 + (-0.1)*0.3 + 0.8*0.4 + 0.2*0.1    = 0.41
```

### Step 2：Softmax 归一化

```text
logits = [-0.09, -0.03, 1.07, 0.41]

softmax(x_i) = exp(x_i) / Σ exp(x_j)

exp:    [0.914, 0.970, 2.915, 1.507]   sum = 6.306

scores: [0.145, 0.154, 0.462, 0.239]
         ↑E0    ↑E1    ↑E2    ↑E3
```

### Step 3：Top-2 选择

```text
选出最高的 2 个：
  Expert_2: score = 0.462  ✅
  Expert_3: score = 0.239  ✅
  Expert_0: score = 0.145  ❌（跳过，不计算）
  Expert_1: score = 0.154  ❌（跳过，不计算）
```

### Step 4：归一化权重（norm_topk_prob=True）

```text
sum = 0.462 + 0.239 = 0.701

g_2 = 0.462 / 0.701 = 0.659
g_3 = 0.239 / 0.701 = 0.341

门控权重向量: [0, 0, 0.659, 0.341]
               ↑E0 ↑E1  ↑E2    ↑E3
```

### Step 5：专家计算与加权求和

```text
Expert_2 前向: FFN_2(u) = [0.5, 0.3, 0.9, 0.4]   （假设）
Expert_3 前向: FFN_3(u) = [0.2, 0.7, 0.1, 0.6]   （假设）

加权输出:
  y_路由 = 0.659 * [0.5, 0.3, 0.9, 0.4]
         + 0.341 * [0.2, 0.7, 0.1, 0.6]
         = [0.330, 0.197, 0.593, 0.264]
           + [0.068, 0.239, 0.034, 0.205]
         = [0.398, 0.436, 0.627, 0.469]

加上残差: h = y_路由 + u = [0.698, 0.336, 1.427, 0.669]
```

---

## 四、路由分配算法原理深析

### 4.1 路由器本质：可学习的分类器

路由器（Gate）回答一个问题：**"对于这个 token 的语义，哪些专家最擅长？"**

```python
# MoEGate.__init__() 中
self.weight = nn.Parameter(
    torch.empty((self.n_routed_experts, self.gating_dim))
)
# shape: [N_experts, hidden_dim]
# 每行 = 一个专家的"代表向量"（Expert Embedding）
```

每个专家有一个可训练的嵌入向量 `e_i`，维度等于 hidden_size。

### 4.2 打分：三步过程

**Step 1 — 点积计算亲和度（Affinity）**

```python
logits = F.linear(hidden_states, self.weight, None)
# 等价于: logits = hidden_states @ weight.T
# 结果 shape: [bsz*seq_len, N_experts]
```

数学本质：计算 token 隐向量 u_t 与每个专家嵌入 e_i 的内积相似度。

**为什么用点积？**

```text
原因1：点积在 softmax 前等价于余弦相似度的缩放版
原因2：梯度计算简单，训练稳定
原因3：天然符合 Transformer 范式（Q·K 也是点积）
原因4：计算高效，可用 F.linear 一次矩阵乘法完成所有专家打分
```

**Step 2 — Softmax 归一化**

```python
scores = logits.softmax(dim=-1)
```

**为什么是 Softmax 而不是直接用 logits？**

```text
原因1：归一化成概率，方便 TopK 后加权求和（权重有意义）

原因2：Softmax 具有"竞争性"——
       某专家分数提高会压低其他专家
       这形成隐性竞争，促进专家间的专门化分工

原因3：梯度能通过 softmax 传回 gate.weight，让路由器可学习
       （TopK 本身不可微，需要依赖 Softmax 传梯度）
```

**Step 3 — TopK 稀疏选择**

```python
topk_weight, topk_idx = torch.topk(
    scores, k=self.top_k, dim=-1, sorted=False
)
```

**TopK 的根本原因**：

```text
如果让所有专家参与计算（Dense MoE）：
  参数量增加 × N   ✅
  计算量增加 × N   ❌（失去意义）

TopK 硬截断：
  只有 K 个专家"开灯"，其余"休眠"
  K=2, N=64 时：每个 token 只激活 3% 的计算量
  → 以极小计算量解锁巨大参数空间
```

**TopK 的不可微问题（moe.png 中标注：Router Non-Differentiable）**：

```text
TopK = argmax，在选择边界处梯度为 0 或无穷大
→ 无法通过"选哪个专家"这个决策直接反向传播梯度
→ 需要辅助损失（Auxiliary Loss）间接监督路由行为
→ 这是 MoE 最核心的工程挑战
```

### 4.3 归一化 Top-K 权重

```python
if self.top_k > 1 and self.norm_topk_prob:
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
```

**为什么需要重新归一化？**

```text
场景：K=2，选出 expert_2(0.55) 和 expert_3(0.20)

不归一化：
  0.55 * out_2 + 0.20 * out_3  → 总权重 0.75
  不同 token 总权重不同 → 引入不必要的幅度变化 → 训练不稳定

归一化后：
  0.733 * out_2 + 0.267 * out_3  → 总权重 = 1.0
  保证所有 token 输出的期望幅度一致
```

---

## 五、负载均衡：辅助损失完整推导

### 5.1 为什么会出现负载崩溃？

```text
训练初期（随机初始化）：
  某些专家恰好对大量 token 评分偏高
  → 这些专家被频繁选中
  → 获得更多梯度更新，变得"更好"
  → 更多 token 倾向于选它们
  → 正反馈循环！

最终：2~3 个"超级专家"承担所有计算
     其余专家几乎闲置，等效于模型退化成小模型
     巨大的参数空间被浪费
```

### 5.2 Aux Loss 的设计哲学

**目标**：让每个专家被选择的"期望频率"和"实际频率"都尽可能均匀

**核心公式（Switch Transformer / Token-level 版）**：

```text
aux_loss = α · Σ(i=1 to N)  P_i · f_i

其中：
  P_i = 专家 i 的平均路由概率（期望被选频率，可微）
      = mean( scores[:, i] )   ← 通过 softmax 传梯度

  f_i = 专家 i 的实际被选比例（历史频率，不可微）
      = count(expert_i被选中) / (T * K) * N   ← 归一化，理想值=1

  α   = 损失系数（控制均衡性 vs 路由质量的权衡）
```

**梯度分析**：

```text
把 f_i 当做常数（stop gradient），只对 P_i 求梯度

当专家 i 被过多选（f_i > 1）：
  P_i · f_i 偏大 → 梯度推动降低 P_i → 下次该专家被选概率降低

当专家 i 被过少选（f_i < 1）：
  P_i · f_i 偏小 → 梯度推动提升 P_i → 下次该专家被选概率提高

→ 间接实现负载均衡（通过调整期望概率来间接影响 TopK 结果）
```

### 5.3 代码实现：Token-level（seq_aux=False）

```python
mask_ce = F.one_hot(
    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
)
# mask_ce: [T*K, N]，独热编码，标记每次选择了哪个专家

ce = mask_ce.float().mean(0)   # [N]，各专家被选频率
Pi = scores_for_aux.mean(0)    # [N]，各专家平均路由概率
fi = ce * self.n_routed_experts # 归一化：理想值为 1.0

aux_loss = (Pi * fi).sum() * self.alpha
```

### 5.4 代码实现：Sequence-level（seq_aux=True，默认，更精细）

```python
scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # [B, S, N]

# 统计每个样本中各专家被选次数，归一化为相对频率
ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
ce.scatter_add_(
    1,
    topk_idx_for_aux_loss,                         # [B, S*K]
    torch.ones(bsz, seq_len * aux_topk, device=...) # 每选一次加 1
).div_(seq_len * aux_topk / self.n_routed_experts)  # 归一化

# 每个样本独立计算均衡损失，再取 batch 平均
aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
#           ↑f_i (per sample)  ↑P_i (per sample, avg over seq)
```

**为什么 Sequence-level 更好？**

```text
Token-level（全局）：
  统计所有样本的专家频率
  样本 A（代码）集中用专家 2，样本 B（数学）集中用专家 3
  → 全局平均看起来很"均衡"，但每个样本内部并不均衡
  → 损失信号被掩盖

Sequence-level（按样本）：
  每个样本独立计算均衡性
  → 惩罚更准确，确保每个样本内部各专家均衡
  → 对多领域混合训练数据更友好
```

### 5.5 超参数 alpha 的权衡

```text
alpha 太大（>> 0.01）：
  均衡性好，但路由质量下降
  专家被强制均分，无法专业化 → 表达能力退化

alpha 太小（<< 0.001）：
  路由自由，但容易崩溃成少数专家垄断

alpha ≈ 0.01：
  MokioMind 默认值，工程经验的平衡点

alpha = 0（Auxiliary-loss-free）：
  DeepSeek-V3 的探索方向：不用 Aux Loss，
  改用 RL 策略或 bias 动态调整来实现均衡
  （moe.png 中右侧标注了"RL Policy"）
```

---

## 六、训练 vs 推理：两套分发实现

### 6.1 训练阶段：repeat_interleave 方案

```python
# Step 1: 每个 token 复制 K 份
x = x.repeat_interleave(K, dim=0)   # [T, h] → [T*K, h]

y = torch.empty_like(x)

# Step 2: 遍历所有专家，找属于自己的 token
for i, expert in enumerate(self.experts):
    mask = (flat_topk_idx == i)
    expert_out = expert(x[mask])
    if expert_out.shape[0] > 0:
        y[mask] = expert_out.to(y.dtype)
    else:
        # DDP 梯度占位：确保未被选中的专家仍有梯度流
        y[mask] = expert_out.to(y.dtype) + 0 * sum(
            p.sum() for p in expert.parameters()
        )

# Step 3: 还原形状并加权求和
y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
```

**数据流可视化（T=3 tokens，K=2，N=4 专家）**：

```text
原始 x:         [t0,  t1,  t2]                    shape: [3, h]
                  ↓
repeat(K=2):    [t0, t0, t1, t1, t2, t2]          shape: [6, h]

flat_topk_idx:  [ 2,  3,  0,  2,  1,  3]
                  ↑   ↑   ↑   ↑   ↑   ↑
                  t0  t0  t1  t1  t2  t2
                  的  的  的  的  的  的
                  第  第  第  第  第  第
                  1   2   1   2   1   2
                  选  选  选  选  选  选

Expert 0: mask=[F,F,T,F,F,F] → 处理 t1 的第1选
Expert 1: mask=[F,F,F,F,T,F] → 处理 t2 的第1选
Expert 2: mask=[T,F,F,T,F,F] → 处理 t0 的第1选 + t1 的第2选
Expert 3: mask=[F,T,F,F,F,T] → 处理 t0 的第2选 + t2 的第2选

加权求和：
  y[t0] = g[t0,2]*Expert2(t0) + g[t0,3]*Expert3(t0)
  y[t1] = g[t1,0]*Expert0(t1) + g[t1,2]*Expert2(t1)
  y[t2] = g[t2,1]*Expert1(t2) + g[t2,3]*Expert3(t2)
```

**训练方案的优缺点**：

```text
优点：
  - 代码简单，逻辑清晰
  - 对自动微分友好，梯度能正确流过每个专家
  - GPU 上可并行执行不同专家的前向

缺点：
  - repeat_interleave 产生冗余数据，内存翻倍
  - 每个专家用 mask 索引，批次碎片化（GPU 利用率不最优）
```

---

### 6.2 推理阶段：分拣（Sort & Dispatch）方案

```python
@torch.no_grad()
def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
    expert_cache = torch.zeros_like(x)

    # Step 1: 按专家索引排序（分拣）
    idxs = flat_expert_indices.argsort()
    # 排序后变成: [E0的token..., E1的token..., E2的token..., ...]

    # Step 2: 统计每个专家分到的 token 数（累积和 = 各专家的切分点）
    tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)

    # Step 3: 计算每个 slot 对应的原始 token 位置
    token_idxs = idxs // self.config.num_experts_per_tok

    # Step 4: 逐专家批量处理
    for i, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
        if start_idx == end_idx:
            continue   # 该专家没有分到任何 token，跳过

        expert = self.experts[i]
        exp_token_idx = token_idxs[start_idx:end_idx]

        expert_tokens = x[exp_token_idx]                    # 取出该专家负责的所有 token
        expert_out = expert(expert_tokens).to(expert_cache.dtype)
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # 加权

        # Step 5: scatter_add_ 将结果累加到对应位置
        expert_cache.scatter_add_(
            0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
        )

    return expert_cache
```

**分拣方案可视化（T=3，K=2，N=4）**：

```text
flat_expert_indices: [2, 3, 0, 2, 1, 3]
                      ↑  ↑  ↑  ↑  ↑  ↑
                      t0 t0 t1 t1 t2 t2
                      选  选  选  选  选  选
                      1   2   1   2   1   2

↓  argsort  ↓

idxs:      [2, 4, 5, 0, 3, 1]     （按专家索引从小到大排序）
专家顺序:   [E0, E1, E2, E2, E3, E3]

bincount:  [1,  1,  2,  2]         (E0:1个, E1:1个, E2:2个, E3:2个)
cumsum:    [1,  2,  4,  6]         (切分点)

Expert 0: idxs[0:1]=[2] → token_idxs=[2//2=1=t1] → 处理 t1，批量1个
Expert 1: idxs[1:2]=[4] → token_idxs=[4//2=2=t2] → 处理 t2，批量1个
Expert 2: idxs[2:4]=[5,0] → token_idxs=[5//2=2=t2, 0//2=0=t0] → 处理 [t2,t0]，批量2个
Expert 3: idxs[4:6]=[3,1] → token_idxs=[3//2=1=t1, 1//2=0=t0] → 处理 [t1,t0]，批量2个

scatter_add_：
  Expert2(t2) * w → cache[t2] += ...
  Expert2(t0) * w → cache[t0] += ...
  Expert3(t1) * w → cache[t1] += ...
  Expert3(t0) * w → cache[t0] += ...   （t0 的第2个专家结果累加进来）
```

**为什么推理用这种方案？**

```text
训练时：
  多个样本并行，T 很大（如 512），内存翻倍可以接受
  关注吞吐量，GPU 利用率优先

推理时（尤其自回归生成阶段）：
  T 通常 = 1（单步生成一个 token），内存极宝贵
  repeat_interleave 产生冗余，浪费内存带宽

分拣方案的优点：
  - 不复制 token，内存高效
  - 每个专家只调用一次 forward（批量），充分利用 GPU tensor core
  - 连续内存访问，cache 友好
```

---

## 七、DeepSeek MoE 三大架构创新

moe.png 底部三张对比图展示了架构演进过程：

### (a) 传统 Top-2 路由（Conventional）

```text
        Output Hidden
           ↑
     [⊗]     [⊗]
      |        |
   Expert_1  Expert_2    (只有2个专家被选，N个专家总数较少)
       \      /
        Router  K=2
           ↑
        Input Hidden

特点：
  N 个大专家，选 K=2 个激活
  专家粒度粗，每个专家参数量大
  组合方式有限：C(N, 2) 种
```

### (b) + Fine-Grained Expert Segmentation（细粒度专家分割）

```text
        Output Hidden
           ↑
  [⊗] [⊗] [⊗] [⊗]     (选 K=4 个，但每个小很多)
   |    |    |    |
  E1   E2   E3  E4    ...   E_2N    (总数翻倍，每个专家缩小一半)
              Router  K=4
           ↑
        Input Hidden

创新思路：
  将每个大专家拆分成 m 个小专家（参数总量不变）
  总专家数: N → mN
  每次激活数: K → mK

为什么更好？
  组合方式: C(N, K) → C(mN, mK)   指数级增长
  每个 token 能组合更丰富、更专业的知识
  例如：N=8, K=2 → C(8,2)=28 种
        mN=64, mK=4 → C(64,4)=635,376 种  ！
```

### (c) + Shared Expert Isolation（共享专家隔离）—— DeepSeekMoE 最终形态

```text
        Output Hidden
              ↑
   [⊕]────────────────[⊕]
    |                   |
   路由专家 (被选中的)    共享专家 (始终激活)
    ↑                   ↑         ↑  绿色方块
   [⊗][⊗]             [E_1][E_2]...[E_Ns]
    |  |
   E_i E_j
       Router  K=3（路由专家部分）
           ↑
        Input Hidden

最终公式:
  h_t = Σ g_{i,t} · FFN_i(u_t)   ← 路由专家（稀疏）
      + Σ FFN_shared_j(u_t)       ← 共享专家（密集）
      + u_t                        ← 残差连接
```

---

## 八、共享专家：路由的兜底机制

### 8.1 为什么需要共享专家？

```text
问题：路由专家在互相竞争专业化时，可能"遗忘"通用知识

场景：
  路由专家逐渐分工：
    Expert_A → 数学推理
    Expert_B → 代码生成
    Expert_C → 中文语言
    Expert_D → 英文语言

  但通用能力（语法、格式、基础逻辑）无人负责
  → 某些 token 被错误路由时，输出质量大幅下降

解决：引入共享专家，始终处理所有 token，提供通用基础能力
```

### 8.2 代码实现

```python
# MoEFeedForward.forward() 末尾
if self.config.n_shared_experts > 0:
    for expert in self.shared_experts:
        y = y + expert(identity)
#                      ↑ identity 是 MoE 层的原始输入 x，不是路由后的结果！
```

**两个关键细节**：

```text
细节1：使用原始输入 identity，不是路由专家的输出
  → 共享专家与路由专家并行计算，结果直接相加
  → 避免梯度依赖链，训练更稳定

细节2：共享专家不经过门控加权，直接加（权重恒为 1）
  → 共享知识无条件贡献，不受路由器质量影响
  → 保证信息通路始终畅通
```

### 8.3 类比理解

```text
共享专家 = 公司基础设施部门（网络、HR、财务）
           人人都用，提供通用服务

路由专家 = 各业务部门（技术、销售、产品）
           按需调用，提供专业服务

两者分工明确，互不干扰，共同服务于每个 token
```

---

## 九、整体架构总结

### 9.1 完整数据流

```text
输入 token u_t
    │
    ▼
┌──────────────────────────────────────────────────┐
│  MoEGate（路由器）                                │
│                                                  │
│  1. logits  = u_t @ W_gate.T    [T, N]           │
│  2. scores  = softmax(logits)   [T, N]           │
│  3. topk_weight, topk_idx = TopK(scores, K)      │
│  4. topk_weight /= sum(topk_weight)  （归一化）  │
│  5. [训练] aux_loss = α·Σ P_i·f_i               │
└──────────────────────────────────────────────────┘
    │ topk_idx, topk_weight
    ▼
┌──────────────────────────────────────────────────┐
│  Dispatch & Compute（分发与计算）                 │
│                                                  │
│  [训练] repeat_interleave → 逐专家 mask 过滤      │
│  [推理] argsort + bincount → 分拣 → 批量处理      │
│                                                  │
│  被选中的 K 个专家：FFN_i(u_t)                    │
│  未被选中的 N-K 个专家：跳过（稀疏性！）           │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Aggregate（聚合）                                │
│                                                  │
│  y_routed = Σ g_{i,t} · FFN_i(u_t)  （路由专家）│
│  y_shared = Σ FFN_shared_j(u_t)     （共享专家） │
│  h_t = y_routed + y_shared + u_t    （残差连接） │
└──────────────────────────────────────────────────┘
    │
    ▼
  输出 h_t  +  aux_loss（仅训练时加入总损失）
```

### 9.2 MoE 路由设计的核心张力

| 目标 | 机制 | 代价 |
|------|------|------|
| 专业化分工 | 竞争性 softmax + TopK 硬选择 | 路由不可微 |
| 负载均衡 | Auxiliary Loss（软约束） | 损失一点路由质量 |
| 计算效率 | 稀疏激活（只选 K 个） | 需要专门的 dispatch 实现 |
| 知识完整性 | 共享专家机制 | 增加少量固定计算 |
| 细粒度组合 | Fine-grained 拆分 | 增加专家总数 |

### 9.3 各代 MoE 演进对比

| 特性 | Switch Transformer | Mixtral 8×7B | DeepSeek-V2/V3 |
|------|--------------------|--------------|-----------------|
| 路由方式 | Top-1 | Top-2 | Top-K（细粒度）|
| 负载均衡 | Aux Loss（Token-level）| Aux Loss | Aux Loss / Aux-loss-free |
| 共享专家 | ❌ | ❌ | ✅ |
| 细粒度分割 | ❌ | ❌ | ✅ |
| 负载均衡策略 | 辅助损失 | 辅助损失 | RL Policy（V3探索）|

---

## 附录：MokioMind 配置参数速查

```python
# MokioMindConfig MoE 相关参数
use_moe: bool = False              # 是否启用 MoE
num_experts_per_tok: int = 2       # K：每 token 激活的专家数
n_routed_experts: int = 4          # N：路由专家总数
n_shared_experts: int = 1          # 共享专家数（始终激活）
scoring_func: str = "softmax"      # 门控评分函数
aux_loss_alpha: float = 0.01       # α：辅助损失系数
seq_aux: bool = True               # 是否使用 sequence-level 辅助损失
norm_topk_prob: bool = True        # 是否归一化 Top-K 权重
```

**计算量分析（默认配置）**：

```text
设 d = hidden_size, d_m = intermediate_size

路由专家参数量：N × 3d·d_m      = 4 × 3d·d_m
共享专家参数量：Ns × 3d·d_m     = 1 × 3d·d_m
总参数量：                        5 × 3d·d_m    ← 是单 FFN 的 5 倍

每 token 路由专家计算：K × 3d·d_m  = 2 × 3d·d_m
每 token 共享专家计算：Ns × 3d·d_m = 1 × 3d·d_m
每 token 总计算量：                  3 × 3d·d_m  ← 是单 FFN 的 3 倍

结论：用 3× 计算量，撬动 5× 参数空间
```
