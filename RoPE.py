import torch


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 所需的 cos 和 sin 值。

    RoPE 将 d 维向量拆分为 d/2 个二维子空间，每个子空间以不同的频率旋转。
    频率公式: θ_i = θ_base^(-2i/d)，其中 i 是子空间索引。

    Args:
        dim:     每个注意力头的维度 (head_dim)，例如 64 或 128
        seq_len: 序列长度，即需要预计算的最大位置数
        theta:   基础频率参数，默认 10000.0（LLaMA 3 用 500000.0）

    Returns:
        (cos, sin): 形状均为 [seq_len, dim//2]
                    cos[m][i] = cos(m · θ_i)，sin[m][i] = sin(m · θ_i)
    """
    # Step 1: 计算每个子空间的基础频率 θ_i = θ_base^(-2i/d)
    #
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ..., dim-2]，即 2i 的值
    # [: (dim // 2)] 是冗余截断（arange 已经只生成 dim//2 个元素），确保安全
    # .float() / dim 得到指数 2i/d
    # theta ** (...) 得到 θ^(2i/d)
    # 1.0 / (...) 取倒数得到 θ^(-2i/d) = θ_i
    #
    # 示例 (dim=64):
    #   i=0:  θ_0  = 10000^(-0/64)  = 1.0      ← 最高频，捕获近距离关系
    #   i=15: θ_15 = 10000^(-30/64) ≈ 0.003    ← 中频
    #   i=31: θ_31 = 10000^(-62/64) ≈ 0.00013  ← 最低频，捕获远距离关系
    # dim//2:
    base_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # base_freqs.shape = [dim//2]  例如 [32]

    # Step 2: 生成位置索引 [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len)
    # t.shape = [seq_len]  例如 [10]

    # Step 3: 外积计算每个 (位置m, 子空间i) 对应的旋转角度 = m · θ_i
    #
    # outer(t, base_freqs)[m][i] = t[m] * base_freqs[i] = m · θ_i
    #
    # 这个矩阵的每一行是一个位置的所有旋转角度，
    # 每一列是一个频率在所有位置上的角度变化：
    #   - 高频列(i=0):  [0, 1, 2, 3, ...]    角度变化快
    #   - 低频列(i=31): [0, 0.00013, ...]     角度变化极慢
    freqs = torch.outer(t, base_freqs).float()
    # freqs.shape = [seq_len, dim//2]  例如 [10, 32]

    # Step 4: 返回预计算的 cos 和 sin 值，供 apply_rotary_emb 使用
    return torch.cos(freqs), torch.sin(freqs)
    # 返回 shape 均为 [seq_len, dim//2]


def rotate_half(x):
    """将向量的前半和后半交叉取负，构造旋转的 sin 项系数。

    这是 "半拆分 (half-split)" 实现风格（GPT-NeoX 风格）：
    - 子空间 i 的配对维度是 (i, i+d/2)，而非交错式的 (2i, 2i+1)

    输入:  x = [x_1, x_2, ..., x_{d/2},  x_{d/2+1}, ..., x_d]
    输出:      [-x_{d/2+1}, ..., -x_d,    x_1, ..., x_{d/2}]

    这样做的目的是为了让 x*cos + rotate_half(x)*sin 等价于旋转矩阵乘法。

    Args:
        x: 形状为 [..., dim] 的张量 (最后一维是 head_dim)
    """
    # 沿最后一维拆成前半和后半
    x1 = x[..., : x.shape[-1] // 2]  # 前半: 维度 [0, d/2)
    x2 = x[..., x.shape[-1] // 2 :]  # 后半: 维度 [d/2, d)

    # 后半取负放前面，前半放后面
    # 这样 rotate_half(x) * sin 就能产生旋转矩阵中的交叉项
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    """对输入张量应用旋转位置编码。

    数学等价于：对每个子空间 i，执行二维旋转
        [x_i, x_{i+d/2}] → [x_i·cos(mθ_i) - x_{i+d/2}·sin(mθ_i),
                             x_{i+d/2}·cos(mθ_i) + x_i·sin(mθ_i)]

    用向量化公式实现：  RoPE(x) = x ⊙ cos + rotate_half(x) ⊙ sin
    其中 ⊙ 表示逐元素乘法。

    Args:
        x:   形状 [batch, n_heads, seq_len, head_dim]，query 或 key 张量
        cos: 形状 [seq_len, head_dim//2]，预计算的 cos 值
        sin: 形状 [seq_len, head_dim//2]，预计算的 sin 值
    """
    # cos/sin 原始形状 [seq_len, dim//2]，需要扩展到 [1, 1, seq_len, dim]
    # 才能和 x [batch, n_heads, seq_len, dim] 进行广播运算

    # 复制一份拼接: [seq_len, dim//2] → [seq_len, dim]
    # 因为 half-split 风格下，前半和后半维度用的是同一组频率的 cos/sin
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(1)
    # cos/sin 形状变为 [1, 1, seq_len, dim]，通过广播匹配 batch 和 head 维度

    # 核心旋转公式: x * cos + rotate_half(x) * sin
    #
    # 展开验证 (以子空间 i 的两个配对维度为例):
    #   结果[i]     = x[i]·cos(mθ_i)     + (-x[i+d/2])·sin(mθ_i)
    #               = x[i]·cos(mθ_i)     - x[i+d/2]·sin(mθ_i)     ✓ 旋转公式
    #   结果[i+d/2] = x[i+d/2]·cos(mθ_i) + x[i]·sin(mθ_i)         ✓ 旋转公式
    return (x * cos) + (rotate_half(x) * sin)


if __name__ == "__main__":
    head_dim = 64  # 每个注意力头的维度
    seq_len = 10  # 序列长度

    # 模拟 query 张量: [batch=2, n_heads=8, seq_len=10, head_dim=64]
    q = torch.randn(2, 8, seq_len, head_dim)

    # 预计算位置编码的 cos/sin
    cos, sin = precompute_freqs_cis(head_dim, seq_len)

    # 对 query 应用旋转位置编码
    q_rotated = apply_rotary_emb(q, cos, sin)

    print(q.shape)  # torch.Size([2, 8, 10, 64])
    print(q_rotated.shape)  # torch.Size([2, 8, 10, 64]) — 形状不变，只是旋转了值
