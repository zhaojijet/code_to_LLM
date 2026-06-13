import torch


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算频率 theta_i = (10000^(-2i/dim))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float()) / dim)

    t = torch.arange(seq_len)

    freqs = torch.outer(t, freqs).float()

    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(1)

    return (x * cos) + (rotate_half(x) * sin)


if __name__ == "__main__":
    head_dim = 64
    seq_len = 10

    q = torch.randn(2, 8, seq_len, head_dim)

    cos, sin = precompute_freqs_cis(head_dim, seq_len)

    q_rotated = apply_rotary_emb(q, cos, sin)

    print(q.shape)
    print(q_rotated.shape)
