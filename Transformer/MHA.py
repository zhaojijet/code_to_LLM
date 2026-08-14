import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        self.head_dim = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.w_k(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.w_v(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_scores = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.head_dim))

        if mask is not None:
            # mask为0的位置置换为极小无穷值
            attn_scores = attn_scores.masked_fill(mask == 0, float("-infinity"))

        attn_scores = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_scores, v)

        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        return self.w_o(output)


def generate_casual_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


if __name__ == "__main__":
    d_model = 128
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)

    x = torch.randn(2, 5, 128)

    mask = generate_casual_mask(5)
    print(mask)

    output = mha(x, mask=mask)

    print(x.shape)
    print(output.shape)
    print(mask.shape)
    print(output)
