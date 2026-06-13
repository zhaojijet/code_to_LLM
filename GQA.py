import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads

        self.w_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def repeat_kv(self, x, n_rep):
        if n_rep == 1:
            return x

        batch_size, n_kv_heads, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim)
            .reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.w_k(x)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.w_v(x)
            .view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        k = self.repeat_kv(k, self.group_size)
        v = self.repeat_kv(v, self.group_size)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-infinity"))

        attn_prob = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_prob, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.w_o(output)


def generate_casual_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


if __name__ == "__main__":

    model = GroupQueryAttention(d_model=128, n_heads=8, n_kv_heads=2)

    x = torch.randn(1, 4, 128)

    out = model(x)

    print(model)
    print(x.shape)
    print(out.shape)
