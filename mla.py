# Multi-head Latent Attention implementation
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelArgs:
    d_model = 512
    n_head = 8
    d_compressed = 64
    d_rope = 16


def apply_rotary_emb(x, position, args: ModelArgs):
    x1, x2 = x[..., 0::2], x[..., 1::2]

    theta = 10000 ** (-2 * torch.arange(args.d_model // 2) / args.d_model)
    angles = position[:, None] * theta[None, :]

    cos, sin = torch.cos(angles), torch.sin(angles)

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        self.d_model = args.d_model
        self.n_head = args.n_head
        self.head_dim = args.d_model / args.n_head
        self.d_compressed = args.d_compressed

        self.dkv = nn.Parameter(torch.zeros(args.d_model, args.d_compressed))
        self.dq = nn.Parameter(torch.zeros(args.d_model, args.d_compressed))

        self.uk = nn.Embedding(args.d_compressed, args.d_model)
        self.uq = nn.Embedding()

        # TODO: finish kv cache setup
        self.register_buffer("kv_cache")
        self.register_buffer("kv_rope_cache")

    def forward(self, x: torch.Tensor, start_pos: int):
        kv_c_new = self.dkv(x[start_pos:])  # (B, seq_len, d_compressed)
        # TODO: RoPE for kv
        q_c = self.dq(x[start_pos:])
        q_c_main, q_c_rope = torch.split(
            q_c, [self.d_model - self.d_compressed, self.d_compressed], -1
        )
        q_c_rope = apply_rotary_emb(q_c_rope)  # TODO

        kv_c_cached = self.kv_cache[:start_pos]
        kv_c = torch.concat(kv_c_new, kv_c_cached)

        # TODO: Finish main Attention calculation part
        q_k_c = q_c @ self.uk
        q_k_c_head = torch.reshape(q_k_c, (-1, -1, self.n_head, self.head_dim))
        kv_c_head = torch.reshape(kv_c, (-1, -1, self.n_head, self.head_dim))
        att_scores = q_k_c @ torch.transpose(kv_c)

        kv_c_head = torch.reshape((self.n_head, self.head_dim))


args = ModelArgs()
MultiHeadLatentAttention(args=args)
