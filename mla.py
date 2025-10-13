# Multi-head Latent Attention implementation
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    d_model = 512
    n_head = 16
    kv_d_compressed = 64
    nope_head_dim = 48
    q_d_compressed = 128
    d_rope = 16


def apply_rotary_emb(x, position, args: ModelArgs):
    x1, x2 = x[..., 0::2], x[..., 1::2]

    theta = 10000 ** (-2 * torch.arange(args.d_model // 2) / args.d_model)
    angles = position[:, None] * theta[None, :]

    cos, sin = torch.cos(angles), torch.sin(angles)

    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.n_head = args.n_head
        self.nope_head_dim = args.nope_head_dim
        self.kv_d_compressed = args.kv_d_compressed
        self.q_d_compressed = args.q_d_compressed
        self.d_rope = args.d_rope

        self.dkv = nn.Parameter(torch.zeros(args.d_model, args.kv_d_compressed)) # Compressed key/value downprojection
        self.dq = nn.Parameter(torch.zeros(args.d_model, args.q_d_compressed))

        self.uk = nn.Parameter(torch.zeros(args.d_compressed, args.d_model)) # Correct dims?
        self.uq = nn.Parameter(torch.zeros(args.q_d_compressed, self.nope_head_dim * self.n_head)) # TODO: Correct head dim used here?

        self.kv_norm = nn.RMSNorm() # TODO: Finish normalization setup
        self.q_norm = nn.RMSNorm()

        self.wo = nn.Parameter(torch.zeros(self.n_head*self.nope_head_dim, self.d_model)) # Final out projection back to d_model

        # TODO: finish kv cache setup
        self.register_buffer("kv_cache")
        self.register_buffer("kv_rope_cache")

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor]):
        _, seq_len, _ = x.shape
        
        # Compressed Key/Value flow:
        kv_c = self.dkv(x[start_pos:])  # (B, seq_len, d_compressed)
        kv_c_main, kv_c_rope = torch.split(
            kv_c, [self.d_model - self.d_compressed, self.d_compressed], -1
        )
        self.kv_cache[:, start_pos:seq_len] = self.kv_norm(kv_c)

        # Query Flow:
        rope_position_list = list(range(start_pos, seq_len))
        q_c = self.dq(x[start_pos:]) # (B, seq_len, n_heads)
        self.q_norm(q_c) # Perform Normalization in compressed space
        q = self.uq(q_c)
        q = q.view(-1, -1, self.n_head, self.nope_head_dim + self.d_rope) # TODO: check if correct head dim is being used
        q_c_main, q_c_rope = torch.split(
            q_c, [self.nope_head_dim, self.d_rope], -1
        ) 
        q_c_rope = apply_rotary_emb(q_c_rope, rope_position_list)

        kv_c_cached = self.kv_cache[:start_pos]
        kv_c = torch.concat(kv_c_main, kv_c_cached)

        kv_c_rope_new = apply_rotary_emb(kv_c_rope, rope_position_list)
        kv_c_rope_cached = self.kv_rope_cache[:start_pos]
        kv_c_rope = torch.concat(kv_c_rope_new, kv_c_rope_cached)

        self.kv_rope_cache[:, start_pos:seq_len] = kv_c_rope_new

        q_k_c = q_c_main @ self.uk
        q_k_c_head = torch.reshape(q_k_c, (-1, -1, self.n_head, self.head_dim)) # (B, seq_len, n_head, head_dim), TODO: kv splitting
        kv_c_head = torch.reshape(kv_c, (-1, -1, self.n_head, self.head_dim))
        att_scores = q_k_c_head @ torch.transpose(kv_c_head) + q_c_rope @ torch.transpose(kv_c_rope) # TODO: fix, utilize einsum
        att_scores += mask



args = ModelArgs()
attention = MultiHeadLatentAttention(args=args)

test_tensor = torch.rand((5, 10, 512))
attention(test_tensor, 0)
