# Multi-head Latent Attention implementation
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    d_model = 512
    n_head = 16
    kv_d_compressed = 96
    nope_head_dim = 48
    q_d_compressed = 128
    d_rope = 16
    max_batch_size = 8
    max_seq_len = 2048


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

        self.dkv = nn.Linear(args.d_model, args.kv_d_compressed + self.d_rope) # Compressed key/value downprojection
        self.dq = nn.Linear(args.d_model, args.q_d_compressed) # Compressed Query downprojection

        self.uk = nn.Linear(torch.zeros(self.n_head, self.nope_head_dim, self.kv_d_compressed)) # projection of compressed kv's to headwise key representations
        self.uk = nn.Sequential(
            nn.Linear(self.n_head*(self.nope_head_dim + self.kv_d_compressed)),
        )
        self.uv = nn.Parameter(torch.zeros(self.n_head, self.nope_head_dim, self.kv_d_compressed)) # projection of compressed kv's to headwise vaule representation

        self.uq = nn.Parameter(torch.zeros(args.q_d_compressed, self.nope_head_dim * self.n_head))

        self.kv_norm = nn.RMSNorm(self.kv_d_compressed)
        self.q_norm = nn.RMSNorm(self.q_d_compressed)

        self.wo = nn.Parameter(torch.zeros(self.n_head*self.nope_head_dim, self.d_model)) # Final out projection back to d_model

        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_d_compressed))
        self.register_buffer("kv_rope_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.q_d_compressed))

    def forward(self, x: torch.Tensor, start_pos: int, mask: torch.Tensor = None):
        _, seq_len, _ = x.shape
        
        # Compressed Key/Value flow, (Important: kv caching/calculation is not performed head-wise):
        kv_c = self.dkv(x[start_pos:])  # Down-project into compressed key/value representation (B, seq_len', (kv_d_compressed + d_rope))
        kv_c_nope, kv_c_rope = torch.split(
            kv_c, [self.kv_d_compressed, self.d_rope], -1
        ) # Split into no positional embedding and positional embedding parts
        self.kv_cache[:, start_pos:seq_len] = self.kv_norm(kv_c) # cache compressed key/value representation

        kv_c_cached = self.kv_cache[:start_pos] # load cached compressed key/value representation
        kv_c = torch.concat(kv_c_cached, kv_c_nope) # join newly calculated key/value representations with cached ones (B, seq_len, kv_d_compressed)

        kv_c_rope_new = apply_rotary_emb(kv_c_rope, rope_position_list) # apply positional embedding to new 
        kv_c_rope_cached = self.kv_rope_cache[:start_pos]
        kv_c_rope = torch.concat(kv_c_rope_new, kv_c_rope_cached) # (B, seq_len, d_rope)

        self.kv_rope_cache[:, start_pos:seq_len] = kv_c_rope_new # cache key/value positional embedding

        # Query Flow:
        rope_position_list = list(range(start_pos, seq_len)) # Positions for positional embedding calcualtion
        q_c = self.dq(x[start_pos:]) # (B, seq_len', n_heads), Queries only needed for non-processed tokens
        q_c_norm = self.q_norm(q_c) # Perform Normalization in compressed space
        q = self.uq(q_c_norm) # Query up-projection (B, seq_len', n_head * (nope_head_dim + d_rope))
        q = q.view(-1, -1, self.n_head, self.nope_head_dim + self.d_rope) # reshape into (B, seq_len', n_head, (nope_head_dim + d_rope))
        q_head_nope, q_head_rope = torch.split(
            q, [self.nope_head_dim, self.d_rope], -1
        ) # Split into no positional embedding and positional embedding parts
        q_head_rope = apply_rotary_emb(q_head_rope, rope_position_list) # apply rotatonal positional embedding 

        # Actual Attention Calculation:
        # Linear Algebra Trick: a @ (b @ c) = (a @ b) @ c, key up-projection step can be "skipped"
        q_nope_proj = torch.einsum("bshd,hdc->bshc", q_head_nope, self.uk) # (B, seq_len', n_head, kv_d_compressed)

        # s represents seq_len' which is the amount of queries, t is the amount kv pairs we want to attend to
        att_nope_scores = torch.einsum("bshc,btc->bsht", q_nope_proj, kv_c) # calculate head-wise attention scores 
        att_rope_scores = torch.einsum("bshc,btc->bsht", q_head_rope, kv_c_rope) # calculate positional embedding attention scores
        att_scores = att_nope_scores + att_rope_scores
        att_scores += mask

        # calculate head-wise values
        values = self.uv(kv_c)
        head_wise_weighted_values = att_nope_scores @ values
        weighted_values = head_wise_weighted_values.squeeze(-2)


args = ModelArgs()
attention = MultiHeadLatentAttention(args=args)

test_tensor = torch.rand((5, 10, 512))
attention(test_tensor, 0)
