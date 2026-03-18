# model/gpt.py
#
# modded-nanoGPT style architecture:
#   - No biases
#   - RMSNorm
#   - RoPE positional embeddings
#   - QK-Norm
#   - ReLU² activation
#   - FlashAttention (via scaled_dot_product_attention)
#   - Embedding skip connections to every transformer block

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0


# ------------------------------------------------------------------ #
# Building blocks                                                     #
# ------------------------------------------------------------------ #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_freqs_cis(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(q, k, freqs_cis):
    # q, k: [B, T, n_head, head_dim]
    # freqs_cis: [T, head_dim/2] complex
    freqs = freqs_cis[:q.shape[1]].unsqueeze(0).unsqueeze(2)  # [1, T, 1, head_dim/2]

    def rotate(x):
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        x_c = torch.view_as_complex(x_)
        x_r = torch.view_as_real(x_c * freqs)  # broadcast: [B, T, n_head, head_dim/2]
        return x_r.flatten(-2).type_as(x)

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # QK-Norm
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # FlashAttention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square()   # ReLU²
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


# ------------------------------------------------------------------ #
# GPT                                                                 #
# ------------------------------------------------------------------ #

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Embedding skip connections: one linear per block (no bias)
        self.embed_skip = nn.ModuleList([
            nn.Linear(config.n_embd, config.n_embd, bias=False)
            for _ in range(config.n_layer)
        ])

        # Weight tying: embedding and lm_head share weights
        self.wte.weight = self.lm_head.weight

        # RoPE frequencies (buffer, not a parameter)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.n_embd // config.n_head, config.block_size),
        )

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size

        x = self.wte(idx)           # [B, T, n_embd]
        x_embed = x                 # save for skip connections
        freqs_cis = self.freqs_cis[:T]

        for i, block in enumerate(self.blocks):
            x = x + self.embed_skip[i](x_embed)   # embedding skip connection
            x = block(x, freqs_cis)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
