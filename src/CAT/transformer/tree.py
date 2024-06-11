import math

import torch
from torch import nn
import torch.nn.functional as F

from .common import *

class TreeAttention(nn.Module):
    def __init__(self, args: AttentionArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.wq = nn.Linear(self.dim, self.dim, bias=args.bias)
        self.wk = nn.Linear(self.dim, self.dim, bias=args.bias)
        self.wv = nn.Linear(self.dim, self.dim, bias=args.bias)
        self.wo = nn.Linear(self.dim, self.dim, bias=args.bias)

        theta = 10000.0
        self.register_buffer('theta_of_dims', 1.0 / (theta ** (torch.arange(0, self.head_dim, 8)[: (self.head_dim // 8)].float() / self.head_dim)), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x   : (batch_size, seq_len, dim)
        # pos : (batch_size, seq_len, 4)
        # mask: (batch_size, seq_len, seq_len)

        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # (batch_size, seq_len, dim)

        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)

        freqs = torch.einsum('bij,k->bikj', pos, self.theta_of_dims)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        # (batch_size, seq_len, head_dim // 8, 4)
        freqs_cis = freqs_cis.view(batch_size, seq_len, 1, -1)
        # (batch_size, seq_len, 1, head_dim // 2)
        xq = xq.reshape(*xq.shape[:-1], -1, 2)
        xk = xk.reshape(*xk.shape[:-1], -1, 2)
        # (batch_size, seq_len, n_heads, head_dim // 2, 2)
        xq = torch.view_as_complex(xq)
        xk = torch.view_as_complex(xk)
        # (batch_size, seq_len, n_heads, head_dim // 2)
        xq = torch.view_as_real(xq * freqs_cis).flatten(3)
        xk = torch.view_as_real(xk * freqs_cis).flatten(3)
        # (batch_size, seq_len, n_heads, head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # (batch_size, n_heads, seq_len, head_dim)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # print(f"scores: {scores[:,0,:,:]}")
        scores = scores + mask.unsqueeze(1)
        # print(f"scores + mask: {scores[:,0,:,:]}")
        scores = F.softmax(scores, dim=-1)
        # print(f"softmax(scores): {scores[:,0,:,:]}")
        # scores = scores.nan_to_num()
        # (batch_size, n_heads, seq_len, seq_len)
        output = torch.matmul(scores, xv)
        # (batch_size, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # (batch_size, seq_len, dim)

        return self.wo(output)

class TreeTransformerBlock(nn.Module):
    def __init__(self, attn_args: AttentionArgs, ffn_args: FeedForwardArgs) -> None:
        super().__init__()
        self.attn = TreeAttention(attn_args)
        self.ffn = FeedForward(ffn_args)
        self.ln1 = nn.LayerNorm(attn_args.dim)
        self.ln2 = nn.LayerNorm(attn_args.dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor):
        h = x + self.attn(self.ln1(x), pos, mask)
        return h + self.ffn(self.ln2(h))

class TreeEncoder(nn.Module):
    def __init__(self, args: TransformerArgs) -> None:
        super().__init__()
        self.embed = nn.Linear(args.feature_dim, args.embedding_dim, bias=args.bias)
        self.blocks = nn.ModuleList([TreeTransformerBlock(args.get_attn_args(), args.get_ffn_args()) for _ in range(args.n_layers)])

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, pos, mask)
        return x